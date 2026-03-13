import argparse
from fastchat.utils import str_to_torch_dtype

from eagle.model.ea_model import EaModel
from eagle.model.kv_cache import initialize_past_key_values
from eagle.model.utils import *
from eagle.model.choices import *

import json
import os
import time
import torch
import numpy as np

from fastchat.llm_judge.common import load_questions
from fastchat.model import get_conversation_template
from tqdm import tqdm


def run_eval(
        model,
        tokenizer,
        forward_func,
        question_file,
        question_begin,
        question_end,
        max_new_tokens,
        num_choices,
        num_gpus_per_model,
        num_gpus_total,
        **kwargs,
):
    print("Entering run_eval")
    
    questions = load_questions(question_file, question_begin, question_end)
    print(f"Loaded {len(questions)} questions from {question_file}")
    
    output_file = question_file.replace('.jsonl', '_with_assist.jsonl')
    
    get_model_answers(
        model,
        tokenizer,
        forward_func,
        questions,
        output_file,
        max_new_tokens,
        num_choices,
        **kwargs,
    )


@torch.inference_mode()
def get_model_answers(
        model,
        tokenizer,
        forward_func,
        questions,
        output_file,
        max_new_tokens,
        num_choices,
        **kwargs,
):
    model.eval()

    if questions:
        first_question = questions[0]
        conv = get_conversation_template("vicuna")
        for j in range(len(first_question["conversations"])):
            role, qs = first_question["conversations"][j]
            if role == "ASSISTANT":
                continue
            conv.append_message(role, qs)
            conv.append_message("ASSISTANT", None)
            conv.stop_str = "</s>"
            prompt = conv.get_prompt()
            inputs = tokenizer([prompt], return_tensors="pt", max_length=2000, truncation=True).to("cuda")
            try:
                _ = forward_func(
                    inputs,
                    model,
                    tokenizer,
                    max_new_tokens,
                    **kwargs,
                )
            except Exception as e:
                print(f"Warm up error: {e}")

    accept_lengths_tree = []
    for question in tqdm(questions):
        choices = []
        for i in range(num_choices):
            cur_accept_lengths_tree = []
            torch.manual_seed(i)
            conv = get_conversation_template("vicuna")
            turns = []
            steps = []
            new_tokens = []
            wall_time = []
            for j in range(len(question["conversations"])):
                role, qs = question["conversations"][j]
                if role == "ASSISTANT":
                    continue
                conv.append_message(role, qs)
                conv.append_message("ASSISTANT", None)
                conv.stop_str = "</s>"
                prompt = conv.get_prompt()

                inputs = tokenizer([prompt], return_tensors="pt", max_length=2000, truncation=True).to("cuda")
                input_ids = inputs.input_ids
                try:
                    torch.cuda.synchronize()
                    start_time = time.time()
                    output_ids, new_token, step, accept_length_tree = forward_func(
                        inputs,
                        model,
                        tokenizer,
                        max_new_tokens,
                        **kwargs,
                    )
                    torch.cuda.synchronize()
                    total_time = time.time() - start_time
                    accept_lengths_tree.extend(accept_length_tree)
                    output_ids = output_ids[0][len(input_ids[0]):]

                    if conv.stop_token_ids:
                        stop_token_ids_index = [
                            i
                            for i, id in enumerate(output_ids)
                            if id in conv.stop_token_ids
                        ]
                        if len(stop_token_ids_index) > 0:
                            output_ids = output_ids[: stop_token_ids_index[0]]

                    output = tokenizer.decode(
                        output_ids,
                        spaces_between_special_tokens=False,
                    )
                    if conv.stop_str and output.find(conv.stop_str) > 0:
                        output = output[: output.find(conv.stop_str)]
                    for special_token in tokenizer.special_tokens_map.values():
                        if isinstance(special_token, list):
                            for special_tok in special_token:
                                output = output.replace(special_tok, "")
                        else:
                            output = output.replace(special_token, "")
                    output = output.strip()

                    if conv.name == "xgen" and output.startswith("Assistant:"):
                        output = output.replace("Assistant:", "", 1).strip()
                except RuntimeError as e:
                    import traceback
                    print(f"ERROR question ID: {question['id']}")
                    print(f"Error details: {str(e)}")
                    print("Full traceback:")
                    traceback.print_exc()
                    output = "ERROR"
                    step = 0
                    new_token = 0
                    total_time = 0
                    accept_length_tree = []

                if role == "USER":
                    turns.append(output)
                    steps.append(int(step))
                    new_tokens.append(int(new_token))
                    wall_time.append(total_time)
                    cur_accept_lengths_tree.extend(accept_length_tree)
                    conv.messages[-1][-1] = output
            choices.append(
                {"index": i, "turns": turns, "decoding_steps": steps, "new_tokens": new_tokens, "wall_time": wall_time,
                 "accept_lengths": cur_accept_lengths_tree})

        new_question = question.copy()
        
        if choices and choices[0]["turns"]:
            model_responses = choices[0]["turns"]
            updated_conversations = []
            assistant_turn_idx = 0
            
            for role, content in question["conversations"]:
                if role == "USER":
                    updated_conversations.append([role, content])
                elif role == "ASSISTANT":
                    if assistant_turn_idx < len(model_responses):
                        updated_conversations.append([role, model_responses[assistant_turn_idx]])
                        assistant_turn_idx += 1
                    else:
                        updated_conversations.append([role, content])
            
            new_question["conversations"] = updated_conversations
            
            conv = get_conversation_template("vicuna")
            conv.messages = []
            
            for role, content in updated_conversations:
                conv.append_message(role, content)
            
            new_question["prompt"] = conv.get_prompt()
            new_question["new_tokens"] = choices[0]["new_tokens"]
            new_question["wall_time"] = choices[0]["wall_time"]
            new_question["accept_lengths"] = choices[0]["accept_lengths"]
        
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
        with open(output_file, "a", encoding="utf-8") as fout:
            fout.write(json.dumps(new_question, ensure_ascii=False) + "\n")

    mean_accept = np.mean(accept_lengths_tree) if accept_lengths_tree else 0
    print(f"# Total accept steps: {len(accept_lengths_tree)}")
    print(f"# Accept lengths stats: min={min(accept_lengths_tree) if accept_lengths_tree else 0}, max={max(accept_lengths_tree) if accept_lengths_tree else 0}")
    print("# Mean accepted tokens: ", mean_accept)
    print(f"# Results saved to: {output_file}")


def ea_forward(inputs, model, tokenizer, max_new_tokens, tree_choices=None, logits_processor=None, max_steps=512):
    """
    Returns:
    """
    input_ids = inputs.input_ids
    assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
    input_ids = input_ids.clone()

    model.ea_layer.reset_kv()
    accept_length_list = []

    if hasattr(model, "tree_choices") and model.tree_choices == tree_choices:
        tree_buffers = model.tree_buffers
    else:
        tree_buffers = generate_tree_buffers(
            tree_choices, device=model.base_model.model.layers[-1].self_attn.q_proj.weight.device
        )
        tree_buffers["retrieve_indices_head"] = tree_buffers["retrieve_indices"].to(
            model.base_model.lm_head.weight.device)
    model.tree_buffers = tree_buffers
    model.tree_choices = tree_choices

    if hasattr(model, "past_key_values"):
        past_key_values = model.past_key_values
        past_key_values_data = model.past_key_values_data
        current_length_data = model.current_length_data
        current_length_data.zero_()
    else:
        (
            past_key_values,
            past_key_values_data,
            current_length_data,
        ) = initialize_past_key_values(model.base_model)
        model.past_key_values = past_key_values
        model.past_key_values_data = past_key_values_data
        model.current_length_data = current_length_data

    input_len = input_ids.shape[1]
    cur_length = input_len

    reset_tree_mode(model)
    tree_logits, logits, hidden_state, sample_token = initialize_tree(
        input_ids, model, tree_buffers["tree_attn_mask"], past_key_values, logits_processor
    )
    new_token = 0
    for idx in range(max_steps):
        candidates, cart_candidates_prob, tree_candidates = generate_candidates(
            tree_logits,
            tree_buffers["tree_indices"],
            tree_buffers["retrieve_indices"],
            sample_token,
            logits_processor
        )
        logits, hidden_state_new, outputs = tree_decoding(
            model,
            tree_candidates,
            past_key_values,
            tree_buffers["tree_position_ids"],
            input_ids,
            tree_buffers["retrieve_indices_head"],
        )
        best_candidate, accept_length, sample_p = evaluate_posterior(
            logits, candidates, logits_processor, cart_candidates_prob,
            tree_logits[2], tree_buffers["p_indices"],
            tree_candidates, tree_buffers["b_indices"]
        )
        input_ids, tree_logits, new_token, hidden_state, sample_token = update_inference_inputs(
            input_ids,
            candidates,
            best_candidate,
            accept_length,
            tree_buffers["retrieve_indices"],
            logits_processor,
            logits,
            tree_logits,
            new_token,
            past_key_values_data,
            current_length_data,
            model,
            hidden_state,
            hidden_state_new,
            sample_p
        )
        t4 = time.time()
        accept_length_tree = input_ids.shape[1] - cur_length
        cur_length = accept_length_tree + cur_length
        accept_length_list.append(accept_length_tree)

        if tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
            for i, id in enumerate(input_ids[0, input_len:]):
                if id == tokenizer.eos_token_id:
                    eos_token_ids_index = i
            invalid_len = len(input_ids[0, input_len:]) - eos_token_ids_index - 1
            if invalid_len > 0:
                accept_length_list[-1] -= invalid_len
                new_token -= invalid_len
            break
        if new_token > max_new_tokens:
            break
        if input_ids.shape[1] > 1960:
            break

    return input_ids, new_token, idx + 1, accept_length_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ea-model-path",
        type=str,
    )
    parser.add_argument(
        "--base-model-path", type=str,
    )
    parser.add_argument(
        "--load-in-8bit", action="store_false",
    )
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
    )
    parser.add_argument(
        "--question-file",
        type=str,
    )
    parser.add_argument(
        "--question-begin",
        type=int,
    )
    parser.add_argument(
        "--question-end", type=int,
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--tree-choices",
        type=str,
        default="mc_sim_7b_63",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float64", "float16", "bfloat16"],
    )
    parser.add_argument(
        "--temp-dir",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    args.tree_choices = eval(args.tree_choices)

    if args.question_file:
        question_file = args.question_file
    else:
        print("Please provide a question file path (--question-file)")
    
    ea_config_path = os.path.join(args.ea_model_path, "config.json")
    source_config_path = "./config/config.json"

    if not os.path.exists(ea_config_path) and os.path.exists(source_config_path):
        print(f"Config file not found at {ea_config_path}, copying from {source_config_path}")
        import shutil
        try:
            os.makedirs(args.ea_model_path, exist_ok=True)
            shutil.copy2(source_config_path, ea_config_path)
            print(f"Successfully copied config.json to {ea_config_path}")
        except Exception as e:
            print(f"Warning: Failed to copy config.json: {e}")

    model = EaModel.from_pretrained(
        base_model_path=args.base_model_path,
        ea_model_path=args.ea_model_path,
        torch_dtype=str_to_torch_dtype(args.dtype),
        low_cpu_mem_usage=True,
        # load_in_8bit=True,
        device_map="auto"
    )

    tokenizer = model.get_tokenizer()

    if args.temperature > 1e-5:
        logits_processor = prepare_logits_processor(temperature=args.temperature)
    else:
        logits_processor = None

    run_eval(
        model=model,
        tokenizer=tokenizer,
        forward_func=ea_forward,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        max_new_tokens=args.max_new_tokens,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        tree_choices=args.tree_choices,
        logits_processor=logits_processor,
        max_steps=args.max_steps,
    )

