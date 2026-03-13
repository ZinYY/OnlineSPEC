import argparse
import json
import os
import time
import torch
from tqdm import tqdm

from fastchat.llm_judge.common import load_questions
from fastchat.model import get_conversation_template

from hydra.model.utils import *
from hydra.model.hydra_model import HydraModel
from hydra.model.kv_cache import initialize_past_key_values
from hydra.model.hydra_choices import *

def hydra_forward(input_ids, model, tokenizer, hydra_choices, temperature, posterior_threshold, posterior_alpha, max_steps = 512):
    assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
    input_ids = input_ids.clone()

    if hasattr(model, "hydra_choices") and model.hydra_choices == hydra_choices:
        hydra_buffers = model.hydra_buffers
    else:
        hydra_buffers = generate_hydra_buffers(
            hydra_choices, device=model.base_model.device
        )
    model.hydra_buffers = hydra_buffers
    model.hydra_choices = hydra_choices

    # Initialize the past key and value states
    if hasattr(model, "past_key_values"):
        past_key_values = model.past_key_values
        past_key_values_data = model.past_key_values_data
        current_length_data = model.current_length_data
        # Reset the past key and value states
        current_length_data.zero_()
    else:
        (
            past_key_values,
            past_key_values_data,
            current_length_data,
        ) = initialize_past_key_values(model.base_model, model.hydra_head_arch)
        model.past_key_values = past_key_values
        model.past_key_values_data = past_key_values_data
        model.current_length_data = current_length_data

    input_len = input_ids.shape[1]
    reset_hydra_mode(model)
    hidden_states, logits = initialize_hydra(
        input_ids, model, hydra_buffers["hydra_attn_mask"], past_key_values, hydra_buffers["proposal_cross_attn_masks"]
    )
    new_token = 0

    for idx in range(max_steps): 
        to_pass_input_ids = None
        if idx == 0:
            to_pass_input_ids = input_ids
        candidates, tree_candidates = model.hydra_head.proposal(logits, hidden_states, hydra_buffers, past_key_values, to_pass_input_ids)
        hidden_states, logits = tree_decoding(
                model,
                tree_candidates,
                past_key_values,
                hydra_buffers["hydra_position_ids"],
                input_ids,
                hydra_buffers["retrieve_indices"],
            )
        best_candidate, accept_length = evaluate_posterior(
                logits, candidates, temperature, posterior_threshold, posterior_alpha, hydra_buffers["max_accepts"]
            )
        input_ids, logits, hidden_states, new_token = update_inference_inputs(
            input_ids,
            candidates,
            best_candidate,
            accept_length,
            hydra_buffers["retrieve_indices"],
            logits,
            hidden_states,
            new_token,
            past_key_values_data,
            current_length_data,
            model.hydra_head_arch
        )
        if tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
            break
        if new_token > 1024:
            break
    return input_ids, new_token, idx

def run_eval(
    model_path,
    base_model_id,
    question_file,
    question_begin,
    question_end,
    answer_file,
    temperature,
    posterior_threshold,
    posterior_alpha,
    grounded_heads,
    hydra_choices,
):
    questions = load_questions(question_file, question_begin, question_end)

    # chunk_size = len(questions)
    ans_handles = []
    # for i in range(0, len(questions), chunk_size):
    ans_handles.append(
        get_model_answers(
            model_path,
            base_model_id,
            questions,
            answer_file,
            temperature,
            posterior_threshold,
            posterior_alpha,
            grounded_heads,
            hydra_choices,
        )
    )

@torch.inference_mode()
def get_model_answers(
    model_path,
    base_model_id,
    questions,
    answer_file,
    temperature,
    posterior_threshold,
    posterior_alpha,
    grounded_heads,
    hydra_choices,
):
    
    # Hydra model setup

    model = HydraModel.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    tokenizer = model.get_tokenizer()
    
    model.eval()
    print('Check model training state:',model.training)
    
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    print('CUDA VISIBLE DEVICES:', cuda_visible_devices)
    
    question = questions[0]

    # warmup
    for _ in range(2):
        torch.manual_seed(0)
        conv = get_conversation_template(base_model_id)
        for j in range(len(question["conversations"])):
            if question["conversations"][j]["from"] == "human":
                qs = question["conversations"][j]["value"]
            else:
                continue
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer([prompt]).input_ids

            # some models may error out when generating long outputs
            try:
                torch.cuda.synchronize()
                start_time = time.time()
                output_ids, new_token, idx = hydra_forward(
                    torch.as_tensor(input_ids).cuda(),
                    model,
                    tokenizer,
                    hydra_choices,
                    temperature,
                    posterior_threshold,
                    posterior_alpha,
                )
                torch.cuda.synchronize()
            except RuntimeError as e:
                print("ERROR question ID: ", question["id"])
                output = "ERROR"
                print(e)
    print('Warmup done')


    for question in tqdm(questions):
        error_occurred = False
        conv = get_conversation_template(base_model_id)
        turns = []
        idxs = []
        idx = 0
        new_tokens = []
        wall_time = []
        choice = {"turns": [], "idxs": [], "new_tokens": [], "wall_time": []}
        for j in range(len(question["conversations"])):
            if question["conversations"][j]["from"] == "human":
                qs = question["conversations"][j]["value"]
            else:
                continue
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer([prompt]).input_ids


            # some models may error out when generating long outputs
            try:
                torch.cuda.synchronize()
                start_time = time.time()
                output_ids, new_token, idx = hydra_forward(
                    torch.as_tensor(input_ids).cuda(),
                    model,
                    tokenizer,
                    hydra_choices,
                    temperature,
                    posterior_threshold,
                    posterior_alpha,
                )
                torch.cuda.synchronize()
                total_time = time.time() - start_time
                # if model.config.is_encoder_decoder:
                #     output_ids = output_ids[0]
                # else:
                output_ids = output_ids[0][len(input_ids[0]) :]

                # be consistent with the template's stop_token_ids
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
                print("ERROR question ID: ", question["id"])
                output = "ERROR"
                print(e)
                new_token = 0
                total_time = 0.0
                error_occurred = True
                break

            if new_token != 0:
                turns.append(output)
                idxs.append(int(idx))
                new_tokens.append(int(new_token))
                wall_time.append(total_time)
                conv.messages[-1][-1] = output
                # torch.cuda.empty_cache()
                
        choice = {"turns": turns, "idxs": idxs, "new_tokens": new_tokens, "wall_time": wall_time}
        
        if error_occurred:
            continue

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            conversations = []
            turn_idx = 0
            for msg in question["conversations"]:
                if msg["from"] == "human":
                    conversations.append({"from": "human", "value": msg["value"]})
                elif msg["from"] == "gpt":
                    if turn_idx < len(choice["turns"]):
                        conversations.append({"from": "gpt", "value": choice["turns"][turn_idx]})
                        turn_idx += 1
                    else:
                        conversations.append({"from": "gpt", "value": ""})
            ans_json = {
                "id": question["id"],
                "conversations": conversations,
                "idxs": choice["idxs"],
                "new_tokens": choice["new_tokens"],
                "wall_time": choice["wall_time"],
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--question-file", 
        type=str, 
        required=True
    )
    parser.add_argument(
        "--answer-file", type=str, help="The output answer file."
    )
    parser.add_argument("--base-model-id", type=str, default="vicuna")

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="The temperature for hydra sampling.",
    )

    parser.add_argument(
        "--posterior-threshold",
        type=float,
        default=0.09,
        help="The posterior threshold for hydra sampling.",
    )
    
    parser.add_argument(
        "--posterior-alpha",
        type=float,
        default=0.3,
        help="The posterior alpha for hydra sampling.",
    )

    parser.add_argument(
        "--grounded-heads",
        action="store_true",
        help="Whether heads are sample conditoned",
    )

    args = parser.parse_args()

    print(f"Output to {args.answer_file}")

    run_eval(
        model_path=args.model_path,
        base_model_id=args.base_model_id,
        answer_file=args.answer_file,
        question_file=args.question_file,
        question_begin=None,
        question_end=None,
        temperature=args.temperature,
        posterior_threshold=args.posterior_threshold,
        posterior_alpha=args.posterior_alpha,
        grounded_heads=args.grounded_heads,
        hydra_choices=eval("mc_sim_7b_63")
    )