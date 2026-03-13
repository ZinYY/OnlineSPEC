import argparse
import json
import os
script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
import time

from fastchat.model import get_conversation_template
from tqdm import tqdm

from eagle3.ea_model import EaModel
from eagle3.utils import *


def detect_model_template(base_model_path):
    model_path_lower = base_model_path.lower()
    if 'llama-2' in model_path_lower or 'llama2' in model_path_lower:
        return 'llama-2'
    if 'vicuna' in model_path_lower:
        return 'vicuna'
    return 'vicuna'


def run_eval(
        base_model_path,
        ea_model_path,
        model_id,
        question_file,
        question_begin,
        question_end,
        answer_file,
        max_new_token,
        num_choices,
        num_gpus_per_model,
        num_gpus_total,
        max_gpu_memory,
        temperature,
        args
):
    print("Loading questions...")
    questions = []
    with open(question_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    
    if question_begin is not None:
        questions = questions[question_begin:]
    if question_end is not None:
        questions = questions[:question_end]
    
    print(f"Loaded {len(questions)} questions from {question_file}")
    
    if not answer_file:
        answer_file = question_file.replace('.jsonl', '_with_assist.jsonl')
    
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model) if (num_gpus_total // num_gpus_per_model) > 1 else len(questions)
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_model_answers(
                base_model_path,
                ea_model_path,
                model_id,
                questions[i: i + chunk_size],
                answer_file,
                max_new_token,
                num_choices,
                num_gpus_per_model,
                max_gpu_memory,
                temperature,
                args
            )
        )

@torch.inference_mode()
def get_model_answers(
        base_model_path,
        ea_model_path,
        model_id,
        questions,
        answer_file,
        max_new_token,
        num_choices,
        num_gpus_per_model,
        max_gpu_memory,
        temperature,
        args
):
    model = EaModel.from_pretrained(
        base_model_path=base_model_path,
        ea_model_path=ea_model_path,
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.top_k,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        # load_in_8bit=True,
        device_map="auto",
        use_eagle3=args.use_eagle3,
    )

    tokenizer = model.get_tokenizer()

    if temperature > 1e-5:
        logits_processor = prepare_logits_processor(temperature=temperature)
    else:
        logits_processor = None

    model.eval()
    print('Check model training state:', model.training)

    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    print('CUDA VISIBLE DEVICES:', cuda_visible_devices)

    template_name = detect_model_template(base_model_path)
    print(f"Using chat template: {template_name}")

    if questions:
        print("Warming up...")
        first_question = questions[0]
        conv = get_conversation_template(template_name)
        
        if "conversations" in first_question:
            for role, content in first_question["conversations"]:
                if role == "ASSISTANT":
                    continue
                conv.append_message(role, content)
                conv.append_message("ASSISTANT", None)
                conv.stop_str = "</s>"
                prompt = conv.get_prompt()
                input_ids = tokenizer([prompt], return_tensors="pt", max_length=2000, truncation=True).input_ids
                
                try:
                    _ = model.eagenerate(
                        input_ids.cuda(),
                        temperature=temperature,
                        log=True
                    )[:3]
                except Exception as e:
                    print(f"Warmup error: {e}")
                break
    
    print('Warmup done')

    accept_lengths_list = []
    for question in tqdm(questions, desc="Evaluating", unit="q"):
        try:
            torch.manual_seed(0)
            conv = get_conversation_template(template_name)
            turns = []
            new_tokens_list = []
            wall_time_list = []
            accept_lengths = []
            
            if "conversations" in question:
                for role, content in question["conversations"]:
                    if role == "ASSISTANT":
                        continue
                    
                    conv.append_message(role, content)
                    conv.append_message("ASSISTANT", None)
                    conv.stop_str = "</s>"
                    prompt = conv.get_prompt()
                    input_ids = tokenizer([prompt], return_tensors="pt", max_length=2000, truncation=True).input_ids
                    
                    torch.cuda.synchronize()
                    start_time = time.time()
                    
                    output_ids, new_token, idx, accept_length_tree = model.eagenerate(
                        input_ids.cuda(),
                        temperature=temperature,
                        log=True
                    )
                    
                    torch.cuda.synchronize()
                    total_time = time.time() - start_time
                    
                    output_ids = output_ids[0][len(input_ids[0]):]
                    
                    if conv.stop_token_ids:
                        stop_token_ids_index = [
                            i for i, id in enumerate(output_ids)
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

                    turns.append(output)
                    new_tokens_list.append(int(new_token))
                    wall_time_list.append(total_time)
                    accept_lengths.extend(accept_length_tree)
                    conv.messages[-1][-1] = output
            
            new_question = question.copy()
            
            if turns:
                updated_conversations = []
                assistant_turn_idx = 0
                
                for role, content in question["conversations"]:
                    if role == "USER":
                        updated_conversations.append([role, content])
                    elif role == "ASSISTANT":
                        if assistant_turn_idx < len(turns):
                            updated_conversations.append([role, turns[assistant_turn_idx]])
                            assistant_turn_idx += 1
                        else:
                            updated_conversations.append([role, content])
                
                new_question["conversations"] = updated_conversations
                
                conv_final = get_conversation_template(template_name)
                conv_final.messages = []
                
                for role, content in updated_conversations:
                    conv_final.append_message(role, content)
                
                new_question["prompt"] = conv_final.get_prompt()
                new_question["new_tokens"] = new_tokens_list
                new_question["wall_time"] = wall_time_list
                new_question["accept_lengths"] = accept_lengths
                accept_lengths_list.extend(accept_lengths)
            
            os.makedirs(os.path.dirname(answer_file) if os.path.dirname(answer_file) else ".", exist_ok=True)
            with open(answer_file, "a", encoding="utf-8") as fout:
                fout.write(json.dumps(new_question, ensure_ascii=False) + "\n")
                
        except Exception as e:
            import traceback
            print(f"ERROR processing question ID: {question.get('id', 'unknown')}")
            print(f"Error details: {str(e)}")
            traceback.print_exc()
            continue
    
    if accept_lengths_list:
        import numpy as np
        mean_accept = np.mean(accept_lengths_list)
        print(f"\n# Total speculative decoding steps: {len(accept_lengths_list)}")
        print(f"# Mean accepted tokens per step: {mean_accept:.2f}")
    print(f"# Results saved to: {answer_file}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EAGLE Evaluation Script")
    
    parser.add_argument(
        "--ea-model-path",
        type=str,
        required=True
    )
    parser.add_argument(
        "--base-model-path", 
        type=str, 
        required=True
    )
    parser.add_argument(
        "--question-file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--answer-file", 
        type=str, 
        default=None,
    )
    parser.add_argument(
        "--load-in-8bit", 
        action="store_true", 
    )
    parser.add_argument(
        "--model-id", 
        type=str, 
        default="eagle-vicuna",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--question-end", 
        type=int, 
        default=None,
    )
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--total-token",
        type=int,
        default=60,
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
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
        "--num-gpus-total", 
        type=int, 
        default=1, 
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        default=None,
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
        "--use-eagle3",
        action="store_true",
    )

    args = parser.parse_args()

    question_file = args.question_file
    answer_file = args.answer_file

    print(f"\nInput:  {question_file}")
    print(f"Output: {answer_file if answer_file else question_file.replace('.jsonl', '_with_assist.jsonl')}\n")

    run_eval(
        args.base_model_path,
        args.ea_model_path,
        args.model_id,
        question_file,
        args.question_begin,
        args.question_end,
        answer_file,
        args.max_new_token,
        args.num_choices,
        args.num_gpus_per_model,
        args.num_gpus_total,
        args.max_gpu_memory,
        args.temperature,
        args
    )
    
    print("\nEvaluation completed successfully!")
