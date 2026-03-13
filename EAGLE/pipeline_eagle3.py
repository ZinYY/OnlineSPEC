import subprocess
import torch
import os
import shutil
from datetime import datetime
import time
import json
import numpy as np
from typing import List

from load_data import save_chunks_to_jsonl
from eagle3.ea_model import EaModel
from eagle3.utils import *
from fastchat.model import get_conversation_template
from tqdm import tqdm


def detect_model_template(base_model_path):
    model_path_lower = base_model_path.lower()
    if 'llama-2' in model_path_lower:
        return 'llama-2'
    if 'vicuna' in model_path_lower:
        return 'vicuna'
    print(f"Cannot detect model type from path '{base_model_path}', defaulting to vicuna template")
    return 'vicuna'


def speed(jsonl_file, report=True):
    data = []
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            data.append(json_obj)

    speeds = []
    accept_lengths_list = []
    for datapoint in data:
        tokens = sum(datapoint['new_tokens'])
        times = sum(datapoint['wall_time'])
        accept_lengths_list.append(np.mean(datapoint['accept_lengths']))
        speeds.append(tokens/times)

    if report:
        print("# Mean accepted tokens: ", np.mean(accept_lengths_list))
        print('# Tokens per second: ', np.mean(speeds))
    return speeds, accept_lengths_list


def evaluate(data_file, base_model_path, ea_model_path, total_token=60, depth=5, top_k=10, temperature=0.0):
    print(f"Loading EAGLE3 model from {ea_model_path}...")
    model = EaModel.from_pretrained(
        base_model_path=base_model_path,
        ea_model_path=ea_model_path,
        total_token=total_token,
        depth=depth,
        top_k=top_k,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        use_eagle3=True,
    )
    tokenizer = model.get_tokenizer()
    
    template_name = detect_model_template(base_model_path)
    print(f"Using chat template: {template_name}")
    
    questions = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    
    print(f"Loaded {len(questions)} questions from {data_file}")
    
    evaluate_start_time = time.time()
    
    output_file = data_file.replace('.jsonl', '_with_assist.jsonl')
    
    if os.path.exists(output_file):
        os.remove(output_file)
    
    if temperature > 1e-5:
        logits_processor = prepare_logits_processor(temperature=temperature)
    else:
        logits_processor = None
    
    model.eval()
    print('Check model training state:', model.training)
    
    # Warmup
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
    for question in tqdm(questions, desc="Evaluating"):
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
            
            os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
            with open(output_file, "a", encoding="utf-8") as fout:
                fout.write(json.dumps(new_question, ensure_ascii=False) + "\n")
                
        except Exception as e:
            import traceback
            print(f"ERROR processing question ID: {question.get('id', 'unknown')}")
            print(f"Error details: {str(e)}")
            traceback.print_exc()
            continue
    
    evaluate_time = time.time() - evaluate_start_time
    
    if accept_lengths_list:
        mean_accept = np.mean(accept_lengths_list)
        print(f"\n# Total speculative decoding steps: {len(accept_lengths_list)}")
        print(f"# Mean accepted tokens per step: {mean_accept:.2f}")
    print(f"# Results saved to: {output_file}")
    
    speeds, accept_lengths = speed(output_file, report=True)
    return speeds, accept_lengths, evaluate_time


def train(data_file, base_model_path, ea_model_path, ea_output_dir, 
          batch_size=4, num_epochs=1, lr=1e-3, max_len=2048, weight_decay=0.01):
    cmd = [
        "python", "traineagle3.py",
        "--basepath", base_model_path,
        "--trainpath", data_file,
        "--savedir", ea_output_dir,
        "--batch_size", str(batch_size),
        "--num_epochs", str(num_epochs),
        "--lr", str(lr),
        "--max_len", str(max_len),
        "--weight_decay", str(weight_decay)
    ]
    
    if ea_model_path and os.path.exists(ea_model_path):
        if os.path.exists(os.path.join(ea_model_path, "config.json")):
            cmd.extend(["--ea-model-path", ea_model_path])
    
    print(f"Executing training command: {' '.join(cmd)}")
    
    env = os.environ.copy()
    env["WANDB_DISABLED"] = "true"
    
    eagle_dir = os.path.dirname(os.path.abspath(__file__))
    
    train_start_time = time.time()
    result = subprocess.run(cmd, env=env, cwd=eagle_dir)
    train_time = time.time() - train_start_time
    
    if result.returncode != 0:
        print(f"Training failed, return code: {result.returncode}")
        return False, train_time
    
    print("Training completed successfully")
    return True, train_time


def run_pipeline(
    data_file, 
    base_model_path, 
    ea_model_path,
    chunk_size=500, 
    chunk_dir="data_chunks", 
    ea_output_dir="outputs", 
    log_file="results.json",
    batch_size=4,
    num_epochs=1, 
    lr=1e-3,
    max_len=2048,
    weight_decay=0.01,
    total_token=60,
    depth=5,
    top_k=10,
    temperature=0.0,
    offline=False
):
    print("EAGLE3 pipeline started")

    print(f"Splitting data file: {data_file}")
    chunk_files = save_chunks_to_jsonl(data_file, chunk_size=chunk_size, output_dir=chunk_dir)
    print(f"Data split into {len(chunk_files)} chunks, saved in: {chunk_dir}")
    
    ea_temp_dir = os.path.join(ea_output_dir, f"ea3_temp_model_{int(time.time())}")
    if os.path.exists(ea_temp_dir):
        shutil.rmtree(ea_temp_dir)
    if os.path.isdir(ea_model_path):
        shutil.copytree(ea_model_path, ea_temp_dir)
    print(f"EA3 model path: {ea_model_path} copied to temporary directory: {ea_temp_dir} for overwriting")
    
    all_mean_speeds = []
    all_mean_accepts = []
    chunk_mean_speeds = []
    chunk_mean_accepts = []
    chunk_evaluate_times = []
    chunk_train_times = []
    total_time = 0.0
    
    for i, chunk_file in enumerate(chunk_files):
        print(f"\nProcessing chunk {i+1}/{len(chunk_files)}: {chunk_file}")
        
        print(f"Evaluating chunk {chunk_file}")
        speeds_list, accept_lengths_list, evaluate_time = evaluate(
            chunk_file, 
            base_model_path, 
            ea_temp_dir,
            total_token=total_token,
            depth=depth,
            top_k=top_k,
            temperature=temperature
        )
        chunk_evaluate_times.append(evaluate_time)
        total_time += evaluate_time
        print(f"Chunk {i+1} evaluation time: {evaluate_time:.2f} seconds")
        
        all_mean_speeds.extend(speeds_list)
        all_mean_accepts.extend(accept_lengths_list)
        chunk_mean_speeds.append(np.mean(speeds_list))
        chunk_mean_accepts.append(np.mean(accept_lengths_list))
        
        print(f"Chunk {i+1} average processing speed (tokens/sec): {np.mean(speeds_list)}")
        print(f"Chunk {i+1} average acceptance rate: {np.mean(accept_lengths_list)}")
        
        if offline:
            print(f"Offline mode: skipping training for chunk {i+1}")
        else:
            chunk_file_with_assist = chunk_file.replace('.jsonl', '_with_assist.jsonl')
            print(f"Training EA3 model with assistance data: {chunk_file_with_assist}")

            train_success, train_time = train(
                chunk_file_with_assist,
                base_model_path,
                ea_temp_dir,
                ea_temp_dir,
                batch_size=batch_size,
                num_epochs=num_epochs,
                lr=lr,
                max_len=max_len,
                weight_decay=weight_decay
            )
            chunk_train_times.append(train_time)
            total_time += train_time
            print(f"Chunk {i+1} training time: {train_time:.2f} seconds")
            
            if train_success:
                print(f"Chunk {i+1} training completed successfully")
            else:
                print(f"Chunk {i+1} training failed")
                return -1
    
    print(f"\nPipeline execution completed")
    print(f"Overall average acceptance: {np.mean(chunk_mean_accepts)}")
    print(f"Overall average processing speed (tokens/sec): {np.mean(chunk_mean_speeds)}")
    print(f"Final EA3 model path: {ea_temp_dir}")
    
    result_dict = {
        "mean_accept": np.mean(chunk_mean_accepts),
        "mean_speed_tokens_per_sec": np.mean(chunk_mean_speeds),
        "chunk_mean_accepts": chunk_mean_accepts,
        "chunk_mean_speeds": chunk_mean_speeds,
        "all_mean_accepts": all_mean_accepts,
        "all_mean_speeds": all_mean_speeds,
        "final_ea_model_path": ea_temp_dir,
        "total_time": total_time,
        "chunk_evaluate_times": chunk_evaluate_times
    }
    if not offline:
        result_dict["chunk_train_times"] = chunk_train_times
    
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, indent=4)
    print(f"Results saved to: {log_file}")
    print(f"Total time (evaluate + training): {total_time:.2f} seconds")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='EAGLE3 pipeline')
    parser.add_argument('--data-file', type=str, required=True, help='Input data file path')
    parser.add_argument('--base-model-path', type=str, required=True, help='Base model path')
    parser.add_argument('--ea-model-path', type=str, required=True, help='EA3 model path')
    parser.add_argument('--chunk-size', type=int, default=8, help='Chunk size')
    parser.add_argument('--chunk-dir', type=str, default='data_chunks', help='Chunk data save directory')
    parser.add_argument('--output-ea-dir', type=str, default='outputs', help='Output base directory')
    
    parser.add_argument('--batch-size', type=int, default=4, help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Training learning rate')
    parser.add_argument('--num-epochs', type=int, default=1, help='Training epochs')
    parser.add_argument('--max-len', type=int, default=2048, help='Maximum sequence length')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay (regularization)')
    parser.add_argument('--total-token', type=int, default=60, help='EAGLE total token number')
    parser.add_argument('--depth', type=int, default=5, help='EAGLE depth')
    parser.add_argument('--top-k', type=int, default=10, help='EAGLE top-k parameter')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature')
    parser.add_argument('--offline', action='store_true', help='Offline mode: do not execute training and always use the original EAGLE model')
    parser.add_argument('--debug-chunks', type=int, help='Debug mode: only process the first N chunks')
    parser.add_argument('--log-file', type=str, default=None, help='Result log file path')
    
    args = parser.parse_args()
    
    chunk_suffix = f"_chunk{args.chunk_size}"
    lr_epoch_suffix = f"_lr{args.lr}_epoch{args.num_epochs}"
    now = datetime.now()
    time_suffix = f"_{now.month}_{now.day}_{now.hour}_{now.minute}"

    chunk_dir_with_params = args.chunk_dir + lr_epoch_suffix + time_suffix
    if not args.log_file:
        log_file = "results_eagle3" + chunk_suffix + lr_epoch_suffix + time_suffix + ".json"
    else:
        log_file = args.log_file + ".json"
    
    start_time = time.time()
    start_datetime = datetime.now()
    
    run_pipeline(
        data_file=args.data_file,
        base_model_path=args.base_model_path,
        ea_model_path=args.ea_model_path,
        chunk_size=args.chunk_size,
        chunk_dir=chunk_dir_with_params,
        ea_output_dir=args.output_ea_dir,
        log_file=log_file,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        max_len=args.max_len,
        weight_decay=args.weight_decay,
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.top_k,
        temperature=args.temperature,
        offline=args.offline
    )
    
    end_time = time.time()
    end_datetime = datetime.now()
    total_duration = end_time - start_time
    
    hours = int(total_duration // 3600)
    minutes = int((total_duration % 3600) // 60)
    seconds = int(total_duration % 60)

    print(f"Start time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time: {hours} hours {minutes} minutes {seconds} seconds (total {total_duration:.2f} seconds)")
