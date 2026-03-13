import subprocess
import torch
import os
import shutil
from datetime import datetime
import time
import json
import numpy as np

from load_data import save_chunks_to_jsonl
from fastchat.llm_judge.common import load_questions
from vanilla_speculative_decoding import get_model_answers, vanilla_speculative_forward
from eagle.model.ea_model import EaModel

def speed(jsonl_file, report=True):
    data = []
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            data.append(json_obj)

    speeds=[]
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


def evaluate(data_file, base_model_path, ea_model_path):
    model = EaModel.from_pretrained(
        base_model_path=base_model_path,
        ea_model_path=ea_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    tokenizer = model.get_tokenizer()
    questions = load_questions(data_file, begin=None, end=None)
    print(f"Loaded {len(questions)} questions from {data_file}")
    output_file = data_file.replace('.jsonl', '_with_assist.jsonl')
    get_model_answers(
        model=model,
        tokenizer=tokenizer,
        forward_func=vanilla_speculative_forward,
        questions=questions,
        output_file=output_file,
        max_new_tokens=1024,
        num_choices=1,
        logits_processor=None,
        max_steps=512
    )
    return speed(output_file, report=True)


def train(data_file, base_model_path, ea_model_path, ea_output_dir, additional_args="", lr=3e-5, num_epochs=1):
    cmd = [
        "accelerate", "launch", "--mixed_precision=bf16", "-m", "train",
        "--basepath", base_model_path,
        "--ea-model-path", ea_model_path,
        "--data_path", data_file,
        "--cpdir", ea_output_dir
    ]
    
    cmd.extend(["--lr", str(lr)])
    cmd.extend(["--num_epochs", str(num_epochs)])
    
    if additional_args:
        cmd.extend(additional_args.split())
    
    print(f"Executing training command: {' '.join(cmd)}")
    
    env = os.environ.copy()
    env["WANDB_DISABLED"] = "true"
    
    result = subprocess.run(cmd, env=env)
    
    if result.returncode != 0:
        print(f"Training failed, return code: {result.returncode}")
        return False
    
    print("Training completed successfully")
    return True


def run_pipeline(
    data_file, 
    base_model_path, 
    ea_model_path,
    chunk_size=500, 
    chunk_dir="data_chunks", 
    ea_output_dir="outputs", 
    log_file="results.json",
    train_args="", 
    lr=3e-5, 
    num_epochs=1, 
    offline=False
):
    print("=== OSD pipeline starts execution ===")

    print(f"Splitting data file: {data_file}")
    chunk_files = save_chunks_to_jsonl(data_file, chunk_size=chunk_size, output_dir=chunk_dir)
    print(f"Data has been split into {len(chunk_files)} chunks, saved in: {chunk_dir}")
    
    ea_temp_dir = os.path.join(ea_output_dir, f"ea_temp_model_{int(time.time())}")
    if os.path.exists(ea_temp_dir):
        shutil.rmtree(ea_temp_dir)
    if os.path.isdir(ea_model_path):
        shutil.copytree(ea_model_path, ea_temp_dir)
    print(f"EA model path: {ea_model_path} copied to temporary directory: {ea_temp_dir} for overwriting")
    
    all_mean_speeds = []
    all_mean_accepts = []
    chunk_mean_speeds = []
    chunk_mean_accepts = []
    
    for i, chunk_file in enumerate(chunk_files):
        print(f"\n=== Processing chunk {i+1}/{len(chunk_files)}: {chunk_file} ===")
        
        print(f"Evaluating chunk {chunk_file}")
        speeds_list, accept_lengths_list = evaluate(chunk_file, base_model_path, ea_temp_dir)
        
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
            print(f"Training EA model with assistance data {chunk_file_with_assist}")

            train_success = train(
                chunk_file_with_assist,
                base_model_path,
                ea_temp_dir,
                ea_temp_dir,
                train_args,
                lr=lr,
                num_epochs=num_epochs
            )
            
            if train_success:
                print(f"Chunk {i+1} training completed successfully")
            else:
                print(f"Chunk {i+1} training failed")
                return -1
    
    print(f"\n=== Pipeline execution completed ===")
    print(f"Overall average acceptance: {np.mean(chunk_mean_accepts)}")
    print(f"Overall average processing speed (tokens/sec): {np.mean(chunk_mean_speeds)}")
    print(f"Final EA model path: {ea_temp_dir}")
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump({
            "mean_accept": np.mean(chunk_mean_accepts),
            "mean_speed_tokens_per_sec": np.mean(chunk_mean_speeds),
            "chunk_mean_accepts": chunk_mean_accepts,
            "chunk_mean_speeds": chunk_mean_speeds,
            "all_mean_accepts": all_mean_accepts,
            "all_mean_speeds": all_mean_speeds,
            "final_ea_model_path": ea_temp_dir
        }, f, indent=4)
    print(f"Results saved to: {log_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='OSD pipeline')
    parser.add_argument('--data-file', type=str, required=True, help='Input data file path')
    parser.add_argument('--base-model-path', type=str, required=True, help='Base model path')
    parser.add_argument('--ea-model-path', type=str, required=True, help='EA model path')
    parser.add_argument('--chunk-size', type=int, default=8, help='Chunk size')
    parser.add_argument('--chunk-dir', type=str, default='data_chunks', help='Chunk data save directory')
    parser.add_argument('--output-ea-dir', type=str, default='outputs', help='Output base directory')
    parser.add_argument('--eval-args', type=str, default='', help='Evaluation additional parameters')
    parser.add_argument('--train-args', type=str, default='', help='Training additional parameters')
    parser.add_argument('--lr', type=float, default=3e-5, help='Training learning rate')
    parser.add_argument('--num-epochs', type=int, default=1, help='Training epochs')
    parser.add_argument('--offline', action='store_true', help='Offline mode: do not execute training and always use the original EAGLE model')
    parser.add_argument('--debug-chunks', type=int, help='Debug mode: only process the first N chunks')
    parser.add_argument('--log-file', type=str, default=None, help='Result log file path')
    
    args = parser.parse_args()
    
    if args.debug_chunks:
        run_pipeline.debug_chunks = args.debug_chunks
    
    chunk_suffix = f"_chunk{args.chunk_size}"
    lr_epoch_suffix = f"_lr{args.lr}_epoch{args.num_epochs}"
    now = datetime.now()
    time_suffix = f"_{now.month}_{now.day}_{now.hour}_{now.minute}"

    chunk_dir_with_params = args.chunk_dir + lr_epoch_suffix + time_suffix
    if not args.log_file:
        log_file = "results" + chunk_suffix + lr_epoch_suffix + time_suffix + ".json"
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
        train_args=args.train_args,
        lr=args.lr,
        num_epochs=args.num_epochs,
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
