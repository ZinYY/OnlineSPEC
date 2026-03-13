import subprocess
import os
import shutil
from datetime import datetime
import time
import json
import numpy as np

from load_data import save_chunks_to_jsonl

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
        idxs = sum(datapoint['idxs'])
        times = sum(datapoint['wall_time'])
        if idxs != 0:
            accept_lengths_list.append(tokens / idxs)
        speeds.append(tokens / times)

    if report:
        if len(accept_lengths_list) > 0:
            print("# Mean accepted tokens: ", np.mean(accept_lengths_list))
        print('# Tokens per second: ', np.mean(speeds))
    return speeds, accept_lengths_list


def evaluate(model_path, question_file, answer_file):
    cmd = [
        "python", "eval.py",
        "--model-path", model_path,
        "--question-file", question_file,
        "--answer-file", answer_file
    ]
    env = os.environ.copy()
    evaluate_start_time = time.time()
    result = subprocess.run(cmd, env=env)
    evaluate_time = time.time() - evaluate_start_time
    
    if result.returncode != 0:
        print(f"Evaluation failed, return code: {result.returncode}")
        return [], [], evaluate_time
    
    print("Evaluation completed successfully")
    speeds, accept_lengths = speed(answer_file, report=True)
    return speeds, accept_lengths, evaluate_time


def train(
    model_path, 
    data_path, 
    output_dir, 
    epoch, 
    lr, 
    with_momentum, 
    teacher_loss_weight, 
    reconstruction_loss_weight,
    weight_decay=0.0
):
    cmd = [
        "python", "train.py",
        "--hydra_model_path", model_path,
        "--data_path", data_path,
        "--bf16", "True",
        "--output_dir", output_dir,
        "--num_train_epochs", str(epoch),
        "--global_batch_size", "5",
        "--per_device_train_batch_size", "5",
        "--per_device_eval_batch_size", "5",
        "--gradient_accumulation_steps", "3",
        "--dataloader_num_workers", "1",
        "--evaluation_strategy", "no",
        "--save_strategy", "no",
        "--learning_rate", str(lr),
        "--logging_steps", "1",
        "--tf32", "True",
        "--model_max_length", "2048",
        "--hydra_num_heads", "4",
        "--hydra_num_layers", "4",
        "--hydra_head_arch", "prefix-mlp",
        "--grounded_heads", "true",
        "--hidden_state_offset", "0",
        "--lm_loss_weight", "0.0",
        "--teacher_loss_weight", str(teacher_loss_weight),
        "--reconstruction_loss_weight", str(reconstruction_loss_weight),
        "--dropout_rate", "0.1",
        "--optim", "sgd",
        "--weight_decay", str(weight_decay),
    ]
    if with_momentum:
        cmd.extend(["--sgd_momentum", "0.9"])
    
    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"
    
    train_start_time = time.time()
    result = subprocess.run(cmd, env=env)
    train_time = time.time() - train_start_time
    
    if result.returncode != 0:
        print(f"Training failed, return code: {result.returncode}")
        return False, train_time
    
    print("Training completed successfully")
    return True, train_time


def run_pipeline(
    data_file, 
    model_path,
    with_momentum,
    teacher_loss_weight,
    reconstruction_loss_weight,
    chunk_size=500, 
    chunk_dir="data_chunks", 
    fined_model_dir="outputs", 
    log_file="results.json",
    lr=3e-5,
    epoch=1, 
    offline=False,
    weight_decay=0.0
):
    """
    Load data and split into chunks, evaluate each chunk, train on the data with assistance, and calculate the overall average acceptance rate.
    """
    print("Pipeline Start")

    print(f"Splitting data file: {data_file}")
    chunk_files = save_chunks_to_jsonl(data_file, chunk_size=chunk_size, output_dir=chunk_dir)
    print(f"Data split into {len(chunk_files)} chunks, saved in: {chunk_dir}")
    
    if offline:
        eval_model_dir = model_path
        print(f"Offline mode: skip training, use the same Hydra model {model_path} for evaluation")
    else:
        shutil.copytree(model_path, fined_model_dir, dirs_exist_ok=True)
        eval_model_dir = fined_model_dir
        print(f"Copied Hydra model to {fined_model_dir}, subsequent training will overwrite this directory")
    
    all_mean_speeds = []
    all_mean_accepts = []
    chunk_mean_speeds = []
    chunk_mean_accepts = []
    chunk_evaluate_times = []
    chunk_train_times = []
    total_time = 0.0
    
    for i, chunk_file in enumerate(chunk_files):
        print(f"\n=== Processing chunk {i+1}/{len(chunk_files)}: {chunk_file} ===")
        
        speeds_list, accept_lengths_list, evaluate_time = evaluate(model_path=eval_model_dir, question_file=chunk_file, answer_file=chunk_file.replace('.jsonl', '_with_assist.jsonl'))
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
            print(f"Offline mode: skip training for chunk {i+1}")
        else:
            train_success, train_time = train(
                model_path=fined_model_dir,
                data_path=chunk_file.replace('.jsonl', '_with_assist.jsonl'),
                output_dir=fined_model_dir,
                epoch=epoch,
                lr=lr,
                with_momentum=with_momentum,
                teacher_loss_weight=teacher_loss_weight,
                reconstruction_loss_weight=reconstruction_loss_weight,
                weight_decay=weight_decay
            )
            chunk_train_times.append(train_time)
            total_time += train_time
            print(f"Chunk {i+1} training time: {train_time:.2f} seconds")
            
            if not train_success:
                print(f"Chunk {i+1} training failed, pipeline terminated")
                return
    
    print(f"\n=== Pipeline completed ===")
    print(f"Overall average acceptance rate: {np.mean(chunk_mean_accepts)}")
    print(f"Overall average processing speed (tokens/sec): {np.mean(chunk_mean_speeds)}")
    result_dict = {
        "mean_accept": np.mean(chunk_mean_accepts),
        "mean_speed_tokens_per_sec": np.mean(chunk_mean_speeds),
        "chunk_mean_accepts": chunk_mean_accepts,
        "chunk_mean_speeds": chunk_mean_speeds,
        "all_mean_accepts": all_mean_accepts,
        "all_mean_speeds": all_mean_speeds,
        "fined_model_path": eval_model_dir,
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
    
    parser = argparse.ArgumentParser(description='Hydra pipeline')
    parser.add_argument('--data-file', type=str, required=True, help='Input data file path')
    parser.add_argument('--hydra-model-path', type=str, required=True, help='Hydra model path')
    parser.add_argument('--chunk-size', type=int, default=8, help='Chunk size')
    parser.add_argument('--chunk-dir', type=str, default='data_chunks', help='Chunk data save directory')
    parser.add_argument('--output-model-dir', type=str, default='outputs', help='Output medusa model directory')
    parser.add_argument('--lr', type=float, default=3e-5, help='Training learning rate')
    parser.add_argument('--num-epochs', type=int, default=1, help='Training epochs')
    parser.add_argument('--offline', action='store_true', help='Offline mode: skip training, use the same Hydra model for evaluation')
    parser.add_argument('--log-file', type=str, default=None, help='Result log file path')
    parser.add_argument('--with-momentum', action='store_true', help='Whether to use momentum during training')
    parser.add_argument('--teacher-loss-weight', type=float, default=1.0, help='Teacher loss weight')
    parser.add_argument('--reconstruction-loss-weight', type=float, default=0.0, help='Reconstruction loss weight')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='Weight decay (L2 regularization)')

    args = parser.parse_args()

    chunk_suffix = f"_chunk{args.chunk_size}"
    lr_epoch_suffix = f"_lr{args.lr}_epoch{args.num_epochs}"
    now = datetime.now()
    time_suffix = f"_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}"

    chunk_dir_with_params = args.chunk_dir + lr_epoch_suffix + time_suffix
    output_ea_dir_with_params = args.output_model_dir + lr_epoch_suffix + time_suffix
    if not args.log_file:
        log_file = "results" + chunk_suffix + lr_epoch_suffix + time_suffix + ".json"
    else:
        log_file = args.log_file + ".json"
    
    start_time = time.time()
    start_datetime = datetime.now()
    
    run_pipeline(
        data_file=args.data_file,
        model_path=args.hydra_model_path,
        chunk_size=args.chunk_size,
        chunk_dir=chunk_dir_with_params,
        fined_model_dir=output_ea_dir_with_params,
        log_file=log_file,
        lr=args.lr,
        epoch=args.num_epochs,
        offline=args.offline,
        with_momentum=args.with_momentum,
        teacher_loss_weight=args.teacher_loss_weight,
        reconstruction_loss_weight=args.reconstruction_loss_weight,
        weight_decay=args.weight_decay
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