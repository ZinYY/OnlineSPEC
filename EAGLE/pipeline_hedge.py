import subprocess
import os
import shutil
from datetime import datetime
import torch
import copy
import time
import json
import numpy as np
import torch.nn.functional as F

from load_data import save_chunks_to_jsonl
from eagle.model.ea_model import EaModel
from eagle.model.cnets import Model, EConfig

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


def compute_softmax_weights_from_losses(losses, temperature=1.0):
    losses_tensor = torch.tensor(losses, dtype=torch.float32)
    neg_losses = -losses_tensor
    weights = F.softmax(neg_losses / temperature, dim=0)
    print(f"loss {losses} (temperature={temperature}) calculated softmax weights: {weights.tolist()}")
    return tuple(weights.tolist())


def ensemble_ea_models(ea_model_path_1, ea_model_path_2, ea_model_path_3, weights=(1.0, 1.0, 1.0), out_dir=None, save_name="pytorch_model.bin"):
    # Load models
    config = EConfig.from_pretrained(ea_model_path_1)
    ea_model_a = Model(config, bias=True)
    load_model_path=os.path.join(ea_model_path_1, "pytorch_model.bin")
    ea_layer_state_dict = torch.load(load_model_path, map_location='cpu')
    ea_model_a.load_state_dict(ea_layer_state_dict, strict=True)

    config = EConfig.from_pretrained(ea_model_path_2)
    ea_model_b = Model(config, bias=True)
    load_model_path=os.path.join(ea_model_path_2, "pytorch_model.bin")
    ea_layer_state_dict = torch.load(load_model_path, map_location='cpu')
    ea_model_b.load_state_dict(ea_layer_state_dict, strict=True)

    config = EConfig.from_pretrained(ea_model_path_3)
    ea_model_c = Model(config, bias=True)
    load_model_path=os.path.join(ea_model_path_3, "pytorch_model.bin")
    ea_layer_state_dict = torch.load(load_model_path, map_location='cpu')
    ea_model_c.load_state_dict(ea_layer_state_dict, strict=True)
    
    wa, wb, wc = weights
    total = float(wa + wb + wc)
    assert total > 0, "The sum of weights must be greater than 0"
    wa, wb, wc = wa / total, wb / total, wc / total

    sd_a = {k: v.cpu() for k, v in ea_model_a.state_dict().items()}
    sd_b = {k: v.cpu() for k, v in ea_model_b.state_dict().items()}
    sd_c = {k: v.cpu() for k, v in ea_model_c.state_dict().items()}

    keys = list(sd_a.keys())

    new_sd = {}
    for k in keys:
        va = sd_a[k]
        if not isinstance(va, torch.Tensor):
            new_sd[k] = va
            continue

        vb = sd_b.get(k, None)
        vc = sd_c.get(k, None)

        if vb is None:
            vb = torch.zeros_like(va)
        if vc is None:
            vc = torch.zeros_like(va)

        new_val = wa * va + wb * vb.to(va.dtype) + wc * vc.to(va.dtype)
        new_sd[k] = new_val

    new_model = copy.deepcopy(ea_model_a)
    first_param = next(new_model.parameters())
    target_device = first_param.device
    new_sd_on_device = {k: v.to(target_device) if isinstance(v, torch.Tensor) else v for k, v in new_sd.items()}
    new_model.load_state_dict(new_sd_on_device)

    save_path = None
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, save_name)
        torch.save(new_model.state_dict(), save_path)
        print(f"EaModel ensemble model saved to: {save_path}")


def evaluate(data_file, base_model_path, ea_model_path_1, ea_model_path_2, ea_model_path_3, ensemble_model_path, weights=(1.0, 1.0, 1.0)):
    print(f"Using weights {weights} for ensemble")
    ensemble_ea_models(
        ea_model_path_1,
        ea_model_path_2,
        ea_model_path_3,
        weights=weights,
        out_dir=ensemble_model_path
    )
    cmd = [
        "python", "evaluation.py",
        "--base-model-path", base_model_path,
        "--ea-model-path", ensemble_model_path,
        "--question-file", data_file
    ]
    print(f"Executing evaluation command: {' '.join(cmd)}")

    env = os.environ.copy()
    result = subprocess.run(cmd, env=env)

    if result.returncode != 0:
        print(f"Evaluation failed, return code: {result.returncode}")
        return False
    
    print("Evaluation completed successfully")
    output_file = data_file.replace('.jsonl', '_with_assist.jsonl')
    return speed(output_file, report=True)


def train(
    data_file, base_model_path, 
    ea_model_path_1, ea_model_path_2, ea_model_path_3, 
    ea_output_dir_1, ea_output_dir_2, ea_output_dir_3, 
    lr_1=3e-5, lr_2=3e-5, lr_3=3e-5, 
    num_epochs=1,
    use_meta_learner=False,
    meta_learner_path=None
):
    ea_model_paths = [ea_model_path_1, ea_model_path_2, ea_model_path_3]
    ea_output_dirs = [ea_output_dir_1, ea_output_dir_2, ea_output_dir_3]
    lrs = [lr_1, lr_2, lr_3]
    
    losses = []
    
    if use_meta_learner and meta_learner_path:
        print(f"\nPreparing training environment: copying meta learner to 3 training directories")
        for idx, ea_output_dir in enumerate(ea_output_dirs, 1):
            if os.path.exists(ea_output_dir):
                shutil.rmtree(ea_output_dir)
            shutil.copytree(meta_learner_path, ea_output_dir)
            print(f"    ├─ [{idx}/3] {os.path.basename(ea_output_dir)}")
        print(f"    └─ Completed! ")
        ea_model_paths = ea_output_dirs
    
    for i, (ea_model_path, ea_output_dir, lr) in enumerate(zip(ea_model_paths, ea_output_dirs, lrs), 1):
        cmd = [
            "accelerate", "launch", "--mixed_precision=bf16", "-m", "train",
            "--basepath", base_model_path,
            "--ea-model-path", ea_model_path,
            "--data_path", data_file,
            "--cpdir", ea_output_dir
        ]
        
        cmd.extend(["--lr", str(lr)])
        cmd.extend(["--num_epochs", str(num_epochs)])
        
        print(f"\nTrainer {i}/3 (lr={lr}):")
        print(f"    ├─ Command: {' '.join(cmd)}")
        
        env = os.environ.copy()
        env["WANDB_DISABLED"] = "true"
        
        result = subprocess.run(cmd, env=env)
        
        if result.returncode != 0:
            print(f"    └─ Training failed, return code: {result.returncode}")
            return False, None
        
        loss_file = os.path.join(ea_output_dir, "training_loss.json")
        if os.path.exists(loss_file):
            with open(loss_file, 'r') as f:
                loss_data = json.load(f)
                model_loss = loss_data.get("total_loss", 0)
                losses.append(model_loss)
                print(f"    └─ Completed, total_loss: {model_loss:.4f}")
        else:
            print(f"    └─ Warning: loss file not found")
            losses.append(float('inf'))
    return True, losses


def run_pipeline(
    data_file, 
    base_model_path, 
    ea_model_path_1,
    ea_model_path_2,
    ea_model_path_3,
    chunk_size=500, 
    chunk_dir="data_chunks", 
    ea_output_dir="outputs", 
    log_file="results.json",
    lr_1=3e-5,
    lr_2=3e-5,
    lr_3=3e-5, 
    num_epochs=1, 
    offline=False,
    temperature=1.0
):
    print("=== EAGLE pipeline started ===")

    print(f"Splitting data file: {data_file}")
    chunk_files = save_chunks_to_jsonl(data_file, chunk_size=chunk_size, output_dir=chunk_dir)
    print(f"Data split into {len(chunk_files)} chunks, saved in: {chunk_dir}")
    
    ea_temp_dir_1 = os.path.join(ea_output_dir, f"1_ea_temp_model_{int(time.time())}") 
    ea_temp_dir_2 = os.path.join(ea_output_dir, f"2_ea_temp_model_{int(time.time())}")
    ea_temp_dir_3 = os.path.join(ea_output_dir, f"3_ea_temp_model_{int(time.time())}")
    ea_ensemble_dir = os.path.join(ea_output_dir, f"ea_ensemble_model_{int(time.time())}")
    for ea_model_path, ea_temp_dir in zip(
        [ea_model_path_1, ea_model_path_2, ea_model_path_3],
        [ea_temp_dir_1, ea_temp_dir_2, ea_temp_dir_3]
    ):
        if os.path.exists(ea_temp_dir):
            shutil.rmtree(ea_temp_dir)
        if os.path.isdir(ea_model_path):
            shutil.copytree(ea_model_path, ea_temp_dir)
        print(f"EA model path: {ea_model_path} copied to temporary directory: {ea_temp_dir} for overwriting")
    
    if os.path.exists(ea_ensemble_dir):
        shutil.rmtree(ea_ensemble_dir)
    os.makedirs(ea_ensemble_dir, exist_ok=True)
    shutil.copy(
        os.path.join(ea_model_path_1, "config.json"),
        os.path.join(ea_ensemble_dir, "config.json")
    )
    print(f"EA ensemble model config file copied to: {ea_ensemble_dir}")
    
    all_mean_speeds = []
    all_mean_accepts = []
    chunk_mean_speeds = []
    chunk_mean_accepts = []
    
    model_losses_history = []
    ensemble_weights_history = []
    
    for i, chunk_file in enumerate(chunk_files):
        print(f"\n{'='*80}")
        print(f"Processing chunk {i+1}/{len(chunk_files)}: {chunk_file}")
        print(f"{'='*80}")
        
        print(f"\n[Step 1/4] Merge - merge 3 base learners to get meta learner")
        if i == 0:
            current_weights = (1.0/3, 1.0/3, 1.0/3)
            print(f"  ├─ First merge, using equal weights: {current_weights}")
        else:
            last_losses = model_losses_history[-1]
            current_weights = compute_softmax_weights_from_losses(last_losses, temperature=temperature)
            print(f"  ├─ Based on last training loss {last_losses}")
            print(f"  └─ Using softmax weights (temperature={temperature}): {current_weights}")
        
        ensemble_weights_history.append(current_weights)
        
        print(f"\n[Step 2/4] Eval - evaluate using meta learner")
        print(f"  ├─ Evaluation data: {chunk_file}")
        print(f"  └─ Meta learner path: {ea_ensemble_dir}")
        speeds_list, accept_lengths_list = evaluate(
            chunk_file, 
            base_model_path, 
            ea_temp_dir_1, 
            ea_temp_dir_2, 
            ea_temp_dir_3,
            ea_ensemble_dir,
            weights=current_weights
        )
        
        all_mean_speeds.extend(speeds_list)
        all_mean_accepts.extend(accept_lengths_list)
        chunk_mean_speeds.append(np.mean(speeds_list))
        chunk_mean_accepts.append(np.mean(accept_lengths_list))
        
        print(f"  ├─ Average processing speed: {np.mean(speeds_list):.2f} tokens/sec")
        print(f"  └─ Average acceptance rate: {np.mean(accept_lengths_list):.4f}")
        
        if offline:
            print(f"\n[Step 3/4] Train - offline mode, skip training")
        else:
            print(f"\n[Step 3/4] Train - train from meta learner, use different learning rates")
            chunk_file_with_assist = chunk_file.replace('.jsonl', '_with_assist.jsonl')
            print(f"  ├─ Training data: {chunk_file_with_assist}")
            print(f"  ├─ Starting point: meta learner ({ea_ensemble_dir})")
            print(f"  ├─ Learning rates: lr_1={lr_1}, lr_2={lr_2}, lr_3={lr_3}")
            print(f"  └─ Training epochs: {num_epochs}")

            train_success, losses = train(
                chunk_file_with_assist,
                base_model_path,
                ea_model_path_1=None,
                ea_model_path_2=None,
                ea_model_path_3=None,
                ea_output_dir_1=ea_temp_dir_1,
                ea_output_dir_2=ea_temp_dir_2,
                ea_output_dir_3=ea_temp_dir_3,
                lr_1=lr_1, lr_2=lr_2, lr_3=lr_3,
                num_epochs=num_epochs,
                use_meta_learner=True,
                meta_learner_path=ea_ensemble_dir
            )
            
            if train_success:
                print(f"\n  Training completed successfully")
                print(f"  ├─ Loss 1: {losses[0]:.4f} (lr={lr_1})")
                print(f"  ├─ Loss 2: {losses[1]:.4f} (lr={lr_2})")
                print(f"  └─ Loss 3: {losses[2]:.4f} (lr={lr_3})")
                model_losses_history.append(losses)
                
                print(f"\n[Step 4/4] Update - update weights for next merge")
                print(f"  └─ Next time will calculate softmax weights based on these losses")
            else:
                print(f"\n  Training failed")
                return -1
    
    print(f"\n{'='*80}")
    print(f"Pipeline executed successfully")
    print(f"{'='*80}")
    print(f"\nOverall performance metrics:")
    print(f"  ├─ Average acceptance: {np.mean(chunk_mean_accepts):.4f}")
    print(f"  └─ Average processing speed: {np.mean(chunk_mean_speeds):.2f} tokens/sec")
    print(f"\nModel saved location:")
    print(f"  ├─ Base Learner 1: {ea_temp_dir_1}")
    print(f"  ├─ Base Learner 2: {ea_temp_dir_2}")
    print(f"  ├─ Base Learner 3: {ea_temp_dir_3}")
    print(f"  └─ Meta Learner (final ensemble): {ea_ensemble_dir}")
    
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump({
            "mean_accept": np.mean(chunk_mean_accepts),
            "mean_speed_tokens_per_sec": np.mean(chunk_mean_speeds),
            "chunk_mean_accepts": chunk_mean_accepts,
            "chunk_mean_speeds": chunk_mean_speeds,
            "all_mean_accepts": all_mean_accepts,
            "all_mean_speeds": all_mean_speeds,
            "final_ea_model_paths": [ea_temp_dir_1, ea_temp_dir_2, ea_temp_dir_3],
            "ensemble_model_path": ea_ensemble_dir,
            "model_losses_history": model_losses_history,
            "ensemble_weights_history": ensemble_weights_history
        }, f, indent=4)
    print(f"Results saved to: {log_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Hedge pipeline')
    parser.add_argument('--data-file', type=str, required=True, help='Input data file path')
    parser.add_argument('--base-model-path', type=str, required=True, help='Base model path')
    parser.add_argument('--ea-model-path-1', type=str, required=True, help='EA model path')
    parser.add_argument('--ea-model-path-2', type=str, required=True, help='EA model path')
    parser.add_argument('--ea-model-path-3', type=str, required=True, help='EA model path')
    parser.add_argument('--chunk-size', type=int, default=8, help='Chunk size')
    parser.add_argument('--chunk-dir', type=str, default='data_chunks', help='Chunk data save directory')
    parser.add_argument('--output-ea-dir', type=str, default='outputs', help='Output base directory')
    parser.add_argument('--eval-args', type=str, default='', help='Evaluation additional parameters')
    parser.add_argument('--lr-1', type=float, default=3e-5, help='Training learning rate')
    parser.add_argument('--lr-2', type=float, default=3e-5, help='Training learning rate')
    parser.add_argument('--lr-3', type=float, default=3e-5, help='Training learning rate')
    parser.add_argument('--num-epochs', type=int, default=1, help='Training epochs')
    parser.add_argument('--temperature', type=float, default=1.2, help='Softmax temperature parameter (< 1.0: more concentrated, = 1.0: standard, > 1.0: more uniform)')
    parser.add_argument('--offline', action='store_true', help='Offline mode: do not execute training and always use the original EAGLE model')
    parser.add_argument('--debug-chunks', type=int, help='Debug mode: only process the first N chunks')
    parser.add_argument('--log-file', type=str, default=None, help='Result log file path')
    
    args = parser.parse_args()
    
    if args.debug_chunks:
        run_pipeline.debug_chunks = args.debug_chunks
    
    chunk_suffix = f"_chunk{args.chunk_size}"
    lr_epoch_suffix = f"_lr{args.lr_1}_epoch{args.num_epochs}"
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
        ea_model_path_1=args.ea_model_path_1,
        ea_model_path_2=args.ea_model_path_2,
        ea_model_path_3=args.ea_model_path_3,
        chunk_size=args.chunk_size,
        chunk_dir=chunk_dir_with_params,
        ea_output_dir=args.output_ea_dir,
        log_file=log_file,
        lr_1=args.lr_1,
        lr_2=args.lr_2,
        lr_3=args.lr_3,
        num_epochs=args.num_epochs,
        offline=args.offline,
        temperature=args.temperature
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
