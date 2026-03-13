import subprocess
import os
import shutil
from datetime import datetime
import torch
import time
import json
import numpy as np
import torch.nn.functional as F
from typing import List

from load_data import save_chunks_to_jsonl
from eagle3.ea_model import EaModel
from eagle3.utils import *
from train_eagle3.configs import EConfig


def detect_model_template(base_model_path):
    model_path_lower = base_model_path.lower()
    if 'llama-2' in model_path_lower or 'llama2' in model_path_lower:
        return 'llama-2'
    if 'vicuna' in model_path_lower:
        return 'vicuna'
    return 'vicuna'


def speed(jsonl_file, report=True):
    """
    Calculate the number of tokens processed per second and the accepted token length list from the answer_file.jsonl.
    """
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


def compute_softmax_weights_from_losses(losses, temperature=1.0):
    """
    Calculate softmax weights from losses. The smaller the loss, the greater the weight.
    """
    losses_tensor = torch.tensor(losses, dtype=torch.float32)
    neg_losses = -losses_tensor
    weights = F.softmax(neg_losses / temperature, dim=0)
    print(f"Softmax weights from losses {losses} (temperature={temperature}): {weights.tolist()}")
    return tuple(weights.tolist())


def ensemble_ea3_models(ea_model_path_1, ea_model_path_2, ea_model_path_3, weights=(1.0, 1.0, 1.0), out_dir=None, save_name="pytorch_model.bin"):
    """
    Weight the parameters of the three EAGLE3 model paths by weights and return the merged state_dict and save it.
    """
    print(f"\nLoading model 1: {ea_model_path_1}")
    load_model_path_1 = os.path.join(ea_model_path_1, "pytorch_model.bin")
    sd_a = torch.load(load_model_path_1, map_location='cpu')
    
    print(f"Loading model 2: {ea_model_path_2}")
    load_model_path_2 = os.path.join(ea_model_path_2, "pytorch_model.bin")
    sd_b = torch.load(load_model_path_2, map_location='cpu')
    
    print(f"Loading model 3: {ea_model_path_3}")
    load_model_path_3 = os.path.join(ea_model_path_3, "pytorch_model.bin")
    sd_c = torch.load(load_model_path_3, map_location='cpu')
    
    wa, wb, wc = weights
    total = float(wa + wb + wc)
    assert total > 0, "The sum of weights must be greater than 0"
    wa, wb, wc = wa / total, wb / total, wc / total
    print(f"Normalized weights: ({wa:.4f}, {wb:.4f}, {wc:.4f})")

    keys = list(sd_a.keys())
    
    new_sd = {}
    original_dtypes = {}
    
    print(f"\nMerging ({len(keys)} parameters)...")
    
    for k in keys:
        va = sd_a[k]
        
        if not isinstance(va, torch.Tensor):
            new_sd[k] = va
            continue

        original_dtypes[k] = va.dtype
        
        vb = sd_b.get(k, None)
        vc = sd_c.get(k, None)

        if vb is None:
            vb = torch.zeros_like(va)
            print(f"Warning: model 2 missing key: {k}")
        if vc is None:
            vc = torch.zeros_like(va)
            print(f"Warning: model 3 missing key: {k}")

        va_bf16 = va.to(torch.bfloat16)
        vb_bf16 = vb.to(torch.bfloat16)
        vc_bf16 = vc.to(torch.bfloat16)
        new_val_bf16 = wa * va_bf16 + wb * vb_bf16 + wc * vc_bf16
        new_val = new_val_bf16.to(torch.float16)
        new_sd[k] = new_val
    
    print(f"Merge completed")
    
    save_path = None
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, save_name)
        torch.save(new_sd, save_path)
        print(f"EAGLE3 ensemble model saved to: {save_path}")
        
        config_path_1 = os.path.join(ea_model_path_1, "config.json")
        if os.path.exists(config_path_1):
            shutil.copy(config_path_1, os.path.join(out_dir, "config.json"))
            print(f"Copied config.json to ensemble directory")
        else:
            print(f"Warning: config.json not found in {ea_model_path_1}")
        
        cache_path_1 = os.path.join(ea_model_path_1, "cache.pt")
        if os.path.exists(cache_path_1):
            shutil.copy(cache_path_1, os.path.join(out_dir, "cache.pt"))
            print(f"Copied cache.pt to ensemble directory")


def evaluate(data_file, base_model_path, ea_model_path_1, ea_model_path_2, ea_model_path_3, 
             ensemble_model_path, weights=(1.0, 1.0, 1.0), total_token=60, depth=5, top_k=10, temperature=0.0):
    """Evaluate the EAGLE3 ensemble model."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("CUDA cache cleared")
    
    print(f"Using weights {weights} for EAGLE3 ensemble")
    ensemble_ea3_models(
        ea_model_path_1,
        ea_model_path_2,
        ea_model_path_3,
        weights=weights,
        out_dir=ensemble_model_path
    )
    
    print(f"Loading EAGLE3 ensemble model from {ensemble_model_path}...")
    
    use_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    model = EaModel.from_pretrained(
        base_model_path=base_model_path,
        ea_model_path=ensemble_model_path,
        total_token=total_token,
        depth=depth,
        top_k=top_k,
        torch_dtype=use_dtype,
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
    
    from fastchat.model import get_conversation_template
    from tqdm import tqdm
    
    if temperature > 1e-5:
        logits_processor = prepare_logits_processor(temperature=temperature)
    else:
        logits_processor = None
    
    model.eval()
    print('Check model training state:', model.training)
    
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
            
            if "CUDA" in str(e) or "cuda" in str(e):
                print("Warning: CUDA error detected, trying to clear CUDA cache...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    print("CUDA cache cleared")
            
            continue
    
    if accept_lengths_list:
        mean_accept = np.mean(accept_lengths_list)
        print(f"\n# Total speculative decoding steps: {len(accept_lengths_list)}")
        print(f"# Mean accepted tokens per step: {mean_accept:.2f}")
    print(f"# Results saved to: {output_file}")
    
    evaluate_time = time.time() - evaluate_start_time
    
    if torch.cuda.is_available():
        del model
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"Evaluation completed, GPU memory cleared")
    
    speeds, accept_lengths = speed(output_file, report=True)
    return speeds, accept_lengths, evaluate_time


def train(data_file, base_model_path, 
          ea_model_path_1, ea_model_path_2, ea_model_path_3, 
          ea_output_dir_1, ea_output_dir_2, ea_output_dir_3, 
          batch_size=4, num_epochs=1, 
          lr_1=1e-3, lr_2=1e-3, lr_3=1e-3, 
          num_epochs_1=None, num_epochs_2=None, num_epochs_3=None,
          max_len=2048,
          use_meta_learner=False,
          meta_learner_path=None):
    ea_model_paths = [ea_model_path_1, ea_model_path_2, ea_model_path_3]
    ea_output_dirs = [ea_output_dir_1, ea_output_dir_2, ea_output_dir_3]
    lrs = [lr_1, lr_2, lr_3]
    num_epochs_list = [
        num_epochs_1 if num_epochs_1 is not None else num_epochs,
        num_epochs_2 if num_epochs_2 is not None else num_epochs,
        num_epochs_3 if num_epochs_3 is not None else num_epochs
    ]
    
    losses = []
    train_time = 0.0
    
    if use_meta_learner and meta_learner_path:
        print(f"\nPreparing training environment: copying meta learner to 3 training directories")
        for idx, ea_output_dir in enumerate(ea_output_dirs, 1):
            if os.path.exists(ea_output_dir):
                shutil.rmtree(ea_output_dir)
            shutil.copytree(meta_learner_path, ea_output_dir)
            print(f"    ├─ [{idx}/3] {os.path.basename(ea_output_dir)}")
        print(f"    └─ Completed! ")
        ea_model_paths = ea_output_dirs
    
    for i, (ea_model_path, ea_output_dir, lr, epochs) in enumerate(zip(ea_model_paths, ea_output_dirs, lrs, num_epochs_list), 1):
        cmd = [
            "python", "traineagle3.py",
            "--basepath", base_model_path,
            "--trainpath", data_file,
            "--savedir", ea_output_dir,
            "--batch_size", str(batch_size),
            "--num_epochs", str(epochs),
            "--lr", str(lr),
            "--max_len", str(max_len)
        ]
        
        if ea_model_path and os.path.exists(ea_model_path):
            if os.path.exists(os.path.join(ea_model_path, "config.json")):
                cmd.extend(["--ea-model-path", ea_model_path])
        
        print(f"\nTrainer {i}/3 (lr={lr}, epochs={epochs}):")
        print(f"    ├─ Command: {' '.join(cmd)}")
        
        env = os.environ.copy()
        env["WANDB_DISABLED"] = "true"
        
        eagle_dir = os.path.dirname(os.path.abspath(__file__))
        
        if i == 1:
            train_start_time = time.time()
        
        result = subprocess.run(cmd, env=env, cwd=eagle_dir)
        
        if i == 1:
            train_time = time.time() - train_start_time
        
        if result.returncode != 0:
            print(f"    └─ Training failed, return code: {result.returncode}")
            return False, None, train_time if i == 1 else 0.0
        
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
    
    return True, losses, train_time


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
    batch_size=4,
    num_epochs=1,
    lr_1=1e-3,
    lr_2=1e-3,
    lr_3=1e-3,
    num_epochs_1=None,
    num_epochs_2=None,
    num_epochs_3=None,
    max_len=2048,
    total_token=60,
    depth=5,
    top_k=10,
    temperature=0.0,
    offline=False,
    softmax_temperature=1.0
):
    print("=== EAGLE3 pipeline execution started ===")

    print(f"Splitting data file: {data_file}")
    chunk_files = save_chunks_to_jsonl(data_file, chunk_size=chunk_size, output_dir=chunk_dir)
    print(f"Data split into {len(chunk_files)} chunks, saved in: {chunk_dir}")
    
    ea_temp_dir_1 = os.path.join(ea_output_dir, f"1_ea3_temp_model_{int(time.time())}") 
    ea_temp_dir_2 = os.path.join(ea_output_dir, f"2_ea3_temp_model_{int(time.time())}")
    ea_temp_dir_3 = os.path.join(ea_output_dir, f"3_ea3_temp_model_{int(time.time())}")
    ea_ensemble_dir = os.path.join(ea_output_dir, f"ea3_ensemble_model_{int(time.time())}")
    
    for ea_model_path, ea_temp_dir in zip(
        [ea_model_path_1, ea_model_path_2, ea_model_path_3],
        [ea_temp_dir_1, ea_temp_dir_2, ea_temp_dir_3]
    ):
        if os.path.exists(ea_temp_dir):
            shutil.rmtree(ea_temp_dir)
        if os.path.isdir(ea_model_path):
            shutil.copytree(ea_model_path, ea_temp_dir)
        print(f"EA3 model path: {ea_model_path} copied to temporary directory: {ea_temp_dir} for overwrite")
    
    if os.path.exists(ea_ensemble_dir):
        shutil.rmtree(ea_ensemble_dir)
    os.makedirs(ea_ensemble_dir, exist_ok=True)
    shutil.copy(
        os.path.join(ea_model_path_1, "config.json"),
        os.path.join(ea_ensemble_dir, "config.json")
    )
    cache_path = os.path.join(ea_model_path_1, "cache.pt")
    if os.path.exists(cache_path):
        shutil.copy(cache_path, os.path.join(ea_ensemble_dir, "cache.pt"))
    print(f"EA3 ensemble model config file copied to: {ea_ensemble_dir}")
    
    all_mean_speeds = []
    all_mean_accepts = []
    chunk_mean_speeds = []
    chunk_mean_accepts = []
    chunk_evaluate_times = []
    chunk_train_times = []
    total_time = 0.0
    
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
            current_weights = compute_softmax_weights_from_losses(
                last_losses, temperature=softmax_temperature
            )
            print(f"  ├─ Based on last training loss {last_losses}")
            print(f"  └─ Using softmax weights (temperature={softmax_temperature}): {current_weights}")
        
        ensemble_weights_history.append(current_weights)
        
        print(f"\n[Step 2/4] Eval - use meta learner for evaluation")
        print(f"  ├─ Evaluation data: {chunk_file}")
        print(f"  └─ Meta learner path: {ea_ensemble_dir}")
        speeds_list, accept_lengths_list, evaluate_time = evaluate(
            chunk_file, 
            base_model_path, 
            ea_temp_dir_1, 
            ea_temp_dir_2, 
            ea_temp_dir_3,
            ea_ensemble_dir,
            weights=current_weights,
            total_token=total_token,
            depth=depth,
            top_k=top_k,
            temperature=temperature
        )
        chunk_evaluate_times.append(evaluate_time)
        total_time += evaluate_time
        print(f"  ├─ Evaluation time: {evaluate_time:.2f} seconds")
        
        all_mean_speeds.extend(speeds_list)
        all_mean_accepts.extend(accept_lengths_list)
        chunk_mean_speeds.append(np.mean(speeds_list))
        chunk_mean_accepts.append(np.mean(accept_lengths_list))
        
        print(f"  ├─ Average processing speed: {np.mean(speeds_list):.2f} tokens/sec")
        print(f"  └─ Average acceptance rate: {np.mean(accept_lengths_list):.4f}")

        if offline:
            print(f"\n[Step 3/4] Train - offline mode, skip training")
        else:
            print(f"\n[Step 3/4] Train - use meta learner as starting point, train with different learning rates")
            chunk_file_with_assist = chunk_file.replace('.jsonl', '_with_assist.jsonl')
            print(f"  ├─ Training data: {chunk_file_with_assist}")
            print(f"  ├─ Starting point: meta learner ({ea_ensemble_dir})")
            print(f"  ├─ Learning rates: lr_1={lr_1}, lr_2={lr_2}, lr_3={lr_3}")
            epochs_1 = num_epochs_1 if num_epochs_1 is not None else num_epochs
            epochs_2 = num_epochs_2 if num_epochs_2 is not None else num_epochs
            epochs_3 = num_epochs_3 if num_epochs_3 is not None else num_epochs
            print(f"  └─ Training epochs: epochs_1={epochs_1}, epochs_2={epochs_2}, epochs_3={epochs_3}")

            train_success, losses, train_time = train(
                chunk_file_with_assist,
                base_model_path,
                ea_model_path_1=None,
                ea_model_path_2=None,
                ea_model_path_3=None,
                ea_output_dir_1=ea_temp_dir_1,
                ea_output_dir_2=ea_temp_dir_2,
                ea_output_dir_3=ea_temp_dir_3,
                batch_size=batch_size,
                num_epochs=num_epochs,
                lr_1=lr_1, lr_2=lr_2, lr_3=lr_3,
                num_epochs_1=num_epochs_1,
                num_epochs_2=num_epochs_2,
                num_epochs_3=num_epochs_3,
                max_len=max_len,
                use_meta_learner=True,
                meta_learner_path=ea_ensemble_dir
            )
            chunk_train_times.append(train_time)
            total_time += train_time
            print(f"  ├─ Training time (only first model): {train_time:.2f} seconds")
            
            if train_success:
                print(f"\nTraining completed successfully")
                print(f"  ├─ Loss 1: {losses[0]:.4f} (lr={lr_1}, epochs={epochs_1})")
                print(f"  ├─ Loss 2: {losses[1]:.4f} (lr={lr_2}, epochs={epochs_2})")
                print(f"  └─ Loss 3: {losses[2]:.4f} (lr={lr_3}, epochs={epochs_3})")
                model_losses_history.append(losses)
                
                print(f"\n[Step 4/4] Update - update weights for next merge")
                print(f"  └─ Next merge will use these losses to calculate softmax weights")
            else:
                print(f"\nTraining failed")
                return -1
    
    print(f"\n{'='*80}")
    print(f"Pipeline execution completed")
    print(f"{'='*80}")
    print(f"\nOverall performance metrics:")
    print(f"  ├─ Average acceptance: {np.mean(chunk_mean_accepts):.4f}")
    print(f"  └─ Average processing speed: {np.mean(chunk_mean_speeds):.2f} tokens/sec")
    print(f"\nModel saved in:")
    print(f"  ├─ Base Learner 1: {ea_temp_dir_1}")
    print(f"  ├─ Base Learner 2: {ea_temp_dir_2}")
    print(f"  ├─ Base Learner 3: {ea_temp_dir_3}")
    print(f"  └─ Meta Learner (final ensemble): {ea_ensemble_dir}")
    
    actual_epochs_1 = num_epochs_1 if num_epochs_1 is not None else num_epochs
    actual_epochs_2 = num_epochs_2 if num_epochs_2 is not None else num_epochs
    actual_epochs_3 = num_epochs_3 if num_epochs_3 is not None else num_epochs
    
    result_dict = {
        "mean_accept": np.mean(chunk_mean_accepts),
        "mean_speed_tokens_per_sec": np.mean(chunk_mean_speeds),
        "chunk_mean_accepts": chunk_mean_accepts,
        "chunk_mean_speeds": chunk_mean_speeds,
        "all_mean_accepts": all_mean_accepts,
        "all_mean_speeds": all_mean_speeds,
        "final_ea_model_paths": [ea_temp_dir_1, ea_temp_dir_2, ea_temp_dir_3],
        "ensemble_model_path": ea_ensemble_dir,
        "model_losses_history": model_losses_history,
        "ensemble_weights_history": ensemble_weights_history,
        "training_config": {
            "lr_1": lr_1, "lr_2": lr_2, "lr_3": lr_3,
            "num_epochs_1": actual_epochs_1,
            "num_epochs_2": actual_epochs_2,
            "num_epochs_3": actual_epochs_3,
            "batch_size": batch_size,
            "max_len": max_len,
            "softmax_temperature": softmax_temperature
        },
        "total_time": total_time,
        "chunk_evaluate_times": chunk_evaluate_times
    }
    if not offline:
        result_dict["chunk_train_times"] = chunk_train_times
    
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, indent=4)
    print(f"Results saved to: {log_file}")
    print(f"Total time (evaluate + training): {total_time:.2f} seconds")
    
    if model_losses_history:
        print(f"\nTraining Loss history (loss after each chunk training):")
        print(f"  {'Chunk':<8} {'Loss 1':<12} {'Loss 2':<12} {'Loss 3':<12}")
        print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*12}")
        for idx, losses in enumerate(model_losses_history):
            print(f"  {idx+1:<8} {losses[0]:<12.4f} {losses[1]:<12.4f} {losses[2]:<12.4f}")

if __name__ == "__main__":
    import argparse
    
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            _ = torch.tensor([1.0], device='cuda')
            print("CUDA initialization successful")
        except Exception as e:
            print(f"CUDA initialization warning: {e}")
            print("If CUDA errors persist, please restart the Python process or use 'nvidia-smi --gpu-reset'")
    
    parser = argparse.ArgumentParser(description='EAGLE3 Online Ensemble pipeline')
    parser.add_argument('--data-file', type=str, required=True, help='Input data file path')
    parser.add_argument('--base-model-path', type=str, required=True, help='Base model path')
    parser.add_argument('--ea-model-path-1', type=str, required=True, help='EA3 model path 1')
    parser.add_argument('--ea-model-path-2', type=str, required=True, help='EA3 model path 2')
    parser.add_argument('--ea-model-path-3', type=str, required=True, help='EA3 model path 3')
    parser.add_argument('--chunk-size', type=int, default=8, help='Chunk size')
    parser.add_argument('--chunk-dir', type=str, default='data_chunks', help='Chunk data save directory')
    parser.add_argument('--output-ea-dir', type=str, default='outputs', help='Output base directory')
    
    parser.add_argument('--batch-size', type=int, default=4, help='Training batch size')
    parser.add_argument('--lr-1', type=float, default=1e-3, help='Model 1 training learning rate')
    parser.add_argument('--lr-2', type=float, default=1e-3, help='Model 2 training learning rate')
    parser.add_argument('--lr-3', type=float, default=1e-3, help='Model 3 training learning rate')
    parser.add_argument('--num-epochs', type=int, default=1, help='Training epochs (default for all models)')
    parser.add_argument('--num-epochs-1', type=int, default=None, help='Model 1 training epochs (if not specified, use --num-epochs)')
    parser.add_argument('--num-epochs-2', type=int, default=None, help='Model 2 training epochs (if not specified, use --num-epochs)')
    parser.add_argument('--num-epochs-3', type=int, default=None, help='Model 3 training epochs (if not specified, use --num-epochs)')
    parser.add_argument('--max-len', type=int, default=2048, help='Maximum sequence length')
    
    # Evaluation parameters
    parser.add_argument('--total-token', type=int, default=60, help='EAGLE total token number')
    parser.add_argument('--depth', type=int, default=5, help='EAGLE depth')
    parser.add_argument('--top-k', type=int, default=10, help='EAGLE top-k parameter')
    parser.add_argument('--temperature', type=float, default=0, help='Sampling temperature')
    
    # Ensemble parameters
    parser.add_argument('--softmax-temperature', type=float, default=0.2, help='Softmax temperature parameter (< 1.0: more concentrated, = 1.0: standard, > 1.0: more uniform)')
    
    # Other options
    parser.add_argument('--offline', action='store_true', help='Offline mode: do not execute training and always use the original EAGLE model')
    parser.add_argument('--debug-chunks', type=int, help='Debug mode: only process the first N chunks')
    parser.add_argument('--log-file', type=str, default=None, help='Result log file path')
    
    args = parser.parse_args()
    
    chunk_suffix = f"_chunk{args.chunk_size}"
    lr_epoch_suffix = f"_lr{args.lr_1}_epoch{args.num_epochs}"
    now = datetime.now()
    time_suffix = f"_{now.month}_{now.day}_{now.hour}_{now.minute}"

    chunk_dir_with_params = args.chunk_dir + lr_epoch_suffix + time_suffix
    if not args.log_file:
        log_file = "results_eagle3_hedge" + chunk_suffix + lr_epoch_suffix + time_suffix + ".json"
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
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr_1=args.lr_1,
        lr_2=args.lr_2,
        lr_3=args.lr_3,
        num_epochs_1=args.num_epochs_1,
        num_epochs_2=args.num_epochs_2,
        num_epochs_3=args.num_epochs_3,
        max_len=args.max_len,
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.top_k,
        temperature=args.temperature,
        offline=args.offline,
        softmax_temperature=args.softmax_temperature
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
