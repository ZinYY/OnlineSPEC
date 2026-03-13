import os
import subprocess
import numpy as np
import json
import random
import argparse


def split_files(source_file, output_dir, chunk_size=20):
    with open(source_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for i in range(0, len(lines), chunk_size):
        chunk = lines[i:i + chunk_size]
        chunk_dir = os.path.join(output_dir, f'chunk_{i // chunk_size}')
        os.makedirs(chunk_dir, exist_ok=True)
        chunk_file = os.path.join(chunk_dir, f'chunk_{i // chunk_size}.jsonl')
        with open(chunk_file, 'w', encoding='utf-8') as cf:
            cf.writelines(chunk)
        print(f"Created {chunk_file} with {len(chunk)} lines.")
    return output_dir


def get_speed_and_accept(data_dir):
    speeds = []
    accepts = []
    avglens = []
    answer_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    data_file = os.path.join(data_dir, 'data.jsonl')
    with open(data_file, 'w') as data_file:
        for file in answer_files:
            file_path = os.path.join(data_dir, file)
            if not os.path.exists(file_path):
                print(f"File {file_path} does not exist.")
                return None
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if data['speed'] > 200:
                continue
            speeds.append(data['speed'])
            if data['total_inference_steps'] == 0:
                print(f"Warning: total_inference_steps is 0 in {file_path}.")
                continue
            accepts.append(data['accepted_steps'] / data['total_inference_steps'])
            avglens.append(len(data['generation_tokens']) / data['total_inference_steps'] * data['accepted_steps'] / data['total_inference_steps'])
            data_file.write(
                json.dumps({
                    'prompt': data['draft_prompt'],
                    'answer': data['generation_text']
                }, ensure_ascii=False) + '\n'
            )
    return speeds, accepts, avglens


def run_lookahead_reasoning(input_file, draft_model_path, target_model):
    output_dir = os.path.dirname(input_file)
    env = os.environ.copy()
    cmd = f"python main.py --dataset {input_file} --model {target_model} --draft_model {draft_model_path} --output_dir {output_dir} --use_spec --enable_n_gram"
    ret = subprocess.run(cmd, shell=True, env=env)
    if ret.returncode != 0:
        raise RuntimeError(f"Lookahead reasoning failed for {input_file}")
    print(f"Processed {input_file} and saved results to {output_dir}")
    return output_dir


def dpo_finetune(data_dir, save_path, model_path):
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '5,6'
    cmd = [
        'deepspeed', '--master_port', '29501', 'dpo_train.py',
        '--pretrain', model_path,
        '--dataset', os.path.join(data_dir, 'verifier_negatives.jsonl'),
        '--save_path', save_path,
        '--train_batch_size', '12',
        '--micro_train_batch_size', '2',
        '--learning_rate', '5e-7',
        '--beta', '0.1',
        '--logging_steps', '1',
        '--save_steps', '-1',
        '--eval_steps', '-1',
        '--zero_stage', '1',
        '--bf16',
        '--max_epochs', '3',
        '--max_len', '2048',
        '--chosen_key', 'chosen',
        '--rejected_key', 'rejected',
        '--attn_implementation', 'flash_attention_2']
    # ret = subprocess.run(' '.join(cmd), shell=True, env=env)
    ret = subprocess.run(cmd, env=env, check=False, start_new_session=True)
    if ret.returncode != 0:
        raise RuntimeError(f"DPO fine-tuning failed for data in {data_dir}")
    print(f"DPO fine-tuning completed. Model saved to {model_path}")
    return model_path


def sft_finetune(data_dir, save_path, model_path):
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '5,6'
    cmd = [
        'deepspeed', '--master_port', '29502', 'sft_train.py',
        '--pretrain', model_path,
        # '--dataset', os.path.join(data_dir, 'data.jsonl'),
        '--dataset', os.path.join(data_dir, 'verifier_negatives.jsonl'),
        '--input_key', 'prompt',
        # '--output_key', 'answer',
        '--output_key', 'chosen',
        '--max_len', '4096',
        '--train_batch_size', '12',
        '--micro_train_batch_size', '2',
        '--save_path', save_path,
        '--save_steps', '-1',
        '--logging_steps', '1',
        '--eval_steps', '-1',
        '--zero_stage', '1',
        '--max_epochs', '1',
        '--bf16',
        '--packing_samples',
        '--learning_rate', '5e-7',
        '--lr_scheduler', 'constant_with_warmup',
        '--lr_warmup_ratio', '0']
    ret = subprocess.run(cmd, env=env, check=False, start_new_session=True)
    if ret.returncode != 0:
        raise RuntimeError(f"SFT fine-tuning failed for data in {data_dir}")
    print(f"SFT fine-tuning completed. Model saved to {model_path}")
    return model_path
    

def run_pipeline(source_file, temp_dir, data_dir, chunk_size=20, draft_model_path=None, method='dpo', target_model=None, log_file=None):
    # Split files
    split_files(source_file, data_dir, chunk_size)
    # copy model to temp_dir
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    subprocess.run(f"cp -r '{draft_model_path}/.' '{temp_dir}/'", shell=True)
    print(f"Copied model from {draft_model_path} to {temp_dir}")
    
    if log_file is not None:
        log_file = log_file if log_file.endswith('.json') else log_file + '.json'
    else:
        random_num = data_dir.split('_')[-1]
        log_file = 'results_' + random_num + '.json'
    all_mean_speeds = []
    chunk_mean_speeds = []
    all_mean_accepts = []
    chunk_mean_accepts = []
    all_mean_avglens = []
    chunk_mean_avglens = []
    # Lookahead Reasoning
    chunk_dirs = [os.path.join(data_dir, f"chunk_{i}") for i in range((len(os.listdir(data_dir))))]
    for chunk_dir in chunk_dirs:
        chunk_file = os.path.join(chunk_dir, os.listdir(chunk_dir)[0])  # Assuming one file per chunk dir
        print(f"Processing {chunk_file} for lookahead reasoning...")
        run_lookahead_reasoning(chunk_file, temp_dir, target_model)
        print(f"Lookahead reasoning completed for {chunk_file}")
        speeds, accepts, avglens = get_speed_and_accept(chunk_dir)
        print(f'Average speeds for {chunk_dir}: {np.mean(speeds)}')
        all_mean_speeds.extend(speeds)
        chunk_mean_speeds.append(np.mean(speeds))
        print(f'Average acceptance rates for {chunk_dir}: {np.mean(accepts)}')
        all_mean_accepts.extend(accepts)
        chunk_mean_accepts.append(np.mean(accepts))
        print(f'Average average lengths for {chunk_dir}: {np.mean(avglens)}')
        all_mean_avglens.extend(avglens)
        chunk_mean_avglens.append(np.mean(avglens))
        with open(log_file, 'w', encoding='utf-8') as lf:
            json.dump({
                'all_mean_speeds': all_mean_speeds,
                'all_mean_accepts': all_mean_accepts,
                'chunk_mean_speeds': chunk_mean_speeds,
                'chunk_mean_accepts': chunk_mean_accepts,
                'all_mean_avglens': all_mean_avglens,
                'chunk_mean_avglens': chunk_mean_avglens
            }, lf, indent=4)

        if method == 'sft':
            print(f"Starting SFT fine-tuning for data in {chunk_dir}...")
            sft_finetune(chunk_dir, temp_dir, temp_dir)
            print(f"SFT fine-tuning completed for data in {chunk_dir}")

        if method == 'dpo':
            print(f"Starting DPO fine-tuning for data in {chunk_dir}...")
            dpo_finetune(chunk_dir, temp_dir, temp_dir)
            print(f"DPO fine-tuning completed for data in {chunk_dir}")

    
def main():
    parser = argparse.ArgumentParser(description="Run the lookahead reasoning pipeline.")
    parser.add_argument('--dataset', type=str, default='data/opc4k.jsonl', help='Path to the original data file')
    parser.add_argument('--draft_model_path', type=str, default='./deepseek-1.5B', help='Initial model path')
    parser.add_argument('--target_model', type=str, default='Qwen/Qwen3-8B', help='Target model for lookahead reasoning')
    parser.add_argument('--method', type=str, choices=['dpo', 'sft'], default='dpo', help='Fine-tuning method')
    parser.add_argument('--chunk_size', type=int, default=25, help='Number of samples per chunk')
    parser.add_argument('--temp_dir', type=str, default='temp_model', help='Temporary model directory')
    parser.add_argument('--data_dir', type=str, default=None, help='Directory for split data chunks')
    parser.add_argument('--log_file', type=str, default=None, help='Log file name (without .json extension). If not specified, uses random number scheme')

    args = parser.parse_args()

    dataset = args.dataset
    temp_dir = args.temp_dir
    if args.data_dir is not None:
        data_dir = args.data_dir
    else:
        data_dir = f'data_chunks_{random.randint(0,10000)}'
    chunk_size = args.chunk_size
    draft_model_path = args.draft_model_path
    method = args.method

    run_pipeline(dataset, temp_dir, data_dir, chunk_size, draft_model_path, method, args.target_model, args.log_file)

if __name__ == "__main__":
    main()
    