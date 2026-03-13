import asyncio
from transformers import AutoTokenizer
import os
import datetime
import argparse
import random
import json
import time

from vllm.inputs import TokensPrompt
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM

class BaselineModel:
    def __init__(self, model_path, eos_ids, gpu_ids):
        self.eos_ids = eos_ids
        self.request_counter = 0
        
        print(f"Initializing BaselineModel: {model_path}")
        print(f"GPU IDs: {gpu_ids} | EOS IDs: {eos_ids}")
        
        engine_args = AsyncEngineArgs(
            model=model_path,
            enforce_eager=True,
            tensor_parallel_size=len(gpu_ids.split(",")),
            data_parallel_size=1,
            gpu_memory_utilization=0.7,
            enable_prefix_caching=True,
            enable_chunked_prefill=True,
            max_num_batched_tokens=8192,
        )
        
        self.engine = AsyncLLM.from_engine_args(engine_args)
        self.prefix = model_path.split("/")[-1].split("-")[0]
    
    async def generate(
        self,
        token_ids,
        max_tokens=100,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        stop=None,
        repetition_penalty=1.0,
    ):
        if stop is None:
            stop = []
        
        prompt = TokensPrompt(prompt_token_ids=token_ids)
        
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            ignore_eos=False,
            output_kind=RequestOutputKind.FINAL_ONLY,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
            seed=int(time.time()),
        )
        
        self.request_counter += 1
        
        async for output in self.engine.generate(
            request_id=f"{self.prefix}-req-{self.request_counter}",
            prompt=prompt,
            sampling_params=sampling_params,
        ):
            await asyncio.sleep(0.0)
        
        result = output.outputs[0]
        token_ids_list = list(result.token_ids)
        num_tokens = len(token_ids_list)
        
        return (
            result.text,
            result.finish_reason,
            result.stop_reason,
            num_tokens,
            token_ids_list,
        )


MODEL_CONFIGS = {
    "qwen3": {
        "name": "qwen3",
        "temperature": 0,
        "top_p": 0.95,
        "top_k": 20,
        "max_tokens": 2048,
        "prompt_template": "qwen3",
        "eos_id": [151643, 151645],
        "stop": ["\n\n"],
        "step_tokens": 100,
        "repetition_penalty": 1.2,
    }
}


def get_model_config(model_name):
    """Get configuration for a specific model."""
    model_name_lower = model_name.lower()

    # Match model configurations
    if "deepseek" in model_name_lower:
        return MODEL_CONFIGS["deepseek"]
    elif "qwen3" in model_name_lower:
        return MODEL_CONFIGS["qwen3"]
    else:
        return MODEL_CONFIGS['qwen3'] # if using qwen3 as default target model

def load_questions(file_path):
    """Load questions from jsonl file"""
    questions = []
    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            questions.append(data)
    return questions

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="aime-2024.jsonl")
parser.add_argument("--start_qid", type=int, default=None)
parser.add_argument("--end_qid", type=int, default=None)
parser.add_argument("--prefix", type=str, default="AIME_NOSPEC")
parser.add_argument("--output_dir", type=str, default="None")

parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
parser.add_argument("--target_gpu_id", type=str, default="7")
parser.add_argument("--max_tokens_len", type=int, default=37000)

args = parser.parse_args()

async def run_problem_no_spec(question, i, target_model, target_tokenizer, target_config, output_dir):
    question_id = question['id'] if 'id' in question else i
    question_text = question['question']
    gold_answer = question['answer'] if 'answer' in question else None

    prompt = question_text + "\nPlease reason step by step, and put your final answer within \\boxed{{}}. Once a full round of reasoning is completed, do not check again, immediately cease reasoning, and output the answer.\n"
    
    target_prompt = target_tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=True, 
    )
    target_prompt += "<think>\n"

    target_token_ids = target_tokenizer.encode(target_prompt, add_special_tokens=False)
    
    current_token_ids = target_token_ids.copy()
    current_text = target_prompt
    all_generation_tokens = []
    
    max_thinking_steps = 10
    step_tokens = target_config.get('step_tokens', 100)
    max_tokens = target_config['max_tokens']
    eos_ids = target_config['eos_id']
    
    THINK_END_MAP = {
        'qwen3': 151668,
        'deepseek': 151649,
    }
    think_end_token_id = THINK_END_MAP.get(target_config['name'], 151668)
    extra_token_len = 0
    
    print(f'Running question: {question_id} | Prompt length: {len(target_token_ids)} tokens')
    # print(f'Step tokens: {step_tokens} | Max thinking steps: {max_thinking_steps}')
    # print(f'EOS token IDs: {eos_ids} | Think end token: {think_end_token_id}')
    
    if any(eos_id in target_token_ids for eos_id in eos_ids):
        print(f'[warn] Prompt already contains EOS token!')

    t0 = time.time()
    current_step = 0
    early_stop_added = False
    
    while True:
        current_step += 1
        
        if current_step > max_thinking_steps:
            if think_end_token_id not in current_token_ids:
                print(f"[info] Thinking budget of {max_thinking_steps} steps reached. Forcing </think>.")
                
                early_stopping_text = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"
                early_stopping_ids = target_tokenizer.encode(early_stopping_text, add_special_tokens=False)
                extra_token_len = len(early_stopping_ids)
                
                current_text += early_stopping_text
                current_token_ids += early_stopping_ids
                all_generation_tokens += early_stopping_ids
        
        if len(all_generation_tokens) > 1024 and not early_stop_added:
            print(f"[info] Token limit of 1024 reached. Adding early stopping prompt.")
            early_stopping_text = "\n\nI have to give the answer directly now:\n\n"
            early_stopping_ids = target_tokenizer.encode(early_stopping_text, add_special_tokens=False)
            extra_token_len += len(early_stopping_ids)
            current_text += early_stopping_text
            current_token_ids += early_stopping_ids
            all_generation_tokens += early_stopping_ids
            early_stop_added = True
        
        if len(all_generation_tokens) >= max_tokens:
            print(f"[info] Reached max_tokens limit: {max_tokens}")
            break
        
        remaining_tokens = max_tokens - len(all_generation_tokens)
        current_step_max_tokens = min(step_tokens, remaining_tokens)
        
        if current_step_max_tokens <= 0:
            break
        
        response = await target_model.generate(
            current_token_ids, 
            max_tokens=current_step_max_tokens,
            temperature=target_config['temperature'],
            top_p=target_config['top_p'], 
            top_k=target_config['top_k'],
            stop=target_config['stop'],
            repetition_penalty=target_config['repetition_penalty']
        )
        
        step_text = response[0]
        finish_reason = response[1]
        stop_reason = response[2]
        step_tokens_list = response[4]
        
        current_text += step_text
        current_token_ids += step_tokens_list
        all_generation_tokens += step_tokens_list
        
        if finish_reason == 'stop' and stop_reason in eos_ids:
            print(f"[info] Found EOS token (ID: {stop_reason}) at step {current_step}")
            break
        
        if finish_reason == 'stop' and stop_reason is None:
            if any(eos_id in step_tokens_list for eos_id in eos_ids):
                print(f"[info] Found EOS token in generated tokens at step {current_step}")
                break
            else:
                print(f"[warn] finish_reason='stop' but no EOS found, stopping at step {current_step}")
                break
        
        if len(step_tokens_list) == 0:
            print(f"[warn] No tokens generated at step {current_step}, stopping")
            break
        
        if current_step > 100:
            print(f"[warn] Exceeded 100 steps, force stopping")
            break
    
    t1 = time.time()
    time_taken = t1 - t0
    
    actual_generated_tokens = len(all_generation_tokens) - extra_token_len
    speed = actual_generated_tokens / time_taken if time_taken > 0 else 0
    
    generation_text = target_tokenizer.decode(all_generation_tokens, skip_special_tokens=False)
    full_text = target_prompt + generation_text
    full_tokens = target_token_ids + all_generation_tokens

    print(f'Finished: {len(all_generation_tokens)} tokens ({actual_generated_tokens} actual) | '
          f'Steps: {current_step} | Time: {time_taken:.2f}s | Speed: {speed:.2f} tokens/s')

    result = {
        'question_id': question_id,
        'question': question_text,
        'target_prompt': target_prompt,
        'generation_tokens': all_generation_tokens,
        'generation_text': generation_text,
        'full_text': full_text,
        'full_tokens': full_tokens,
        'gold': gold_answer,
        'time_taken': time_taken,
        'speed': speed,
        'total_inference_steps': current_step,
        'extra_token_len': extra_token_len,
        'target_config': target_config,
    }

    output_file = os.path.join(output_dir, f"{question_id}.json")
    with open(output_file, 'w') as f:
        json.dump(result, f)

async def main():
    if args.output_dir != "None":
        output_dir = args.output_dir
    else:
        output_dir = (
            args.prefix
            + "_"
            + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            + "_"
            + str(random.randint(100000, 999999))
        )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    questions = load_questions(args.dataset)[args.start_qid : args.end_qid]

    target_tokenizer = AutoTokenizer.from_pretrained(args.model)
    target_config = get_model_config(args.model)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.target_gpu_id
    target_model = BaselineModel(
        model_path=args.model,
        eos_ids=target_config["eos_id"],
        gpu_ids=args.target_gpu_id,
    )

    print(f"Target Model: {args.model}")
    print(f"Config: {target_config}")

    for i in range(len(questions)):
        await run_problem_no_spec(
            questions[i],
            i,
            target_model,
            target_tokenizer,
            target_config,
            output_dir
        )

    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    asyncio.run(main())