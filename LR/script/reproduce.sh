# set up judge model
CUDA_VISIBLE_DEVICES=7 vllm serve Qwen/Qwen2.5-7B-Instruct --enable-prefix-caching --port 6767

# You may need to modify the prompt in src/lr.py when changing the dataset. (Including problem dependent prompt and judge prompt)

# gsm8k

# offline
python main.py --dataset data/gsm2.5k.jsonl --draft_model ./Qwen3-0.6B-Base --model Qwen/Qwen3-8B --prefix GSM8k --use_spec
# online sft
python pipeline.py --dataset data/gsm2.5k.jsonl --draft_model_path ./Qwen3-0.6B-Base --target_model Qwen/Qwen3-8B --method sft
# offline dpo
python pipeline.py --dataset data/gsm2.5k.jsonl --draft_model_path ./Qwen3-0.6B-Base --target_model Qwen/Qwen3-8B --method dpo
# target model
python baseline.py --dataset data/gsm2.5k.jsonl --prefix GSM8k_TARGET --model Qwen/Qwen3-8B
# draft model
python baseline.py --dataset data/gsm2.5k.jsonl --prefix GSM8k_DRAFT --model ./Qwen3-0.6B

# mbpp
# offline
python main.py --dataset data/mbpp.jsonl --draft_model ./Qwen3-0.6B-Base --model Qwen/Qwen3-8B --prefix MBPP --use_spec
# online sft
python pipeline.py --dataset data/mbpp.jsonl --draft_model_path ./Qwen3-0.6B-Base --target_model Qwen/Qwen3-8B --method sft
# offline dpo
python pipeline.py --dataset data/mbpp.jsonl --draft_model_path ./Qwen3-0.6B-Base --target_model Qwen/Qwen3-8B --method dpo
# target model
python baseline.py --dataset data/mbpp.jsonl --prefix MBPP_TARGET --model Qwen/Qwen3-8B
# draft model
python baseline.py --dataset data/mbpp.jsonl --prefix MBPP_DRAFT --model ./Qwen3-0.6B

# math
# offline
python main.py --dataset data/math.jsonl --draft_model ./Qwen3-0.6B-Base --model Qwen/Qwen3-8B --prefix MATH --use_spec
# online sft
python pipeline.py --dataset data/math.jsonl --draft_model_path ./Qwen3-0.6B-Base --target_model Qwen/Qwen3-8B --method sft
# offline dpo
python pipeline.py --dataset data/math.jsonl --draft_model_path ./Qwen3-0.6B-Base --target_model Qwen/Qwen3-8B --method dpo
# target model
python baseline.py --dataset data/math.jsonl --prefix MATH_TARGET --model Qwen/Qwen3-8B
# draft model
python baseline.py --dataset data/math.jsonl --prefix MATH_DRAFT --model ./Qwen3-0.6B

# mmlu
# offline
python main.py --dataset data/mmlu.jsonl --draft_model ./Qwen3-0.6B-Base --model Qwen/Qwen3-8B --prefix MMLU --use_spec
# online sft
python pipeline.py --dataset data/mmlu.jsonl --draft_model_path ./Qwen3-0.6B-Base --target_model Qwen/Qwen3-8B --method sft
# offline dpo
python pipeline.py --dataset data/mmlu.jsonl --draft_model_path ./Qwen3-0.6B-Base --target_model Qwen/Qwen3-8B --method dpo
# target model
python baseline.py --dataset data/mmlu.jsonl --prefix MMLU_TARGET --model Qwen/Qwen3-8B
# draft model
python baseline.py --dataset data/mmlu.jsonl --prefix MMLU_DRAFT --model ./Qwen3-0.6B