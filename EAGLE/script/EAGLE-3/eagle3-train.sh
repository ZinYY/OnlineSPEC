# Offline training warmup

python traineagle3.py \
  --basepath ~/PTM/vicuna-7b-v1.3 \
  --trainpath data/gsm8k_offline.jsonl \
  --savedir ./eagle3_ckpts/vicuna/gsm \
  --batch_size 4 \
  --num_epochs 1 \
  --lr 1e-4

python traineagle3.py \
  --basepath ~/PTM/vicuna-7b-v1.3 \
  --trainpath data/spider_offline.jsonl \
  --savedir ./eagle3_ckpts/vicuna/spider \
  --batch_size 4 \
  --num_epochs 1 \
  --lr 1e-4

python traineagle3.py \
  --basepath ~/PTM/vicuna-7b-v1.3 \
  --trainpath data/code_search_offline.jsonl \
  --savedir ./eagle3_ckpts/vicuna/code \
  --batch_size 4 \
  --num_epochs 1 \
  --lr 1e-4

python traineagle3.py \
  --basepath ~/PTM/vicuna-7b-v1.3 \
  --trainpath data/finance_offline.jsonl \
  --savedir ./eagle3_ckpts/vicuna/finance \
  --batch_size 4 \
  --num_epochs 1 \
  --lr 1e-4

python traineagle3.py \
  --basepath ~/PTM/Llama-2-7b-chat-hf \
  --trainpath data/gsm8k_offline.jsonl \
  --savedir ./eagle3_ckpts/llama/gsm \
  --batch_size 4 \
  --num_epochs 1 \
  --lr 2e-4

python traineagle3.py \
  --basepath ~/PTM/Llama-2-7b-chat-hf \
  --trainpath data/spider_offline.jsonl \
  --savedir ./eagle3_ckpts/llama/spider \
  --batch_size 4 \
  --num_epochs 1 \
  --lr 2e-4

python traineagle3.py \
  --basepath ~/PTM/Llama-2-7b-chat-hf \
  --trainpath data/code_search_offline.jsonl \
  --savedir ./eagle3_ckpts/llama/code \
  --batch_size 4 \
  --num_epochs 1 \
  --lr 2e-4

python traineagle3.py \
  --basepath ~/PTM/Llama-2-7b-chat-hf \
  --trainpath data/finance_offline.jsonl \
  --savedir ./eagle3_ckpts/llama/finance \
  --batch_size 4 \
  --num_epochs 1 \
  --lr 2e-4