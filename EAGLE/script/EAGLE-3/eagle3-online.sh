##################### Llama #####################
python pipeline_eagle3.py \
  --data-file data/spider_online_4k.jsonl \
  --base-model-path ~/PTM/Llama-2-7b-chat-hf \
  --ea-model-path ./eagle3_ckpts/llama/spider \
  --chunk-size 40 \
  --batch-size 2 \
  --lr 1e-3 \
  --num-epochs 2 \
  --log-file llama_spider_lr1e-3_epoch2

python pipeline_eagle3.py \
  --data-file data/gsm_online_4k.jsonl \
  --base-model-path ~/PTM/Llama-2-7b-chat-hf \
  --ea-model-path ./eagle3_ckpts/llama/gsm \
  --chunk-size 40 \
  --batch-size 2 \
  --lr 1e-3 \
  --num-epochs 2 \
  --log-file llama_gsm_lr1e-3_epoch2

python pipeline_eagle3.py \
  --data-file data/code_search_online_4k.jsonl \
  --base-model-path ~/PTM/Llama-2-7b-chat-hf \
  --ea-model-path ./eagle3_ckpts/llama/code \
  --chunk-size 40 \
  --batch-size 2 \
  --lr 1e-3 \
  --num-epochs 2 \
  --log-file llama_code_search_lr1e-3_epoch2

python pipeline_eagle3.py \
  --data-file data/finance_online_4k.jsonl \
  --base-model-path ~/PTM/Llama-2-7b-chat-hf \
  --ea-model-path ./eagle3_ckpts/llama/finance \
  --chunk-size 40 \
  --batch-size 2 \
  --lr 1e-3 \
  --num-epochs 2 \
  --log-file llama_finance_lr1e-3_epoch2

##################### Vicuna #####################

python pipeline_eagle3.py \
  --data-file data/spider_online_4k.jsonl \
  --base-model-path ~/PTM/vicuna-7b-v1.3 \
  --ea-model-path ./eagle3_ckpts/vicuna/spider \
  --chunk-size 40 \
  --batch-size 2 \
  --lr 1e-3 \
  --num-epochs 2 \
  --log-file vicuna_spider_lr1e-3_epoch2

python pipeline_eagle3.py \
  --data-file data/gsm_online_4k.jsonl \
  --base-model-path ~/PTM/vicuna-7b-v1.3 \
  --ea-model-path ./eagle3_ckpts/vicuna/gsm \
  --chunk-size 40 \
  --batch-size 2 \
  --lr 1e-3 \
  --num-epochs 2 \
  --log-file vicuna_gsm_lr1e-3_epoch2

python pipeline_eagle3.py \
  --data-file data/code_search_online_4k.jsonl \
  --base-model-path ~/PTM/vicuna-7b-v1.3 \
  --ea-model-path ./eagle3_ckpts/vicuna/code \
  --chunk-size 40 \
  --batch-size 2 \
  --lr 1e-3 \
  --num-epochs 2 \
  --log-file vicuna_code_search_lr1e-3_epoch2

python pipeline_eagle3.py \
  --data-file data/finance_online_4k.jsonl \
  --base-model-path ~/PTM/vicuna-7b-v1.3 \
  --ea-model-path ./eagle3_ckpts/vicuna/finance \
  --chunk-size 40 \
  --batch-size 2 \
  --lr 1e-3 \
  --num-epochs 2 \
  --log-file vicuna_finance_lr1e-3_epoch2