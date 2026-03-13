##################### Llama #####################
python pipeline_eagle3.py \
  --data-file data/spider_online_4k.jsonl \
  --base-model-path ~/PTM/Llama-2-7b-chat-hf \
  --ea-model-path ./eagle3_ckpts/llama/spider \
  --chunk-size 40 \
  --offline \
  --log-file llama_spider_offline
  
python pipeline_eagle3.py \
  --data-file data/gsm_online_4k.jsonl \
  --base-model-path ~/PTM/Llama-2-7b-chat-hf \
  --ea-model-path ./eagle3_ckpts/llama/gsm \
  --chunk-size 40 \
  --offline \
  --log-file llama_gsm_offline

python pipeline_eagle3.py \
  --data-file data/code_search_online_4k.jsonl \
  --base-model-path ~/PTM/Llama-2-7b-chat-hf \
  --ea-model-path ./eagle3_ckpts/llama/code \
  --offline \
  --chunk-size 40 \
  --log-file llama_code_search_offline

python pipeline_eagle3.py \
  --data-file data/finance_online_4k.jsonl \
  --base-model-path ~/PTM/Llama-2-7b-chat-hf \
  --ea-model-path ./eagle3_ckpts/llama/finance \
  --offline \
  --chunk-size 40 \
  --log-file llama_finance_offline

##################### Vicuna #####################

python pipeline_eagle3.py \
  --data-file data/spider_online_4k.jsonl \
  --base-model-path ~/PTM/vicuna-7b-v1.3 \
  --ea-model-path ./eagle3_ckpts/vicuna/spider \
  --offline \
  --chunk-size 40 \
  --log-file vicuna_spider_offline

python pipeline_eagle3.py \
  --data-file data/gsm_online_4k.jsonl \
  --base-model-path ~/PTM/vicuna-7b-v1.3 \
  --ea-model-path ./eagle3_ckpts/vicuna/gsm \
  --offline \
  --chunk-size 40 \
  --log-file vicuna_gsm_offline

python pipeline_eagle3.py \
  --data-file data/code_search_online_4k.jsonl \
  --base-model-path ~/PTM/vicuna-7b-v1.3 \
  --ea-model-path ./eagle3_ckpts/vicuna/code \
  --offline \
  --chunk-size 40 \
  --log-file vicuna_code_search_offline

python pipeline_eagle3.py \
  --data-file data/finance_online_4k.jsonl \
  --base-model-path ~/PTM/vicuna-7b-v1.3 \
  --ea-model-path ./eagle3_ckpts/vicuna/finance \
  --offline \
  --chunk-size 40 \
  --log-file vicuna_finance_offline