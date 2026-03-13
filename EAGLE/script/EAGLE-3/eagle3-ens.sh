############################# Vicuna #################################

# gsm8k
python pipeline_eagle3_hedge.py \
  --data-file data/gsm_online_4k.jsonl \
  --base-model-path ~/PTM/vicuna-7b-v1.3 \
  --ea-model-path-1 ./eagle3_ckpts/vicuna/gsm \
  --ea-model-path-2 ./eagle3_ckpts/vicuna/gsm \
  --ea-model-path-3 ./eagle3_ckpts/vicuna/gsm \
  --chunk-size 40 \
  --batch-size 2 \
  --lr-1 2e-4 \
  --lr-2 1e-4 \
  --lr-3 4e-4 \
  --num-epochs-1 2 \
  --num-epochs-2 2 \
  --num-epochs-3 2 \
  --log-file vicuna_gsm_hedge_lr2e-4_1e-4_4e-4_epoch2

# spider
python pipeline_eagle3_hedge.py \
  --data-file data/spider_online_4k.jsonl \
  --base-model-path ~/PTM/vicuna-7b-v1.3 \
  --ea-model-path-1 ./eagle3_ckpts/vicuna/spider \
  --ea-model-path-2 ./eagle3_ckpts/vicuna/spider \
  --ea-model-path-3 ./eagle3_ckpts/vicuna/spider \
  --chunk-size 40 \
  --batch-size 2 \
  --lr-1 2e-4 \
  --lr-2 1e-4 \
  --lr-3 4e-4 \
  --num-epochs-1 2 \
  --num-epochs-2 2 \
  --num-epochs-3 2 \
  --log-file vicuna_spider_hedge_lr2e-4_1e-4_4e-4_epoch2

# code search
python pipeline_eagle3_hedge.py \
  --data-file data/code_search_online_4k.jsonl \
  --base-model-path ~/PTM/vicuna-7b-v1.3 \
  --ea-model-path-1 ./eagle3_ckpts/vicuna/code \
  --ea-model-path-2 ./eagle3_ckpts/vicuna/code \
  --ea-model-path-3 ./eagle3_ckpts/vicuna/code \
  --chunk-size 40 \
  --batch-size 2 \
  --lr-1 2e-4 \
  --lr-2 1e-4 \
  --lr-3 4e-4 \
  --num-epochs-1 2 \
  --num-epochs-2 2 \
  --num-epochs-3 2 \
  --log-file vicuna_code_search_hedge_lr2e-4_1e-4_4e-4_epoch2

# finance
python pipeline_eagle3_hedge.py \
  --data-file data/finance_online_4k.jsonl \
  --base-model-path ~/PTM/vicuna-7b-v1.3 \
  --ea-model-path-1 ./eagle3_ckpts/vicuna/finance \
  --ea-model-path-2 ./eagle3_ckpts/vicuna/finance \
  --ea-model-path-3 ./eagle3_ckpts/vicuna/finance \
  --chunk-size 40 \
  --batch-size 2 \
  --lr-1 2e-4 \
  --lr-2 1e-4 \
  --lr-3 4e-4 \
  --num-epochs-1 2 \
  --num-epochs-2 2 \
  --num-epochs-3 2 \
  --log-file vicuna_finance_hedge_lr2e-4_1e-4_4e-4_epoch2

############################# Llama ######################################

# gsm8k
python pipeline_eagle3_hedge.py \
  --data-file data/gsm_online_4k.jsonl \
  --base-model-path ~/PTM/Llama-2-7b-chat-hf \
  --ea-model-path-1 ./eagle3_ckpts/llama/gsm \
  --ea-model-path-2 ./eagle3_ckpts/llama/gsm \
  --ea-model-path-3 ./eagle3_ckpts/llama/gsm \
  --chunk-size 40 \
  --batch-size 2 \
  --lr-1 2e-4 \
  --lr-2 1e-4 \
  --lr-3 4e-4 \
  --num-epochs-1 2 \
  --num-epochs-2 2 \
  --num-epochs-3 2 \
  --log-file llama_gsm8k_hedge_lr2e-4_1e-4_4e-4_epoch2

# spider
python pipeline_eagle3_hedge.py \
  --data-file data/spider_online_4k.jsonl \
  --base-model-path ~/PTM/Llama-2-7b-chat-hf \
  --ea-model-path-1 ./eagle3_ckpts/llama/spider \
  --ea-model-path-2 ./eagle3_ckpts/llama/spider \
  --ea-model-path-3 ./eagle3_ckpts/llama/spider \
  --chunk-size 40 \
  --batch-size 2 \
  --lr-1 2e-4 \
  --lr-2 1e-4 \
  --lr-3 4e-4 \
  --num-epochs-1 2 \
  --num-epochs-2 2 \
  --num-epochs-3 2 \
  --log-file llama_spider_hedge_lr2e-4_1e-4_4e-4_epoch2

# code
python pipeline_eagle3_hedge.py \
  --data-file data/code_search_online_4k.jsonl \
  --base-model-path ~/PTM/Llama-2-7b-chat-hf \
  --ea-model-path-1 ./eagle3_ckpts/llama/code \
  --ea-model-path-2 ./eagle3_ckpts/llama/code \
  --ea-model-path-3 ./eagle3_ckpts/llama/code \
  --chunk-size 40 \
  --batch-size 2 \
  --lr-1 2e-4 \
  --lr-2 1e-4 \
  --lr-3 4e-4 \
  --num-epochs-1 2 \
  --num-epochs-2 2 \
  --num-epochs-3 2 \
  --log-file llama_code_search_hedge_lr2e-4_1e-4_4e-4_epoch2

# finance
python pipeline_eagle3_hedge.py \
  --data-file data/finance_online_4k.jsonl \
  --base-model-path ~/PTM/Llama-2-7b-chat-hf \
  --ea-model-path-1 ./eagle3_ckpts/llama/finance \
  --ea-model-path-2 ./eagle3_ckpts/llama/finance \
  --ea-model-path-3 ./eagle3_ckpts/llama/finance \
  --chunk-size 40 \
  --batch-size 2 \
  --lr-1 2e-4 \
  --lr-2 1e-4 \
  --lr-3 4e-4 \
  --num-epochs-1 2 \
  --num-epochs-2 2 \
  --num-epochs-3 2 \
  --log-file llama_finance_hedge_lr2e-4_1e-4_4e-4_epoch2
