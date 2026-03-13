############################# Llama #############################
python pipeline_hedge.py \
--data-file data/spider_online_4k.jsonl \
--base-model-path ~/PTM/Llama-2-7b-chat-hf \
--ea-model-path-1 ckpts/llama/spider \
--ea-model-path-2 ckpts/llama/spider \
--ea-model-path-3 ckpts/llama/spider \
--chunk-size 40 \
--chunk-dir tmp/chunk \
--output-ea-dir fined_model_3lr \
--lr-1 3e-5 \
--lr-2 6e-5 \
--lr-3 1.2e-4 \
--num-epochs 5 \
--log-file llama_spider_chunk40_lr3e-5_6e-5_1.2e-4_epoch5


python pipeline_hedge.py \
--data-file data/gsm_online_4k.jsonl \
--base-model-path ~/PTM/Llama-2-7b-chat-hf \
--ea-model-path-1 ckpts/llama/gsm \
--ea-model-path-2 ckpts/llama/gsm \
--ea-model-path-3 ckpts/llama/gsm \
--chunk-size 40 \
--chunk-dir tmp/chunk \
--output-ea-dir fined_model_3lr \
--lr-1 3e-5 \
--lr-2 6e-5 \
--lr-3 1.2e-4 \
--num-epochs 5 \
--log-file llama_gsm_chunk40_lr3e-5_6e-5_1.2e-4_epoch5

python pipeline_hedge.py \
--data-file data/code_search_online_4k.jsonl \
--base-model-path ~/PTM/Llama-2-7b-chat-hf \
--ea-model-path-1 ckpts/llama/code \
--ea-model-path-2 ckpts/llama/code \
--ea-model-path-3 ckpts/llama/code \
--chunk-size 40 \
--chunk-dir tmp/chunk \
--output-ea-dir fined_model_3lr \
--lr-1 3e-5 \
--lr-2 6e-5 \
--lr-3 1.2e-4 \
--num-epochs 5 \
--log-file llama_code_search_chunk40_lr3e-5_6e-5_1.2e-4_epoch5

python pipeline_hedge.py \
--data-file data/finance_online_4k.jsonl \
--base-model-path ~/PTM/Llama-2-7b-chat-hf \
--ea-model-path-1 ckpts/llama/finance \
--ea-model-path-2 ckpts/llama/finance \
--ea-model-path-3 ckpts/llama/finance \
--chunk-size 40 \
--chunk-dir tmp/chunk \
--output-ea-dir fined_model_3lr \
--lr-1 3e-5 \
--lr-2 6e-5 \
--lr-3 1.2e-4 \
--num-epochs 5 \
--log-file llama_finance_chunk40_lr3e-5_6e-5_1.2e-4_epoch5

############################# Vicuna #############################

python pipeline_hedge.py \
--data-file data/spider_online_4k.jsonl \
--base-model-path ~/PTM/vicuna-7b-v1.3 \
--ea-model-path-1 ckpts/vicuna/spider \
--ea-model-path-2 ckpts/vicuna/spider \
--ea-model-path-3 ckpts/vicuna/spider \
--chunk-size 40 \
--chunk-dir tmp/chunk \
--output-ea-dir fined_model_3lr \
--lr-1 3e-5 \
--lr-2 6e-5 \
--lr-3 1.2e-4 \
--num-epochs 5 \
--log-file vicuna_spider_chunk40_lr3e-5_6e-5_1.2e-4_epoch5

python pipeline_hedge.py \
--data-file data/gsm_online_4k.jsonl \
--base-model-path ~/PTM/vicuna-7b-v1.3 \
--ea-model-path-1 ckpts/vicuna/gsm \
--ea-model-path-2 ckpts/vicuna/gsm \
--ea-model-path-3 ckpts/vicuna/gsm \
--chunk-size 40 \
--chunk-dir tmp/chunk \
--output-ea-dir fined_model_3lr \
--lr-1 5e-4 \
--lr-2 1e-3 \
--lr-3 1e-1 \
--num-epochs 1 \
--log-file vicuna_gsm_chunk40_lr5e-4_1e-3_1e-1_epoch1

python pipeline_hedge.py \
--data-file data/code_search_online_4k.jsonl \
--base-model-path ~/PTM/vicuna-7b-v1.3 \
--ea-model-path-1 ckpts/vicuna/code \
--ea-model-path-2 ckpts/vicuna/code \
--ea-model-path-3 ckpts/vicuna/code \
--chunk-size 40 \
--chunk-dir tmp/chunk \
--output-ea-dir fined_model_3lr \
--lr-1 3e-5 \
--lr-2 6e-5 \
--lr-3 1.2e-4 \
--num-epochs 5 \
--log-file vicuna_code_search_chunk40_lr3e-5_6e-5_1.2e-4_epoch5

python pipeline_hedge.py \
--data-file data/finance_online_4k.jsonl \
--base-model-path ~/PTM/vicuna-7b-v1.3 \
--ea-model-path-1 ckpts/vicuna/finance \
--ea-model-path-2 ckpts/vicuna/finance \
--ea-model-path-3 ckpts/vicuna/finance \
--chunk-size 40 \
--chunk-dir tmp/chunk \
--output-ea-dir fined_model_3lr \
--lr-1 3e-5 \
--lr-2 6e-5 \
--lr-3 1.2e-4 \
--num-epochs 5 \
--log-file vicuna_finance_chunk40_lr3e-5_6e-5_1.2e-4_epoch5