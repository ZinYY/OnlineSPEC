############################# Llama #############################

python pipeline.py \
--data-file data/spider_online_4k.jsonl \
--base-model-path ~/PTM/Llama-2-7b-chat-hf \
--ea-model-path ckpts/llama/spider \
--chunk-size 40 \
--chunk-dir tmp/chunk \
--output-ea-dir fined_model \
--lr 3e-5 \
--num-epochs 5 \
--log-file llama_spider_chunk40_lr3e-5_epoch5

python pipeline.py \
--data-file data/gsm_online_4k.jsonl \
--base-model-path ~/PTM/Llama-2-7b-chat-hf \
--ea-model-path ckpts/llama/gsm \
--chunk-size 40 \
--chunk-dir tmp/chunk \
--output-ea-dir fined_model \
--lr 3e-5 \
--num-epochs 5 \
--log-file llama_gsm_chunk40_lr3e-5_epoch5

python pipeline.py \
--data-file data/code_search_online_4k.jsonl \
--base-model-path ~/PTM/Llama-2-7b-chat-hf \
--ea-model-path ckpts/llama/code \
--chunk-size 40 \
--chunk-dir tmp/chunk \
--output-ea-dir fined_model \
--lr 3e-5 \
--num-epochs 5 \
--log-file llama_code_search_chunk40_lr3e-5_epoch5

python pipeline.py \
--data-file data/finance_online_4k.jsonl \
--base-model-path ~/PTM/Llama-2-7b-chat-hf \
--ea-model-path ckpts/llama/finance \
--chunk-size 40 \
--chunk-dir tmp/chunk \
--output-ea-dir fined_model \
--lr 3e-5 \
--num-epochs 5 \
--log-file llama_finance_chunk40_lr3e-5_epoch5

############################# Vicuna #############################

python pipeline.py \
--data-file data/spider_online_4k.jsonl \
--base-model-path ~/PTM/vicuna-7b-v1.3 \
--ea-model-path ckpts/vicuna/spider \
--chunk-size 40 \
--chunk-dir tmp/chunk \
--output-ea-dir fined_model \
--lr 3e-5 \
--num-epochs 5 \
--log-file vicuna_spider_chunk40_lr3e-5_epoch5

python pipeline.py \
--data-file data/gsm_online_4k.jsonl \
--base-model-path ~/PTM/vicuna-7b-v1.3 \
--ea-model-path ckpts/vicuna/gsm \
--chunk-size 40 \
--chunk-dir tmp/chunk \
--output-ea-dir fined_model \
--lr 1e-2 \
--num-epochs 1 \
--log-file vicuna_gsm_chunk40_lr1e-2_epoch1

python pipeline.py \
--data-file data/code_search_online_4k.jsonl \
--base-model-path ~/PTM/vicuna-7b-v1.3 \
--ea-model-path ckpts/vicuna/code \
--chunk-size 40 \
--chunk-dir tmp/chunk \
--output-ea-dir fined_model \
--lr 3e-5 \
--num-epochs 5 \
--log-file vicuna_code_search_chunk40_lr3e-5_epoch5

python pipeline.py \
--data-file data/finance_online_4k.jsonl \
--base-model-path ~/PTM/vicuna-7b-v1.3 \
--ea-model-path ckpts/vicuna/finance \
--chunk-size 40 \
--chunk-dir tmp/chunk \
--output-ea-dir fined_model \
--lr 3e-5 \
--num-epochs 5 \
--log-file vicuna_finance_chunk40_lr3e-5_epoch5