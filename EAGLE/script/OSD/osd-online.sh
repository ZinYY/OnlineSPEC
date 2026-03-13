############################# Llama #############################
python pipeline_osd.py \
--data-file data/spider_online_4k.jsonl \
--base-model-path ~/PTM/Llama-2-7b-chat-hf \
--ea-model-path ckpts/llama/spider \
--chunk-size 40 \
--chunk-dir tmp/chunk \
--output-ea-dir fined_model \
--lr 6e-5 \
--num-epochs 5 \
--log-file llama_spider_osd_chunk40_lr6e-5_epoch5

python pipeline_osd.py \
--data-file data/gsm_online_4k.jsonl \
--base-model-path ~/PTM/Llama-2-7b-chat-hf \
--ea-model-path ckpts/llama/gsm \
--chunk-size 40 \
--chunk-dir tmp/chunk \
--output-ea-dir fined_model \
--lr 6e-5 \
--num-epochs 5 \
--log-file llama_gsm_osd_chunk40_lr6e-5_epoch5

python pipeline_osd.py \
--data-file data/finance_online_4k.jsonl \
--base-model-path ~/PTM/Llama-2-7b-chat-hf \
--ea-model-path ckpts/llama/finance \
--chunk-size 40 \
--chunk-dir tmp/chunk \
--output-ea-dir fined_model \
--lr 6e-5 \
--num-epochs 5 \
--log-file llama_finance_osd_chunk40_lr6e-5_epoch5

python pipeline_osd.py \
--data-file data/code_search_online_4k.jsonl \
--base-model-path ~/PTM/Llama-2-7b-chat-hf \
--ea-model-path ckpts/llama/code \
--chunk-size 40 \
--chunk-dir tmp/chunk \
--output-ea-dir fined_model \
--lr 1e-4 \
--num-epochs 5 \
--log-file llama_code_search_osd_chunk40_lr1e-4_epoch5

############################# Vicuna #############################

python pipeline_osd.py \
--data-file data/spider_online_4k.jsonl \
--base-model-path ~/PTM/vicuna-7b-v1.3 \
--ea-model-path ckpts/vicuna/spider \
--chunk-size 40 \
--chunk-dir tmp/chunk \
--output-ea-dir fined_model \
--lr 6e-5 \
--num-epochs 5 \
--log-file vicuna_spider_osd_chunk40_lr6e-5_epoch5

python pipeline_osd.py \
--data-file data/gsm_online_4k.jsonl \
--base-model-path ~/PTM/vicuna-7b-v1.3 \
--ea-model-path ckpts/vicuna/gsm \
--chunk-size 40 \
--chunk-dir tmp/chunk \
--output-ea-dir fined_model \
--lr 6e-4 \
--num-epochs 3 \
--log-file vicuna_gsm_osd_chunk40_lr6e-4_epoch3

python pipeline_osd.py \
--data-file data/finance_online_4k.jsonl \
--base-model-path ~/PTM/vicuna-7b-v1.3 \
--ea-model-path ckpts/vicuna/finance \
--chunk-size 40 \
--chunk-dir tmp/chunk \
--output-ea-dir fined_model \
--lr 6e-5 \
--num-epochs 5 \
--log-file vicuna_finance_osd_chunk40_lr6e-5_epoch5

python pipeline_osd.py \
--data-file data/code_search_online_4k.jsonl \
--base-model-path ~/PTM/vicuna-7b-v1.3 \
--ea-model-path ckpts/vicuna/code \
--chunk-size 40 \
--chunk-dir tmp/chunk \
--output-ea-dir fined_model \
--lr 1e-4 \
--num-epochs 5 \
--log-file vicuna_code_search_osd_chunk40_lr1e-4_epoch5




