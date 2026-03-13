# 5e-4
python pipeline.py \
--data-file data/gsm_online_4k.jsonl \
--base-model-path ~/PTM/vicuna-7b-v1.3 \
--ea-model-path ckpts/vicuna/gsm \
--chunk-size 40 \
--chunk-dir tmp/chunk \
--output-ea-dir fined_model/fined_model \
--lr 5e-4 \
--num-epochs 1 \
--log-file gsm_chunk40_lr5e-4_epoch1

# 1e-3
python pipeline.py \
--data-file data/gsm_online_4k.jsonl \
--base-model-path ~/PTM/vicuna-7b-v1.3 \
--ea-model-path ckpts/vicuna/gsm \
--chunk-size 40 \
--chunk-dir tmp/chunk \
--output-ea-dir fined_model/fined_model \
--lr 1e-3 \
--num-epochs 1 \
--log-file gsm_chunk40_lr1e-3_epoch1

# 1e-2
python pipeline.py \
--data-file data/gsm_online_4k.jsonl \
--base-model-path ~/PTM/vicuna-7b-v1.3 \
--ea-model-path ckpts/vicuna/gsm \
--chunk-size 40 \
--chunk-dir tmp/chunk \
--output-ea-dir fined_model/fined_model \
--lr 1e-2 \
--num-epochs 1 \
--log-file gsm_chunk40_lr1e-2_epoch1

# 2e-2
python pipeline.py \
--data-file data/gsm_online_4k.jsonl \
--base-model-path ~/PTM/vicuna-7b-v1.3 \
--ea-model-path ckpts/vicuna/gsm \
--chunk-size 40 \
--chunk-dir tmp/chunk \
--output-ea-dir fined_model/fined_model \
--lr 2e-2 \
--num-epochs 1 \
--log-file gsm_chunk40_lr2e-2_epoch1

# 5e-2 
python pipeline.py \
--data-file data/gsm_online_4k.jsonl \
--base-model-path ~/PTM/vicuna-7b-v1.3 \
--ea-model-path ckpts/vicuna/gsm \
--chunk-size 40 \
--chunk-dir tmp/chunk \
--output-ea-dir fined_model/fined_model \
--lr 5e-2 \
--num-epochs 1 \
--log-file gsm_chunk40_lr5e-2_epoch1

# 1e-1 
python pipeline.py \
--data-file data/gsm_online_4k.jsonl \
--base-model-path ~/PTM/vicuna-7b-v1.3 \
--ea-model-path ckpts/vicuna/gsm \
--chunk-size 40 \
--chunk-dir tmp/chunk \
--output-ea-dir fined_model/fined_model \
--lr 1e-1 \
--num-epochs 1 \
--log-file gsm_chunk40_lr1e-1_epoch1

# 5e-1
python pipeline.py \
--data-file data/gsm_online_4k.jsonl \
--base-model-path ~/PTM/vicuna-7b-v1.3 \
--ea-model-path ckpts/vicuna/gsm \
--chunk-size 40 \
--chunk-dir tmp/chunk \
--output-ea-dir fined_model/fined_model \
--lr 5e-1 \
--num-epochs 1 \
--log-file gsm_chunk40_lr5e-1_epoch1

# ensemble
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
--log-file gsm_hedge_5e-4_1e-3_1e-1_epoch1