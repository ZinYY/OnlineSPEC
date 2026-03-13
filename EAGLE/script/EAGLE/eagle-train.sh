############################### Llama #############################
WANDB_DISABLED=true accelerate launch --mixed_precision=bf16 train.py \
--basepath ~/PTM/Llama-2-7b-chat-hf \
--ea-model-path None \
--data_path data/spider_offline.jsonl \
--cpdir ckpts/llama/spider \
--lr 4e-5 \
--num_epochs 1

WANDB_DISABLED=true accelerate launch --mixed_precision=bf16 train.py \
--basepath ~/PTM/Llama-2-7b-chat-hf \
--ea-model-path None \
--data_path data/gsm_offline.jsonl \
--cpdir ckpts/llama/gsm \
--lr 4e-5 \
--num_epochs 1

WANDB_DISABLED=true accelerate launch --mixed_precision=bf16 train.py \
--basepath ~/PTM/Llama-2-7b-chat-hf \
--ea-model-path None \
--data_path data/code_search_offline.jsonl \
--cpdir ckpts/llama/code \
--lr 4e-5 \
--num_epochs 1

WANDB_DISABLED=true accelerate launch --mixed_precision=bf16 train.py \
--basepath ~/PTM/Llama-2-7b-chat-hf \
--ea-model-path None \
--data_path data/finance_offline.jsonl \
--cpdir ckpts/llama/finance \
--lr 4e-5 \
--num_epochs 1

############################### Vicuna #############################

WANDB_DISABLED=true accelerate launch --mixed_precision=bf16 train.py \
--basepath ~/PTM/vicuna-7b-v1.3 \
--ea-model-path None \
--data_path data/spider_offline.jsonl \
--cpdir ckpts/vicuna/spider \
--lr 4e-5 \
--num_epochs 1

WANDB_DISABLED=true accelerate launch --mixed_precision=bf16 train.py \
--basepath ~/PTM/vicuna-7b-v1.3 \
--ea-model-path None \
--data_path data/gsm_offline.jsonl \
--cpdir ckpts/vicuna/gsm \
--lr 4e-5 \
--num_epochs 1

WANDB_DISABLED=true accelerate launch --mixed_precision=bf16 train.py \
--basepath ~/PTM/vicuna-7b-v1.3 \
--ea-model-path None \
--data_path data/code_search_offline.jsonl \
--cpdir ckpts/vicuna/code \
--lr 4e-5 \
--num_epochs 1

WANDB_DISABLED=true accelerate launch --mixed_precision=bf16 train.py \
--basepath ~/PTM/vicuna-7b-v1.3 \
--ea-model-path None \
--data_path data/finance_offline.jsonl \
--cpdir ckpts/vicuna/finance \
--lr 4e-5 \
--num_epochs 1