# model path: 
# vicuna: ckpts/vicuna/spider, ckpts/vicuna/gsm, ckpts/vicuna/finance, ckpts/vicuna/code
# llama: ckpts/llama/spider, ckpts/llama/gsm, ckpts/llama/finance, ckpts/llama/code

export WANDB_MODE=disabled
torchrun --nproc_per_node=4 train.py \
--data_path data/spider_offline.jsonl \
--bf16 True \
--output_dir ckpts/vicuna/spider \
--num_train_epochs 1 \
--global_batch_size 12 \
--per_device_train_batch_size 3 \
--per_device_eval_batch_size 3 \
--gradient_accumulation_steps 1 \
--dataloader_num_workers 4 \
--evaluation_strategy "no" \
--save_strategy "no" \
--learning_rate 1e-1 \
--logging_steps 1 \
--tf32 True \
--model_max_length 2048 \
--hydra_num_heads 4 \
--hydra_num_layers 4 \
--hydra_head_arch prefix-mlp \
--grounded_heads true \
--hidden_state_offset 0 \
--lm_loss_weight 0.0 \
--teacher_loss_weight 1.0 \
--reconstruction_loss_weight 0.0 \
--dropout_rate 0.1 \
--optim sgd \
--sgd_momentum 0.9

export WANDB_MODE=disabled
torchrun --nproc_per_node=4 train.py \
--data_path data/gsm8k_offline.jsonl \
--bf16 True \
--output_dir ckpts/vicuna/gsm \
--num_train_epochs 1 \
--global_batch_size 12 \
--per_device_train_batch_size 3 \
--per_device_eval_batch_size 3 \
--gradient_accumulation_steps 1 \
--dataloader_num_workers 4 \
--evaluation_strategy "no" \
--save_strategy "no" \
--learning_rate 1e-1 \
--logging_steps 1 \
--tf32 True \
--model_max_length 2048 \
--hydra_num_heads 4 \
--hydra_num_layers 4 \
--hydra_head_arch prefix-mlp \
--grounded_heads true \
--hidden_state_offset 0 \
--lm_loss_weight 0.0 \
--teacher_loss_weight 1.0 \
--reconstruction_loss_weight 0.0 \
--dropout_rate 0.1 \
--optim sgd \
--sgd_momentum 0.9

export WANDB_MODE=disabled
torchrun --nproc_per_node=4 train.py \
--data_path data/finance_offline.jsonl \
--bf16 True \
--output_dir ckpts/vicuna/finance \
--num_train_epochs 1 \
--global_batch_size 12 \
--per_device_train_batch_size 3 \
--per_device_eval_batch_size 3 \
--gradient_accumulation_steps 1 \
--dataloader_num_workers 4 \
--evaluation_strategy "no" \
--save_strategy "no" \
--learning_rate 1e-1 \
--logging_steps 1 \
--tf32 True \
--model_max_length 2048 \
--hydra_num_heads 4 \
--hydra_num_layers 4 \
--hydra_head_arch prefix-mlp \
--grounded_heads true \
--hidden_state_offset 0 \
--lm_loss_weight 0.0 \
--teacher_loss_weight 1.0 \
--reconstruction_loss_weight 0.0 \
--dropout_rate 0.1 \
--optim sgd \
--sgd_momentum 0.9

export WANDB_MODE=disabled
torchrun --nproc_per_node=4 train.py \
--data_path data/code_search_offline.jsonl \
--bf16 True \
--output_dir ckpts/vicuna/code \
--num_train_epochs 1 \
--global_batch_size 12 \
--per_device_train_batch_size 3 \
--per_device_eval_batch_size 3 \
--gradient_accumulation_steps 1 \
--dataloader_num_workers 4 \
--evaluation_strategy "no" \
--save_strategy "no" \
--learning_rate 1e-1 \
--logging_steps 1 \
--tf32 True \
--model_max_length 2048 \
--hydra_num_heads 4 \
--hydra_num_layers 4 \
--hydra_head_arch prefix-mlp \
--grounded_heads true \
--hidden_state_offset 0 \
--lm_loss_weight 0.0 \
--teacher_loss_weight 1.0 \
--reconstruction_loss_weight 0.0 \
--dropout_rate 0.1 \
--optim sgd \
--sgd_momentum 0.9