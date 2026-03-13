# Change the base model:
# Change the model_name_or_path in `Hydra/train.py` (line 367) and `Hydra/hydra/model/hydra_model.py` (line 30, 58, 157) accordingly.

################################## Vicuna ##################################

# spider
python eval.py  --model-path ckpts/vicuna/spider --question-file data/spider_4k.jsonl --answer-file answer/vicuna_spider_offline.jsonl

python pipeline.py --data-file data/spider_4k.jsonl --hydra-model-path ckpts/vicuna/spider --chunk-size 80 --chunk-dir tmp/data_chunks --output-model-dir tmp/outputs --lr 1e-1 --num-epochs 3 --log-file vicuna_spider_baseline

python pipeline.py --data-file data/spider_4k.jsonl --hydra-model-path ckpts/vicuna/spider --chunk-size 80 --chunk-dir tmp/data_chunks --output-model-dir tmp/outputs --lr 1e-1 --num-epochs 3 --with-momentum --log-file vicuna_spider_opt

# gsm
python eval.py  --model-path ckpts/vicuna/gsm --question-file data/gsm_4k.jsonl --answer-file answer/vicuna_gsm_offline.jsonl

python pipeline.py --data-file data/gsm_4k.jsonl --hydra-model-path ckpts/vicuna/gsm --chunk-size 80 --chunk-dir tmp/data_chunks --output-model-dir tmp/outputs --lr 1e-1 --num-epochs 3 --log-file vicuna_gsm_baseline

python pipeline.py --data-file data/gsm_4k.jsonl --hydra-model-path ckpts/vicuna/gsm --chunk-size 80 --chunk-dir tmp/data_chunks --output-model-dir tmp/outputs --lr 1e-1 --num-epochs 3 --with-momentum --log-file vicuna_gsm_opt

# finance
python eval.py  --model-path ckpts/vicuna/finance --question-file data/finance_4k.jsonl --answer-file answer/vicuna_finance_offline.jsonl

python pipeline.py --data-file data/finance_4k.jsonl --hydra-model-path ckpts/vicuna/finance --chunk-size 80 --chunk-dir tmp/data_chunks --output-model-dir tmp/outputs --lr 1e-1 --num-epochs 3 --log-file vicuna_finance_baseline

python pipeline.py --data-file data/finance_4k.jsonl --hydra-model-path ckpts/vicuna/finance --chunk-size 80 --chunk-dir tmp/data_chunks --output-model-dir tmp/outputs --lr 1e-1 --num-epochs 3 --with-momentum --log-file vicuna_finance_opt

# code
python eval.py  --model-path ckpts/vicuna/code --question-file data/code_search_4k.jsonl --answer-file answer/vicuna_code_offline.jsonl

python pipeline.py --data-file data/code_search_4k.jsonl --hydra-model-path ckpts/vicuna/code --chunk-size 80 --chunk-dir tmp/data_chunks --output-model-dir tmp/outputs --lr 1e-1 --num-epochs 3 --log-file vicuna_code_baseline

python pipeline.py --data-file data/code_search_4k.jsonl --hydra-model-path ckpts/vicuna/code --chunk-size 80 --chunk-dir tmp/data_chunks --output-model-dir tmp/outputs --lr 1e-1 --num-epochs 3 --with-momentum --log-file vicuna_code_opt


################################## Llama ##################################

# spider
python eval.py  --model-path ckpts/llama/spider --question-file data/spider_4k.jsonl --answer-file answer/llama_spider_offline.jsonl

python pipeline.py --data-file data/spider_4k.jsonl --hydra-model-path ckpts/llama/spider --chunk-size 80 --chunk-dir tmp/data_chunks --output-model-dir tmp/outputs --lr 1e-1 --num-epochs 3 --log-file llama_spider_baseline

python pipeline.py --data-file data/spider_4k.jsonl --hydra-model-path ckpts/llama/spider --chunk-size 80 --chunk-dir tmp/data_chunks --output-model-dir tmp/outputs --lr 1e-1 --num-epochs 3 --with-momentum --log-file llama_spider_opt

# gsm
python eval.py  --model-path ckpts/llama/gsm --question-file data/gsm_4k.jsonl --answer-file answer/llama_gsm_offline.jsonl

python pipeline.py --data-file data/gsm_4k.jsonl --hydra-model-path ckpts/llama/gsm --chunk-size 80 --chunk-dir tmp/data_chunks --output-model-dir tmp/outputs --lr 1e-1 --num-epochs 3 --log-file llama_gsm_baseline

python pipeline.py --data-file data/gsm_4k.jsonl --hydra-model-path ckpts/llama/gsm --chunk-size 80 --chunk-dir tmp/data_chunks --output-model-dir tmp/outputs --lr 1e-1 --num-epochs 3 --with-momentum --log-file llama_gsm_opt

# finance
python eval.py  --model-path ckpts/llama/finance --question-file data/finance_4k.jsonl --answer-file answer/llama_finance_offline.jsonl

python pipeline.py --data-file data/finance_4k.jsonl --hydra-model-path ckpts/llama/finance --chunk-size 80 --chunk-dir tmp/data_chunks --output-model-dir tmp/outputs --lr 1e-1 --num-epochs 3 --log-file llama_finance_baseline

python pipeline.py --data-file data/finance_4k.jsonl --hydra-model-path ckpts/llama/finance --chunk-size 80 --chunk-dir tmp/data_chunks --output-model-dir tmp/outputs --lr 1e-1 --num-epochs 3 --with-momentum --log-file llama_finance_opt

# code
python eval.py  --model-path ckpts/llama/code --question-file data/code_search_4k.jsonl --answer-file answer/llama_code_offline.jsonl

python pipeline.py --data-file data/code_search_4k.jsonl --hydra-model-path ckpts/llama/code --chunk-size 80 --chunk-dir tmp/data_chunks --output-model-dir tmp/outputs --lr 1e-1 --num-epochs 3 --log-file llama_code_baseline

python pipeline.py --data-file data/code_search_4k.jsonl --hydra-model-path ckpts/llama/code --chunk-size 80 --chunk-dir tmp/data_chunks --output-model-dir tmp/outputs --lr 1e-1 --num-epochs 3 --with-momentum --log-file llama_code_opt