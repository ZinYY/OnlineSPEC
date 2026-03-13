# OnlineSpec: Speculative Decoding Meets Online Learning

**OnlineSpec** is a unified framework that integrates speculative decoding with online learning to achieve enhanced performance in large language model inference and adaptation.

This repository contains implementations of three key approaches:
- **EAGLE/EAGLE-3/OSD**: Speculative decoding methods for efficient inference
- **Hydra**: Online learning framework with adaptive model updates
- **Lookahead Reasoning (LR)**: Advanced techniques for reasoning models speedup.

It also includes a shared module directory:
- **ospec_common**: Reusable common implementations shared across subprojects.

## 📑 Table of Contents

- [Repository Structure](#-repository-structure)
- [Proposed Methods](#-proposed-methods)
- [Reproduce the Results](#-reproduce-the-results)
  - [EAGLE, EAGLE-3, and OSD](#-eagle-eagle-3-and-osd)
  - [Hydra](#-hydra)
  - [Lookahead Reasoning](#-lookahead-reasoning-lr)
- [Additional Notes](#-additional-notes)

## Repository Structure Guide

This repository is organized as a multi-project workspace with shared common modules.

### Top-level layout

- `EAGLE/`: EAGLE, EAGLE-3, and OSD methods
- `Hydra/`: Hydra online learning framework
- `LR/`: Lookahead Reasoning
- `ospec_common/`: shared modules used by multiple projects

### Shared module policy

To keep logic unchanged while reducing duplicate code, common implementations are centralized in `ospec_common/`:

- `ospec_common/chunking.py`
  - Shared implementation for splitting question files into JSONL chunks.
  - Used by `EAGLE/load_data.py` and `Hydra/load_data.py` via compatibility imports.

- `ospec_common/prepare_data.py`
  - Shared data preparation entry used by both EAGLE and Hydra.
  - `EAGLE/data/prepare_data.py` and `Hydra/data/prepare_data.py` are now thin wrappers that preserve original script paths.

- `ospec_common/eagle_config.py`
  - Shared `EConfig` class for EAGLE variants.
  - Existing config modules keep original paths and export `EConfig` for backward compatibility.

### Backward compatibility

All existing script entry paths are preserved:

- `python EAGLE/data/prepare_data.py`
- `python Hydra/data/prepare_data.py`
- imports from original config files and `load_data.py`

So existing reproduction scripts can continue to run without command changes.

## 📝 Proposed Methods

This section introduces the key methodological contributions of OnlineSpec, which enhance the performance of speculative decoding through online learning techniques.

### 🎯 Online Ensemble

**Overview**: Online ensemble combines multiple draft models with different learning rates to improve speculative decoding performance. This method leverages ensemble learning principles to achieve better accuracy than single-model approaches.

**Key Benefits**:
- Outperforms offline and simple online update methods
- Robust performance through model diversity
- Effective for both EAGLE and EAGLE-3 frameworks

**Usage Example**:

**EAGLE**:
```bash
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
```

**EAGLE-3**:
```bash
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
```

> **Note**: For complete examples and additional configurations, see:
> - `EAGLE/script/EAGLE/eagle-ens.sh`
> - `EAGLE/script/EAGLE-3/eagle3-ens.sh`

### 🚀 Online Learning with Optimism

**Overview**: Online learning with optimism (momentum-based updates) enhances the Hydra head model's adaptation to streaming data. This approach maintains an optimistic update direction based on historical gradients, leading to faster convergence and better performance.

**Key Benefits**:
- Superior performance compared to offline and simple online update methods
- Faster adaptation to new data patterns
- Stable training dynamics through momentum

**Usage Example**:

```bash
python pipeline.py \
  --data-file data/gsm_4k.jsonl \
  --hydra-model-path ckpts/vicuna/gsm \
  --chunk-size 80 \
  --chunk-dir tmp/data_chunks \
  --output-model-dir tmp/outputs \
  --lr 1e-1 \
  --num-epochs 3 \
  --with-momentum \
  --log-file vicuna_gsm_opt
```

### 🧠 Lookahead Reasoning with Online DPO

**Overview**: Online Direct Preference Optimization (DPO) fine-tunes the draft model in Lookahead Reasoning, enabling better alignment with target model preferences. This method improves upon offline and simple online supervised fine-tuning (SFT) approaches.

**Key Benefits**:
- Better alignment with target model preferences
- Improved reasoning accuracy through preference learning
- More effective than standard SFT methods

**Usage Example**:

```bash
python pipeline.py \
  --dataset data/math.jsonl \
  --draft_model_path ./Qwen3-0.6B-Base \
  --target_model Qwen/Qwen3-8B \
  --method dpo
```

## Reproduce the Results

The following section covers the steps to reproduce the results in the paper.

### 🦅 EAGLE, EAGLE-3, and OSD

This section covers the implementation of EAGLE, EAGLE-3, and OSD (Online Speculative Decoding) methods for efficient speculative decoding.

#### 🔧 Environment Setup

```bash
conda create -n eagle python=3.12 -y
conda activate eagle
cd EAGLE
pip install -r requirements.txt
```

#### 📥 Model Preparation

The experiments require the following base models:
- `vicuna-7b-v1.3`
- `Llama-2-7b-chat-hf`

Download the models to the local directory `~/PTM/`:

```bash
# Download Vicuna-7B-v1.3
huggingface-cli download lmsys/vicuna-7b-v1.3 --local-dir ~/PTM/vicuna-7b-v1.3 --local-dir-use-symlinks False

# Download Llama-2-7B-Chat-HF (requires access approval)
# Visit https://huggingface.co/meta-llama/Llama-2-7b-chat-hf to request access first
huggingface-cli download meta-llama/Llama-2-7b-chat-hf --local-dir ~/PTM/Llama-2-7b-chat-hf --local-dir-use-symlinks False
```

> **Note**: For Llama-2 models, you need to request access on Hugging Face first. After approval, use `huggingface-cli login` to authenticate.

#### 📊 Data Preparation

The following datasets are used in the experiments:
- **GSM8K**: Math word problems
- **Spider**: Text-to-SQL tasks
- **Alpaca-Finance**: Financial instruction following
- **Code-search-net (Python)**: Code generation from documentation

Prepare the data:

```bash
cd EAGLE/data/
python prepare_data.py
```

This script will:
- Download and process all required datasets
- Convert data to the required format
- Generate offline (800 samples) and online (4000 samples) training splits

#### 🚀 Running Experiments

##### EAGLE

**1. Warmup Training (Offline)**
```bash
cd EAGLE/script/EAGLE/
bash eagle-train.sh
```
Train the EAGLE model using offline data. Follow the instructions in the script to configure training parameters.

**2. Offline Evaluation**
```bash
bash eagle-offline.sh
```
Evaluate the model performance on the offline test set.

**3. Online Evaluation and Update**
```bash
bash eagle-online.sh
```
Run online evaluation and perform model updates with streaming data.

**4. Online Ensemble**
```bash
bash eagle-ens.sh
```
Apply ensemble methods for improved online performance.

##### EAGLE-3

EAGLE-3 follows the same workflow as EAGLE. Use the corresponding scripts in `EAGLE/script/EAGLE-3/`:

- `eagle-train.sh` - Warmup training
- `eagle-offline.sh` - Offline evaluation
- `eagle-online.sh` - Online evaluation and update
- `eagle-ens.sh` - Online ensemble

##### OSD (Online Speculative Decoding)

**1. Offline Evaluation**
```bash
cd EAGLE/script/OSD/
bash osd-offline.sh
```

**2. Online Evaluation and Update**
```bash
bash osd-online.sh
``` 

### 🐉 Hydra

Hydra is an online learning framework that enables adaptive model updates during inference.

#### 🔧 Environment Setup

```bash
conda create -n hydra python=3.10 -y
conda activate hydra
cd Hydra
pip install -e ".[train]"
```

#### 📥 Model Preparation

The experiments require the same base models as EAGLE:
- `vicuna-7b-v1.3`
- `Llama-2-7b-chat-hf`

Download models to `~/PTM/`:

```bash
huggingface-cli download lmsys/vicuna-7b-v1.3 --local-dir ~/PTM/vicuna-7b-v1.3 --local-dir-use-symlinks False
huggingface-cli download meta-llama/Llama-2-7b-chat-hf --local-dir ~/PTM/Llama-2-7b-chat-hf --local-dir-use-symlinks False
```

> **Important**: When changing the base model, update the `model_name_or_path` in:
> - `Hydra/train.py` (line 367)
> - `Hydra/hydra/model/hydra_model.py` (lines 30, 58, 157)

#### 📊 Data Preparation

The same datasets are used: GSM8K, Spider, Alpaca-Finance, and Code-search-net (Python).

```bash
cd Hydra/data/
pip install 'datasets<3.0.0'  # Required for code_search_net dataset compatibility
python prepare_data.py
```

This script will:
- Download and process all datasets
- Generate offline (800 samples) and online (4000 samples) splits
- Save data in the required format

#### 🚀 Running Experiments

**1. Offline Training Warmup**

Train Hydra models for each dataset using offline data:

```bash
cd Hydra/script/
bash train.sh
```

Model checkpoints will be saved in `Hydra/ckpts/`.

**2. Online Evaluation and Update**

Run online evaluation and model updates:

```bash
bash run.sh
```

Follow the instructions in the script to reproduce the paper results.

### 🧠 Lookahead Reasoning (LR)

Lookahead Reasoning implements advanced reasoning techniques to improve model accuracy through speculative decoding and online learning.

> **Note**: All experiments in this section should be run from the `LR/` directory.

#### 🔧 Environment Setup

```bash
conda create -n lar python=3.10 -y
conda activate lar
cd LR
pip install -r requirements.txt
```

#### 📊 Data Preparation

Prepare the datasets:

```bash
cd LR/data/
python prepare_data.py
```

This script generates the following data files:
- `math.jsonl` - Mathematics problems
- `gsm2.5k.jsonl` - GSM8K subset (2.5K samples)
- `mbpp.jsonl` - Python code generation tasks
- `mmlu.jsonl` - Multi-task language understanding

These files are used for both offline evaluation and online updates.

#### 📥 Model Preparation

The experiments use the following models:
- **Qwen3-8B** - Main reasoning model
- **Qwen3-0.6B-Base** - Base model (requires local download)
- **Qwen2.5-7B-Instruct** - Instruction-tuned model

Download the Qwen3-0.6B-Base model:

```bash
cd LR/
huggingface-cli download Qwen/Qwen3-0.6B-Base --local-dir ./Qwen3-0.6B-Base --local-dir-use-symlinks False
```

#### 🚀 Running Experiments

**Reproduce Paper Results**

Run the main reproduction script:

```bash
cd LR/script/
bash reproduce.sh
```

This script supports three modes:
- **LR**: Standard Lookahead Reasoning
- **OSD-LR**: Online Speculative Decoding with LR
- **Online-LR**: Online learning with LR

Follow the instructions in the script to configure and run experiments.

**Test Accuracy**

Evaluate model accuracy on the four datasets:

```bash
cd LR/test_accuracy/
# Run the corresponding Python scripts for each dataset
python test_<dataset>.py
```

## 📝 Additional Notes

- Check individual script files for detailed parameter configurations
- Model checkpoints and logs are typically saved in respective `ckpts/` and `logs/` directories