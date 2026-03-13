# This code is based on tatsu-lab/stanford_alpaca. Below is the original copyright:
#
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# Adapted from: lm-sys/FastChat/blob/main/fastchat/train/train.py

from dataclasses import dataclass, field
import json
import os
import math
import pathlib
import tempfile
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import Trainer, BitsAndBytesConfig
from transformers.trainer_utils import EvalLoopOutput, denumpify_detensorize, has_length, seed_worker, PREFIX_CHECKPOINT_DIR
from transformers.trainer_pt_utils import LabelSmoother, nested_detach, find_batch_size, nested_concat, nested_numpify, IterableDatasetShard
from composer.utils import maybe_create_object_store_from_uri
from composer.utils.file_helpers import parse_uri
import wandb

from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template
from torch.nn import CrossEntropyLoss, SmoothL1Loss
from torch.nn import functional as F
from hydra.model.hydra_model import HydraModel, HydraConfig
from hydra.train.utils import get_scheduler
from transformers.trainer_pt_utils import get_parameter_names

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
EVAL_SPLIT_SEED = 42

# Customized for training Hydra heads
class CustomizedTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        class DummyScheduler:
            def step(self): pass
            def state_dict(self): return {}
            def load_state_dict(self, state_dict): pass
            def get_last_lr(self):
                if hasattr(self, "optimizer") and hasattr(self.optimizer, "param_groups"):
                    return [group["lr"] for group in self.optimizer.param_groups]
                return [0.0]
        self.lr_scheduler = DummyScheduler()
        self.lr_scheduler.optimizer = self.optimizer

    # Functions related to pre-computed data
    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": False,
            "prefetch_factor": None if self.args.dataloader_num_workers == 0 else self.args.dataloader_prefetch_factor,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    # Metrics and loss functions
    def _score_preds(
        self,
        logits,
        labels,
        teacher_logits,
        teacher_labels,
        predict_hidden_states,
        label_hidden_states,
        lm_loss_fct,
        teacher_loss_fct,
        reconstruct_loss_fct,
        shift,
        log,
        log_key
    ):
        """
        Compute metrics such as acc and loss for given predictions.

        Args:
            logits (torch.Tensor): Predictions to compute metrics on.
            labels (torch.Tensor): True labels.
            loss_fct (torch.nn.Loss): Loss function to compute loss.
            log (dict): Dictionary to store computed metrics.
            log_key (str): Prefix for logging.
        Returns:
            torch.Tensor: The computed loss.
        """

        # Building LM terms
        hydra_logits = logits[:, : -shift].contiguous()
        teacher_logits = teacher_logits[:, shift - 1 : -1].contiguous()
        teacher_labels = teacher_labels[:, shift - 1 : -1].contiguous()
        hydra_labels = labels[..., shift :].contiguous()

        hydra_logits = hydra_logits.view(-1, hydra_logits.shape[-1])
        teacher_logits = teacher_logits.view(-1, teacher_logits.shape[-1])
        teacher_labels = teacher_labels.view(-1)
        hydra_labels = hydra_labels.view(-1)

        hydra_labels = hydra_labels.to(hydra_logits.device)
        not_ignore_lm = hydra_labels.ne(IGNORE_TOKEN_ID)

        # Building reconstruction terms
        reconstruct_labels = labels[..., shift - 1 :].contiguous().view(-1)
        not_ignore_reconstruct = reconstruct_labels.ne(IGNORE_TOKEN_ID)

        # Hack for metrics on og states
        if shift - 1 == 0:
            hydra_pred_hidden_states = predict_hidden_states.contiguous()
        else:
            hydra_pred_hidden_states = predict_hidden_states[:, : -(shift - 1)].contiguous()
        hydra_label_hidden_states = label_hidden_states[:, shift - 1 :].contiguous()
        hydra_pred_hidden_states = hydra_pred_hidden_states.view(-1, hydra_pred_hidden_states.shape[-1])
        hydra_label_hidden_states = hydra_label_hidden_states.view(-1, hydra_label_hidden_states.shape[-1])

        # Compute losses
        cur_lm_loss = lm_loss_fct(hydra_logits, hydra_labels)
        cur_teacher_loss = teacher_loss_fct(hydra_logits[not_ignore_lm], teacher_logits[not_ignore_lm]) # When distilling, no ignore label
        cur_reconstruct_loss = reconstruct_loss_fct(hydra_pred_hidden_states[not_ignore_reconstruct], hydra_label_hidden_states[not_ignore_reconstruct])

        # Computing acc
        hydra_labels = hydra_labels[not_ignore_lm]
        teacher_labels = teacher_labels[not_ignore_lm]

        # Add top-k accuracy
        for k in range(1, 6):
            _, topk = hydra_logits.topk(k, dim=-1)
            topk = topk[not_ignore_lm]
            correct = topk.eq(hydra_labels.unsqueeze(-1)).any(-1)
            teacher_correct = topk.eq(teacher_labels.unsqueeze(-1)).any(-1)
            log[f"{log_key}_top{k}"] = correct.float().mean().item()
            log[f"{log_key}_teacher_top{k}"] = teacher_correct.float().mean().item()
        
        # logging losses
        log[f"{log_key}_lm_loss"] = cur_lm_loss.item()
        log[f"{log_key}_teacher_loss"] = cur_teacher_loss.item()
        log[f"{log_key}_reconstruct_loss"] = cur_reconstruct_loss.item()

        cur_loss = self.args.lm_loss_weight * cur_lm_loss + self.args.teacher_loss_weight * cur_teacher_loss + self.args.reconstruction_loss_weight * cur_reconstruct_loss
        log[f"{log_key}_loss"] = cur_loss.item()

        return cur_loss

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        return_log=False,
    ):
        """
        Compute the training loss for the model.

        Args:
            model (torch.nn.Module): The model for which to compute the loss.
            inputs (dict): The input data, including input IDs, attention mask, and labels.
            return_outputs (bool): Whether to return model outputs along with the loss.

        Returns:
            Union[float, Tuple[float, torch.Tensor]]: The computed loss, optionally with model outputs.
        """

        # Unpack the data
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        attention_mask = inputs["attention_mask"]
        base_hidden_states = None
        if "base_hidden_states" in inputs:
            base_hidden_states = inputs["base_hidden_states"].to(torch.bfloat16)

        # DDP will give us model.module
        if hasattr(model, "module"):
            hydra = model.module.hydra
        else:
            hydra = model.hydra

        all_hydra_logits, all_hydra_hidden_states, _, orig_logits, base_hidden_states = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            base_hidden_states=base_hidden_states,
            run_hydra_head=True,
            output_orig=True,
            noise_alpha=self.args.noise_alpha if model.training else 0.0,
        )

        # Fix for smooth l1 loss
        all_hydra_logits = all_hydra_logits.to(torch.float32)
        all_hydra_hidden_states = all_hydra_hidden_states.to(torch.float32)
        orig_logits = orig_logits.to(torch.float32)
        base_hidden_states = base_hidden_states.to(torch.float32)

        # Get teacher probs
        teacher_probs = F.softmax(orig_logits, dim=-1)
        teacher_labels = teacher_probs.argmax(dim=-1)

        # Shift so that tokens < n predict n
        loss = 0
        lm_loss_fct = CrossEntropyLoss()
        teacher_loss_fct = CrossEntropyLoss()
        reconstruct_loss_fct = SmoothL1Loss()
        log = {}

        # Get base model perf
        _ = self._score_preds(
            orig_logits, 
            labels, 
            teacher_probs,
            teacher_labels,
            base_hidden_states, 
            base_hidden_states, 
            lm_loss_fct, 
            teacher_loss_fct,
            reconstruct_loss_fct, 
            1, 
            log, 
            "orig"
        )

        for i in range(hydra):
            shift = 2 + i
            loss += self._score_preds(
                all_hydra_logits[i],
                labels,
                teacher_probs,
                teacher_labels,
                all_hydra_hidden_states[i],
                base_hidden_states,
                lm_loss_fct,
                teacher_loss_fct,
                reconstruct_loss_fct,
                shift,
                log,
                f"hydra{i}"
            )

        self.log(log)

        if return_log:
            return (loss, all_hydra_logits, log)
        return (loss, all_hydra_logits) if return_outputs else loss
    
    # Overwriting save to only save the hydra heads
    def _save_checkpoint(self, model, trial, metrics=None):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{round(self.state.epoch)}"

        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
            print(
                f"Checkpoint destination directory {output_dir} already exists and is non-empty."
                "Saving will proceed but saved results may be invalid."
            )
        self.save_model(output_dir, _internal_call=True)
        
        if self.args.should_save:
            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        torch.save(
            self.model.hydra_head.state_dict(),
            os.path.join(output_dir, "hydra_lm_head.pt"),
        )
        self.args.hydra_config.save_pretrained(output_dir)
        
        if hasattr(self, 'optimizer') and self.optimizer is not None:
            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            print(f"Saved optimizer state to {output_dir}/optimizer.pt")
        if hasattr(self, 'lr_scheduler') and self.lr_scheduler is not None:
            torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
            print(f"Saved scheduler state to {output_dir}/scheduler.pt")
    
    # Overwriting scheduler so final lr can be specified
    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        self._created_lr_scheduler = True
        return self.lr_scheduler

    def train(self, *args, **kwargs):
        return super().train(*args, **kwargs)
    
    # Overwriting optimizer to add momentum for SGD
    def create_optimizer(self):
        if self.args.optim == "sgd":
            decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if (n in decay_parameters and p.requires_grad)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = torch.optim.SGD(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                momentum=self.args.sgd_momentum,
                weight_decay=self.args.weight_decay
            )
            return self.optimizer
        else:
            return super().create_optimizer()

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="~/PTM/vicuna-7b-v1.3")
    # model_name_or_path: Optional[str] = field(default="~/PTM/Llama-2-7b-chat-hf")
    hydra_model_path: Optional[str] = field(default=None)
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "Load in 4 bit."},
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Load in 8 bit."},
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default="sharegpt_clean.json",
        metadata={"help": "Path to the training data."},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=1024,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    hydra_num_heads: int = field(
        default=1,
        metadata={"help": "Number of Hydra heads."},
    )
    hydra_num_layers: int = field(
        default=1,
        metadata={"help": "Number of layers for each Hydra head."},
    )
    hydra_head_arch: str = field(
        default="mlp",
        metadata={"help": "What model architecture to use for Hydra heads."},
    )
    grounded_heads: bool = field(
        default=True,
        metadata={"help": "Whether to ground the Hydra heads on previous head predictions."},
    )
    hidden_state_offset: int = field(
        default=0,
        metadata={"help": "Number of layers back from final layer to use as embeddings"},
    )
    remote_upload_base: str = field(
        default=None,
        metadata={"help": "Remote location for training artifacts"}
    )
    dataloader_prefetch_factor: int = field(
        default=2,
    )
    global_batch_size: int = field(
        default=12,
        metadata={"help": "Global batch size."},
    )
    final_lr_multiplier: float = field(
        default=0.0,
        metadata={"help": "Final learning rate multiplier."},
    )
    noise_alpha: int = field(
        default=0,
        metadata={"help": "Noise std for input."},
    )
    dropout_rate: float = field(
        default=0.0,
        metadata={"help": "Dropout rate."},
    )
    lm_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for LM loss"},
    )
    teacher_loss_weight: float = field(
        default=0.0,
        metadata={"help": "Weight for teacher distilation loss"},
    )
    reconstruction_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for reconstruction loss"},
    )
    lr_scheduler_type: str = field(
        default="cosine",
        metadata={"help": "Type of learning rate scheduler."},
    )
    sgd_momentum: float = field(
        default=0.0,
        metadata={"help": "Momentum for SGD optimizer."},
    )


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """
    Save the model's state dictionary to a specified directory.

    Args:
        trainer (transformers.Trainer): The Hugging Face Trainer object.
        output_dir (str): The directory where the model state dictionary will be saved.
    """
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def save_model(lm_head: torch.nn.Module, output_dir: str):
    """
    Save LM heads to potentially remote object store

    Args:
        lm_head (torch.Module): The LM head module.
        output_dir (str): The directory where the model state dictionary will be saved.
    """
    state_dict = lm_head.state_dict()
    object_store = maybe_create_object_store_from_uri(output_dir)
    if object_store is None:
        torch.save(state_dict, os.path.join(output_dir, "hydra_lm_head.pt"))
    else:
        with tempfile.NamedTemporaryFile() as tmp:
            torch.save(state_dict, tmp)
            object_store.upload_object("hydra_lm_head.pt", tmp.name)


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    base_conv = get_conversation_template("vicuna")
    roles = {"human": base_conv.roles[0], "gpt": base_conv.roles[1]}

    all_input_ids = []
    all_labels = []
    all_attention_masks = []

    max_len = tokenizer.model_max_length

    for i, source in enumerate(sources):
        if len(source) == 0:
            empty_ids = torch.full((max_len,), tokenizer.pad_token_id, dtype=torch.long)
            empty_labels = torch.full_like(empty_ids, IGNORE_TOKEN_ID)
            empty_attn = empty_ids.ne(tokenizer.pad_token_id)
            all_input_ids.append(empty_ids)
            all_labels.append(empty_labels)
            all_attention_masks.append(empty_attn)
            continue

        if roles.get(source[0]["from"], None) != base_conv.roles[0] and len(source) > 1:
            source = source[1:]

        full_conv = get_conversation_template("vicuna")
        full_conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == full_conv.roles[j % 2], f"{i}, {j}, {role}, {full_conv.roles[j % 2]}"
            full_conv.append_message(role, sentence["value"])
        full_prompt = full_conv.get_prompt()

        input_ids = tokenizer(
            full_prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=max_len,
            truncation=True,
        ).input_ids[0]
        attention_mask = input_ids.ne(tokenizer.pad_token_id)
        total_len = int(attention_mask.sum().item())

        labels = torch.full_like(input_ids, IGNORE_TOKEN_ID)

        step_conv = get_conversation_template("vicuna")
        step_conv.messages = []
        cum_lens = []

        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            step_conv.append_message(role, sentence["value"])
            partial_prompt = step_conv.get_prompt()
            partial_ids = tokenizer(
                partial_prompt,
                return_tensors="pt",
                padding="max_length",
                max_length=max_len,
                truncation=True,
            ).input_ids[0]
            partial_len = int(partial_ids.ne(tokenizer.pad_token_id).sum().item())
            partial_len = min(partial_len, total_len)
            cum_lens.append(partial_len)

        prev = 0
        for j, sentence in enumerate(source):
            cur = cum_lens[j]
            cur = min(cur, total_len)

            if sentence["from"] == "gpt":
                if cur > prev:
                    labels[prev:cur] = input_ids[prev:cur]
            prev = cur

        all_input_ids.append(input_ids)
        all_labels.append(labels)
        all_attention_masks.append(attention_mask)

    input_ids = torch.stack(all_input_ids, dim=0)
    labels = torch.stack(all_labels, dim=0)
    attention_mask = torch.stack(all_attention_masks, dim=0)

    return dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning.

    Args:
        raw_data (list): A list of raw data examples.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for data preprocessing.
    """

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )

def read_jsonl(path: str) -> Sequence[Dict]:
    """Read a JSONL file.

    Args:
        path (str): Path to the JSONL file.

    Returns:
        list: A list of dictionaries.
    """
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

def make_raw_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for data preprocessing.
        data_args: Data arguments.

    Returns:
        dict: A dictionary containing train dataset only.
    """
    dataset_cls = SupervisedDataset
    rank0_print("Loading data...")

    train_json = read_jsonl(data_args.data_path)
    train_dataset = dataset_cls(train_json, tokenizer=tokenizer)

    return dict(train_dataset=train_dataset)

def make_supervised_data_module(tokenizer, data_args, training_args, hidden_size):
    return make_raw_supervised_data_module(tokenizer, data_args)

def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    config.use_cache = False

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config if model_args.load_in_4bit else None,
        load_in_4bit=model_args.load_in_4bit,
        load_in_8bit=model_args.load_in_8bit,
        device_map=f"cuda:{local_rank}"
    )

    for param in model.parameters():
        param.requires_grad = False

    hydra_lm_head = HydraModel(
        model,
        hydra_num_heads=training_args.hydra_num_heads,
        hydra_num_layers=training_args.hydra_num_layers,
        hydra_head_arch=training_args.hydra_head_arch,
        base_model_name_or_path=model_args.model_name_or_path,
        grounded_heads=training_args.grounded_heads,
        hidden_state_offset=training_args.hidden_state_offset,
        dropout_rate=training_args.dropout_rate,
    )

    if not (model_args.hydra_model_path is None):
        print(f"Loading Hydra LM head from {model_args.hydra_model_path}")
        hydra_model_path = os.path.join(model_args.hydra_model_path, "hydra_lm_head.pt")
        state_dict = torch.load(hydra_model_path, map_location="cpu")
        hydra_lm_head.hydra_head.load_state_dict(state_dict)
    
    print("Saving ckpt locally to: ", training_args.output_dir)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    # modify tokenizer for LLaMA models
    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.eos_token
    # print("Tokenizer pad token: ", tokenizer.pad_token)
    # print("Tokenizer pad_token_id: ", tokenizer.pad_token_id)

    # Load data
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, training_args=training_args, hidden_size=config.hidden_size)

    hydra_config = HydraConfig(
        hydra_num_heads=training_args.hydra_num_heads,
        hydra_num_layers=training_args.hydra_num_layers,
        hydra_head_arch=training_args.hydra_head_arch,
        base_model_name_or_path=model_args.model_name_or_path,
        grounded_heads=training_args.grounded_heads,
        hidden_state_offset=training_args.hidden_state_offset,
    )

    hydra_config.save_pretrained(training_args.output_dir)
    training_args.hydra_config = hydra_config # For saving during checkpointing

    training_args.remove_unused_columns = False
    trainer = CustomizedTrainer(
        model=hydra_lm_head, tokenizer=tokenizer, args=training_args, train_dataset=data_module["train_dataset"]
    )
    
    if trainer.optimizer is None:
        trainer.create_optimizer()
    if trainer.lr_scheduler is None:
        trainer.create_scheduler(num_training_steps=trainer.get_num_train_epochs() * len(trainer.get_train_dataloader()))
    
    if not (model_args.hydra_model_path is None):
        optimizer_path = os.path.join(model_args.hydra_model_path, "optimizer.pt")
        scheduler_path = os.path.join(model_args.hydra_model_path, "scheduler.pt")
        
        if os.path.exists(optimizer_path):
            print(f"Loading optimizer state from {optimizer_path}")
            optimizer_state = torch.load(optimizer_path, map_location="cpu")
            trainer.optimizer.load_state_dict(optimizer_state)
            print("Optimizer state loaded (momentum preserved!)")
        else:
            print("No optimizer state found, starting with fresh optimizer")
        
        if os.path.exists(scheduler_path):
            print(f"Loading scheduler state from {scheduler_path}")
            scheduler_state = torch.load(scheduler_path, map_location="cpu")
            trainer.lr_scheduler.load_state_dict(scheduler_state)
            print("Scheduler state loaded")

    trainer.train()

    if hasattr(hydra_lm_head, "module"):
        lm_head = hydra_lm_head.module.hydra_head
    else:
        lm_head = hydra_lm_head.hydra_head

    torch.save(
        lm_head.state_dict(),
        os.path.join(training_args.output_dir, "hydra_lm_head.pt"),
    )
    
    torch.save(
        trainer.optimizer.state_dict(),
        os.path.join(training_args.output_dir, "optimizer.pt"),
    )
    print(f"Saved optimizer state to {training_args.output_dir}/optimizer.pt")
    
    torch.save(
        trainer.lr_scheduler.state_dict(),
        os.path.join(training_args.output_dir, "scheduler.pt"),
    )
    print(f"Saved scheduler state to {training_args.output_dir}/scheduler.pt")

if __name__ == "__main__":
    train()
