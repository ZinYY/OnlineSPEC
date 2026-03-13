import argparse
import json
import os
import torch
from transformers import AutoTokenizer
from accelerate.utils import set_seed
from datasets import load_dataset
from typing import Any, Dict, List
from torch.utils.data import DataLoader
from tqdm import tqdm

from cnets import Model
from configs import EConfig

parser = argparse.ArgumentParser(description='EAGLE Training - Single GPU')
parser.add_argument('--basepath', type=str, required=True, help='Path to base model')
parser.add_argument('--trainpath', type=str, required=True, help='Path to training data')
parser.add_argument('--ea-model-path', type=str, default=None, help='Path to pretrained EAGLE model (optional, for fine-tuning)')
parser.add_argument('--savedir', type=str, default='./checkpoints', help='Directory to save eagle model weights')
parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
parser.add_argument('--num_epochs', type=int, default=1, help='Number of training epochs')
parser.add_argument('--max_len', type=int, default=2048, help='Maximum sequence length')
parser.add_argument('--lr', type=float, default=6e-5, help='Learning rate')
args = parser.parse_args()

ds_config = None
train_config = {
    "bs": args.batch_size,
    "num_epochs": args.num_epochs,
    "num_workers": 4,
    "max_len": args.max_len,
    "config_path": args.config,
    "gradient_checkpoint": True
}

torch.backends.cuda.matmul.allow_tf32 = True
set_seed(0)

device = torch.device('cuda')
print(f"Using device: {device}")


def build_dataset(tokenizer, datapath):
    from fastchat.model.model_adapter import get_conversation_template

    ds = load_dataset('json', data_files=datapath)
    ds = ds['train']
    ds = ds.shuffle(seed=42)
    ds1 = ds
    original_columns1 = ds1.column_names
    num_proc = 4

    def preprocess_function(examples):
        new_examples = {
            "attention_mask": [],
            "input_ids": [],
            "loss_mask": []
        }
        for i in range(len(examples['id'])):
            source = examples['conversations'][i]
            if not source:
                continue
            
            conv = get_conversation_template("vicuna")
            
            for j, turn in enumerate(source):
                role = turn[0]
                content = turn[1]
                
                if role.upper() in ["USER", "HUMAN"]:
                    conv.append_message(conv.roles[0], content)
                elif role.upper() in ["ASSISTANT", "GPT"]:
                    conv.append_message(conv.roles[1], content)
            
            conversation = conv.get_prompt()

            if not tokenizer.pad_token_id:
                tokenizer.pad_token_id = tokenizer.unk_token_id

            input_ids = tokenizer(
                conversation,
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids[0]
            if len(input_ids) > train_config["max_len"]:
                continue
            loss_mask = torch.ones_like(input_ids)

            conv = get_conversation_template("vicuna")
            sep = conv.sep + conv.roles[1] + ":"
            sep2 = conv.sep2

            total_len = len(input_ids)
            turns = conversation.split(sep2)
            
            cur_len = 1
            loss_mask[:cur_len] = 0
            
            for i, turn in enumerate(turns):
                if turn == "":
                    break
                turn_len = len(tokenizer(turn).input_ids)

                parts = turn.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

                loss_mask[cur_len: cur_len + instruction_len] = 0
                cur_len += turn_len

            loss_mask[cur_len:] = 0
            attention_mask = torch.ones_like(loss_mask)

            new_examples["input_ids"].append(input_ids[None, :])
            new_examples["loss_mask"].append(loss_mask[None, :])
            new_examples["attention_mask"].append(attention_mask[None, :])

        return new_examples

    ds1 = ds1.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns1,
        load_from_cache_file=True
    )

    ds1.set_format(type="torch")
    return ds1


class DataCollatorWithPadding:

    def paddingtensor2D(self, intensors, N):
        B, n = intensors.shape
        padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(item['input_ids'].shape[1] for item in features)
        batch_input_ids = torch.cat([self.paddingtensor2D(item['input_ids'], max_length) for item in features])
        batch_attention_mask = torch.cat(
            [self.paddingtensor2D(item['attention_mask'], max_length) for item in features])
        batch_loss_mask = torch.cat(
            [self.paddingtensor2D(item['loss_mask'], max_length) for item in features])

        batch = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
        }
        return batch


print("Loading tokenizer and building dataset...")
tokenizer = AutoTokenizer.from_pretrained(args.basepath)
traindataset = build_dataset(tokenizer, args.trainpath)

print("Initializing model...")
config = EConfig.from_pretrained(train_config["config_path"])
model = Model(config, ds_config, train_config, path=args.basepath, load_emb=True, load_head=True)

print("Scanning dataset to build draft vocabulary...")
print("(This will process the data again to count token frequencies)")
model.scandata(args.trainpath, args.basepath)
print("Draft vocabulary built successfully!")

if args.ea_model_path:
    print(f"\nLoading pretrained EAGLE weights from {args.ea_model_path}...")
    eagle_weights_path = os.path.join(args.ea_model_path, "pytorch_model.bin")
    
    if os.path.exists(eagle_weights_path):
        pretrained_state_dict = torch.load(eagle_weights_path, map_location='cpu')
        
        model_state_dict = model.state_dict()
        loaded_keys = []
        missing_keys = []
        
        for key, value in pretrained_state_dict.items():
            if key in model_state_dict:
                if model_state_dict[key].shape == value.shape:
                    model_state_dict[key] = value
                    loaded_keys.append(key)
                else:
                    print(f"  ⚠️  Skipping {key}: shape mismatch ({model_state_dict[key].shape} vs {value.shape})")
            else:
                missing_keys.append(key)
        
        model.load_state_dict(model_state_dict)
        print(f"  Loaded {len(loaded_keys)} parameters from pretrained model")
        if missing_keys:
            print(f"  Skipped {len(missing_keys)} keys not in current model")
        print("  Starting fine-tuning from pretrained weights")
    else:
        print(f"  Warning: {eagle_weights_path} not found, training from scratch")
else:
    print("\nTraining from scratch (no pretrained weights)")

model = model.to(device)

num_epochs = train_config["num_epochs"]

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

os.makedirs(args.savedir, exist_ok=True)

train_loader = DataLoader(
    traindataset, 
    batch_size=train_config["bs"], 
    shuffle=True,
    num_workers=train_config["num_workers"],
    pin_memory=True,
    collate_fn=DataCollatorWithPadding()
)


def save_eagle_model(model, config, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    config_dict = {
        "architectures": ["EAGLEModel"],
        "auto_map": {
            "AutoConfig": "configuration_eagle.EAGLEConfig",
            "AutoModelForCausalLM": "modeling_eagle.EAGLEForCausalLM"
        },
        "bias": False,
        "vocab_size": config.vocab_size,
        "draft_vocab_size": config.draft_vocab_size,
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "num_hidden_layers": config.num_hidden_layers,
        "max_position_embeddings": config.max_position_embeddings,
        "rms_norm_eps": config.rms_norm_eps,
        "hidden_act": config.hidden_act,
        "pad_token_id": config.pad_token_id,
        "bos_token_id": config.bos_token_id,
        "eos_token_id": config.eos_token_id,
        "torch_dtype": "float32",
        "transformers_version": "4.0.0",
    }
    
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)
    
    state_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            state_dict[name] = param.cpu().detach()
    
    if hasattr(model, 'd2t'):
        state_dict['d2t'] = model.d2t.cpu()
    if hasattr(model, 't2d'):
        state_dict['t2d'] = model.t2d.cpu()
    
    torch.save(state_dict, os.path.join(save_dir, "pytorch_model.bin"))
    
    if hasattr(model, 'd2t') and hasattr(model, 't2d'):
        cache_dict = {
            "d2t": model.d2t.cpu(),
            "t2d": model.t2d.cpu()
        }
        torch.save(cache_dict, os.path.join(save_dir, "cache.pt"))
    
    print(f"  - Saved config.json")
    print(f"  - Saved pytorch_model.bin ({len(state_dict)} parameters)")
    if hasattr(model, 'd2t'):
        print(f"  - Saved cache.pt (draft vocabulary)")


print("Starting training...")
for epoch in range(num_epochs):
    print(f"\n{'='*60}")
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f"{'='*60}")

    model.train()
    epoch_acces = [[] for _ in range(model.length)]
    epoch_plosses = [[] for _ in range(model.length)]

    for batch_idx, data in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}")):
        optimizer.zero_grad()

        plosses, vlosses, acces = model(
            input_ids=data["input_ids"].to(device),
            attention_mask=data["attention_mask"].to(device),
            loss_mask=data["loss_mask"].to(device),
        )

        ploss_weight = [0.8 ** i for i in range(len(plosses))]
        ploss = sum([ploss_weight[i] * plosses[i] for i in range(len(plosses))])
        loss = ploss

        loss.backward()
        optimizer.step()
        
        epoch_acces = [epoch_acces[i] + [acces[i]] for i in range(len(acces))]
        epoch_plosses = [epoch_plosses[i] + [plosses[i].item()] for i in range(len(plosses))]

    print(f"\n--- Epoch {epoch + 1} Summary ---")
    for i in range(len(epoch_acces)):
        acc_i = sum(epoch_acces[i]) / len(epoch_acces[i])
        print(f"Position {i} - Acc: {acc_i:.4f}")

    for i in range(len(epoch_plosses)):
        loss_i = sum(epoch_plosses[i]) / len(epoch_plosses[i])
        print(f"Position {i} - Loss: {loss_i:.4f}")
    
    torch.cuda.empty_cache()

print("\nSaving final EAGLE model...")
final_model_dir = f"{args.savedir}/eagle_model_final"
save_eagle_model(model, config, final_model_dir)
print(f"Final EAGLE model saved to {final_model_dir}")

print("\nTraining completed!")
