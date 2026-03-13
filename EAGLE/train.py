import argparse
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    AutoConfig,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset
from fastchat.model.model_adapter import get_conversation_template
from torch import nn, optim
from accelerate import Accelerator
from accelerate.utils import set_seed
from safetensors import safe_open
import json

from eagle.model.cnets import Model
from eagle.model.configs import EConfig

parser = argparse.ArgumentParser(description='Training script')
parser.add_argument('--basepath', type=str)
parser.add_argument('--ea-model-path', type=str, required=True, help='EA model path')
parser.add_argument('--data_path', type=str, help='Training data file path')
parser.add_argument('--lr', type=float, default=3e-5)
parser.add_argument('--num_epochs', type=int, default=1)
parser.add_argument('--cpdir', type=str, default='0')
args = parser.parse_args()

train_config = {
    "lr": args.lr,
    "bs": 4,
    "gradient_accumulation_steps": 1,
    "is_warmup": True,
    "num_epochs": args.num_epochs,
    "num_warmup_steps": 2000,
    "total_steps": 800000,
    "p_w": 0.1,
    "v_w": 1.0,
    "head_w": 0.1,
    "data_noise": True,
    "noise": "uniform",
    "mean": 0.0,
    "std": 0.2,
    "max_len": 2048,
    "max_len_threshold": 2048,
    "enable_length_filter": True,
    "adaptive_batch_size": True,
    "min_batch_size": 1,
    "memory_efficient": True,
    "b1": 0.9,
    "b2": 0.95,
    "grad_clip": 0.5,
    "save_freq": 5
}

def process_batch_conversations_truncate(examples, tokenizer, max_length=None):
    if max_length is None:
        max_length = train_config["max_len"]
    
    batch_input_ids = []
    batch_hidden_states = []
    batch_targets = []
    batch_attention_masks = []
    batch_loss_masks = []
    
    skipped_samples = 0
    processed_samples = 0
    split_samples = 0
    
    for example in examples:
        conversation = example['prompt']
        
        try:
            inputs = tokenizer(
                conversation,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=False
            )
            input_ids = inputs.input_ids[0]
            
            sub_sample = process_single_sample(input_ids, conversation, tokenizer)
            if sub_sample is None:
                skipped_samples += 1
                continue
                    
            batch_input_ids.append(sub_sample['input_ids'])
            batch_hidden_states.append(sub_sample['hidden_states'])
            batch_targets.append(sub_sample['target'])
            batch_attention_masks.append(sub_sample['attention_mask'])
            batch_loss_masks.append(sub_sample['loss_mask'])
            processed_samples += 1
                
        except Exception as e:
            print(f"Error processing sample, skipping: {str(e)}")
            skipped_samples += 1
            continue
    
    if len(batch_input_ids) == 0:
        print(f"All {len(examples)} samples were skipped")
        return None
    
    if skipped_samples > 0 or split_samples > 0:
        print(f"Batch statistics: processed {processed_samples} samples, skipped {skipped_samples} invalid samples, split {split_samples} extra samples")
    
    max_seq_len = max(ids.shape[0] for ids in batch_input_ids)
    max_seq_len = min(max_seq_len, train_config["max_len_threshold"])
    
    padded_input_ids = []
    padded_hidden_states = []
    padded_targets = []
    padded_attention_masks = []
    padded_loss_masks = []
    
    for i in range(len(batch_input_ids)):
        if batch_input_ids[i].shape[0] > max_seq_len:
            batch_input_ids[i] = batch_input_ids[i][:max_seq_len]
            batch_hidden_states[i] = batch_hidden_states[i][:max_seq_len]
            batch_targets[i] = batch_targets[i][:max_seq_len]
            batch_attention_masks[i] = batch_attention_masks[i][:max_seq_len]
            batch_loss_masks[i] = batch_loss_masks[i][:max_seq_len]
        
        seq_len = batch_input_ids[i].shape[0]
        pad_len = max_seq_len - seq_len
        
        if pad_len > 0:
            padded_input_ids.append(F.pad(batch_input_ids[i], (0, pad_len), value=tokenizer.pad_token_id))
            
            hidden_pad = torch.zeros(pad_len, batch_hidden_states[i].shape[1]).to(batch_hidden_states[i].device)
            padded_hidden_states.append(torch.cat([batch_hidden_states[i], hidden_pad], dim=0))
            
            target_pad = torch.zeros(pad_len, batch_targets[i].shape[1]).to(batch_targets[i].device)
            padded_targets.append(torch.cat([batch_targets[i], target_pad], dim=0))
            
            padded_attention_masks.append(F.pad(batch_attention_masks[i], (0, pad_len), value=0))
            padded_loss_masks.append(F.pad(batch_loss_masks[i], (0, pad_len), value=0))
        else:
            padded_input_ids.append(batch_input_ids[i])
            padded_hidden_states.append(batch_hidden_states[i])
            padded_targets.append(batch_targets[i])
            padded_attention_masks.append(batch_attention_masks[i])
            padded_loss_masks.append(batch_loss_masks[i])
    
    return {
        "input_ids": torch.stack(padded_input_ids),
        "hidden_states": torch.stack(padded_hidden_states),
        "target": torch.stack(padded_targets),
        "attention_mask": torch.stack(padded_attention_masks),
        "loss_mask": torch.stack(padded_loss_masks)
    }

def process_single_sample(input_ids, conversation, tokenizer, is_chunk=False):
    try:
        loss_mask = torch.ones_like(input_ids)
        
        if not is_chunk:
            conv = get_conversation_template("vicuna")
            sep = conv.sep + conv.roles[1] + ":"
            sep = conv.roles[1] + ":"

            total_len = int(input_ids.ne(tokenizer.pad_token_id).sum())
            turns = conversation.split(conv.sep2)
            cur_len = 1
            loss_mask[:cur_len] = 0
            
            for i, turn in enumerate(turns):
                if turn == "":
                    break
                turn_len = len(tokenizer(turn).input_ids)
                
                parts = turn.split(sep)
                if len(parts) != 2:
                    parts = turn.rsplit(sep, maxsplit=1)
                    if len(parts) != 2:
                        break
                parts[0] += sep
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2
                
                if i != 0 and not tokenizer.legacy:
                    instruction_len -= 1
                    
                loss_mask[cur_len: cur_len + instruction_len] = 0
                cur_len += turn_len
                
                if i != 0 and not tokenizer.legacy:
                    cur_len -= 1
                    
            loss_mask[cur_len:] = 0
        else:
            chunk_len = len(input_ids)
            start_loss_idx = chunk_len * 2 // 3
            loss_mask[:start_loss_idx] = 0
        
        if loss_mask.sum().item() == 0:
            return None

        with torch.no_grad():
            outs = base_model(input_ids.unsqueeze(0).cuda(), output_hidden_states=True)
            hidden_states = outs.hidden_states[-1].squeeze(0)
            
            if train_config["memory_efficient"]:
                del outs
                torch.cuda.empty_cache()
            
        if train_config["data_noise"]:
            if train_config["noise"] == "uniform":
                noise = (torch.rand_like(hidden_states) - 0.5) * train_config["std"] * 512 / hidden_states.shape[0]
                hidden_states = hidden_states + noise

        target = hidden_states[1:]
        zeropadding = torch.zeros(1, target.shape[1]).to(target.device)
        target = torch.cat((target, zeropadding), dim=0)
        
        return {
            'input_ids': input_ids,
            'hidden_states': hidden_states,
            'target': target,
            'attention_mask': torch.ones_like(input_ids),
            'loss_mask': loss_mask
        }
        
    except Exception as e:
        print(f"Error processing single sample: {str(e)}")
        return None
    

def top_accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k)
        return res

if __name__ == "__main__":
    set_seed(0)
    accelerator = Accelerator(mixed_precision='bf16')

    WANDB_DISABLED_FLAG = os.environ.get("WANDB_DISABLED", "").lower() in ("1", "true", "yes", "y", "on")
    if accelerator.is_main_process and not WANDB_DISABLED_FLAG:
        import wandb
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "ess"), 
            entity=os.environ.get("WANDB_ENTITY", ""),
            config=train_config
        )
    else:
        class _WBStub:
            @staticmethod
            def log(*args, **kwargs):
                return None
        wandb = _WBStub()

    base_model = AutoModelForCausalLM.from_pretrained(
        args.basepath,
        device_map="auto",
        torch_dtype=torch.float16
    )
    base_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.basepath, use_fast=False)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    baseconfig = AutoConfig.from_pretrained(args.basepath)
    head = torch.nn.Linear(baseconfig.hidden_size, baseconfig.vocab_size, bias=False)

    try:
        with open(os.path.join(args.basepath, "model.safetensors.index.json"), "r") as f:
            index_json = json.loads(f.read())
            head_path = index_json["weight_map"]["lm_head.weight"]
        with safe_open(os.path.join(args.basepath, head_path),
                    framework="pt",
                    device="cpu") as f:
            tensor_slice = f.get_slice("lm_head.weight")
            vocab_size, hidden_dim = tensor_slice.get_shape()
            tensor = tensor_slice[:, :hidden_dim].float()
    except:
        with open(os.path.join(args.basepath, "pytorch_model.bin.index.json"), "r") as f:
            index_json = json.loads(f.read())
            head_path = index_json["weight_map"]["lm_head.weight"]
        weights = torch.load(os.path.join(args.basepath, head_path))
        tensor = weights["lm_head.weight"].float()

    head.weight.data = tensor
    head.eval()

    for param in head.parameters():
        param.requires_grad = False

    if accelerator.is_main_process:
        if not os.path.exists(args.cpdir):
            os.makedirs(args.cpdir)

    config = EConfig.from_pretrained('config/config.json')

    model = Model(config, load_emb=False)

    checkpoint_path = os.path.join(args.ea_model_path, "pytorch_model.bin")
    if os.path.exists(checkpoint_path):
        print(f"load from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint)
    else:
        print("start from scratch")

    criterion = nn.SmoothL1Loss(reduction="none")
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_config["lr"],
        betas=(train_config["b1"], train_config["b2"])
    )

    if train_config["is_warmup"]:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=train_config["num_warmup_steps"],
            num_training_steps=train_config["total_steps"]
        )
        model, head, optimizer, scheduler = accelerator.prepare(model, head, optimizer, scheduler)
    else:
        model, head, optimizer = accelerator.prepare(model, head, optimizer)

    if args.data_path:
        data_file = args.data_path
    else:
        print("No data path specified, using default data file")

    ds = load_dataset('json', data_files=data_file)
    ds = ds['train']

    for epoch in range(train_config["num_epochs"]):
        top_3acc = [0 for _ in range(3)]
        correct = 0
        total = 0
        epoch_loss = 0
        num_batches = 0
        model.train()
        
        batch_size = train_config["bs"]
        num_samples = len(ds)
        
        total_skipped_samples = 0
        total_processed_samples = 0
        
        batch_start = 0
        progress_bar = tqdm(total=num_samples, desc=f"Epoch {epoch+1}")
        
        while batch_start < num_samples:
            batch_end = min(batch_start + batch_size, num_samples)
            batch_examples = [ds[i] for i in range(batch_start, batch_end)]
            
            try:
                data = process_batch_conversations_truncate(batch_examples, tokenizer)
                
                if data is None:
                    total_skipped_samples += len(batch_examples)
                    batch_start = batch_end
                    progress_bar.update(len(batch_examples))
                    continue
                
                data = {k: v.to(accelerator.device) for k, v in data.items()}
                
                actual_batch_size = data["input_ids"].shape[0]
                max_seq_len = data["input_ids"].shape[1]
                total_processed_samples += actual_batch_size
                
                optimizer.zero_grad()

                predict = model(data["hidden_states"], input_ids=data["input_ids"], attention_mask=data["attention_mask"])
                
                with torch.no_grad():
                    target_head = head(data["target"])
                    target_p = nn.Softmax(dim=2)(target_head)
                    target_p = target_p.detach()
                    
                out_head = head(predict)
                out_logp = nn.LogSoftmax(dim=2)(out_head)
                loss_mask = data["loss_mask"][:, :, None]
                
                plogp = target_p * out_logp
                ploss = -torch.sum(torch.sum(loss_mask * plogp, 2)) / (loss_mask.sum() + 1e-5)
                
                vloss = criterion(predict, data["target"])
                vloss = torch.sum(torch.mean(loss_mask * vloss, 2)) / (loss_mask.sum() + 1e-5)
                
                loss = train_config["v_w"] * vloss + train_config["p_w"] * ploss
                
                if loss_mask.sum().item() == 0:
                    print(f"Warning: loss_mask is all 0, skipping batch")
                    batch_start = batch_end
                    progress_bar.update(len(batch_examples))
                    continue
                    
                if torch.isnan(loss):
                    print(f"Error: detected NaN loss, skipping batch")
                    batch_start = batch_end
                    progress_bar.update(len(batch_examples))
                    continue

                accelerator.backward(loss)
                accelerator.clip_grad_value_(model.parameters(), train_config["grad_clip"])
                optimizer.step()
                
                if train_config["is_warmup"]:
                    scheduler.step()
                    
                topkacc = None
                with torch.no_grad():
                    _, predicted = torch.max(out_head, 2)
                    _, target = torch.max(target_head, 2)
                    ct = loss_mask.sum().item()
                    cc = ((predicted == target) * loss_mask.squeeze(-1)).sum().item()
                    
                    valid_mask = loss_mask.view(-1) == 1
                    out_head_flat = out_head.view(-1, target_head.shape[-1])[valid_mask]
                    target_flat = target.view(-1)[valid_mask]
                    
                    if len(out_head_flat) > 0:
                        topkacc = top_accuracy(out_head_flat, target_flat, (1, 2, 3))
                        
                        for top_i in range(len(topkacc)):
                            top_3acc[top_i] += topkacc[top_i]
                        total += ct
                        correct += cc
                    
                if accelerator.is_main_process and ct != 0:
                    logdict = {
                        "train/lr": optimizer.optimizer.param_groups[0]["lr"],
                        "train/vloss": vloss.item(),
                        "train/ploss": ploss.item(),
                        "train/loss": loss.item(),
                        "train/acc": cc / ct,
                        "train/batch_size": actual_batch_size,
                        "train/max_seq_len": max_seq_len
                    }
                    if topkacc is not None:
                        for id, acc in enumerate(topkacc):
                            logdict[f'train/top_{id + 1}_acc'] = acc.item() / ct
                    wandb.log(logdict)
                
                if train_config["memory_efficient"]:
                    del predict, target_head, target_p, out_head, out_logp, loss_mask, plogp, vloss
                    torch.cuda.empty_cache()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"Error: out of memory, skipping batch")
                    torch.cuda.empty_cache()
                    continue
                else:
                    print(f"Error: runtime error, skipping batch: {str(e)}")
            except Exception as e:
                print(f"Error: unknown error, skipping batch: {str(e)}")
            
            batch_start = batch_end
            progress_bar.update(len(batch_examples))
        
        progress_bar.close()
        
        print(f"Epoch {epoch+1} statistics:")
        print(f"# Processed samples: {total_processed_samples}")
        print(f"# Skipped samples: {total_skipped_samples}")
        print(f"# Total samples: {num_samples}")
        print(f"# Processing rate: {total_processed_samples/num_samples*100:.1f}%")
        
        if accelerator.is_local_main_process:
            for id, i in enumerate(top_3acc):
                wandb.log({f'train/epochtop_{id + 1}_acc': i / total})
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, train_config["num_epochs"], epoch_loss))
            print('Training accuracy: {:.2f}%'.format(100 * correct / total))
            print('correct: {}, total: {}'.format(correct, total))
            wandb.log({"train/epochacc": correct / total, "train/epochloss": epoch_loss})

    if accelerator.is_local_main_process:
        accelerator.save_state(output_dir=f"{args.cpdir}")
        print(f"Model saved to: {args.cpdir}")
        
        loss_file = os.path.join(args.cpdir, "training_loss.json")
        with open(loss_file, 'w') as f:
            json.dump({
                "total_loss": epoch_loss,
                "num_batches": num_batches,
                "avg_loss": epoch_loss / num_batches if num_batches > 0 else 0
            }, f, indent=4)
        print(f"Training loss saved to: {loss_file}")