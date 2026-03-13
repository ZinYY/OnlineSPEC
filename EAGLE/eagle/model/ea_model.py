import copy
import json
import time

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig,AutoConfig
from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from .modeling_mixtral_kv import MixtralForCausalLM as KVMixtralForCausalLM
from .utils import *
from .kv_cache import initialize_past_key_values
from .choices import mc_sim_7b_63
from transformers import AutoTokenizer
import os
from huggingface_hub import hf_hub_download
from .cnets import Model
from .configs import EConfig
from huggingface_hub import hf_hub_download




class EaModel(nn.Module):

    def __init__(
            self,
            base_model,
            base_model_name_or_path,
            ea_model_path,
    ):

        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = base_model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.lm_head.weight.shape[0]
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path)
        config = EConfig.from_pretrained(ea_model_path)
        with open(ea_model_path,"r") as f:
            con=json.loads(f.read())
        try:
            bias=con["bias"]
        except:
            bias=True
        self.ea_layer = Model(config,bias=bias)

        low_memory=False

        device = base_model.model.layers[-1].self_attn.q_proj.weight.device
        if device!=base_model.lm_head.weight.device:
            self.ea_layer.diff_device = True
            if not low_memory:
                # self.ea_layer.head=nn.Linear(base_model.lm_head.in_features,base_model.lm_head.out_features,bias=False)
                # self.ea_layer.head.weight=copy.deepcopy(base_model.lm_head.weight)
                # self.ea_layer.head.to(device)
                self.ea_layer.headweight = base_model.lm_head.weight.clone().to(device)
            else:
                self.ea_layer.layer_device = device

        else:
            self.ea_layer.diff_device = False
        self.ea_layer.to(self.base_model.dtype).to(device)
        self.ea_layer.init_tree()

    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer

    @classmethod
    def from_pretrained(
            cls,
            Type="LLaMA",
            base_model_path=None,
            ea_model_path=None,
            **kwargs,
    ):
        #assert Type=="LLaMA" or "Mixtral"
        Type=AutoConfig.from_pretrained(base_model_path).architectures[0]
        if Type=='LlamaForCausalLM':
            base_model = KVLlamaForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        else:
            base_model = KVMixtralForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )

        configpath=os.path.join(ea_model_path,"config.json")
        if not os.path.exists(configpath):
            configpath = hf_hub_download(ea_model_path, "config.json")
        model = cls(
            base_model,
            base_model_path,
            configpath
        )
        load_model_path=os.path.join(ea_model_path, "pytorch_model.bin")
        if not os.path.exists(load_model_path):
            load_model_path=hf_hub_download(ea_model_path, "pytorch_model.bin")
        ea_layer_state_dict = torch.load(load_model_path,
                                         map_location=base_model.device)
        model.ea_layer.load_state_dict(ea_layer_state_dict, strict=True)

        return model

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            labels=None,
            past_key_values=None,
            output_orig=False,
            position_ids=None,
            init=True,
            logits_processor=None
    ):

        with torch.inference_mode():
            # Pass input through the base model
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
            if output_orig:
                orig = self.base_model.lm_head(outputs[0])
            hidden_states = outputs[0].clone()
        if init:
            if logits_processor is not None:
                logits = orig[:, -1]
                logits = logits_processor(None, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                token = torch.multinomial(probabilities, 1)
            else:
                token = torch.argmax(orig[:, -1])
                token = token[None, None]
            input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1)
            # Clone the output hidden states
            # print("input_ids shape for ea_layer:", input_ids.shape)  # [1, L+1]
            ea_logits = self.ea_layer.topK_genrate(hidden_states, input_ids, self.base_model.lm_head, logits_processor)
            if output_orig:
                return ea_logits, outputs, orig, hidden_states, token
            return ea_logits, hidden_states, token
        else:
            if output_orig:
                return outputs, orig, hidden_states

    @torch.no_grad()
    def eagenerate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            tree_choices=mc_sim_7b_63,

    ):
        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        #assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        if hasattr(self, "tree_choices") and self.tree_choices == tree_choices:
            tree_buffers = self.tree_buffers
        else:
            tree_buffers = generate_tree_buffers(
                tree_choices, device=self.base_model.model.layers[-1].self_attn.q_proj.weight.device
            )
            tree_buffers["retrieve_indices_head"] = tree_buffers["retrieve_indices"].to(
                self.base_model.lm_head.weight.device)
        self.tree_buffers = tree_buffers
        self.tree_choices = tree_choices

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        tree_logits, logits, hidden_state, sample_token = initialize_tree(
            input_ids, self, tree_buffers["tree_attn_mask"], past_key_values, logits_processor
        )
        new_token = 0

        for idx in range(max_length):
            candidates, cart_candidates_prob, tree_candidates = generate_candidates(
                tree_logits,
                tree_buffers["tree_indices"],
                tree_buffers["retrieve_indices"],
                sample_token,
                logits_processor
            )
            logits, hidden_state_new, outputs = tree_decoding(
                self,
                tree_candidates,
                past_key_values,
                tree_buffers["tree_position_ids"],
                input_ids,
                tree_buffers["retrieve_indices_head"],
            )
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor, cart_candidates_prob, tree_logits[2], tree_buffers["p_indices"],
                tree_candidates, tree_buffers["b_indices"]
            )
            input_ids, tree_logits, new_token, hidden_state, sample_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                tree_buffers["retrieve_indices"],
                logits_processor,
                logits,
                tree_logits,
                new_token,
                past_key_values_data,
                current_length_data,
                self,
                hidden_state,
                hidden_state_new,
                sample_p
            )

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                return input_ids
            if new_token > max_new_tokens:
                return input_ids
            if input_ids.shape[1] > max_length:
                return input_ids

    @torch.no_grad()
    def ea_generate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_steps=512,
            tree_choices=mc_sim_7b_63,

    ):
        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        if hasattr(self, "tree_choices") and self.tree_choices == tree_choices:
            tree_buffers = self.tree_buffers
        else:
            tree_buffers = generate_tree_buffers(
                tree_choices, device=self.base_model.model.layers[-1].self_attn.q_proj.weight.device
            )
            tree_buffers["retrieve_indices_head"] = tree_buffers["retrieve_indices"].to(
                self.base_model.lm_head.weight.device)
        self.tree_buffers = tree_buffers
        self.tree_choices = tree_choices

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        tree_logits, logits, hidden_state, sample_token = initialize_tree(
            input_ids, self, tree_buffers["tree_attn_mask"], past_key_values, logits_processor
        )
        new_token = 0

        for idx in range(max_steps):
            candidates, cart_candidates_prob, tree_candidates = generate_candidates(
                tree_logits,
                tree_buffers["tree_indices"],
                tree_buffers["retrieve_indices"],
                sample_token,
                logits_processor
            )
            logits, hidden_state_new, outputs = tree_decoding(
                self,
                tree_candidates,
                past_key_values,
                tree_buffers["tree_position_ids"],
                input_ids,
                tree_buffers["retrieve_indices_head"],
            )

            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor, cart_candidates_prob, tree_logits[2], tree_buffers["p_indices"],
                tree_candidates, tree_buffers["b_indices"]
            )
            #print("post", time.time() - s)
            input_ids, tree_logits, new_token, hidden_state, sample_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                tree_buffers["retrieve_indices"],
                logits_processor,
                logits,
                tree_logits,
                new_token,
                past_key_values_data,
                current_length_data,
                self,
                hidden_state,
                hidden_state_new,
                sample_p
            )

            yield input_ids

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > 1024:
                break
            if input_ids.shape[1] > 1960:
                break

    @torch.no_grad()
    def naive_generate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_steps=512,
            tree_choices=mc_sim_7b_63,

    ):
        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        if hasattr(self, "tree_choices") and self.tree_choices == tree_choices:
            tree_buffers = self.tree_buffers
        else:
            tree_buffers = generate_tree_buffers(
                tree_choices, device=self.base_model.model.layers[-1].self_attn.q_proj.weight.device
            )
            tree_buffers["retrieve_indices_head"] = tree_buffers["retrieve_indices"].to(
                self.base_model.lm_head.weight.device)
        self.tree_buffers = tree_buffers
        self.tree_choices = tree_choices

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        outputs = self.base_model(input_ids, past_key_values=past_key_values, use_cache=True)
        new_token = 0

        for idx in range(max_steps):
            input_id = outputs.logits[:, -1:].argmax(dim=-1)
            outputs = self.base_model(input_id, use_cache=True, past_key_values=past_key_values)
            input_ids = torch.cat([input_ids, input_id], dim=-1)

            yield input_ids

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > 1024:
                break
            if input_ids.shape[1] > 1960:
                break


class EaModel3(nn.Module):
    def __init__(self, base_model, base_model_name_or_path, ea_layers):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = base_model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.lm_head.weight.shape[0]
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path)
        self.ea_layers = nn.ModuleList(ea_layers)

        self._active_ea_idx = 0
        self._last_draft_idx = 0
        self.ea_layer = self.ea_layers[0]
        self.tree_buffers = None
        self.tree_choices = None

    @classmethod
    def from_pretrained(cls, base_model_path, ea_model_paths, **kwargs):
        assert len(ea_model_paths) == 3, "ea_model_paths must contain exactly three directories."
        Type = AutoConfig.from_pretrained(base_model_path).architectures[0]
        if Type == "LlamaForCausalLM":
            base_model = KVLlamaForCausalLM.from_pretrained(base_model_path, **kwargs)
        else:
            base_model = KVMixtralForCausalLM.from_pretrained(base_model_path, **kwargs)

        device = base_model.model.layers[-1].self_attn.q_proj.weight.device
        ea_layers = []

        for ea_path in ea_model_paths:
            config_path = os.path.join(ea_path, "config.json")
            if not os.path.exists(config_path):
                config_path = hf_hub_download(ea_path, "config.json")
            econfig = EConfig.from_pretrained(config_path)

            with open(config_path, "r") as fin:
                cfg_dict = json.load(fin)
            bias = cfg_dict.get("bias", True)

            ea = Model(econfig, bias=bias)
            if device != base_model.lm_head.weight.device:
                ea.diff_device = True
                ea.headweight = base_model.lm_head.weight.clone().to(device)
            else:
                ea.diff_device = False

            ea.to(base_model.dtype).to(device)
            ea.init_tree()

            weight_path = os.path.join(ea_path, "pytorch_model.bin")
            if not os.path.exists(weight_path):
                weight_path = hf_hub_download(ea_path, "pytorch_model.bin")
            state = torch.load(weight_path, map_location=device)
            ea.load_state_dict(state, strict=True)
            ea_layers.append(ea)

        return cls(base_model, base_model_path, ea_layers)

    def get_tokenizer(self):
        return self.tokenizer

    def get_ea_layer(self, idx):
        return self.ea_layers[idx]

    def reset_all_kv(self):
        for ea in self.ea_layers:
            if hasattr(ea, "reset_kv"):
                ea.reset_kv()

    def topK_genrate_for(self, idx, hidden_states, input_ids_with_token, head=None, logits_processor=None):
        return self.ea_layers[idx].topK_genrate(
            hidden_states, input_ids_with_token, head=head, logits_processor=logits_processor
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        past_key_values=None,
        output_orig=False,
        position_ids=None,
        init=True,
        logits_processor=None,
    ):
        with torch.inference_mode():
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
            if output_orig:
                orig = self.base_model.lm_head(outputs[0])
            hidden_states = outputs[0].clone()

        if init:
            if output_orig:
                logits_last = orig[:, -1]
            else:
                logits_last = self.base_model.lm_head(outputs[0])[:, -1]

            if logits_processor is not None:
                logits_last = logits_processor(None, logits_last)
                probs = torch.nn.functional.softmax(logits_last, dim=1)
                token = torch.multinomial(probs, 1)
            else:
                token = torch.argmax(logits_last, dim=1, keepdim=True)

            input_ids_cat = torch.cat((input_ids, token.to(input_ids.device)), dim=1)

            ea_logits = self.ea_layers[self._active_ea_idx].topK_genrate(
                hidden_states, input_ids_cat, head=self.base_model.lm_head, logits_processor=logits_processor
            )

            if output_orig:
                return ea_logits, outputs, orig, hidden_states, token
            return ea_logits, hidden_states, token

        if output_orig:
            return outputs, orig, hidden_states
        return outputs

    @torch.inference_mode()
    def ea_forward_draft3(
        self,
        inputs,
        tokenizer,
        max_new_tokens,
        tree_choices=None,
        logits_processor=None,
        max_steps=512,
    ):
        results = []
        accept_totals = []
        errors = []

        for layer_idx in range(min(3, len(self.ea_layers))):
            try:
                result = self._run_single_draft(
                    layer_idx=layer_idx,
                    inputs=inputs,
                    tokenizer=tokenizer,
                    max_new_tokens=max_new_tokens,
                    tree_choices=tree_choices,
                    logits_processor=logits_processor,
                    max_steps=max_steps,
                )
                accept_sum = sum(max(val, 0) for val in result[3])
                results.append((layer_idx, result))
                accept_totals.append(accept_sum)
            except Exception as exc:
                errors.append(exc)

        if not results:
            raise errors[0] if errors else RuntimeError("No valid draft result obtained.")

        best_idx = max(range(len(results)), key=lambda i: accept_totals[i])
        self._last_draft_idx, best_result = results[best_idx]
        return best_result

    def _run_single_draft(
        self,
        layer_idx,
        inputs,
        tokenizer,
        max_new_tokens,
        tree_choices,
        logits_processor,
        max_steps,
    ):
        input_ids = inputs.input_ids
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        input_ids = input_ids.clone()

        tree_choices = tree_choices or mc_sim_7b_63
        ea_layer = self.ea_layers[layer_idx]
        ea_layer.reset_kv()

        device = self.base_model.model.layers[-1].self_attn.q_proj.weight.device
        tree_buffers = generate_tree_buffers(tree_choices, device=device)
        tree_buffers["retrieve_indices_head"] = tree_buffers["retrieve_indices"].to(
            self.base_model.lm_head.weight.device
        )

        past_key_values, past_key_values_data, current_length_data = initialize_past_key_values(self.base_model)

        prev_active_idx = self._active_ea_idx
        prev_ea_layer = self.ea_layer
        self._active_ea_idx = layer_idx
        self.ea_layer = ea_layer

        try:
            input_len = input_ids.shape[1]
            cur_length = input_len

            reset_tree_mode(self)
            tree_logits, logits, hidden_state, sample_token = initialize_tree(
                input_ids, self, tree_buffers["tree_attn_mask"], past_key_values, logits_processor
            )
            new_token = 0
            accept_length_list = []

            for idx in range(max_steps):
                candidates, cart_candidates_prob, tree_candidates = generate_candidates(
                    tree_logits,
                    tree_buffers["tree_indices"],
                    tree_buffers["retrieve_indices"],
                    sample_token,
                    logits_processor,
                )

                logits, hidden_state_new, _ = tree_decoding(
                    self,
                    tree_candidates,
                    past_key_values,
                    tree_buffers["tree_position_ids"],
                    input_ids,
                    tree_buffers["retrieve_indices_head"],
                )

                best_candidate, accept_length, sample_p = evaluate_posterior(
                    logits,
                    candidates,
                    logits_processor,
                    cart_candidates_prob,
                    tree_logits[2],
                    tree_buffers["p_indices"],
                    tree_candidates,
                    tree_buffers["b_indices"],
                )

                input_ids, tree_logits, new_token, hidden_state, sample_token = update_inference_inputs(
                    input_ids,
                    candidates,
                    best_candidate,
                    accept_length,
                    tree_buffers["retrieve_indices"],
                    logits_processor,
                    logits,
                    tree_logits,
                    new_token,
                    past_key_values_data,
                    current_length_data,
                    self,
                    hidden_state,
                    hidden_state_new,
                    sample_p,
                )

                accept_length_tree = input_ids.shape[1] - cur_length
                cur_length += accept_length_tree
                accept_length_list.append(accept_length_tree)

                if tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                    eos_idx = None
                    for i, tok in enumerate(input_ids[0, input_len:]):
                        if tok == tokenizer.eos_token_id:
                            eos_idx = i
                            break
                    if eos_idx is not None:
                        invalid = len(input_ids[0, input_len:]) - eos_idx - 1
                        if invalid > 0:
                            accept_length_list[-1] -= invalid
                            new_token -= invalid
                    break

                if new_token > max_new_tokens:
                    break
                if input_ids.shape[1] > 1960:
                    break

            steps = idx + 1 if "idx" in locals() else 0
        finally:
            reset_tree_mode(self)
            self._active_ea_idx = prev_active_idx
            self.ea_layer = prev_ea_layer

        return input_ids, new_token, steps, accept_length_list
