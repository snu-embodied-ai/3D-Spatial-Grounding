import torch
import torch.nn as nn
import torch.nn.functional as F

from omegaconf import OmegaConf, DictConfig
import pandas as pd

import einops
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer

from modules.ca_uni3d import CA_Uni3D
from modules.Uni3D.models.uni3d import create_uni3d
from modules.gfp import GeometricFeaturePropagation
from modules.gem import GeometricEnhancer
from modules.utils.util import maybe_autocast

from data.dataset_consts import TASKS, TASK_TYPE, SCENE_IDS

class SegPoint(nn.Module):
    def __init__(self, cfg: DictConfig,
                 instruct_cfg: DictConfig):
        super().__init__()
        self.prompts = instruct_cfg

        # 1. Load Llama2
        self.LLM_name = cfg.LLM.model_name
        self.LLM_new_name = cfg.LLM.new_model_name
        self.LLM_device_map = cfg.LLM.device_map

        base_model = AutoModelForCausalLM.from_pretrained(
            self.LLM_name,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.float16,
            device_map=self.LLM_device_map,
            truncation_side=cfg.LLM.truncation_side,
        )

        self.LLM_tokenizer = AutoTokenizer.from_pretrained(self.LLM_name, trust_remote_code=True)
        self.LLM_tokenizer.pad_token = self.LLM_tokenizer.eos_token
        self.LLM_tokenizer.add_special_tokens(cfg.LLM.special_tokens)
        self.LLM_tokenizer.padding_side = cfg.LLM.padding_size

        if cfg.LLM.LoRA.train_with:
            lora_cfg = LoraConfig(
                r=cfg.LLM.LoRA.rank,
                lora_alpha=cfg.LLM.LoRA.alpha,
                target_modules=cfg.LLM.LoRA.target_modules,
                lora_dropout=cfg.LLM.LoRA.dropout,
                bias='none',
                modules_to_save=[],
            )
            self.LLM = get_peft_model(base_model, peft_config=lora_cfg)

        # 2. Load PCD encoder
        pretrained_encoder = create_uni3d(cfg.Point_encoder.Uni3D, cfg.distributed)
        self.pcd_encoder = CA_Uni3D(pretrained_uni3d=pretrained_encoder,
                                    cfg=cfg.Point_encoder.CA_Uni3D)

        # 2-1. PCD projector
        self.pcd_proj = nn.Linear(cfg.Point_encoder.PointcloudEncoder.pc_encoder_dim, self.LLM.config.hidden_size)

        # 3. Load Geometric Enhancer Module
        self.GEM = GeometricEnhancer(cfg.GEM)

        # 4. Load Geometric-guided Feature Propagation Module
        self.GFP = GeometricFeaturePropagation(cfg.GFP)

        # 5. <SEG> token projector
        self.seg_projector = nn.Sequential(
            nn.Linear(self.LLM.config.hidden_size, self.LLM.config.hidden_size),
            nn.GELU(),
            nn.Linear(self.LLM.config.hidden_size, cfg.GFP.gem_dim)
        )

    def generate_input_embs(self,
                            data_dict: dict,
                            category_mapping: pd.DataFrame,
                            include_targets: bool = True):
        """
        Concatenate prompts and visual embeddings and special tokens
        System Prompts + USER prompts (with point embedding) + ASSISTANT prompts (expected output)
        """
        batch_size = len(data_dict["scene_id"])
        device = self.device
        # seq_len = 0
        
        # 1. Tokenize system prompts
        if self.prompts.system_prompts.add_sys_delimiters:
            sys_prompt = f"<<SYS>>{self.prompts.system_prompts.text}<<\SYS>>"
        else:
            sys_prompt = self.prompts.system_prompts.text

        system_prompt_tokens = self.LLM_tokenizer(
            sys_prompt,
            return_tensors='pt',
            # padding='longest'
        ).to(device)
        # seq_len += len(system_prompt_tokens)
        system_prompt_tokens = einops.repeat(system_prompt_tokens, 't -> b t', b=batch_size)

        main_prompt = self.prompts[TASKS[data_dict["task"]]][TASK_TYPE[data_dict["task_type"]]]

        # 2. Tokenize USER token and USER prompts (+ ASSISTANT token)
        # "USER: "
        user_token = self.LLM_tokenizer(
            "USER: ",
            return_tensor='pt',
            # padding='longest'
        )
        # seq_len += len(user_token)
        user_token = einops.repeat(user_token, 't -> b t', b=batch_size)

        # " Can you .... ASSISTANT: "
        user_prompt = main_prompt.USER + "ASSISTANT: "
        if data_dict["task"] == 0 and data_dict["task_type"] == 0:
            # SemanticSegmentation & specific
            user_prompt = user_prompt.format(data_dict["category"])
            
        user_prompt_tokens = self.LLM_tokenizer(
            user_prompt,
            return_tensors='pt',
            # padding='longest'
        ).to(device)
        # seq_len += len(user_prompt_tokens)
        user_prompt_tokens = einops.repeat(user_prompt_tokens, 't -> b t', b=batch_size)

        if include_targets:
            # 3. Tokenize ASSISTANT token and ASSISTANT prompts
            batch_assistant_prompts = []
            assistant_prompt_template = main_prompt.ASSISTANT
            if data_dict["task"] == 0 and data_dict["task_type"] == 1:
                # SemanticSegmentation & all_categories
                for i in range(batch_size):
                    num_cats = data_dict["num_category"][i]
                    assistant_prompt = assistant_prompt_template * num_cats
                    assistant_prompt.rstrip(", ")       # Remove last commas
                    assistant_prompt += self.LLM_tokenizer.eos_token

                    categories = data_dict["category"][i, :num_cats]
                    categories = category_mapping["raw_category"][categories].values

                    for cat in categories:
                        assistant_prompt = assistant_prompt.replace("{category}", cat, 1)

                    batch_assistant_prompts.append(assistant_prompt)

            self.LLM_tokenizer.padding_side = 'right'
            # self.LLM_tokenizer.truncation_side = 'right'
            assistant_prompt_tokens = self.LLM_tokenizer(
                batch_assistant_prompts,
                return_tensor='pt',
                padding='longest'
                # truncation=True
            )           # (B, T)


        # 4. Remove BOS (<s> in Llama2) to concatenate subsequences into a whole sequence
        user_token.input_ids = user_token.input_ids[:, 1:]
        user_token.attention_mask = user_token.attention_mask[:, 1:]

        user_prompt_tokens.input_ids = user_prompt_tokens.input_ids[:, 1:]
        user_prompt_tokens.attention_mask = user_prompt_tokens.attention_mask[:, 1:]

        # if include_targets:
        #     assistant_prompt_tokens.input_ids = assistant_prompt_tokens.input_ids[:, 1:]
        #     assistant_prompt_tokens.attention_mask = assistant_prompt_tokens.attention_mask[:, 1:]

        # 5. Get embeddings for each subsequences
        system_prompt_embed = self.LLM.get_input_embeddings()(system_prompt_tokens.input_ids)                                      # (B, T1, hidden_dim)
        user_embed = self.LLM.get_input_embeddings()(user_token.input_ids)  # (B, 1, hidden_dim)
        user_prompt_embed = self.LLM.get_input_embeddings()(user_prompt_tokens.input_ids)                                      # (B, T2, hidden_dim)

        if include_targets:
            assistant_prompt_embed = self.LLM.get_input_embeddings()(assistant_prompt_tokens.input_ids)         # (B, T3, hidden_dim)

        # 6. Get point embeddings and concatenate all to generate prompt embeddings
        point_embed = data_dict["point_embed"].unsqueeze(dim=1)     # (B, num_groups, hidden_dim)
        point_mask = data_dict["point_mask"]                        # (B, num_groups)         

        # 6-1. Concat fixed length sequences first
        # Assuming task/task types are all the same within a single mini-batch
        inputs_embeds = torch.cat([system_prompt_embed, user_embed, point_embed, user_prompt_embed], dim=1)
        attention_mask = torch.cat([
            system_prompt_tokens.attention_mask,
            user_token.attention_mask,
            point_mask,
            user_prompt_tokens.attention_mask
        ], dim=1)

        if include_targets:
            # 6-2. Concat assistant prompt embeddings
            inputs_embeds = torch.cat([inputs_embeds, assistant_prompt_embed], dim=1)
            attention_mask = torch.cat([attention_mask, assistant_prompt_tokens.attention_mask], dim=1)

            # Construct targets
            targets = torch.zeros_like(attention_mask).long().fill_(-100)

            # Only apply loss to answer tokens (after ASSISTANT: )
            targets_idx = assistant_prompt_tokens.attention_mask.bool()
            targets[:, -targets_idx.size(1):][targets_idx] = assistant_prompt_tokens.input_ids[targets_idx]

            # Do not predict BOS token
            targets[:, -targets_idx.size(1)] = -100
        else:
            targets = None

        return inputs_embeds, attention_mask, targets

    def forward(self,
                data_dict: dict):
        """
        
        """


        device = self.device
        batch_size = len(data_dict["scene_id"])
        # TODO
        # This should be added right before the forward operation
        cat_mapping = data_dict["category_mapping"]

        # 1. Get GEM features
        gem_feats = self.GEM(data_dict["features"])

        # 2. Get Pointcloud embeddings
        point_embed = self.pcd_encoder(inputs=data_dict["features"],
                                       gem_features=gem_feats)
        data_dict["point_embed"] = self.pcd_proj(point_embed)

        # 3. Generate input embeddings for LLM
        inputs_embeds, attention_mask, targets = self.generate_input_embs(data_dict, cat_mapping, include_targets=True)         # (B, T, D), (B, T), (B, T)

        # 4. Forward
        with maybe_autocast(self):
            outputs = self.LLM(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_state=True,
            )

        # 4. Compute Loss inside forward function
        logits = outputs.logits.float()

        # TODO: get hidden states of POINT sequences and <SEG>


