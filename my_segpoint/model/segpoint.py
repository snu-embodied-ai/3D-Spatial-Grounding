import math
from typing import List, Dict
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from omegaconf import OmegaConf, DictConfig
import pandas as pd

import einops
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from accelerate.logging import get_logger

from model.modules.ca_uni3d import CA_Uni3D
from model.modules.Uni3D.models.uni3d import create_uni3d
from model.modules.gfp import GeometricFeaturePropagation
from model.modules.gem import GeometricEnhancer
from model.modules.llama2 import LlamaWithBeamHidden
from model.modules.utils.util import disabled_train, maybe_autocast
from trainer.losses import masked_cross_entropy, DiceLoss

from data.constants.dataset_consts import TASKS, TASK_TYPE

logger = get_logger(__name__)

class SegPoint(nn.Module):
    def __init__(self, cfg: DictConfig,
                 instruct_cfg: DictConfig):
        super().__init__()
        self.prompts = instruct_cfg

        # 1. Load Llama2
        self.LLM_name = cfg.LLM.model_name
        self.LLM_new_name = cfg.LLM.new_model_name

        base_model = LlamaWithBeamHidden.from_pretrained(
            self.LLM_name,
            low_cpu_mem_usage=True,
            return_dict=True,
            # torch_dtype=torch.bfloat16,
            # device_map=self.LLM_device_map,
            # truncation_side=cfg.LLM.truncation_side,
        )

        self.LLM_tokenizer = AutoTokenizer.from_pretrained(self.LLM_name, trust_remote_code=True)
        self.LLM_tokenizer.pad_token = self.LLM_tokenizer.eos_token
        
        self.LLM_tokenizer.add_special_tokens({
            "additional_special_tokens" : [cfg.LLM.seg_token, cfg.LLM.point_token]
        })
        self.LLM_tokenizer.padding_side = cfg.LLM.padding_side
        base_model.resize_token_embeddings(len(self.LLM_tokenizer))

        self.seg_token_id = self.LLM_tokenizer.convert_tokens_to_ids(cfg.LLM.seg_token)
        self.point_token_id = self.LLM_tokenizer.convert_tokens_to_ids(cfg.LLM.point_token)

        if cfg.LLM.LoRA.train_with:
            # 1-1. FREEZE the LLM parameters when training LoRA layers
            for param in base_model.parameters():
                param.requires_grad = False

            # 1-2. make text_hidden_fcs, mask_decoder, lm_head, embed_tokens trainable
            for n, p in base_model.named_parameters():
                if any(
                    [
                        x in n
                        for x in ["lm_head", "embed_tokens"]
                    ]
                ):
                    p.requires_grad = True

            base_model.eval()
            base_model.train = disabled_train

            # 1-3. Add LoRA layers
            lora_cfg = LoraConfig(
                r=cfg.LLM.LoRA.rank,
                lora_alpha=cfg.LLM.LoRA.alpha,
                target_modules=cfg.LLM.LoRA.target_modules,
                lora_dropout=cfg.LLM.LoRA.dropout,
                bias='none',
                modules_to_save=[],
            )
            self.LLM = get_peft_model(base_model, peft_config=lora_cfg)

        # 2. Load PCD encoder (parameters are frozen inside the create function)
        pretrained_encoder = create_uni3d(cfg.Point_encoder.Uni3D, cfg.distributed)
        self.pcd_encoder = CA_Uni3D(pretrained_uni3d=pretrained_encoder,
                                    cfg=cfg.Point_encoder.CA_Uni3D)

        # 2-1. PCD projector
        if cfg.Point_encoder.CA_Uni3D.align_clip_dim:
            pcd_embed_dim = cfg.Point_encoder.Uni3D.PointcloudEncoder.embed_dim
        else:
            pcd_embed_dim = cfg.Point_encoder.Uni3D.PointcloudEncoder.pc_feat_dim
        self.pcd_proj = nn.Linear(pcd_embed_dim, self.LLM.config.hidden_size)

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

        # 6. Loss weights
        self.loss_weights = cfg.loss_weights

    def count_params(self, parameters):
        tot = sum([math.prod(p.shape) for p in parameters])
        return tot

    def show_params_size(self, tot):
        if tot >= 1e9:
            return '{:.1f}B'.format(tot / 1e9)
        elif tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}k'.format(tot / 1e3)

    def get_learnable_named_params(self):
        learnable_named_params = {}
        frozen_named_params = {}
        for n, p in self.named_parameters():
            if p.requires_grad:
                learnable_named_params.update({n: p})
            else:
                frozen_named_params.update({n: p})
        learnable_params_size = self.count_params(learnable_named_params.values())
        frozen_params_size = self.count_params(frozen_named_params.values())

        logger.info(
            f"Build LEO with {self.show_params_size(learnable_params_size+frozen_params_size)} parameters, " +
            f"{self.show_params_size(learnable_params_size)} learnable and " +
            f"{self.show_params_size(frozen_params_size)} frozen"
        )
        logger.info(f"🧊 Frozen parameters: {list(frozen_named_params.keys())}")
        logger.info(f"🔥 Tuned parameters: {list(learnable_named_params.keys())}")

        return learnable_named_params
    
    def to_device(self,
                  data_dict: dict):
        device = next(self.parameters()).device
        self.device = device

        data_dict['xyz'] = data_dict['xyz'].to(device)
        data_dict['features'] = data_dict['features'].to(device)
        data_dict['mask'] = data_dict['mask'].to(device)
        data_dict['category'] = data_dict['category'].to(device)
        data_dict['padding_mask'] = data_dict['padding_mask'].to(device)
        data_dict['valid_gt_mask'] = data_dict['valid_gt_mask'].to(device)

        return data_dict

    def generate_input_embs(self,
                            data_dict: dict,
                            category_mapping: dict,
                            train: bool = True):
        """
        Concatenate prompts and visual embeddings and special tokens
        System Prompts + USER prompts (with point embedding) + ASSISTANT prompts (expected output)
        """
        batch_size = len(data_dict["num_category"])
        device = next(self.parameters()).device

        main_prompt = self.prompts[TASKS[data_dict["task"]]][TASK_TYPE[data_dict["task_type"]]]

        # 1. Tokenize USER token and USER prompts (+ ASSISTANT token)
        # 1-1. "USER: " (USER token)
        user_token = self.LLM_tokenizer(
            ["USER: "] * batch_size,
            return_tensors='pt',
            # padding='longest'
        ).to(device)

        # 1-2. " Can you .... ASSISTANT: " (USER prompt)
        user_prompt = main_prompt.USER
        if data_dict["task"] == 0 and data_dict["task_type"] == 0:
            # SemanticSegmentation & specific
            batch_user_prompt = []
            for i in range(batch_size):
                category = category_mapping[data_dict["category"][i].item()]
                batch_user_prompt.append(user_prompt.format(category=category))
        elif data_dict["task"] == 0 and data_dict["task_type"] == 1:
            # SemanticSegmentation & all_categories
            batch_user_prompt = [user_prompt] * batch_size
            
        user_prompt_tokens = self.LLM_tokenizer(
            batch_user_prompt,
            return_tensors='pt',
            padding='longest'
        ).to(device)

        # 2. Tokenize all category names
        cat_list_prompt = f"Only objects from the following list appears in the point cloud. [{', '.join(category_mapping.values())}]"
        cat_list_tokens = self.LLM_tokenizer(
            [cat_list_prompt] * batch_size,
            return_tensors='pt'
        ).to(device)

        # 3. Tokenize ASSISTANT token and ASSISTANT prompts
        assistant_token = self.LLM_tokenizer(
            ["ASSISTANT: "] * batch_size,
            return_tensors='pt'
        ).to(device)

        batch_assistant_prompts = []
        assistant_prompt_template = main_prompt.ASSISTANT
        if data_dict["task"] == 0 and data_dict["task_type"] == 1:
            # SemanticSegmentation & all_categories
            for i in range(batch_size):
                num_cats = data_dict["num_category"][i]
                assistant_prompt = assistant_prompt_template * num_cats
                assistant_prompt.rstrip(", ")       # Remove last commas
                # assistant_prompt += "." + self.LLM_tokenizer.eos_token

                categories = data_dict["category"][i, :num_cats]
                categories = [category_mapping[cat.item()] for cat in categories]

                for cat in categories:
                    assistant_prompt = assistant_prompt.replace("{category}", cat, 1)

                batch_assistant_prompts.append(assistant_prompt)
        elif data_dict["task"] == 0 and data_dict["task_type"] == 0:
            # SemanticSegmentation & specific
            for i in range(batch_size):
                # assistant_prompt = assistant_prompt_template + self.LLM_tokenizer.eos_token
                category = data_dict["category"][i].item()
                category = category_mapping[category]
                
                assistant_prompt = assistant_prompt_template.format(category=category)
                batch_assistant_prompts.append(assistant_prompt)

        self.LLM_tokenizer.padding_side = 'right'
        # self.LLM_tokenizer.truncation_side = 'right'
        assistant_prompt_tokens = self.LLM_tokenizer(
            batch_assistant_prompts,
            return_tensors='pt',
            padding='longest',
            # truncation=True,
            return_special_tokens_mask=True
        ).to(device)           # (B, T)

        # (Optional) Few shot examples
        if main_prompt.few_shot:
            if data_dict["task"] == 0 and data_dict["task_type"] == 0:
                # SemanticSegmentation & specific
                rand_cat = random.choice(list(category_mapping.values()))
                fewshot_user_prompt = user_prompt.format(category=rand_cat)
                fewshot_assistant_prompt = assistant_prompt_template.format(category=rand_cat)
                few_shot_prompt = f"USER: {fewshot_user_prompt} {cat_list_prompt} ASSISTANT: {fewshot_assistant_prompt}"
            elif data_dict["task"] == 0 and data_dict["task_type"] == 1:
                # SemanticSegmentation & all_categories
                rand_cats = random.sample(list(category_mapping.values()), k=random.randint(1,6))

                fewshot_assistant_prompt = assistant_prompt_template * len(rand_cats)
                fewshot_assistant_prompt.rstrip(", ")       # Remove last commas

                for cat in rand_cats:
                    fewshot_assistant_prompt = assistant_prompt.replace("{category}", cat, 1)

                few_shot_prompt = f"USER: {user_prompt} {cat_list_prompt} ASSISTANT: {fewshot_assistant_prompt}"
        else:
            few_shot_prompt = ""

        # 4. Tokenize system prompts
        if self.prompts.system_prompts.add_sys_delimiters:
            sys_prompt = (
                f"<<SYS>>{self.prompts.system_prompts.text}\n"
                "Examples:\n"
                f"{few_shot_prompt}<</SYS>>"
                )
        else:
            sys_prompt = (
                f"{self.prompts.system_prompts.text}\n"
                "Examples:\n"
                f"{few_shot_prompt}"
            )

        system_prompt_tokens = self.LLM_tokenizer(
            [sys_prompt] * batch_size,
            return_tensors='pt',
            padding='longest'
        ).to(device)

        # 5. Remove BOS (<s> in Llama2) to concatenate subsequences into a whole sequence
        user_token.input_ids = user_token.input_ids[:, 1:]
        user_token.attention_mask = user_token.attention_mask[:, 1:]

        user_prompt_tokens.input_ids = user_prompt_tokens.input_ids[:, 1:]
        user_prompt_tokens.attention_mask = user_prompt_tokens.attention_mask[:, 1:]

        cat_list_tokens.input_ids = cat_list_tokens.input_ids[:, 1:]
        cat_list_tokens.attention_mask = cat_list_tokens.attention_mask[:, 1:]

        assistant_token.input_ids = assistant_token.input_ids[:, 1:]
        assistant_token.attention_mask = assistant_token.attention_mask[:, 1:]

        # 6. Get embeddings for each subsequences
        system_prompt_embed = self.LLM.get_input_embeddings()(system_prompt_tokens.input_ids)                                      # (B, T1, hidden_dim)
        user_embed = self.LLM.get_input_embeddings()(user_token.input_ids)  # (B, 1, hidden_dim)
        user_prompt_embed = self.LLM.get_input_embeddings()(user_prompt_tokens.input_ids)                                      # (B, T2, hidden_dim)
        cat_list_embed = self.LLM.get_input_embeddings()(cat_list_tokens.input_ids)                             # (B, T', hidden_dim)

        assistant_embed = self.LLM.get_input_embeddings()(assistant_token.input_ids)        # (B, 1, hidden_dim)
        assistant_prompt_embed = self.LLM.get_input_embeddings()(assistant_prompt_tokens.input_ids)         # (B, T3, hidden_dim)

        # 7. Get point embeddings/tokens and concatenate all to generate prompt embeddings
        point_embed = data_dict["point_embed"]                # (B, num_groups, hidden_dim)
        point_token_ids = torch.ones((point_embed.size()[:2]), device=device, dtype=torch.int64) * self.point_token_id
        point_mask = torch.ones(point_embed.size()[:2], device=device)                        # (B, num_groups)         

        # 7-1. Concat fixed length sequences first
        # Assuming task/task types are all the same within a single mini-batch
        inputs_embeds = torch.cat([
            system_prompt_embed, 
            user_embed, 
            point_embed, 
            user_prompt_embed, 
            cat_list_embed,
            assistant_embed], dim=1)
        input_ids = torch.cat([
            system_prompt_tokens.input_ids, 
            user_token.input_ids, 
            point_token_ids, 
            user_prompt_tokens.input_ids, 
            cat_list_tokens.input_ids,
            assistant_token.input_ids], dim=1)
        attention_mask = torch.cat([
            system_prompt_tokens.attention_mask,
            user_token.attention_mask,
            point_mask,
            user_prompt_tokens.attention_mask,
            cat_list_tokens.attention_mask,
            assistant_token.attention_mask], dim=1)

        if train:
            # 7-2. Concat assistant prompt embeddings and tokens
            inputs_embeds = torch.cat([inputs_embeds, assistant_prompt_embed], dim=1)
            input_ids = torch.cat([input_ids, assistant_prompt_tokens.input_ids], dim=1)

            attention_mask = torch.cat([attention_mask, assistant_prompt_tokens.attention_mask], dim=1)

            # 7-3. Construct targets
            targets = torch.zeros_like(attention_mask).long().fill_(-100)

            # Only apply loss to answer tokens
            targets_idx = assistant_prompt_tokens.attention_mask.bool()
            targets[:, -targets_idx.size(1):] = assistant_prompt_tokens.input_ids

            # Do not predict BOS token
            targets[:, -targets_idx.size(1)] = -100

            # Create mask for point embeddings and seg tokens (shift to match the indices for output sequences)
            point_embed_mask = input_ids[:,1:] == self.point_token_id
            
            seg_token_mask = torch.zeros_like(point_embed_mask).bool()
            seg_token_mask[:, -targets_idx.size(1):] = input_ids[:,-targets_idx.size(1):] == self.seg_token_id
            
            special_token_masks = {
                "POINT": point_embed_mask,
                "SEG": seg_token_mask
            }

        else:
            # Remove BOS token since BOS token would be added to input_ids during generate()
            targets = assistant_prompt_tokens.input_ids[:, 1:]

            # Create mask for point embeddings and seg tokens (shift to match the indices for output sequences)
            point_embed_mask = input_ids[:,1:] == self.point_token_id
            
            special_token_masks = {
                "POINT": point_embed_mask
            }

        return inputs_embeds, input_ids, attention_mask, targets, special_token_masks
    
    def seg_batched_projection(self,
                               all_seg_hidden: List[torch.Tensor]):
        """
        Padding the sequences for batched projection. After projection, unpad the sequences back to their original shape.

        Parameters
        ---
        all_seg_hidden: list[torch.Tensor]
            List of hidden state of <SEG> token for each batch.

        Returns
        ---
        projected_seg_hidden: torch.Tensor
            Padded projected hidden state of <SEG> token (batched)
        mask: torch.Tensor
            Mask for valid indices (non-padded) in hidden state
        """

        original_lengths = [seg_h.shape[0] for seg_h in all_seg_hidden]
        max_len = max(original_lengths)

        padded = [F.pad(seg_h, (0, 0, 0, max_len - seg_h.shape[0])) for seg_h in all_seg_hidden]
        stacked = torch.stack(padded)  # Shape: (B, max_len, D)

        B, S, D = stacked.size()
        projected_seg_hidden = self.seg_projector(stacked)

        mask = torch.zeros((B,S), dtype=bool)
        for orig_len in original_lengths:
            mask[:,:orig_len] = True

        return projected_seg_hidden, mask

    def forward(self,
                data_dict: dict):
        """
        
        """
        
        device = next(self.parameters()).device

        batch_size = len(data_dict["num_category"])
        # TODO
        # This should be added right before the forward operation (in Trainer)
        cat_mapping = data_dict["category_mapping"]
        point_padding_mask = data_dict["padding_mask"]

        # data_dict = self.to_device(data_dict)

        # 1. Get GEM features
        gem_feats = self.GEM(data_dict["features"]) * point_padding_mask.unsqueeze(dim=-1)

        # 2. Get Pointcloud embeddings
        point_embed, intermed_feats, intermed_points = self.pcd_encoder(input=data_dict["features"],
                                                                        gem_feature=gem_feats,
                                                                        padding_mask=point_padding_mask)
        data_dict["point_embed"] = self.pcd_proj(point_embed)

        # 3. Generate input embeddings for LLM
        # (B, T, D), (B, T), (B, T), (B, T)
        inputs_embeds, input_ids, attention_mask, targets, special_token_ids = self.generate_input_embs(data_dict, cat_mapping, train=True)

        # 4. Forward
        with maybe_autocast(self):
            outputs = self.LLM(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True,
            )
        logits = outputs.logits.bfloat16()

        # 5. Get hidden states of POINT sequences and <SEG>
        # 5-1. Get indices of point tokens for each batch
        # point_indices[:, 0] -> batch indices, point_indices[:, 1] -> sequence indices
        point_indices = special_token_ids["POINT"].nonzero(as_tuple=False)  # (batch_size * num_points, 2)

        point_hidden = outputs.hidden_states[-1][point_indices[:, 0], point_indices[:, 1]]  # (batch_size * num_points, hidden_dim)

        point_hidden = einops.rearrange(point_hidden, '(b n) d -> b n d', b=batch_size)

        # 5-2. Get hidden states of <SEG> 
        # Since number of seg tokens will vary for each batch, so it is saved as a list of tensors instead of batched tensor
        all_seg_hidden = [h[mask] for h, mask in zip(outputs.hidden_states[-1][:,:-1,:], special_token_ids["SEG"])]

        # 6. Geometric-guided Feature Propagation
        gfp_feats = self.GFP(intermediate_features=intermed_feats,
                             intermediate_points=intermed_points,
                             hidden_point=point_hidden,
                             gem_features=gem_feats,
                             xyz=data_dict["xyz"])

        # 7. Project <SEG> hidden state into the GFP feature dimension
        pad_seg_hidden_proj, valid_inds = self.seg_batched_projection(all_seg_hidden)

        # 8. Dot product between <SEG> hidden state and gfp feats
        output_mask = torch.matmul(pad_seg_hidden_proj, gfp_feats.transpose(1,2))       # (B, G, N) -> padded version

        # Save the <SEG> token masks for results
        data_dict["output_mask"] = output_mask
        data_dict["valid_output_mask_indices"] = valid_inds.unsqueeze(dim=-1).to(device) * data_dict["valid_gt_mask"]

        # 9. Loss calcuation
        # TODO: Do I have to filter valid category predictions..? -> Then zero out the invalid category prediction categories
        # A. Text Generation Loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = targets[..., 1:].contiguous()
        num_tokens_for_loss = (shift_labels >= 0).int().sum(1)      # (B,)

        shift_logits = einops.rearrange(shift_logits, 'b t d -> (b t) d')
        shift_labels = einops.rearrange(shift_labels, 'b t -> (b t)')

        shift_labels = shift_labels.to(device)
        text_loss = F.cross_entropy(shift_logits, shift_labels, reduction='none')
        text_loss = einops.rearrange(text_loss, '(b t) -> b t', b=batch_size)
        text_loss = text_loss.sum(1) / num_tokens_for_loss   # (B,)

        # B. Cross Entropy Loss and Mask Loss
        gt_mask = data_dict["mask"]     # (B, G, N)
        ce_loss = masked_cross_entropy(output_mask, gt_mask, data_dict["valid_output_mask_indices"], reduction='batch_mean')                  # (B,)
        dice_loss = DiceLoss(output_mask, gt_mask, data_dict["valid_output_mask_indices"], reduction="none")                        # (B,)

        # C. Weighted Sum
        loss = text_loss * self.loss_weights[0] + ce_loss * self.loss_weights[1] + dice_loss * self.loss_weights[2]
        
        # Not averaging the loss inside the forward (train) function. Mean operation will be done in Trainer
        data_dict.update({'loss': loss})

        return data_dict
    
    @torch.no_grad()
    def generate(self,
                 data_dict,
                 use_nucleus_sampling: bool = False,
                 num_beams: int = 5,
                 max_new_tokens: int = 64,
                 min_new_tokens: int = 1,
                 repetition_penalty: float = 3.0,
                 length_penalty: float = 1.0,
                 num_captions: int = 1,
                 temperature: float = 1.0):
        """
        Generating output sequence. `generate()` requires same keys for `data_dict` but doesn't use ground truth categories for the answer
        """
        
        device = next(self.parameters()).device

        batch_size = len(data_dict["num_category"])
        # TODO
        # This should be added right before the forward operation (in Trainer)
        cat_mapping = data_dict["category_mapping"]
        point_padding_mask = data_dict["padding_mask"]

        # data_dict = self.to_device(data_dict)

        # 1. Get GEM features
        gem_feats = self.GEM(data_dict["features"]) * point_padding_mask.unsqueeze(dim=-1)

        # 2. Get Pointcloud embeddings
        point_embed, intermed_feats, intermed_points = self.pcd_encoder(input=data_dict["features"],
                                                                        gem_feature=gem_feats,
                                                                        padding_mask=point_padding_mask)
        data_dict["point_embed"] = self.pcd_proj(point_embed)

        # 3. Generate input embeddings for LLM
        # (B, T, D), (B, T), None, (B, T)
        inputs_embeds, input_ids, attention_mask, targets, special_token_ids = self.generate_input_embs(data_dict, cat_mapping, train=False)

        # 4. Generate POINT hidden states
        with maybe_autocast(self):
            pre_outputs = self.LLM(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True,
            )
        
        # Get indices of point tokens for each batch
        # point_indices[:, 0] -> batch indices, point_indices[:, 1] -> sequence indices
        point_indices = special_token_ids["POINT"].nonzero(as_tuple=False)  # (batch_size * num_points, 2)

        point_hidden = pre_outputs.hidden_states[-1][point_indices[:, 0], point_indices[:, 1]]  # (batch_size * num_points, hidden_dim)

        point_hidden = einops.rearrange(point_hidden, '(b n) d -> b n d', b=batch_size)

        # 4. BOS token as condition
        bos_tokens = self.LLM_tokenizer(
            [self.LLM_tokenizer.bos_token] * batch_size,
            return_tensors='pt',
        ).to(device)
        bos_tokens_ids = bos_tokens.input_ids[:, 0:1]   # (B, 1)
        bos_tokens_attn = bos_tokens.attention_mask[:, 0:1]   # (B, 1)
        bos_embeds = self.LLM.get_input_embeddings()(bos_tokens_ids)   # (B, 1, D)

        # Concatenate the BOS token
        input_ids = torch.cat([input_ids, bos_tokens_ids], dim=1)       # (B, T1+O+T2+1)
        inputs_embeds = torch.cat([inputs_embeds, bos_embeds], dim=1)   # (B, T1+O+T2+1, D)
        attention_mask = torch.cat([attention_mask, bos_tokens_attn], dim=1)   # (B, T1+O+T2+1)

        # Think about formatting the output
        # cutting off the irrelevant outputs
        # OR adding some more system prompts -> YOu should output {category} <SEG>, .... 
        # OR refer to strict output format (JSON) in Llama
        with maybe_autocast(self):
            outputs = self.LLM.generate(
                inputs=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict_in_generate=True,
                output_hidden_states=True,
                detach_hidden=True,
                pad_token_id=self.LLM_tokenizer.pad_token_id,
                eos_token_id=self.LLM_tokenizer.eos_token_id,
                # output_scores=True,
                output_all_beams=False,
                do_sample=use_nucleus_sampling,
                # top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
        
        outputs.sequences[outputs.sequences == self.LLM_tokenizer.unk_token_id] = self.LLM_tokenizer.eos_token_id

        # Remove all EOS tokens (</s>)
        generated_seq = outputs.sequences[:, inputs_embeds.size(1):-1]
        seg_token_mask = generated_seq == self.seg_token_id
        valid_seq_len = generated_seq.size(1)

        # Trim EOS tokens that are removed in generated_seq
        final_hidden_state = outputs.hidden_states[:, :valid_seq_len, :]      # (batch_size, valid_seq_len, hidden_dim)

        # Get the hidden states of <SEG> token for each batch
        all_seg_hidden = [h[mask] for h, mask in zip(final_hidden_state, seg_token_mask)]

        # 5. Geometric-guided Feature Propagation
        gfp_feats = self.GFP(intermediate_features=intermed_feats,
                             intermediate_points=intermed_points,
                             hidden_point=point_hidden,
                             gem_features=gem_feats,
                             xyz=data_dict["xyz"])

        # 6. Project <SEG> hidden state into the GFP feature dimension
        pad_seg_hidden_proj, valid_inds = self.seg_batched_projection(all_seg_hidden)

        # 7. Dot product between <SEG> hidden state and gfp feats
        output_mask = torch.matmul(pad_seg_hidden_proj, gfp_feats.transpose(1,2))       # (B, G, N) -> padded version

        # FINAL OUTPUT
        # all_output_seq = [seq[~mask] for seq, mask in zip(generated_seq, seg_token_mask)]
        all_output_txt = self.LLM_tokenizer.batch_decode(generated_seq)
        data_dict["gen_answers"] = all_output_txt
        data_dict["output_mask"] = output_mask
        data_dict["valid_output_mask_indices"] = valid_inds

        all_gt_txt = self.LLM_tokenizer.batch_decode(targets)
        data_dict["gt_text"] = all_gt_txt

        return data_dict