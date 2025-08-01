import torch
import torch.nn as nn
import torch.nn.functional as F

from omegaconf import OmegaConf, DictConfig

from einops import rearrange
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer

from modules.ca_uni3d import CA_Uni3D
from modules.Uni3D.models.uni3d import create_uni3d
from modules.gfp import GeometricFeaturePropagation
from modules.gem import GeometricEnhancer

class SegPoint(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()

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

        self.tokenizer = AutoTokenizer.from_pretrained(self.LLM_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = cfg.LLM.padding_size

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

        # 3. Load Geometric Enhancer Module
        self.GEM = GeometricEnhancer(cfg.GEM)

        # 4. Load Geometric-guided Feature Propagation Module
        self.GFP = GeometricFeaturePropagation(cfg.GFP)

        # 5. <SEG> token projector
        self.seg_projector = nn.Sequential(
            nn.Linear(self.LLM.hidden_size, self.LLM.hidden_size),
            nn.GELU(),
            nn.Linear(self.LLM.hidden_size, cfg.GFP.gem_dim)
        )