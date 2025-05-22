import math

import torch
import torch.nn as nn
from torch.nn import functional as F

import os, sys
CUR_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(CUR_DIR, os.pardir))
sys.path.extend([ROOT_DIR, CUR_DIR])

from attention import CrossAttentionLayer, SelfAttentionLayer

import einops

class AttUNet(nn.Module):
    def __init__(self, 
                 unet_cfg: dict,
                 kv_dim: int,
                 dropout: float,
                 activation: str,
                 normalize_before: bool,
                 batch_first: bool,
                 is_occupancy: bool = False):
        super(AttUNet, self).__init__()

        self.dropout = unet_cfg["dropout"]

        self.blocks = nn.ModuleList()
        self.map_size = unet_cfg["heatmap_shape"]
        self.kernel_size = unet_cfg["kernel_size"]
        self.scale = math.prod(self.kernel_size)
        if is_occupancy:
            self.conv_channels = unet_cfg["occ_conv_channels"]
        else:
            self.conv_channels = unet_cfg["conv_channels"]

        self.init_H = self.map_size[0] // self.scale
        self.init_W = self.map_size[1] // self.scale

        H, W = self.init_H, self.init_W

        self.sizeMatcher = nn.Sequential(
            nn.Conv1d(
                in_channels=unet_cfg["num_tokens"],
                out_channels=self.init_H * self.init_W,
                kernel_size=3,
                padding=1,
                stride=1
            ),
            nn.Dropout2d(self.dropout)
        )

        for i in range(len(self.kernel_size)):
            conv_block = nn.ModuleList()

            up_conv = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=self.conv_channels[i],
                    out_channels=self.conv_channels[i+1],
                    kernel_size=self.kernel_size[i],
                    stride=unet_cfg["strides"][i]
                ),
                nn.GELU(),
                nn.Dropout2d(self.dropout)
            )
            conv_block.append(up_conv)
            H *= self.kernel_size[i]
            W *= self.kernel_size[i]

            conv = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.conv_channels[i+1],
                    out_channels=self.conv_channels[i+1],
                    kernel_size=3,
                    padding=1,
                    stride=1
                ),
                nn.GELU()
            )
            conv_block.append(conv)

            if i < len(self.kernel_size)- 1:
                conv_block.append(nn.LayerNorm([self.conv_channels[i+1], H, W]))
            else:
                conv_block.append(nn.Sequential(
                    nn.Conv2d(
                        in_channels=self.conv_channels[i+1],
                        out_channels=1,
                        kernel_size=1,
                        stride=1
                    ),
                ))

            self.blocks.append(conv_block)

        self.heatmap_pos = nn.Linear(2,1)


        # for i in range(len(unet_cfg["kernel_size"])):
        #     att_unet = nn.ModuleList()

        #     # 0. Position Embedding of heatmap tokens
        #     heatmap_pos = nn.Embedding((H*W), self.conv_channels[i])
        #     att_unet.append(heatmap_pos)

        #     H *= unet_cfg["kernel_size"][i]
        #     W *= unet_cfg["kernel_size"][i]
            
        #     # 1. Attention between heatmap embeddings and candidate embeddings
        #     cross_att = CrossAttentionLayer(
        #         d_model=self.conv_channels[i],
        #         nhead=unet_cfg["conv_n_heads"][i],
        #         kdim=kv_dim,
        #         vdim=kv_dim,
        #         dropout=dropout,
        #         activation=activation,
        #         normalize_before=normalize_before,
        #         batch_first=batch_first
        #     )
        #     att_unet.append(cross_att)

        #     # 2. Up Convolution to upsample the heatmap embeddings
        #     up_conv = nn.ConvTranspose2d(
        #         self.conv_channels[i],
        #         self.conv_channels[i+1],
        #         kernel_size=unet_cfg["kernel_size"][i],
        #         stride=unet_cfg["strides"][i]
        #     )
        #     att_unet.append(up_conv)

        #     # 3. Batch Normalization
        #     att_unet.append(nn.BatchNorm2d(self.conv_channels[i+1]))

        #     self.blocks.append(att_unet)

    def forward(self,
                heatmap_grid_centers: torch.Tensor,
                candidates: torch.Tensor,
                candidates_pos: torch.Tensor):    
        # 1. Size matching for heatmap generation
        # B, G, F  -> B, H'*W', F
        heatmap = self.sizeMatcher(candidates + candidates_pos)
        heatmap = einops.rearrange(heatmap, 'b (h w) f -> b f h w', h=self.init_H, w=self.init_W)

        for i, block in enumerate(self.blocks):
            # 2. Up convolution to heatmap
            # B, F_i, H_i, W_i  ->  B H, F_{i+1}, H_{i+1}, W_{i+1}, 
            heatmap = block[0](heatmap)
            
            if i < len(self.blocks) - 1:
                # 3. Convolution
                heatmap = block[1](heatmap)

                # 4. Normalization
                heatmap = block[2](heatmap)

            else:
                pos = self.heatmap_pos(heatmap_grid_centers)
                pos = einops.rearrange(pos, 'b h w c -> b c h w')
                # 3. Convolution
                heatmap = block[1](heatmap + pos)

                # 4. Final Convolution for logits
                heatmap = block[2](heatmap + pos)

        # 6. Rearrangement
        heatmap = einops.rearrange(heatmap, 'b 1 h w -> b h w')
        return heatmap
