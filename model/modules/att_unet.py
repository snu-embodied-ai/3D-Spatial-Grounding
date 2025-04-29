import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from .attention import CrossAttentionLayer

import einops

class AttUNet(nn.Module):
    def __init__(self, unet_cfg: dict,
                 kv_dim: int,
                 dropout: float,
                 activation: str,
                 normalize_before: bool,
                 batch_first: bool):
        super(AttUNet, self).__init__()

        self.blocks = nn.ModuleList()
        self.scale = math.prod(unet_cfg["kernel_size"])

        for i in range(len(unet_cfg["kernel_size"])):
            att_unet = nn.ModuleList()
            
            # 1. Attention between heatmap embeddings and candidate embeddings
            attention = CrossAttentionLayer(
                d_model=unet_cfg["conv_channels"][i],
                nhead=unet_cfg["conv_n_heads"][i],
                kdim=kv_dim,
                vdim=kv_dim,
                dropout=dropout,
                activation=activation,
                normalize_before=normalize_before,
                batch_first=batch_first
            )
            att_unet.append(attention)

            # 2. Up Convolution to upsample the heatmap embeddings
            up_conv = nn.ConvTranspose2d(
                unet_cfg["conv_channels"][i],
                unet_cfg["conv_channels"][i+1],
                kernel_size=unet_cfg["kernel_size"][i],
                stride=unet_cfg["strides"][i]
            )
            att_unet.append(up_conv)

            # 3. Batch Normalization
            att_unet.append(nn.BatchNorm2d(unet_cfg["conv_channels"][i+1]))

            self.blocks.append(att_unet)

    def forward(self,
                heatmap_grid_centers: torch.Tensor,
                candidates: torch.Tensor):
        B, H, W = heatmap_grid_centers.size()[:3]

        init_H, init_W = H // self.scale, W // self.scale
        heatmap = torch.zeros((B, init_H, init_W, candidates.size(-1)))

        for i, block in enumerate(self.blocks):
            att, conv, norm = block
            _, h, w = heatmap.size()

            # 1. Attention Layer
            if i == len(self.blocks)-1:
                heatmap = att(tgt=einops.rearrange('b h w c -> b (h w) c'),
                              memory=candidates,
                              query_pos=heatmap_grid_centers)
                heatmap = einops.rearrange(heatmap, 'b (h w) c -> b c h w')
            else:
                heatmap = att(tgt=einops.rearrange(heatmap, 'b h w c -> b (h w) c'),
                            memory=candidates,)
                heatmap = einops.rearrange(heatmap, 'b (h w) c -> b c h w', h=h)
            
            # 2. Up Convolution
            heatmap = conv(heatmap)

            # 3. Batch normalization
            heatmap = norm(heatmap)

            heatmap = einops.rearrange(heatmap, 'b c h w -> b h w c')

        return heatmap
