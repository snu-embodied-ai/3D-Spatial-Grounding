import torch
import torch.nn as nn

import einops

import os, sys
CUR_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(CUR_DIR, os.pardir))
sys.path.extend([ROOT_DIR, CUR_DIR])

from attention import CrossAttentionLayer
from detr_layers import DetrTransformerEncoderLayer
from att_unet import AttUNet


class OccupancyMapGenerator(nn.Module):
    def __init__(self,
                 embedding_dim: int,
                 nhead: int,
                 activation: str,
                 pre_norm: bool,
                 taskhead_cfg: dict,
                 dropout: float,
                 ):
        super(OccupancyMapGenerator, self).__init__()

        self.map_gen = taskhead_cfg["heatmap_gen_method"]

        if self.map_gen == "self attention":
            self.heatmap_generator = DetrTransformerEncoderLayer(
                self_attn_cfg=dict(num_heads=nhead, 
                                   embed_dims=embedding_dim, 
                                   dropout=dropout, 
                                   batch_first=True),
                ffn_cfg=dict(embed_dims=embedding_dim, 
                             feedforward_channels=embedding_dim, 
                             ffn_drop=0.0),
            )
            # TODO
            # SHOULD FINISH HERE. CANNOT USE THIS

        elif self.map_gen == "mlp":
            self.aggregate_mlp = nn.Sequential(
                nn.Linear(embedding_dim, 1),
                nn.GELU(),
            )
            self.map_mlp = nn.Linear(taskhead_cfg["num_tokens"], taskhead_cfg["heatmap_shape"][0] * taskhead_cfg["heatmap_shape"][1])
            pass
        elif self.map_gen == "basic_unet":
            # self.conv_block = nn.ModuleList()
            # input_dim = embedding_dim
            # output_dim = embedding_dim

            # for i in range(3):
            #     input_dim = output_dim
            #     output_dim = input_dim // 2

            #     self.conv_block.append(nn.Sequential(
            #         nn.Conv1d(in_channels=input_dim,
            #                   out_channels=output_dim,
            #                   kernel_size=3,
            #                   stride=2,
            #                   padding=1),
            #         nn.BatchNorm1d(output_dim),
            #         nn.GELU()
            #     ))
            # self.projector = nn.Linear(embedding_dim, taskhead_cfg["conv_channels"][0])
            self.heatmap_generator = AttUNet(unet_cfg=taskhead_cfg,
                                             kv_dim=embedding_dim,
                                             dropout=dropout,
                                             activation=activation,
                                             normalize_before=pre_norm,
                                             batch_first=True,
                                             is_occupancy=True)
        else:
            raise NotImplementedError("Other task heads aren't implemented besides Attention, MLP, UNet")
        
    def forward(self, 
                heatmap_grid_centers: torch.Tensor,
                vision_embedding: torch.Tensor,
                vision_pos: torch.Tensor):
        b, h, w, _ = heatmap_grid_centers.size()
        if self.map_gen == "self attention":
            pass
            # heatmap = self.heatmap_generator(query=vision_embedding,
            #                                  query_pos=vision_pos_enc,
            #                                  attn_masks=None,
            #                                  key_padding_mask=None)
            # heatmap = einops.rearrange(heatmap, "b (h w) 1 -> b h w", h=h, w=w)
            
        elif self.map_gen == "basic_unet":
            # vision_embedding = self.projector(vision_embedding)
            heatmap = self.heatmap_generator(heatmap_grid_centers, 
                                             vision_embedding,
                                             vision_pos)
        elif self.map_gen == "mlp":
            agg_queries = self.aggregate_mlp(vision_embedding)
            heatmap = self.map_mlp(einops.rearrange(agg_queries, 'b n 1 -> b n'))
            heatmap = einops.rearrange(heatmap, 'b (h w) -> b h w', h=h, w=w)
        else:
            NotImplementedError("Other task heads aren't implemented besides Attention, MLP, UNet")

        return heatmap
    


class HeatMapGenerator(nn.Module):
    def __init__(self,
                 nhead: int,
                 hidden_dim: int,
                 activation: str,
                 pre_norm: bool,
                 taskhead_cfg: dict,
                 dropout: float,
                 ):
        super(HeatMapGenerator, self).__init__()

        self.map_gen = taskhead_cfg["heatmap_gen_method"]
        heatmap_dimension = len(taskhead_cfg["heatmap_shape"])

        if self.map_gen == "cross attention":
            self.heatmap_generator = CrossAttentionLayer(
                d_model=heatmap_dimension,
                nhead=nhead,
                kdim=hidden_dim,
                vdim=hidden_dim,
                activation=activation,
                normalize_before=pre_norm,
                batch_first=True,
            )
            self.heatmap_mlp = nn.Linear(heatmap_dimension, 1)
        elif self.map_gen == "mlp":
            self.aggregate_mlp = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.GELU(),
            )
            self.map_mlp = nn.Linear(taskhead_cfg["num_tokens"], taskhead_cfg["heatmap_shape"][0] * taskhead_cfg["heatmap_shape"][1])
            pass
        elif self.map_gen == "basic_unet":
            # self.projector = nn.Linear(hidden_dim, taskhead_cfg["conv_channels"][0])
            self.heatmap_generator = AttUNet(unet_cfg=taskhead_cfg,
                                             kv_dim=hidden_dim,
                                             dropout=dropout,
                                             activation=activation,
                                             normalize_before=pre_norm,
                                             batch_first=True)
        else:
            raise NotImplementedError("Other task heads aren't implemented besides Attention, MLP, UNet")
        
    def forward(self, heatmap_grid_centers: torch.Tensor,
                query_embedding: torch.Tensor,
                query_pos: torch.Tensor):
        b, h, w, _ = heatmap_grid_centers.size()

        if self.map_gen == "cross attention":
            heatmap_pos = einops.rearrange(heatmap_grid_centers, 
                                           "b h w c -> b (h w) c")
            heatmap = self.heatmap_generator(tgt=heatmap_pos,
                                             memory=query_embedding,
                                             memory_mask=None,
                                             memory_key_padding_mask=None,
                                             pos=None,
                                             query_pos=None
                                             )
            
            heatmap = einops.rearrange(heatmap, "b (h w) c -> b h w c", h=h, w=w)
            heatmap = self.heatmap_mlp(heatmap).reshape(b, h, w)

        elif self.map_gen == "mlp":
            agg_queries = self.aggregate_mlp(query_embedding)
            heatmap = self.map_mlp(einops.rearrange(agg_queries, 'b n 1 -> b n'))
            heatmap = einops.rearrange(heatmap, 'b (h w) -> b h w', h=h, w=w)
            
        elif self.map_gen == "basic_unet":
            # query_proj = self.projector(query_embedding)
            heatmap = self.heatmap_generator(heatmap_grid_centers, query_embedding, query_pos)

        else:
            NotImplementedError("Other task heads aren't implemented besides Attention, MLP, UNet")

        return heatmap