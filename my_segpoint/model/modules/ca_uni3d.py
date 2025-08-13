import torch
import torch.nn as nn
from omegaconf import DictConfig

from pointnet2_ops import pointnet2_utils
from Uni3D.models.uni3d import Uni3D

import logging

class CA_Uni3D(nn.Module):
    def __init__(self,
                 pretrained_uni3d: Uni3D,
                 cfg: DictConfig):
        """
        Point Encoder for SegPoint. Backbone from Uni3D and Cross Attention Layers integrated between the Transformer blocks of the backbone.

        Parameters
        ---
        pretrained_uni3d: Uni3D
            Backbone of the Point Encoder. Point Encoder Backbones should be frozen before passing to this modified model
        cfg: omegaConf.DictConfig
            config for CA_Uni3D
        """
        self.uni3d_logit_scale = pretrained_uni3d.logit_scale
        self.uni3d = pretrained_uni3d.point_encoder

        # self.group_divider = self.uni3d.group_divider
        # self.encoder = self.uni3d.encoder

        # self.encoder2trans = self.uni3d.encoder2trans

        # self.trans2embed = self.uni3d.trans2embed
        # self.cls_token = self.uni3d.cls_token
        # self.cls_pos = self.uni3d.cls_pos

        self.num_CA_layers = cfg.num_CA_layers
        self.transformers_per_block = len(self.uni3d.visual.blocks) // self.num_CA_layers

        self.CA_layers = nn.ModuleList()
        self.CA_gating_factors = nn.ModuleList()
        for i in range(len(self.num_CA_layers)):
            self.CA_layers.append(nn.MultiheadAttention(
                embed_dim=self.uni3d.trans_dim,
                num_heads=cfg.num_heads,
                dropout=cfg.dropout_rate,
                kdim=cfg.gem_dim,
                vdim=cfg.gem_dim,
                batch_first=True
            ))
            self.CA_gating_factors.append(nn.Parameter(torch.zeros(1)))

        self.align_clip_dim = cfg.align_clip_dim

    def forward(self, input: torch.Tensor,
                gem_feature: torch.Tensor):
        """
        Parameters
        ---
        input: torch.Tensor
            Input point cloud xyz and features (rgb + @). Shape of `(B, N, 3+F)`
        gem_feature: torch.Tensor
            Resultant geometric features (xyz not included explicitly). Shape of `(B, N, D)`

        Returns
        ---
        x: torch.Tensor 
            Feature Embedding of the point cloud. Shape of `(B, G, D)`. If `align_clip_dim = True`, `(B, G, D_clip)`
        intermediate_features: list[torch.Tensor]
            List of the interemediate features for further computes (GFP). List of length `num_CA_layers` and each tensors having shape of `(B, G, D)`
        """
        pts = input[:,:,:3]
        colors = input[:,:,3:6]

        # Divide the point cloud
        _, center, features = self.uni3d.group_divider(pts, colors)

        # Encode the input point cloud patches
        group_input_tokens = self.uni3d.encoder(features)  # (B, G, pc_encoder_dim)
        group_input_tokens = self.uni3d.encoder2trans(group_input_tokens)

        # Prepare CLS token
        cls_tokens = self.uni3d.cls_token.expand(group_input_tokens.size(0), -1, -1)  
        cls_pos = self.uni3d.cls_pos.expand(group_input_tokens.size(0), -1, -1)
        
        # Add pos embedding
        pos = self.uni3d.pos_embed(center)
        # Final input
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)

        x = x + pos
        # Patch_dropout of 0. would mean it is disabled and this function becomes an identity function
        x = self.uni3d.patch_dropout(x)

        x = self.uni3d.pos_drop(x)      # (B, G, pc_feat_dim)

        intermediate_features = []
        for i, blk in enumerate(self.uni3d.visual.blocks):
            if i % self.transformers_per_block:
                intermediate_features.append(x)

                # Cross Attention between point features and GEM features
                fusion = self.CA_layers[i](query=x,
                                           key=gem_feature,
                                           value=gem_feature)
                
                x = x + self.CA_gating_factors[i] * fusion
            
            x = blk(x)
        
        intermediate_features.append(x)

        if self.align_clip_dim:
            x = self.uni3d.trans2embed(x)

        return x, intermediate_features, center

            


