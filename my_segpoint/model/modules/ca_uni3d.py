import torch
import torch.nn as nn
from omegaconf import DictConfig

from torch.nn.attention import SDPBackend, sdpa_kernel

from .Uni3D.models.uni3d import Uni3D

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
        super().__init__()

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
        self.use_flash_att = cfg.flash_attention

        self.CA_layers = nn.ModuleList()
        self.CA_gating_factors = nn.ParameterList()
        for i in range(self.num_CA_layers):
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
                gem_feature: torch.Tensor,
                padding_mask: torch.Tensor):
        """
        Parameters
        ---
        input: torch.Tensor
            Input point cloud xyz and features (rgb + @). Shape of `(B, N, 3+F)`
        gem_feature: torch.Tensor
            Resultant geometric features (xyz not included explicitly). Shape of `(B, N, D)`
        padding_mask: torch.Tensor
            Padding mask of points. Shape of `(B,N)`

        Returns
        ---
        x: torch.Tensor 
            Feature Embedding of the point cloud. Shape of `(B, G, D)`. If `align_clip_dim = True`, `(B, G, D_clip)`
        intermediate_features: list[torch.Tensor]
            List of the interemediate features for further computes (GFP). List of length `num_CA_layers` and each tensors having shape of `(B, G, D)`
        """
        B, _, _ = input.size()
        device = input.device
        input_dtype = input.dtype

        pts = input[:,:,:3].contiguous()
        colors = input[:,:,3:6].contiguous()

        # Divide the point cloud
        center = []
        features = []
        # To apply masking
        for i in range(B):
            masked_pts = pts[i][padding_mask[i]]
            masked_colors = colors[i][padding_mask[i]]
            _, single_center, single_features = self.uni3d.group_divider(masked_pts.unsqueeze(dim=0).to(torch.float), masked_colors.unsqueeze(dim=0).to(torch.float))
            center.append(single_center)
            features.append(single_features)

        center = torch.cat(center, dim=0).to(device).to(input_dtype)
        features = torch.cat(features, dim=0).to(device).to(input_dtype)

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

        x = self.uni3d.visual.pos_drop(x)      # (B, G, pc_feat_dim)

        intermediate_features = []
        ca_idx = 0
        for i, blk in enumerate(self.uni3d.visual.blocks):
            if i % self.transformers_per_block == 0:
                intermediate_features.append(x[:, 1:, :])

                # Cross Attention between point features and GEM features
                if self.use_flash_att:
                    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                        fusion, _ = self.CA_layers[ca_idx](query=x,
                                                           key=gem_feature,
                                                           value=gem_feature)
                else:
                    fusion, _ = self.CA_layers[ca_idx](query=x,
                                                       key=gem_feature,
                                                       value=gem_feature)
                    
                x = x + self.CA_gating_factors[ca_idx] * fusion

                ca_idx += 1
            
            x = blk(x)
        
        intermediate_features.append(x[:, 1:, :])

        if self.align_clip_dim:
            x = self.uni3d.trans2embed(x)

        return x[:, 1:, :], intermediate_features, center

            


