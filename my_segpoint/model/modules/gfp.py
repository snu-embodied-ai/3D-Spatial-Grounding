import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from omegaconf import OmegaConf, DictConfig
import einops
from typing import List, Dict

from pointnet2_ops.pointnet2_modules import PointnetFPModule
from pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation
from model.modules.utils.point_utils import fps, knn_point, index_points

class GeometricFeaturePropagation(nn.Module):
    def __init__(self,
                 cfg: DictConfig):
        super().__init__()

        self.selected_idx = cfg.selected_intermed_idx
        self.num_intermed_feats = cfg.num_intermed_feats
        self.gem_dim = cfg.gem_dim
        self.pc_feat_dim = cfg.pc_feat_dim
        self.K = cfg.num_neighbors

        self.use_flash_att = cfg.flash_attention

        # Upsampling layers for the third/fourth intermediate features, excluding the final(fifth) feature
        num_upsample_layers = len(self.selected_idx) - 1
        self.upsample_layers = nn.ModuleList()
        for i in range(num_upsample_layers):
            self.upsample_layers.append(
                PointnetFPModule(mlp=[self.gem_dim + self.pc_feat_dim, 
                                      self.pc_feat_dim, 
                                      self.gem_dim, 
                                      self.gem_dim]))
            
        num_attentive_props = len(self.selected_idx)
        self.attentive_props = nn.ModuleList()
        self.norm = nn.ModuleList()

        # Change the indices in reverse order (the last feature becomes the first index)
        for i in range(num_attentive_props):
            if i == 0:
                kv_dim = cfg.LLM_hidden_dim + self.pc_feat_dim
            else:
                kv_dim = self.gem_dim

            self.attentive_props.append(
                nn.MultiheadAttention(
                    embed_dim=self.gem_dim,
                    num_heads=cfg.num_heads,
                    dropout=cfg.dropout_rate,
                    kdim=kv_dim,
                    vdim=kv_dim,
                    batch_first=True
                )
            )
            self.norm.append(
                nn.LayerNorm(self.gem_dim)
            )


    def forward(self,
                intermediate_features: List[torch.Tensor],
                intermediate_points: torch.Tensor,
                hidden_point: torch.Tensor,
                gem_features: torch.Tensor,
                xyz: torch.Tensor):
        """
        Parameters
        ---
        intermediate_features: list[torch.Tensor]
            List of Intermediate Features from the point encoder, each tensors having shape of `(B, G, pc_feat_dim)`
        intermediate_points: torch.Tensor
            Intermediate centers from the point encoder. Since the number of center points are all the same through the transformer layers, only single tensor is required instead of a list
        hidden_point: torch.Tensor
            Final hidden state of <POINT> tokens from the LLM output. Shape of `(B, G, LLM_dim)`
        gem_features: torch.Tensor
            Resultant geometric features (xyz not included explicitly). Shape of `(B, N, D)`
        xyz: torch.Tensor
        """
        B, _, _ = gem_features.size()
        input_dtype = hidden_point.dtype

        # 1. Concatenate the last intermediate features with hidden_point
        fused_feats = torch.cat([intermediate_features[-1], hidden_point], dim=-1)
        kv_xyz = intermediate_points

        for i in range(len(self.selected_idx)):
            if i < len(self.selected_idx) -1:
                # 2. Downsampling gem features
                fps_idx = furthest_point_sample(xyz.float(), self.num_intermed_feats[i])

                feat_flipped = gem_features.transpose(1,2).contiguous().float()
                downsampled_gem_feat = gather_operation(feat_flipped, fps_idx)

                xyz_flipped = xyz.transpose(1,2).contiguous().float()
                downsampled_xyz = gather_operation(xyz_flipped, fps_idx).transpose(1,2).contiguous()

                intermed_feat = intermediate_features[self.selected_idx[i]].transpose(1,2).contiguous()            
                intermed_xyz = intermediate_points

                query = self.upsample_layers[i](downsampled_xyz, intermed_xyz.float(), downsampled_gem_feat, intermed_feat.float()).transpose(1,2).contiguous().to(input_dtype)
                query_xyz = downsampled_xyz

            else:
                query = gem_features
                query_xyz = xyz.contiguous()

            # Attentive Propagation
            # 1. Group the points
            # TODO: Should fix here. Why is intermediate keep going in as new_xyz? Should change
            group_idx = knn_point(self.K, kv_xyz, query_xyz)
            grouped_feats = index_points(fused_feats, group_idx).to(input_dtype)

            # 2. Reshape the tensors for attetion
            query_flat = einops.rearrange(query, 'B N D -> (B N) 1 D')
            keyval_flat = einops.rearrange(grouped_feats, 'B N K D -> (B N) K D')

            if self.use_flash_att:
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    att_vals, _ = self.attentive_props[i](query=query_flat,
                                                          key=keyval_flat,
                                                          value=keyval_flat)
            else:
                att_vals, _ = self.attentive_props[i](query=query_flat,
                                                      key=keyval_flat,
                                                      value=keyval_flat)
            
            # att_vals = self.norm[i](att_vals)
            att_vals = einops.rearrange(att_vals, '(B N) 1 D -> B N D', B=B)
            fused_feats = self.norm[i](query + att_vals)
            kv_xyz = query_xyz

        # for i in range(-2, -len(self.selected_idx) - 2, -1):
        #     if i > -len(self.selected_idx) -1:
        #         # Upsampling and Downsampling
        #         # Ensure the dtype is maintained after upsampling
        #         fps_idx = furthest_point_sample(xyz.float(), self.num_intermed_feats[i])

        #         feat_flipped = gem_features.transpose(1,2).contiguous()
        #         downsampled_gem_feat = gather_operation(feat_flipped.float(), fps_idx)

        #         xyz_flipped = xyz.transpose(1,2).contiguous()
        #         downsampled_xyz = gather_operation(xyz_flipped.float(), fps_idx).transpose(1,2).contiguous()

        #         intermed_feat = intermediate_features[self.selected_idx[i]].transpose(1,2).contiguous()            
        #         intermed_xyz = intermediate_points

        #         query = self.upsample_layers[i+1](downsampled_xyz, intermed_xyz.float(), downsampled_gem_feat, intermed_feat.float()).transpose(1,2).contiguous().to(input_dtype)
        #     else:
        #         downsampled_xyz = xyz
        #         query = gem_features

        #     # Attentive Propagation
        #     # 1. Group the points
        #     # TODO: Should fix here. Why is intermediate keep going in as new_xyz? Should change
        #     group_idx = knn_point(self.K, intermediate_points, downsampled_xyz)
        #     grouped_feats = index_points(fused_feats, group_idx).to(input_dtype)

        #     # 2. Reshape the tensors for attetion
        #     query_flat = einops.rearrange(query, 'B N D -> (B N) 1 D')
        #     keyval_flat = einops.rearrange(grouped_feats, 'B N K D -> (B N) K D')

        #     if self.use_flash_att:
        #         with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        #             att_vals, _ = self.attentive_props[i+1](query=query_flat,
        #                                                     key=keyval_flat,
        #                                                     value=keyval_flat)
        #     else:
        #         att_vals, _ = self.attentive_props[i+1](query=query_flat,
        #                                                 key=keyval_flat,
        #                                                 value=keyval_flat)
            
        #     att_vals = self.norm[i+1](att_vals)
        #     att_vals = einops.rearrange(att_vals, '(B N) 1 D -> B N D', B=B)
        #     fused_feats = query + att_vals

        return fused_feats



