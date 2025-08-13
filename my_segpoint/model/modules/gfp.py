import torch
import torch.nn as nn

from omegaconf import OmegaConf, DictConfig
import einops

from pointnet2_ops.pointnet2_modules import PointnetFPModule
from pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation
from modules.utils.point_utils import fps, knn_point, index_points

class GeometricFeaturePropagation(nn.Module):
    def __init__(self,
                 cfg: DictConfig):
        super().__init__()

        self.selected_idx = cfg.selected_intermed_idx
        self.num_intermed_feats = cfg.num_intermed_feats
        self.gem_dim = cfg.gem_dim
        self.pc_feat_dim = cfg.pc_feat_dim
        self.K = cfg.num_neighbors

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


    def forward(self,
                intermediate_features: list[torch.Tensor],
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
        """
        B, _, _ = gem_features.size()

        # 1. Concatenate the last intermediate features with hidden_point
        fused_feats = torch.cat([intermediate_features[-1], hidden_point], dim=-1)

        for i in range(-2, -len(self.selected_idx) - 2, -1):
            if i > -len(self.selected_idx) -1:
                # Upsampling and Downsampling
                fps_idx = furthest_point_sample(xyz, self.num_intermed_feats[i])

                feat_flipped = gem_features.transpose(1,2).contiguous()
                downsampled_gem_feat = gather_operation(feat_flipped, fps_idx).transpose(1,2).contiguous()            

                xyz_flipped = xyz.transpose(1,2).contiguous()
                downsampled_xyz = gather_operation(xyz_flipped, fps_idx).transpose(1,2).contiguous()

                intermed_feat = intermediate_features[self.selected_idx[i]]
                intermed_xyz= intermediate_points

                query = self.upsample_layers[i](downsampled_xyz, intermed_xyz, downsampled_gem_feat, intermed_feat)
            else:
                downsampled_xyz = xyz
                query = gem_features

            # Attentive Propagation
            # 1. Group the points
            group_idx = knn_point(self.K, intermediate_points, downsampled_xyz)
            grouped_feats = index_points(fused_feats, group_idx)

            # 2. Reshape the tensors for attetion
            query_flat = einops.rearrange(query, 'B N D -> (B N) 1 D')
            keyval_flat = einops.rearrange(grouped_feats, 'B N K D -> (B N) K D')

            att_vals = self.attentive_props[i+1](query=query_flat,
                                                 key=keyval_flat,
                                                 val=keyval_flat)
            
            att_vals = einops.rearrange(att_vals, '(B N) 1 D -> B N D', B=B)
            fused_feats = query + att_vals

        return fused_feats



