import torch
import torch.nn as nn

from omegaconf import OmegaConf, DictConfig
import einops

from .kpconv import KPConvBlock
from pointnet2_ops.pointnet2_utils import ball_query


class GeometricEnhancer(nn.Module):
    def __init__(
            self,
            cfg: DictConfig,
    ):
        super().__init__()

        num_layers = cfg.num_layers
        deform_radius = cfg.deform_radius
        self.num_samples = cfg.num_samples
        KPConv_cfg = cfg.KPConv

        self.r = deform_radius * KPConv_cfg.other.radius
        self.conv_layers = nn.ModuleList()
        self.in_channels = KPConv_cfg.in_channels
        self.out_channels = KPConv_cfg.out_channels

        for i in range(num_layers):
            self.conv_layers.append(
                KPConvBlock(in_channels=self.in_channels[i],
                            out_channels=self.out_channels[i],
                            **KPConv_cfg.other)
            )

    def forward(self,
                input: torch.Tensor):
        """
        Parameters
        ---
        input: torch.Tensor
            Input Pointcloud including xyz and features(rgb,...). Shape: (B, N, 3+F)

        Returns
        ---
        output: torch.Tensor
            Output Tensor including enhanced features (excluding xyz) Shape: (B, N, D)
        """
        B, N, D = input.size()
        input_dtype = input.dtype

        # TRY knn sampling if this doesn't work
        points = input[:,:,:3].contiguous()

        neighbors = ball_query(self.r,
                               self.num_samples,
                               points.to(torch.float32),
                               points.to(torch.float32))
        
        stacked_input = einops.rearrange(input, 'B N D -> (B N) D')
        neighbors = einops.rearrange(neighbors, 'B N M -> (B N) M').to(torch.int64)
        points = stacked_input[:,:3]
        features = stacked_input[:,3:]
        
        for conv in self.conv_layers:
            features = conv(features, points, neighbors)
            
        output = einops.rearrange(features, '(B N) D -> B N D', B=B).to(input_dtype)

        return output

