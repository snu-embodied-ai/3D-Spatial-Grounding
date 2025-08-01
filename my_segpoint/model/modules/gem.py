import torch
import torch.nn as nn

from omegaconf import OmegaConf, DictConfig
import einops

from kpconv import KPConvBlock
from KPConv.datasets.common import batch_neighbors

class GeometricEnhancer(nn.Module):
    def __init__(
            self,
            cfg: DictConfig,
    ):
        super().__init__()

        num_layers = cfg.num_layers
        deform_radius = cfg.deform_radius
        KPConv_cfg = cfg.KPConv_cfg

        self.r = deform_radius * KPConv_cfg.radius
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

        batch_inds = torch.ones(B, device=input.device) * N
        stacked_input = einops.rearrange(input, 'B N D -> (B N) D')

        points = stacked_input[:,:3]
        features = stacked_input[:,3:]

        neighbors = batch_neighbors(queries=points,
                                    supports=points,
                                    q_batches=batch_inds,
                                    s_batches=batch_inds,
                                    radius=self.r)
        
        for conv in self.conv_layers:
            features = conv(features, points, neighbors)
            
        output = einops.rearrange(features, '(B N) D -> B N D', B=B)

        return output

