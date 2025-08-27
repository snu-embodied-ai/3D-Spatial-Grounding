import torch
import torch.nn as nn

from einops import rearrange

from .KPConv.models.blocks import KPConv, BatchNormBlock

class KPConvBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            radius: float = 0.03,
            sigma: float = 1.2,
            num_kernel_points: int = 15,
            in_points_dim: int = 3,
            fixed_kernel_points: str = 'center',
            KP_influence: str = 'linear',
            aggregation_mode: str = 'sum',
            use_batchnorm: bool = True,
            bn_momentum: float = 0.02,
    ):
        super().__init__()

        # Get model hyperparameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius
        self.sigma = sigma
        self.num_kernel_points = num_kernel_points
        self.in_points_dim = in_points_dim
        self.fixed_kernel_points = fixed_kernel_points
        self.KP_influence = KP_influence
        self.aggregation_mode = aggregation_mode
        self.use_bn = use_batchnorm
        if self.use_bn:
            self.bn_momentum = bn_momentum

        # Define the KPConv class
        self.kpconv = KPConv(kernel_size=self.num_kernel_points,
                             p_dim=self.in_points_dim,
                             in_channels=self.in_channels,
                             out_channels=self.out_channels,
                             KP_extent=self.radius * self.sigma,
                             radius=self.radius,
                             fixed_kernel_points=self.fixed_kernel_points,
                             KP_influence=self.KP_influence,
                             aggregation_mode=self.aggregation_mode,
                             deformable=True,
                             modulated=False
                             )
        
        if self.use_bn:
            self.batch_norm = BatchNormBlock(in_dim=self.out_channels,
                                             use_bn=self.use_bn,
                                             bn_momentum=self.bn_momentum)
            
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x: torch.Tensor, 
                input_points: torch.Tensor, 
                input_neighbors: torch.Tensor):
        """
        Parameters
        ---
        x: torch.Tensor
            Features of the input point cloud, which are the tensors being modified. Shape: (B*N, F)
        input_points: torch.Tensor
            XYZ coordinates of the input point cloud, which are intact in the forward process. Shape: (B*N, 3)
        input_neighbors: torch.Tensor
            Indices of the neighbors for each points of the pointcloud. Shape: (B*N, M)
        """
        x = self.kpconv(input_points, input_points, input_neighbors, x)
        return self.leaky_relu(self.batch_norm(x))