import torch
import timm
import numpy as np
from torch import nn

from .point_encoder import PointcloudEncoder, RegionPointcloudEncoder

class Uni3D(nn.Module):
    def __init__(self, point_encoder):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.point_encoder = point_encoder

    def encode_pc(self, vision_dict):
        vision_dict["divided_xyz"] = vision_dict["divided_xyz"].contiguous()
        vision_dict["divided_features"] = vision_dict["divided_features"].contiguous()

        return self.point_encoder(vision_dict)

    def forward(self, vision_dict):
        cls_feat, region_feat = self.encode_pc(vision_dict)
        
        vision_dict["cls_embedding"] = cls_feat
        vision_dict["region_embedding"] = region_feat

        return vision_dict


def create_uni3d(cfg):  
    # create transformer blocks for point cloud via timm
    point_transformer = timm.create_model(cfg["pc_model"], checkpoint_path=cfg["pretrained_pc"], drop_path_rate=cfg["drop_path_rate"])

    if cfg["PointcloudEncoder"]["token_unit"] == "region":
        point_encoder = RegionPointcloudEncoder(point_transformer, cfg["PointcloudEncoder"])
    else:
        point_encoder = PointcloudEncoder(point_transformer, cfg["PointcloudEncoder"])

    # uni3d model
    model = Uni3D(point_encoder=point_encoder,)
    return model


