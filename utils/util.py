import wandb
import os, sys

import torch
import torch.nn.functional as F
import ruamel.yaml
import einops
import numpy as np
import datetime

import open3d as o3d

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')        # FOR MAC
    else:
        device = torch.device('cpu')

    return device

def load_config(root, path):
    """Load a YAML config file from the given path."""
    full_path = os.path.join(root, path)
    yaml = ruamel.yaml.YAML()

    try:
        with open(full_path, 'r') as f:
            return yaml.load(f)
    except (IOError, yaml.YAMLError) as e:
        print(f"Error loading config from {full_path}: {e}")
        return {}
    
def accuracy_and_save_points(v_dict: dict, 
                             l_dict: dict,
                             top_k: list = [1,3,5],
                             save_points: bool = False,
                             table: wandb.Table = None,
                             local_save_dir: str = None,
                             run_type: str = None):
    """
    
    """
    all_acc = [0, 0, 0]

    pred_pcd = v_dict["output_logits"]
    label_pcd = v_dict["divided_labels"]
    all_xyz = v_dict["divided_xyz"]
    centers = v_dict["centers"]
    scales = v_dict["scales"]
    all_masks = v_dict["mask"].bool()

    # B, G, num_points = pred_points.size()

    for idx, pred, label, xyz, center, scale, mask in zip(l_dict["index"], pred_pcd, label_pcd, all_xyz, centers, scales, all_masks):
        # 1. Un-normalize & Mask invalid points (paddings)
        #  - masked_pred_pcd, masked_label_pcd : (N, 1)
        xyz = xyz * scale[:,None,None] + center.unsqueeze(dim=1)
        masked_pred_pcd = pred[mask]
        masked_label_pcd = label[mask]
        masked_xyz = xyz[mask]

        # ==== 2. ACCURACY CALCULATION ===========
        for i, k in enumerate(top_k):
            # 1. Select top k points of prediction
            pred_point_indices = torch.topk(masked_pred_pcd, k=k, dim=0).indices

            # 2. Collect corresponding points from the GT heatmap
            pred_point_on_label = masked_label_pcd[pred_point_indices]
        
            # 3. Compute the number of correct correspondences
            included = torch.any(pred_point_on_label > 0)

            all_acc[i] += included

        # ==== 3. SAVE PREDICTED POINTS ==========
        if save_points:
            # 1. Apply sigmoid for proper visualization
            proba_pred = F.sigmoid(masked_pred_pcd)

            # 2. Concatentate xyz and probabilties
            rgb_pred = torch.cat([proba_pred] * 3, dim=-1) * 255
            rgb_label = torch.cat([masked_label_pcd] * 3, dim=-1) * 255

            pcd_pred = torch.cat([masked_xyz, rgb_pred], dim=-1)
            pcd_label = torch.cat([masked_xyz, rgb_label], dim=-1)

            # 3. Detach tensors and convert to numpy arrays
            idx = idx.detach().clone().cpu().numpy()
            pcd_pred = pcd_pred.detach().clone().cpu().numpy()
            pcd_label = pcd_label.detach().clone().cpu().numpy()

            # 4. Log to wandb
            pcd_name = f"{run_type}_{idx}"
            table.add_data(
                pcd_name,
                wandb.Object3D(pcd_pred),
                wandb.Object3D(pcd_label)
            )

            # 5. Save to PLY format
            ply_pred = o3d.geometry.PointCloud()

            ply_pred.points = o3d.utility.Vector3dVector(pcd_pred[:,:3])
            ply_pred.colors = o3d.utility.Vector3dVector(pcd_pred[:,3:] / 255)

            o3d.io.write_point_cloud(os.path.join(local_save_dir, f"{pcd_name}_pred_pcd.ply"), ply_pred, write_ascii=True)

    return all_acc
    

def accuracy_and_save_img(v_dict: dict, 
                          l_dict: dict,
                          top_k: list = [1,3,5],
                          save_heatmap_img: bool = False,
                          table: wandb.Table = None,
                          local_save_dir: str = None,
                          run_type: str = None):
    """
    Calculate the accuracy of the predicted points.
    The accuracy is computed as:
    1. Select top k points from output heatmap
    2. Check if at least one point is inside the ground truth region among the top k points
    3. Check all batches
    4. Create wandb table for heatmap visualization if `save_heatmap_img` is True.
    """
    all_acc = []

    pred_occupancy_map = v_dict["output_occupancy_map"]
    label_occupancy_map = v_dict["occupancy_label"]

    pred_heatmap = v_dict["output_heatmap"]
    label_heatmap = v_dict["heatmap_label"]

    B, H, W = pred_heatmap.size()

    # 1. Rearrange the prediction and label for topk operation
    flat_pred_occupancy_map = einops.rearrange(pred_occupancy_map, "b h w -> b (h w)")
    flat_label_occupancy_map = einops.rearrange(label_occupancy_map, "b h w -> b (h w)")
    flat_pred_heatmap = einops.rearrange(pred_heatmap, "b h w -> b (h w)")
    flat_label_heatmap = einops.rearrange(label_heatmap, "b h w -> b (h w)")

    # ==== 2. ACCURACY CALCULATION ===========
    for num_points in top_k:
        # 1. Select top k points of predicted heatmap
        pred_point_indices_occ = torch.topk(flat_pred_occupancy_map, k=num_points).indices
        pred_point_indices = torch.topk(flat_pred_heatmap, k=num_points).indices

        # 2. Collect corresponding points from the GT heatmap
        pred_point_on_label_occ = flat_label_occupancy_map[torch.arange(B).unsqueeze(dim=1), pred_point_indices_occ]
        pred_point_on_label = flat_label_heatmap[torch.arange(B).unsqueeze(dim=1), pred_point_indices]

        # 3. Compute the number of correct correspondences
        included_occ = pred_point_on_label_occ.sum(dim=-1) > 0
        included = pred_point_on_label.sum(dim=-1) > 0

        acc_occ = included_occ.sum()
        acc = included.sum()
        all_acc.append([acc_occ, acc])

    # ==== 4. SAVE PREDICTED HEATMAP ==========
    # 1. Apply sigmoid for proper visualization
    pred_occupancy_map = F.sigmoid(pred_occupancy_map)
    pred_heatmap = F.sigmoid(pred_heatmap)
    if save_heatmap_img:
        for idx, pred_occ, label_occ, pred, label in zip(l_dict["index"], pred_occupancy_map, label_occupancy_map, pred_heatmap, label_heatmap):
            if pred_occ.min() < 0 or pred_occ.max() > 1:
                    raise Exception("Predictions are out of bound! - (0,1)")
            if pred.min() < 0 or pred.max() > 1:
                    raise Exception("Predictions are out of bound! - (0,1)")
            
            # 2. Detach tensors and convert to numpy arrays
            idx = idx.detach().cpu().numpy()
            pred_occ = pred_occ.detach().clone().cpu().numpy()
            label_occ = label_occ.detach().cpu().numpy()
            pred = pred.detach().clone().cpu().numpy()
            label = label.detach().cpu().numpy()
            
            # 3. Rearrange the heatmaps back to their original shape
            pred_occ_img = (pred_occ * 255).astype(np.int32)
            label_occ_img = (label_occ * 255).astype(np.int32)
            pred_img = (pred * 255).astype(np.int32)
            label_img = (label * 255).astype(np.int32)

            # 4. Save numpy array and add to wandb Table
            img_name = f"{run_type}_{idx}"

            np.save(os.path.join(local_save_dir, f"{img_name}_occupancy_map.npy"), pred_occ_img)
            np.save(os.path.join(local_save_dir, f"{img_name}_heatmap.npy"), pred_img)
            table.add_data(img_name, 
                           wandb.Image(pred_occ_img), 
                           wandb.Image(label_occ_img), 
                           wandb.Image(pred_img), 
                           wandb.Image(label_img))

    return all_acc


def init_wandb(cfg, model_cfg, data_cfg):
    """
    Initialize wandb to log loss values and predictions and store hyperparameters to wandb config file.
    """
    cur_time = datetime.datetime.now()
    date = cur_time.strftime("%m%d%Y-%H:%M")

    wandb_cfg = {
        "project": cfg["project"],
        "name": f"{cfg['project']}_batch{data_cfg['train']['batch_size']}_epochs{cfg['num_epochs']}_{date}",
        "notes": f"Spatial 3D with batch size {data_cfg['train']['batch_size']} and running {cfg['num_epochs']} epochs",
        "config": {
            "name": f"{cfg['project']}_batch{data_cfg['train']['batch_size']}_epochs{cfg['num_epochs']}_{date}",
            "model": model_cfg,
            "dataset": data_cfg,
        }
    }

    wandb.init(**wandb_cfg)
    wandb.config.update(cfg)


    
class Log:
    def __init__(self,
                ouput_dir: str):
        if wandb.run is not None:
            self.path = os.path.join(ouput_dir, "log", f"{wandb.config['name']}_log.txt")
        else:
            raise Exception("Initialize wandb first to open log file!! Call init_wandb() first")
        
        self.file = open(self.path, "w+")

    def write(self, msg):
        # print(msg)
        self.file.write(msg+"\n")
        self.file.flush()