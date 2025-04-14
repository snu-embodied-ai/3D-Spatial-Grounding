import wandb
import os, sys

import torch
import ruamel.yaml

import datetime


def init_wandb(cfg, model_cfg, data_cfg):
    """
    Initialize wandb to log loss values and predictions and store hyperparameters to wandb config file.
    """
    today = datetime.datetime.today()
    date = f"{today.month}-{today.day}-{today.year}"

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