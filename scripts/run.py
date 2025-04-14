import argparse
import yaml

import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group

import os, sys
CUR_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(CUR_DIR, os.pardir))
sys.path.extend([ROOT_DIR, CUR_DIR])

from data.dataloader import create_dataloader
from utils.trainer import Trainer
from utils.util import *

from model.noLLM import Spatial3D_lang
from model.text_encoder.CLIP_text_encoder import load_text_encoder

"""
TODO:
    - DDP(Distributed Data Parallel) should be implemented -> Even though you get more GPUs, you need to run in multiple GPUs to train faster and get enough batch size (if you are not going to run with batch size of 1/2/4)
    - However, implement the full code without DDP first. Then, insert more codes to enable DDP. I think finishing the code is much more important, and check if my code is running well.
"""

def ddp_setup():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_process_group(backend="nccl")

def main(args):
    ddp_setup()

    # ======= 1. Selecting the device for train/evaluation ======================
    # device = get_device()

    # ======= 2. Load configuration files =================================
    config, model_config, data_config = [
        load_config(ROOT_DIR, path) for path in [args.config, args.model_config, args.data_config]
    ]

    # ======= 3. Initialize wandb and open log file ===============================
    init_wandb(config, model_config, data_config)

    if not os.path.exists(config["output_dir"]):
        os.mkdir(config["output_dir"])
        os.mkdir(os.path.join(config["output_dir"], "log"))
    log = Log(config["output_dir"])

    # ======= 4. Initialize the model ======================================
    log.write("Creating Spatial3DLang Model..!!")
    model = Spatial3D_lang(model_config)
    text_encoder = load_text_encoder(model_config)

    if config["freeze_pretrain"]:
        trainable_param_groups = model.freeze_pretrained_modules()
        for param in text_encoder.parameters():
            param.requires_grad = False
        config["Optimizer"]["trainable_param_groups"] = trainable_param_groups

    tokenizer = text_encoder.tokenizer

    # ======= 5. Create Dataloaders ======================================
    if data_config["only_test"]:
        _, _, test_Dataloader = create_dataloader(data_config, tokenizer)
    else:
        train_Dataloader, val_Dataloader, test_Dataloader = create_dataloader(data_config, tokenizer)

    
    # ======= 6. Train OR Inference ======================================
    if args.run_type == "train":
        trainer = Trainer(model=model,
                          text_encoder=text_encoder,
                          train_cfg=config,
                          model_cfg=model_config,
                        #   device=device,
                          )
        trainer.fit(train_Dataloader, val_Dataloader, log)
    elif args.run_type == "inference":
        pass
    else:
        raise Exception("CODE ERROR. Wrond argument has been inserted to run_type")
    
    destroy_process_group()
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_type", 
                        type=str, default="train",
                        choices=["train", "inference"],
                        help="Determine to TRAIN the model or run the model for INFERENCE")
    
    parser.add_argument("--config",
                        type=str, required=True,
                        help="Path of the configuration file for TRAIN or INFERENCE"
                        )
    
    parser.add_argument("--model_config",
                        type=str, required=True,
                        help="Path of the model's configuration file ")
    
    parser.add_argument("--data_config",
                        type=str, required=True,
                        help="Path of the model's configuration file ")

    args = parser.parse_args()

    main(args)