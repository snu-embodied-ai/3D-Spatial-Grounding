import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
import einops
from tqdm import tqdm
import time
import wandb

import os, sys
CUR_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(CUR_DIR, os.pardir))
sys.path.extend([ROOT_DIR, CUR_DIR])

from utils.util import *
from utils.losses import create_loss_func
from utils.optimizer_and_scheduler import create_optimizer_and_scheduler
from utils.util import accuracy_and_save_img

class Trainer():
    def __init__(self, 
                 model: nn.Module,
                 text_encoder: nn.Module,
                 train_cfg: dict,
                 model_cfg: dict,
                #  device: torch.device,
                 ):
        # Training Parameters
        self.output_dir = os.path.join(ROOT_DIR, train_cfg["output_dir"])
        self.ckpt_dir = os.path.join(ROOT_DIR, train_cfg["ckpt_dir"])
        self.output_heatmap_dir = os.path.join(ROOT_DIR, train_cfg["heatmap_dir"])

        self.gpu_id = int(os.environ["LOCAL_RANK"])

        self.model = model.to(self.gpu_id)
        self.text_encoder = text_encoder
        # if model_cfg["CLIP"]["to_CPU"]:
        #     self.text_encoder = text_encoder
        # else:
        #     self.text_encoder = text_encoder.to(self.gpu_id)

        self.model_cfg = model_cfg

        self.loss_types = train_cfg["loss_type"]
        self.imbalance_weight = train_cfg["imbalance_weight"]
        self.smooth_eps = train_cfg["label_smoothing_eps"]

        self.max_epochs = train_cfg["num_epochs"]
        self.epochs_run = 0
        self.print_freq = train_cfg["print_freq"]
        self.clip_grad_max_norm = train_cfg["clip_grad_max_norm"]

        if wandb.run is not None:
            self.save_name = wandb.config["name"]
        else:
            raise Exception("Initialize wandb first to call Trainer class!! Call init_wandb() first")
        
        # 1. Load snapshot to resume previous training
        if train_cfg["Resume"]["resume_training"] is True:
            self.epochs_run = train_cfg["Resume"]["resume_epoch"]
            self._load_snapshot()

        if train_cfg["freeze_pretrain"]:
            self.trainable_params = self.model.freeze_pretrained_modules()

        # self.model = DDP(self.model, device_ids=[self.gpu_id], find_unused_parameters=True)
        
        # 2. Creating the output directory & checkpoint directory if doesn't exist
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        if not os.path.exists(self.ckpt_dir):
            os.mkdir(self.ckpt_dir)
        if not os.path.exists(self.output_heatmap_dir):
            os.mkdir(self.output_heatmap_dir)

        # 3. Creating Loss functions
        self.loss_functions = create_loss_func(self.loss_types, self.imbalance_weight)

        # 4. Create optimizer and scheduler (transferred to current device)
        self.optimizer, self.scheduler = create_optimizer_and_scheduler(
            param_groups=self.model.parameters(),
            optimizer_name=train_cfg["Optimizer"]["name"],
            optimizer_kwargs=train_cfg["Optimizer"]["kwargs"],
            scheduler_name=train_cfg["Scheduler"]["name"],
            scheduler_kwargs=train_cfg["Scheduler"]["kwargs"],
            )
        
        
    def _load_snapshot(self,):
        snapshot = torch.load(os.path.join(self.ckpt_dir, f"checkpoint_{self.epochs_run}.pth"), map_location=f"cuda:{self.gpu_id}")
        
        assert self.epochs_run == snapshot["epoch"], "Checkpoint Epoch doesn't match with the Snapshot's Epoch!!"

        self.model.load_state_dict(snapshot["model"])

    def compute_loss(self, v_dict):
        pred_occ = v_dict["output_occupancy_map"]
        target_occ = v_dict["occupancy_label"]
        pred = v_dict["output_heatmap"]
        target = v_dict["heatmap_label"]
        _, H, W = pred.size()
        
        if self.smooth_eps:
            target_occ = target_occ.clamp(self.smooth_eps / 2, 1 - self.smooth_eps / 2)
            target = target.clamp(self.smooth_eps / 2, 1 - self.smooth_eps / 2)

        loss = 0

        for loss_func in self.loss_functions:
            loss += loss_func(pred_occ, target_occ) / (H*W)
            loss += loss_func(pred, target) / (H*W)

        return loss
    
    def single_step(self, v_dict, l_dict):
        # 1. Send Data dictionaries to current device
        new_v_dict, new_l_dict = {}, {}
        for key, data in v_dict.items():
            new_v_dict[key] = data.to(self.gpu_id)
        v_dict.update(new_v_dict)

        for key, data in l_dict.items():
            new_l_dict[key] = data.to(self.gpu_id)
        l_dict.update(new_l_dict)

        # 2. Encode text descriptions
        l_dict = self.text_encoder(l_dict)
        
        # 3. Forward operation and loss computation
        v_dict, l_dict = self.model(v_dict, l_dict)
        loss = self.compute_loss(v_dict)

        v_dict["loss"] = loss

        return v_dict, l_dict

    def train(self, cur_epoch: int, 
              train_loader: DataLoader, 
              log: Log):

        self.model.train()

        loss = 0
        occ_all_acc1, occ_all_acc3, occ_all_acc5 = [], [], []
        all_acc1, all_acc3, all_acc5 = [], [], []

        table = wandb.Table(
            columns=["Index", "Occupancy Prediction", "Occupancy Ground Truth", "Prediction", "Ground Truth"]
        )

        for i, (vision_dict, lang_dict) in enumerate(train_loader):
            # 0. ZERO GRAD optimizer
            self.optimizer.zero_grad()

            # 1. Single forward pass
            vision_dict, lang_dict = self.single_step(vision_dict, lang_dict)
            cur_loss = vision_dict["loss"]

            # 4. Back propagation & Optimizer Step
            cur_loss.backward()
            nn.utils.clip_grad_norm_(self.trainable_params, self.clip_grad_max_norm)
            self.optimizer.step()

            loss += cur_loss.item()

            if i % self.print_freq == 0:
                log.write(f"{i}-th iteration :  Loss {cur_loss.item()}")

            # 5. Scheduler step
            self.scheduler.step()

            # 2. Prediction and Save heatmap prediction images
            if i == len(train_loader)-1:
                save_dir = os.path.join(self.output_heatmap_dir, f"train_epoch_{cur_epoch}/sample_{i}")
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)

                acc_1, acc_3, acc_5 = accuracy_and_save_img(vision_dict, lang_dict, 
                                                            top_k=[1,3,5], 
                                                            save_heatmap_img=True, 
                                                            table=table, 
                                                            local_save_dir=save_dir, 
                                                            run_type="Train")
            else:
                acc_1, acc_3, acc_5 = accuracy_and_save_img(vision_dict, lang_dict, 
                                                            top_k=[1,3,5], 
                                                            save_heatmap_img=False)
            occ_all_acc1.append(acc_1[0])
            occ_all_acc3.append(acc_3[0])
            occ_all_acc5.append(acc_5[0])
            all_acc1.append(acc_1[1])
            all_acc3.append(acc_3[1])
            all_acc5.append(acc_5[1])

        # 6. Compute total loss
        loss /= len(train_loader.dataset)

        occ_all_acc1 = sum(occ_all_acc1) / len(train_loader.dataset)
        occ_all_acc3 = sum(occ_all_acc3) / len(train_loader.dataset)
        occ_all_acc5 = sum(occ_all_acc5) / len(train_loader.dataset)
        all_acc1 = sum(all_acc1) / len(train_loader.dataset)
        all_acc3 = sum(all_acc3) / len(train_loader.dataset)
        all_acc5 = sum(all_acc5) / len(train_loader.dataset)

        return loss, [occ_all_acc1, occ_all_acc3, occ_all_acc5], [all_acc1, all_acc3, all_acc5], table
        
    @torch.no_grad()
    def evaluate(self, cur_epoch: int,
                 val_loader: DataLoader):

        self.model.eval()

        loss = 0
        occ_all_acc1, occ_all_acc3, occ_all_acc5 = [], [], []
        all_acc1, all_acc3, all_acc5 = [], [], []

        table = wandb.Table(
            columns=["Index", "Occupancy Prediction", "Occupancy Ground Truth", "Prediction", "Ground Truth"]
        )

        for i, (vision_dict, lang_dict) in enumerate(val_loader):
            # 1. Single forward pass
            vision_dict, lang_dict = self.single_step(vision_dict, lang_dict)
            cur_loss = vision_dict["loss"]
            loss += cur_loss.item()

            # 2. Prediction and Save heatmap prediction images
            save_dir = os.path.join(self.output_heatmap_dir, f"eval_epoch_{cur_epoch}/sample_{i}")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)

            acc_1, acc_3, acc_5 = accuracy_and_save_img(vision_dict, lang_dict, 
                                                        top_k=[1,3,5], 
                                                        save_heatmap_img=True, 
                                                        table=table, 
                                                        local_save_dir=save_dir, 
                                                        run_type="Val")
            occ_all_acc1.append(acc_1[0])
            occ_all_acc3.append(acc_3[0])
            occ_all_acc5.append(acc_5[0])
            all_acc1.append(acc_1[1])
            all_acc3.append(acc_3[1])
            all_acc5.append(acc_5[1])

        # 3. Compute total loss
        loss /= len(val_loader.dataset)

        occ_all_acc1 = sum(occ_all_acc1) / len(val_loader.dataset)
        occ_all_acc3 = sum(occ_all_acc3) / len(val_loader.dataset)
        occ_all_acc5 = sum(occ_all_acc5) / len(val_loader.dataset)
        all_acc1 = sum(all_acc1) / len(val_loader.dataset)
        all_acc3 = sum(all_acc3) / len(val_loader.dataset)
        all_acc5 = sum(all_acc5) / len(val_loader.dataset)

        return loss, [occ_all_acc1, occ_all_acc3, occ_all_acc5], [all_acc1, all_acc3, all_acc5], table
    
    def _save_snapshot(self, 
                      cur_epoch: int,
                      losses: list,
                      accuracies: list,
                      occ_accuracies: list):
        snapshot = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            # 'scheduler': self.scheduler.state_dict(),
            'config': self.model_cfg,
            'epoch': cur_epoch,
            'losses': losses,
            'accuracies': accuracies,
            'occupancy_accuracies': occ_accuracies
        }
        torch.save(snapshot, os.path.join(self.ckpt_dir, f"checkpoint_{cur_epoch}.pth"))

    def _write_log(self, start_time: float,
                  log: Log,
                  cur_epoch: int,
                  train_loss,
                  train_accuracies: list,
                  train_occ_accuracies: list,
                  val_loss,
                  val_accuracies: list,
                  val_occ_accuracies: list):
        elapsed = time.time() - start_time

        log.write(f"Epoch {cur_epoch} : {elapsed/60:.4f} minutes")

        log.write(f"Training Loss : {train_loss},  Acc@1 : {train_accuracies[0]}, Acc@3 : {train_accuracies[1]}, Acc@5 : {train_accuracies[2]}")
        log.write(f"Heatmap Acc@1 : {train_accuracies[0]}, Heatmap Acc@3 : {train_accuracies[1]}, Heatmap Acc@5 : {train_accuracies[2]}")
        log.write(f"Occupancy Map Acc@1 : {train_occ_accuracies[0]}, Occupancy Map Acc@3 : {train_occ_accuracies[1]}, Occupancy Map Acc@5 : {train_occ_accuracies[2]}")

        log.write(f"Validation Loss : {val_loss},  Acc@1 : {val_accuracies[0]}, Acc@3 : {val_accuracies[1]}, Acc@5 : {val_accuracies[2]}")
        log.write(f"Heatmap Acc@1 : {val_accuracies[0]}, Heatmap Acc@3 : {val_accuracies[1]}, Heatmap Acc@5 : {val_accuracies[2]}")
        log.write(f"Occupancy Map Acc@1 : {val_occ_accuracies[0]}, Occupancy Map Acc@3 : {val_occ_accuracies[1]}, Occupancy Map Acc@5 : {val_occ_accuracies[2]}")

    def _log_wandb(self,
                  cur_epoch: int,
                  train_loss,
                  train_accuracies: list,
                  train_occ_accuracies: list,
                  train_table: wandb.Table,
                  val_loss,
                  val_accuracies: list,
                  val_occ_accuracies: list,
                  val_table: wandb.Table,):
        wandb_log = {
            "epoch": cur_epoch,
            "train_loss": train_loss,
            "train_occ_accuracy": train_occ_accuracies[0],
            "train_accuracy": train_accuracies[0],
            "train_heatmaps": train_table,
            "validation_loss": val_loss,
            "validation_occ_accuracy": val_occ_accuracies[0],
            "validation_accuracy": val_accuracies[0],
            "validation_heatmaps": val_table
        }
        wandb.log(wandb_log)

    def save_and_log(self,
                     cur_epoch: int, 
                     losses: list,
                     accuracies: list,
                     occ_accuracies: list,
                     tables: list,
                     start_time: float,
                     log: Log,
                     ):
        
        # 1. Save the trained model and other related data via snapshot
        if self.gpu_id == 0:
            self._save_snapshot(cur_epoch=cur_epoch,
                                losses=losses,
                                accuracies=accuracies,
                                occ_accuracies=occ_accuracies)
        
            # 2. Save training metrics to log file
            self._write_log(start_time=start_time,
                        log=log,
                        cur_epoch=cur_epoch,
                        train_loss=losses[0],
                        train_accuracies=accuracies[0],
                        train_occ_accuracies=occ_accuracies[0],
                        val_loss=losses[1],
                        val_accuracies=accuracies[1],
                        val_occ_accuracies=occ_accuracies[0])
            
            # 3. Log metrics to wandb to track the training process
            self._log_wandb(cur_epoch=cur_epoch,
                        train_loss=losses[0],
                        train_accuracies=accuracies[0],
                        train_occ_accuracies=occ_accuracies[0],
                        train_table=tables[0],
                        val_loss=losses[1],
                        val_accuracies=accuracies[1],
                        val_occ_accuracies=occ_accuracies[0],
                        val_table=tables[1])


    def fit(self,
            train_loader: DataLoader,
            val_loader: DataLoader, 
            log: Log):
        """
        Train and validate the model
        """
        start_time = time.time()
        log.write(f"Start Training Model...")

        for epoch in tqdm(range(self.epochs_run, self.max_epochs), desc="Model Training Loop"):
            epoch_start = time.time()

            # 1. Train and Validate the model
            # train_loader.sampler.set_epoch(epoch)
            # val_loader.sampler.set_epoch(epoch)
            train_loss, train_occ_accuracies, train_accuracies, train_table = self.train(epoch, train_loader, log)
            val_loss, val_occ_accuracies, val_accuracies, val_table = self.evaluate(epoch, val_loader)
            
            # 2. Save model snapshot and log metrics (to .txt and wandb)
            self.save_and_log(cur_epoch=epoch,
                              losses=[train_loss, val_loss],
                              accuracies=[train_accuracies, val_accuracies],
                              occ_accuracies=[train_occ_accuracies, val_occ_accuracies],
                              tables=[train_table, val_table],
                              start_time=epoch_start,
                              log=log)
            
            log.write(f"Epoch {epoch} Done")

        total_elapsed = time.time() - start_time
        msg = f"Finished Training - time elapsed {total_elapsed/60:.4f} minutes"
        log.write(msg)