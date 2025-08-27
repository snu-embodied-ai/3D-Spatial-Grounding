import os
from datetime import timedelta
from math import ceil
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import InitProcessGroupKwargs, ProjectConfiguration, set_seed
from omegaconf import OmegaConf, DictConfig
from tqdm import trange

from data.datasets import SemanticSegmentationDataset, ReferringSegmentationDataset, Instruct3DSegmentationDataset, build_dataloaders
from model.segpoint import SegPoint
from misc.accelerator import CustomAccelerator
from trainer.build import Tracker, latest_checkpoint, build_optim
from trainer.metrics import compute_mIoU_binary_groups
from trainer.util import split_by_SEG, pad_sequences

from data.constants.dataset_consts import DATASETS
from data.constants.scannet import CLASS_LABELS as SCANNET_CLASSES, VALID_CLASS_ID_TO_LABEL as SCANNET_ID_TO_LABEL

# TODO: STUDY LOGGER
logger = get_logger(__name__)

model_parallel_classes = (
    nn.parallel.DistributedDataParallel,
    nn.DataParallel,
)

class SegPointTrainer:
    def __init__(self,
                 cfg: DictConfig):
        set_seed(cfg.rng_seed)
        self.exp_dir = cfg.exp_dir
        self.mode = cfg.mode

        # initialize accelerator
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=800))
        kwargs = ([ddp_kwargs] if cfg.num_gpu > 1 else []) + [init_kwargs]
        gradient_accumulation_steps = cfg.train.get('gradient_accumulation_steps', 1)

        # TODO: ACCELERATOR
        self.accelerator = CustomAccelerator(
            project_config=ProjectConfiguration(
                project_dir=self.exp_dir,
                automatic_checkpoint_naming=True,
                total_limit=1,
            ),
            gradient_accumulation_steps=gradient_accumulation_steps,
            log_with=cfg.logger.name,
            kwargs_handlers=kwargs
        )

        # 1. Generate Dataset and Dataloader
        self.dataset_cfg = OmegaConf.load(cfg.dataset_config_path)
        self.instruction_cfg = OmegaConf.load(cfg.instructions_config_path)

        self.train_loaders = []
        self.val_loaders = []
        self.test_loaders = []
        num_steps_per_epoch = 0

        for task in cfg.tasks:
            for task_type in cfg.task_type:
                loader_dict = build_dataloaders(task, task_type, cfg)

                train_loader = self.accelerator.prepare(loader_dict['train'])
                val_loader = self.accelerator.prepare(loader_dict['val'])
                test_loader = self.accelerator.prepare(loader_dict['test'])
                
                self.train_loaders.append((task, task_type, train_loader))
                self.val_loaders.append((task, task_type, val_loader))
                self.test_loaders.append((task, task_type, test_loader))

                num_steps_per_epoch += 1

        # 2. Build model
        self.model_cfg = OmegaConf.load(cfg.model_config_path)
        self.model = SegPoint(self.model_cfg, self.instruction_cfg)

        # 3. Get learnable parameters for building the optimizer
        learnable_named_params = self.model.get_learnable_named_params()
        self.accelerator.learn_params_list = list(learnable_named_params.keys())
        optim_params = list(learnable_named_params.values())

        # 4. Get number of total steps and build Optimizer and Scheduler
        total_steps = ceil(num_steps_per_epoch / gradient_accumulation_steps) * cfg.train.epochs

        self.optimizer, self.scheduler = build_optim(cfg, optim_params, total_steps=total_steps)
        
        # 5. Prepare accelerator
        self.model, self.optimizer, self.scheduler = self.accelerator.prepare(self.model, self.optimizer, self.scheduler)

        # 6. Build Tracker and load checkpoints
        # TODO: Fix the Tracker class or fix my configs as LEO
        self.exp_tracker = Tracker(cfg)
        self.accelerator.register_for_checkpointing(self.exp_tracker)

        # load checkpoints
        self.load_checkpoints(cfg.pretrained_ckpt_path)

        # 7.. OTHER
        self.epochs = cfg.train.epochs
        self.grad_norm = cfg.train.grad_norm
        self.val_interval = cfg.eval.val_interval
        self.num_batch_val = cfg.eval.num_batch_val

        # TODO: ACCELERATOR
        self.accelerator.init_trackers(
            project_name=cfg.name,
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            init_kwargs={
                'wandb': {
                    'name': self.exp_tracker.exp_name, 'entity': cfg.logger.entity,
                    'id': self.exp_tracker.run_id, 'resume': True
                }
            }
        )

    def load(self, path: str, model_only: bool = False):
        if model_only:
            model_state_dict = torch.load(os.path.join(path, 'pytorch_model.bin'))
            if isinstance(self.model, model_parallel_classes):
                self.model.module.load_state_dict(model_state_dict, strict=False)
            else:
                self.model.load_state_dict(model_state_dict, strict=False)
        else:
            # resume training
            self.accelerator.load_state(path, strict=False)
            self.accelerator.project_configuration.iteration = int(str(path)[-1]) + 1
        logger.info(f"Successfully loaded from {str(path)}, load_model_only = {model_only}")

    def load_checkpoints(self,
                         pretrained_ckpt_path: str):
        # load checkpoints
        resume_ckpt = latest_checkpoint(os.path.join(self.exp_dir, 'checkpoints'))
        self_best_ckpt = os.path.join(self.exp_dir, 'best.pth')

        if self.mode == 'train':
            if resume_ckpt:
                load_model_only = False
                self.pretrained_ckpt_path = resume_ckpt
                logger.info(f"Train: resume and load state from {self.pretrained_ckpt_path}")
            elif pretrained_ckpt_path and os.path.exists(pretrained_ckpt_path):
                load_model_only = True
                self.pretrained_ckpt_path = pretrained_ckpt_path
                logger.info(f"Train: start and load model from {self.pretrained_ckpt_path}")
            else:
                self.pretrained_ckpt_path = None
                logger.info("Train: start from scratch")

        else:
            if os.path.exists(self_best_ckpt):
                self.pretrained_ckpt_path = self_best_ckpt
            elif pretrained_ckpt_path and os.path.exists(pretrained_ckpt_path):
                self.pretrained_ckpt_path = pretrained_ckpt_path
            else:
                raise ValueError("No checkpoint to load for evaluation")
            load_model_only = True
            logger.info(f"Eval: load model from {self.pretrained_ckpt_path}")

        if self.pretrained_ckpt_path is not None:
            self.load(path=self.pretrained_ckpt_path, model_only=load_model_only)

    
    def forward(self, data_dict, inference=False):
        if data_dict["dataset_idx"] == 0:
            # ScanNet
            data_dict["category_mapping"] = SCANNET_ID_TO_LABEL
        else:
            # OTHER
            raise Exception("NOT IMPLEMENTED OTHER DATASETS inside forward() of trainer")
        
        if inference:
            if isinstance(self.model, model_parallel_classes):
                return self.model.module.generate(data_dict)
            else:
                return self.model.generate(data_dict)
        else:
            return self.model(data_dict)
    
    def backward(self, loss):
        self.optimizer.zero_grad()
        # TODO: ACCELERATOR
        self.accelerator.backward(loss)
        if self.grad_norm is not None and self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.grad_norm)
        self.optimizer.step()
        self.scheduler.step()

    def evaluate(self,
                 data_dict: dict,):
        """
        Parameters
        ---
        data_dict: dict
            Dictionary containing all data
            In this function, following keys are required:
            - `gt_text` (list): List of ground truth text sequences (only answers)
            - `gen_answers` (list) : List of generated text sequences (only answers)
            - `mask` (torch.Tensor) : Ground truth segmentation mask for all categories. Shape : (B, num_categories, num_points)
            - `output_mask` (torch.Tensor) : Generated segmentation mask for all categories the model predicted. Shape : (B, num_pred_segs, num_points)
            - `valid_output_mask_indices` (torch.Tensor): Indices of valid segments. 1 indicates valid segments and 0 indicates padding.
            - `dataset_idx` (int): Integer indicating the dataset name

        Returns
        ---

        """
        B, num_pred_segs, num_points = data_dict["output_mask"].size()
        _, num_cats, _ = data_dict["mask"].size()

        valid_output_mask_indices = data_dict["valid_output_mask_indices"]      # (B, num_output_seg_tokens)

        # ScanNet
        if data_dict["dataset_idx"] == 0:
            valid_class_labels = sorted(SCANNET_CLASSES)
        # Others (Add later)
        else:
            valid_class_labels = None

        batch_valid_seg_output = []
        batch_valid_seg_gt = []
        num_matches = torch.zeros(B)

        # 1. Compare ground truth text and output texts
        print("gen answers: ", data_dict["gen_answers"])
        print("gt text: ", data_dict["gt_text"])
        output_split = split_by_SEG(data_dict["gen_answers"])
        gt_split = split_by_SEG(data_dict["gt_text"])

        print("output_splits:", output_split)
        print("gt_splits:", gt_split)

        for batch_idx, split in enumerate(output_split):
            valid_seg_output = []
            valid_seg_gt = []

            for i in range(len(split)):
                output = split[i]
                if output in valid_class_labels:
                    gt = gt_split[batch_idx]

                    if output in gt:
                        valid_seg_output.append(data_dict["output_mask"][batch_idx, i])
                        valid_seg_gt.append(data_dict["mask"][batch_idx, gt.find(output)])
                        num_matches[batch_idx, i] += 1

            if len(valid_seg_output) == 0:
                out_stacked = torch.zeros((1, num_points))
                gt_stacked = torch.ones_like(out_stacked)
            else:
                out_stacked = torch.stack(valid_seg_output)
                gt_stacked = torch.stack(valid_seg_gt)

            batch_valid_seg_output.append(out_stacked)
            batch_valid_seg_gt.append(gt_stacked)

        padded_seg_output, padding_mask = pad_sequences(batch_valid_seg_output)
        padded_seg_gt, _ = pad_sequences(batch_valid_seg_gt)

        batch_mIoU, IoU = compute_mIoU_binary_groups(padded_seg_output, padded_seg_gt, padding_mask)

        # Compute Accuracy
        accurate_pred = IoU > 0.5                           # (B, G')
        accuracy = accurate_pred.sum(dim=1) / num_cats      # (B, )

        return batch_mIoU, accuracy
    

    def train_step(self, epoch, loader):
        logger.info(f"Start training epoch {epoch+1}")
        self.model.train()

        # IMPORTANT: make shuffling different per epoch across ranks
        if hasattr(loader, "sampler") and hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)
        pbar = trange(len(loader), disable=(not self.accelerator.is_main_process))

        if self.exp_tracker.loader_step > 0:
            logger.info(f"Skip the first {self.exp_tracker.loader_step} batches")
            loader_iter = iter(loader)
            for _ in range(self.exp_tracker.loader_step):
                next(loader_iter)
            pbar.update(self.exp_tracker.loader_step)
        else:
            loader_iter = loader

        for data_dict in loader_iter:
            with self.accelerator.accumulate(self.model):
                # 1. Forward (forward())
                data_dict = self.forward(data_dict, inference=False)

                # 2. Calculate loss and optimize
                loss = data_dict['loss']        # all losses are in shape of (batch_size,)
                loss_all = loss.mean()
                self.backward(loss_all)

                # 3. record
                # Save PLY for the training outputs..?
                loss_dict = {'overall': loss_all}
                self.log(loss_dict, mode='train', task='loss')
                self.exp_tracker.step_loader()
                pbar.update(1)

        logger.info(f"Finish training epoch {epoch+1}")

    @torch.no_grad
    def val_step(self, epoch, loader):
        logger.info(f"Start validation epoch {epoch+1}")
        self.model.eval()

        all_mIoU = 0
        all_acc = 0
        total_len = len(loader.dataset)

        pbar = trange(len(loader), disable=(not self.accelerator.is_main_process))

        for i, data_dict in enumerate(loader):
            if i >= self.num_batch_val:
                break
           
            # 1. Inference (generate())
            data_dict = self.forward(data_dict, inference=True)

            # 2. Gather for metrcis (among different devices)
            data_dict_non_tensor = {k: v for k, v in data_dict.items() if not isinstance(v, torch.Tensor)}
            data_dict_non_tensor = self.accelerator.gather_for_metrics(data_dict_non_tensor)
            # for k, v in data_dict_non_tensor.items():
            #     print(f"key : {k}")
            #     print(f"value : {v}")

            data_dict = {k: v for k, v in data_dict.items() if isinstance(v, torch.Tensor)}
            # for k, v in data_dict.items():
            #     print(f"key: {k}")
            #     print(f"value shape: {v.size()}")
            # data_dict = self.accelerator.gather_for_metrics(data_dict)
            data_dict.update(data_dict_non_tensor)

            # 3. Collect relevant outputs for metric and filter valid outputs
            batch_mIoU, accuracy = self.evaluate(data_dict)
            all_mIoU += batch_mIoU.sum().item()
            all_acc += accuracy.sum().item()
            
            # 4. Log batch metrics...?

            pbar.update(1)

        mIoU = all_mIoU / total_len
        mean_acc = all_acc / total_len

        results = {'mIoU': mIoU, 'accuracy': mean_acc}
        self.log(results, mode='val')
        logger.info(f"Validation: {results}")

        # simply summing up
        overall_avg_metrics = sum(list(results.values())) / len(results)
        self.log({'avg_metrics': overall_avg_metrics}, mode='val')
        if overall_avg_metrics > self.exp_tracker.overall_best_result:
            is_best = True
            self.exp_tracker.overall_best_result = overall_avg_metrics
        else:
            is_best = False
        logger.info(f"Finish validation epoch {epoch+1}, is_best = {is_best}")
        return is_best

    @torch.no_grad
    def test_step(self, loader, task, task_type):
        logger.info("Start final testing")
        self.model.eval()

        all_mIoU = 0
        all_acc = 0
        total_len = len(loader.dataset)

        pbar = trange(len(loader), disable=(not self.accelerator.is_main_process))

        for i, data_dict in enumerate(loader):
            if i >= self.num_batch_val:
                break

            # 1. Inference (generate())
            data_dict = self.forward(data_dict, inference=True)

            # 2. Gather for metrcis (among different devices)
            data_dict_non_tensor = {k: v for k, v in data_dict.items() if not isinstance(v, torch.Tensor)}
            data_dict_non_tensor = self.accelerator.gather_for_metrics(data_dict_non_tensor)

            data_dict = {k: v for k, v in data_dict.items() if isinstance(v, torch.Tensor)}
            data_dict = self.accelerator.gather_for_metrics(data_dict)
            data_dict.update(data_dict_non_tensor)

            # 3. Collect relevant outputs for metric and filter valid outputs
            batch_mIoU, accuracy = self.evaluate(data_dict)
            all_mIoU += batch_mIoU.sum().item()
            all_acc += accuracy.sum().item()
            
            # 4. Log batch metrics...?

            pbar.update(1)

        mIoU = all_mIoU / total_len
        mean_acc = all_acc / total_len
        
        results = {'task': task,
                   'task_type': task_type,
                   'mIoU': mIoU, 'accuracy': mean_acc}
        self.log(results, mode='test')
        logger.info(f"TEST for {task_type} {task}: {results}")
        # self.evaluators[task_name].reset()

        logger.info("Finish testing")

    """
    STUDY ACCELERATOR to use this!!
    """
    def log(self, results, mode='train', task='default'):
        log_dict = {}
        for key, val in results.items():
            log_dict[f'{mode}/{task}/{key}'] = val

        if mode == 'train':
            lrs = self.scheduler.get_lr()
            for i, lr in enumerate(lrs):
                log_dict[f'train/lr/group_{i}'] = lr

        self.accelerator.log(log_dict)

    def save(self, name='best.pth', model_only=False):
        if model_only:
            path = os.path.join(self.exp_dir, name)
            os.makedirs(path, exist_ok=True)
            model_state_dict = self.accelerator.get_state_dict(self.model)
            # automatically filter non-learnable params, and save on main_process
            self.accelerator.save(model_state_dict, os.path.join(path, 'pytorch_model.bin'))
        else:
            self.accelerator.save_state()   # automatic_checkpoint_naming = True -> self.exp_dir / checkpoints

    def run(self):
        if self.mode == 'train':
            start_epoch = self.exp_tracker.epoch
            for epoch in range(start_epoch, self.epochs):
                
                # 1. Training - train for all dataloaders (all task/task types)
                for i, (task, task_type, train_loader) in enumerate(self.train_loaders):
                    
                    # 1-1. Train steps
                    self.train_step(epoch, train_loader)

                    # 1-2. Validations
                    if (len(self.train_loaders) * epoch + i+1) % self.val_interval == 0:
                        _, _, val_loader = self.val_loaders[i]
                        is_best = self.val_step(epoch, val_loader)
                        if is_best:
                            self.save('best.pth', model_only=True)   # save the best checkpoint
                            self.accelerator.wait_for_everyone()

                self.exp_tracker.step()
                self.save(f"epoch_{epoch}.pth", model_only=False)   # automatic checkpointing
                self.accelerator.wait_for_everyone()

            # load best checkpoint for test
            logger.info("Training finished, load best checkpoint for testing")
            self.load(os.path.join(self.exp_dir, 'best.pth'), model_only=True)
        for i, (task, task_type, test_loader) in enumerate(self.test_loaders):
            self.test_step(test_loader, task, task_type)

        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()