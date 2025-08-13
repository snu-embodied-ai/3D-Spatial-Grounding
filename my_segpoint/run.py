import os
from datetime import datetime
import json
from pathlib import Path
import argparse

# import hydra
from accelerate.logging import get_logger
from omegaconf import OmegaConf

# import common.io_utils as iu
# from misc import rgetattr
from trainer.trainer import SegPointTrainer

logger = get_logger(__name__)


def main(cfg):
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'   # suppress hf tokenizer warning
    # naming_keys = [cfg.name]
    # for name in cfg.naming_keywords:
    #     key = str(rgetattr(cfg, name))
    #     if key:
    #         naming_keys.append(key)
    exp_name = "SegPoint"

    # Record the experiment
    cfg.exp_dir = os.path.join(
        cfg.base_dir, exp_name,
        f"{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}" if 'time' in cfg.naming_keywords else ""
    )
    os.makedirs(cfg.exp_dir, exist_ok=True)

    with Path(os.path.join(cfg.exp_dir, 'config.json')).open("w") as f:
        json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=2)

    trainer = SegPointTrainer(cfg)
    trainer.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer_cfg", default="cfg/trainer.yaml", type=str)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.trainer_cfg)
    main(cfg)
