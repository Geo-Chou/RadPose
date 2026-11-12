import argparse
import os

import torch
from mmengine.config import Config, DictAction
from mmengine.runner import Runner

original_torch_load = torch.load


def load_checkpoint(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return original_torch_load(*args, **kwargs)


torch.load = load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--work-dir")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", nargs="?", const="auto")
    parser.add_argument(
        "--launcher", choices=["none", "pytorch", "slurm", "mpi"], default="none"
    )
    parser.add_argument("--cfg-options", nargs="+", action=DictAction)
    parser.add_argument("--local-rank", type=int, default=0)
    args = parser.parse_args()
    os.environ.setdefault("LOCAL_RANK", str(args.local_rank))
    return args


def main():
    args = parse_args()
    config = Config.fromfile(args.config)
    config.launcher = args.launcher
    if args.cfg_options:
        config.merge_from_dict(args.cfg_options)
    if args.work_dir:
        config.work_dir = args.work_dir
    if args.amp:
        config.optim_wrapper.type = "AmpOptimWrapper"
        config.optim_wrapper.loss_scale = "dynamic"
    if args.resume:
        config.resume = True
        config.load_from = None if args.resume == "auto" else args.resume
    Runner.from_cfg(config).train()


if __name__ == "__main__":
    main()
