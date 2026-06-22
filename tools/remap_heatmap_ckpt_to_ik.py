#!/usr/bin/env python3
"""Remap CNNHeatmapHead checkpoint keys for HeatmapIKHead.

This script converts a checkpoint trained with:
    model.head = CNNHeatmapHead
to a checkpoint compatible with:
    model.head = HeatmapIKHead(heatmap_head=..., ik_head=...)

Default behavior:
    head.xxx -> head.heatmap_head.xxx
"""

from __future__ import annotations

import argparse
import os
import pickle
from collections import OrderedDict
from typing import Dict

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Remap checkpoint keys from CNNHeatmapHead to HeatmapIKHead.')
    parser.add_argument('src', help='Source checkpoint path.')
    parser.add_argument('dst', help='Output checkpoint path.')
    parser.add_argument(
        '--src-prefix',
        default='head.',
        help='Prefix to match in source state_dict. Default: head.')
    parser.add_argument(
        '--dst-prefix',
        default='head.heatmap_head.',
        help='Replacement prefix in destination state_dict. '
             'Default: head.heatmap_head.')
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite destination if it already exists.')
    parser.add_argument(
        '--map-all-head',
        action='store_true',
        help='Map all keys under src-prefix. Default behavior is already this; '
             'kept for explicitness in command history.')
    parser.add_argument(
        '--trusted-source',
        action='store_true',
        help='Allow loading full checkpoint metadata with pickle. Required for '
             'MMEngine checkpoints on PyTorch 2.6+ when weights_only loading '
             'rejects objects such as HistoryBuffer. Only use for checkpoints '
             'from a trusted source.')
    parser.add_argument(
        '--keep-checkpoint-state',
        action='store_true',
        help='Keep resume-only runner state such as optimizer, scheduler, and '
             'message_hub. By default these are stripped so the output can be '
             'used with load_from under PyTorch 2.6+ weights_only loading.')
    return parser.parse_args()


def remap_state_dict(state_dict: Dict[str, torch.Tensor],
                     src_prefix: str,
                     dst_prefix: str) -> tuple[OrderedDict, int]:
    remapped = OrderedDict()
    mapped_count = 0
    for key, value in state_dict.items():
        if key.startswith(src_prefix):
            new_key = dst_prefix + key[len(src_prefix):]
            mapped_count += 1
        else:
            new_key = key
        remapped[new_key] = value
    return remapped, mapped_count


def _safe_torch_load(path: str, trusted_source: bool = False):
    """Load checkpoint in the same runtime env as training (mmengine present)."""
    try:
        if trusted_source:
            return torch.load(path, map_location='cpu', weights_only=False)
        return torch.load(path, map_location='cpu')
    except pickle.UnpicklingError as exc:
        raise RuntimeError(
            'Failed to load checkpoint with PyTorch weights_only mode. '
            'PyTorch 2.6+ defaults torch.load(weights_only=True), which can '
            'reject MMEngine checkpoint metadata such as HistoryBuffer. If this '
            'checkpoint was produced by your own training run or another trusted '
            'source, re-run with --trusted-source to load it with '
            'weights_only=False.'
        ) from exc
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            'Failed to load checkpoint due to missing dependency module. '
            'Please run this script in the same conda/env used for training '
            '(where mmengine/mmpose are installed). '
            f'Original error: {exc}'
        ) from exc


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.src):
        raise FileNotFoundError(f'Source checkpoint not found: {args.src}')
    if os.path.exists(args.dst) and not args.overwrite:
        raise FileExistsError(
            f'Destination exists: {args.dst}. Use --overwrite to replace it.')

    ckpt = _safe_torch_load(args.src, trusted_source=args.trusted_source)
    if not isinstance(ckpt, dict):
        raise TypeError('Checkpoint must be a dict-like object.')
    if 'state_dict' not in ckpt:
        raise KeyError('Checkpoint does not contain key "state_dict".')

    old_state = ckpt['state_dict']
    if not isinstance(old_state, dict):
        raise TypeError('checkpoint["state_dict"] must be dict-like.')

    new_state, mapped_count = remap_state_dict(
        old_state, args.src_prefix, args.dst_prefix)
    ckpt['state_dict'] = new_state
    if not args.keep_checkpoint_state:
        for key in ('message_hub', 'optimizer', 'optim_wrapper',
                    'param_schedulers', 'scheduler'):
            ckpt.pop(key, None)

    os.makedirs(os.path.dirname(args.dst) or '.', exist_ok=True)
    torch.save(ckpt, args.dst)

    total = len(old_state)
    print(f'Saved remapped checkpoint to: {args.dst}')
    print(f'Total keys: {total}')
    print(f'Mapped keys ({args.src_prefix} -> {args.dst_prefix}): {mapped_count}')
    print(f'Unchanged keys: {total - mapped_count}')


if __name__ == '__main__':
    main()
