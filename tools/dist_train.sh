#!/usr/bin/env bash

set -euo pipefail

CONFIG=$1
GPUS=$2

PYTHONPATH="$(dirname "$0")/..:${PYTHONPATH:-}" torchrun \
    --nnodes="${NNODES:-1}" \
    --node-rank="${NODE_RANK:-0}" \
    --master-addr="${MASTER_ADDR:-127.0.0.1}" \
    --master-port="${PORT:-29500}" \
    --nproc-per-node="$GPUS" \
    "$(dirname "$0")/train.py" "$CONFIG" --launcher pytorch "${@:3}"
