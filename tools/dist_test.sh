#!/usr/bin/env bash

set -euo pipefail

CONFIG=$1
CHECKPOINT=$2
GPUS=$3

PYTHONPATH="$(dirname "$0")/..:${PYTHONPATH:-}" torchrun \
    --nnodes="${NNODES:-1}" \
    --node-rank="${NODE_RANK:-0}" \
    --master-addr="${MASTER_ADDR:-127.0.0.1}" \
    --master-port="${PORT:-29500}" \
    --nproc-per-node="$GPUS" \
    "$(dirname "$0")/test.py" "$CONFIG" "$CHECKPOINT" \
    --launcher pytorch "${@:4}"
