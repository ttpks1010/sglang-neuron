#!/bin/bash
# Muse Glimmer 30B (text) on one trn2.3xlarge, aggregated (prefill and decode in one server), TP4 on four NeuronCores.
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/models/Muse-Glimmer-30B}
# 127.0.0.1: clients on this host only; give an address of the node for clients elsewhere, reachable only by them
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-30000}
# the longest prompt plus answer; every decode graph compiles over it, so a shorter one compiles faster
CONTEXT_LENGTH=${CONTEXT_LENGTH:-16384}
# a persistent compile cache: the same path on every start, one directory per image version
CACHE_DIR=${CACHE_DIR:-/cache}
export TORCH_NEURONX_NEFF_CACHE_DIR=$CACHE_DIR/neff TORCH_NEURONX_HLO_CACHE_DIR=$CACHE_DIR/hlo \
    NKI_COMPILE_CACHE_URL=$CACHE_DIR/nki

# --language-model-only: SGLang's own text-only switch for this architecture (the model's launch rules refuse a launch
# without it): no vision tower, no multimodal processor, media requests refused. --reasoning-parser muse: the model
# writes its reasoning in a separate channel before the answer; the parser puts it in reasoning_content.
exec python3 -m sglang.launch_server --model-path "$MODEL_PATH" \
    --language-model-only --reasoning-parser muse \
    --tp-size 4 --context-length "$CONTEXT_LENGTH" --chunked-prefill-size 2048 \
    --max-running-requests 8 --max-total-tokens 32768 --mem-fraction-static 0.9 \
    --host "$HOST" --port "$PORT" "$@"
