#!/bin/bash
# Qwen-Image-Edit online serving startup script

MODEL="${MODEL:-Qwen/Qwen-Image-Edit}"
PORT="${PORT:-8092}"
QUANTIZATION="${QUANTIZATION:-}"
IGNORED_LAYERS="${IGNORED_LAYERS:-}"
EXTRA_FLAGS=()

if [[ -n "$QUANTIZATION" ]]; then
    EXTRA_FLAGS+=(--quantization "$QUANTIZATION")
fi

if [[ -n "$IGNORED_LAYERS" ]]; then
    EXTRA_FLAGS+=(--ignored-layers "$IGNORED_LAYERS")
fi

echo "Starting Qwen-Image-Edit server..."
echo "Model: $MODEL"
echo "Port: $PORT"
if [[ -n "$QUANTIZATION" ]]; then
    echo "Quantization: $QUANTIZATION"
fi
if [[ -n "$IGNORED_LAYERS" ]]; then
    echo "Ignored layers: $IGNORED_LAYERS"
fi

vllm serve "$MODEL" --omni \
    --port "$PORT" \
    "${EXTRA_FLAGS[@]}"
