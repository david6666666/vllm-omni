#!/bin/bash
# Qwen-Image online serving startup script

MODEL="${MODEL:-Qwen/Qwen-Image}"
PORT="${PORT:-8091}"
QUANTIZATION="${QUANTIZATION:-}"
LOAD_FORMAT="${LOAD_FORMAT:-auto}"

echo "Starting Qwen-Image server..."
echo "Model: $MODEL"
echo "Port: $PORT"
if [ -n "$QUANTIZATION" ]; then
  echo "Quantization: $QUANTIZATION"
fi
echo "Load format: $LOAD_FORMAT"

CMD=(vllm serve "$MODEL" --omni --port "$PORT" --load-format "$LOAD_FORMAT")
if [ -n "$QUANTIZATION" ]; then
  CMD+=(--quantization "$QUANTIZATION")
fi

"${CMD[@]}"
