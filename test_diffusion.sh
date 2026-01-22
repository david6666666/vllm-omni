#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASSETS_DIR="${ROOT_DIR}/assets"
OUTPUT_DIR="${ROOT_DIR}/outputs"
SERVER_LOG_DIR="${OUTPUT_DIR}/server_logs"

mkdir -p "${ASSETS_DIR}" "${OUTPUT_DIR}" "${SERVER_LOG_DIR}"

start_server() {
  local model=$1
  local port=$2
  local log_file=$3
  vllm serve "${model}" --omni --port "${port}" > "${log_file}" 2>&1 &
  echo $!
}

wait_for_server() {
  local port=$1
  local pid=$2
  local retries=60
  local delay=2

  for _ in $(seq 1 "${retries}"); do
    if curl -sf "http://localhost:${port}/v1/health" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "${pid}" >/dev/null 2>&1; then
      return 1
    fi
    sleep "${delay}"
  done
  return 1
}

stop_server() {
  local pid=$1
  if kill -0 "${pid}" >/dev/null 2>&1; then
    kill "${pid}" >/dev/null 2>&1 || true
    wait "${pid}" >/dev/null 2>&1 || true
  fi
}

cleanup_pids=()
cleanup() {
  for pid in "${cleanup_pids[@]}"; do
    stop_server "${pid}"
  done
}
trap cleanup EXIT

# Example image assets for image editing / image-to-video
if [ ! -f "${ASSETS_DIR}/qwen-bear.png" ]; then
  curl -L -o "${ASSETS_DIR}/qwen-bear.png" \
    https://vllm-public-assets.s3.us-west-2.amazonaws.com/omni-assets/qwen-bear.png
fi

# ---------------------------
# Text-to-Image models
# ---------------------------
python examples/offline_inference/text_to_image/text_to_image.py \
  --model /workspace/models/ByteDance-Seed/BAGEL-7B-MoT \
  --prompt "a cup of coffee on the table" \
  --seed 42 \
  --cfg_scale 4.0 \
  --num_images_per_prompt 1 \
  --num_inference_steps 50 \
  --height 1024 \
  --width 1024 \
  --output "${OUTPUT_DIR}/bagel_coffee.png"

python examples/offline_inference/text_to_image/text_to_image.py \
  --model /workspace/models/Qwen/Qwen-Image \
  --prompt "a cup of coffee on the table" \
  --seed 42 \
  --cfg_scale 4.0 \
  --num_images_per_prompt 1 \
  --num_inference_steps 50 \
  --height 1024 \
  --width 1024 \
  --output "${OUTPUT_DIR}/qwen_image_coffee.png"

python examples/offline_inference/text_to_image/text_to_image.py \
  --model /workspace/models/Tongyi-MAI/Z-Image-Turbo \
  --prompt "a cup of coffee on the table" \
  --seed 42 \
  --cfg_scale 4.0 \
  --num_images_per_prompt 1 \
  --num_inference_steps 50 \
  --height 1024 \
  --width 1024 \
  --output "${OUTPUT_DIR}/zimage_coffee.png"

python examples/offline_inference/text_to_image/text_to_image.py \
  --model /workspace/models/AIDC-AI/Ovis-Image-7B \
  --prompt "a cup of coffee on the table" \
  --seed 42 \
  --cfg_scale 4.0 \
  --num_images_per_prompt 1 \
  --num_inference_steps 50 \
  --height 1024 \
  --width 1024 \
  --output "${OUTPUT_DIR}/ovis_image_coffee.png"

python examples/offline_inference/text_to_image/text_to_image.py \
  --model /workspace/models/meituan-longcat/LongCat-Image \
  --prompt "a cup of coffee on the table" \
  --seed 42 \
  --cfg_scale 4.0 \
  --num_images_per_prompt 1 \
  --num_inference_steps 50 \
  --height 1024 \
  --width 1024 \
  --output "${OUTPUT_DIR}/longcat_image_coffee.png"

python examples/offline_inference/text_to_image/text_to_image.py \
  --model /workspace/models/stabilityai/stable-diffusion-3.5-medium \
  --prompt "a cup of coffee on the table" \
  --seed 42 \
  --cfg_scale 4.0 \
  --num_images_per_prompt 1 \
  --num_inference_steps 50 \
  --height 1024 \
  --width 1024 \
  --output "${OUTPUT_DIR}/sd3_coffee.png"

python examples/offline_inference/text_to_image/text_to_image.py \
  --model /workspace/models/black-forest-labs/FLUX.2-klein-9B \
  --prompt "a cup of coffee on the table" \
  --seed 42 \
  --cfg_scale 4.0 \
  --num_images_per_prompt 1 \
  --num_inference_steps 50 \
  --height 1024 \
  --width 1024 \
  --output "${OUTPUT_DIR}/flux2_klein_9b_coffee.png"

# ---------------------------
# Image-Editing models
# ---------------------------
python examples/offline_inference/image_to_image/image_edit.py \
  --model /workspace/models/Qwen/Qwen-Image-Edit \
  --image "${ASSETS_DIR}/qwen-bear.png" \
  --prompt "Let this mascot dance under the moon, surrounded by floating stars and poetic bubbles such as 'Be Kind'" \
  --output "${OUTPUT_DIR}/qwen_image_edit.png" \
  --num_inference_steps 50 \
  --cfg_scale 4.0

python examples/offline_inference/image_to_image/image_edit.py \
  --model /workspace/models/Qwen/Qwen-Image-Edit-2509 \
  --image "${ASSETS_DIR}/qwen-bear.png" "${OUTPUT_DIR}/qwen_image_coffee.png" \
  --prompt "Combine these images into a single scene" \
  --output "${OUTPUT_DIR}/qwen_image_edit_2509.png" \
  --num_inference_steps 50 \
  --cfg_scale 4.0 \
  --guidance_scale 1.0

python examples/offline_inference/image_to_image/image_edit.py \
  --model /workspace/models/Qwen/Qwen-Image-Layered \
  --image "${OUTPUT_DIR}/qwen_image_edit.png" \
  --prompt "Decompose the image into layered RGBA outputs" \
  --output "${OUTPUT_DIR}/qwen_image_layered" \
  --num_inference_steps 50 \
  --cfg_scale 4.0 \
  --layers 4 \
  --color-format "RGBA"

python examples/offline_inference/image_to_image/image_edit.py \
  --model /workspace/models/meituan-longcat/LongCat-Image-Edit \
  --image "${ASSETS_DIR}/qwen-bear.png" \
  --prompt "Stylize this image into a colorful illustration" \
  --output "${OUTPUT_DIR}/longcat_image_edit.png" \
  --num_inference_steps 50 \
  --cfg_scale 4.0

python examples/offline_inference/image_to_image/image_edit.py \
  --model /workspace/models/black-forest-labs/FLUX.2-klein-9B \
  --image "${ASSETS_DIR}/qwen-bear.png" \
  --prompt "Stylize this image into a colorful illustration" \
  --output "${OUTPUT_DIR}/flux2_klein_9b_edit.png" \
  --num_inference_steps 50 \
  --cfg_scale 4.0

# ---------------------------
# Text-to-Video (Wan2.2 T2V)
# ---------------------------
python examples/offline_inference/text_to_video/text_to_video.py \
  --model /workspace/models/Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --prompt "Two anthropomorphic cats in comfy boxing gear fight intensely on a spotlighted stage." \
  --negative_prompt "" \
  --height 480 \
  --width 640 \
  --num_frames 32 \
  --guidance_scale 4.0 \
  --guidance_scale_high 3.0 \
  --num_inference_steps 40 \
  --fps 16 \
  --output "${OUTPUT_DIR}/wan22_t2v.mp4"

# ---------------------------
# Image-to-Video (Wan2.2 I2V / TI2V)
# ---------------------------
python examples/offline_inference/image_to_video/image_to_video.py \
  --model /workspace/models/Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --image "${ASSETS_DIR}/qwen-bear.png" \
  --prompt "A bear playing with yarn, smooth motion" \
  --negative_prompt "" \
  --height 480 \
  --width 832 \
  --num_frames 48 \
  --guidance_scale 4.0 \
  --num_inference_steps 40 \
  --flow_shift 12.0 \
  --fps 16 \
  --output "${OUTPUT_DIR}/wan22_ti2v.mp4"

# ---------------------------
# Text-to-Audio (Stable Audio Open)
# ---------------------------
python examples/offline_inference/text_to_audio/text_to_audio.py \
  --model /workspace/models/stabilityai/stable-audio-open-1.0 \
  --prompt "The sound of a hammer hitting a wooden surface." \
  --negative_prompt "Low quality." \
  --seed 42 \
  --guidance_scale 7.0 \
  --audio_length 10.0 \
  --num_inference_steps 100 \
  --num_waveforms 1 \
  --output "${OUTPUT_DIR}/stable_audio_open.wav"

# ---------------------------
# Online serving tests (non-hanging)
# ---------------------------

ZIMAGE_PORT=8091
ZIMAGE_MODEL="/workspace/models/Tongyi-MAI/Z-Image-Turbo"
ZIMAGE_LOG="${SERVER_LOG_DIR}/zimage_server.log"
ZIMAGE_OUTPUT="${OUTPUT_DIR}/zimage_coffee_online.png"

echo "Starting online serving test: Z-Image-Turbo (port ${ZIMAGE_PORT})"
ZIMAGE_PID=$(start_server "${ZIMAGE_MODEL}" "${ZIMAGE_PORT}" "${ZIMAGE_LOG}")
cleanup_pids+=("${ZIMAGE_PID}")
if ! wait_for_server "${ZIMAGE_PORT}" "${ZIMAGE_PID}"; then
  echo "Z-Image-Turbo server failed to start. See ${ZIMAGE_LOG}"
  stop_server "${ZIMAGE_PID}"
  exit 1
fi

curl -s --max-time 600 "http://localhost:${ZIMAGE_PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "A cup of coffee"}
    ],
    "extra_body": {
      "height": 1024,
      "width": 1024,
      "num_inference_steps": 50,
      "true_cfg_scale": 4.0,
      "seed": 42
    }
  }' | jq -r '.choices[0].message.content[0].image_url.url' \
  | cut -d',' -f2- | base64 -d > "${ZIMAGE_OUTPUT}"

stop_server "${ZIMAGE_PID}"

QWEN_EDIT_PORT=8092
QWEN_EDIT_MODEL="/workspace/models/Qwen/Qwen-Image-Edit"
QWEN_EDIT_LOG="${SERVER_LOG_DIR}/qwen_image_edit_server.log"
QWEN_EDIT_OUTPUT="${OUTPUT_DIR}/qwen_image_edit_online.png"

echo "Starting online serving test: Qwen-Image-Edit (port ${QWEN_EDIT_PORT})"
QWEN_EDIT_PID=$(start_server "${QWEN_EDIT_MODEL}" "${QWEN_EDIT_PORT}" "${QWEN_EDIT_LOG}")
cleanup_pids+=("${QWEN_EDIT_PID}")
if ! wait_for_server "${QWEN_EDIT_PORT}" "${QWEN_EDIT_PID}"; then
  echo "Qwen-Image-Edit server failed to start. See ${QWEN_EDIT_LOG}"
  stop_server "${QWEN_EDIT_PID}"
  exit 1
fi

QWEN_EDIT_IMG_B64=$(base64 -w0 "${ASSETS_DIR}/qwen-bear.png")
QWEN_EDIT_REQUEST_JSON=$(
  jq -n --arg prompt "Stylize this image into a colorful illustration" --arg img "${QWEN_EDIT_IMG_B64}" '{
    messages: [{
      role: "user",
      content: [
        {"type": "text", "text": $prompt},
        {"type": "image_url", "image_url": {"url": ("data:image/png;base64," + $img)}}
      ]
    }],
    extra_body: {
      height: 1024,
      width: 1024,
      num_inference_steps: 50,
      guidance_scale: 1,
      seed: 42
    }
  }'
)

curl -s --max-time 600 "http://localhost:${QWEN_EDIT_PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "${QWEN_EDIT_REQUEST_JSON}" \
  | jq -r '.choices[0].message.content[0].image_url.url' \
  | cut -d',' -f2- | base64 -d > "${QWEN_EDIT_OUTPUT}"

stop_server "${QWEN_EDIT_PID}"
