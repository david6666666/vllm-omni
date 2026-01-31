#!/usr/bin/env bash
# usage bash test_diffusion.sh --model-prefix /workspace/models 2>&1 |tee test_examples.log
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ROOT_DIR}/../../.." && pwd)"
ASSETS_DIR="${ROOT_DIR}/assets"
OUTPUT_DIR="${ROOT_DIR}/outputs"
SERVER_LOG_DIR="${OUTPUT_DIR}/server_logs"
MODEL_PREFIX=""

mkdir -p "${ASSETS_DIR}" "${OUTPUT_DIR}" "${SERVER_LOG_DIR}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--model-prefix PATH]

Options:
  --model-prefix, -m  Model root prefix (default: empty)
  -h, --help          Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-prefix|-m)
      if [[ $# -lt 2 ]]; then
        echo "Error: --model-prefix requires a value" >&2
        exit 1
      fi
      MODEL_PREFIX="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -n "${MODEL_PREFIX}" ]]; then
  MODEL_PREFIX="${MODEL_PREFIX%/}/"
fi

echo "ROOT_DIR:$ROOT_DIR"
echo "REPO_ROOT:$REPO_ROOT"
echo "ASSETS_DIR:$ASSETS_DIR"
echo "OUTPUT_DIR:$OUTPUT_DIR"
echo "SERVER_LOG_DIR:$SERVER_LOG_DIR"
echo "MODEL_PREFIX:$MODEL_PREFIX"

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
    if curl -sf "http://localhost:${port}/health" >/dev/null 2>&1; then
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

python "${REPO_ROOT}/examples/offline_inference/text_to_image/text_to_image.py" \
  --model "${MODEL_PREFIX}Qwen/Qwen-Image" \
  --prompt "a cup of coffee on the table" \
  --seed 42 \
  --cfg_scale 4.0 \
  --num_images_per_prompt 1 \
  --num_inference_steps 50 \
  --height 1024 \
  --width 1024 \
  --output "${OUTPUT_DIR}/qwen_image_coffee.png"

python "${REPO_ROOT}/examples/offline_inference/text_to_image/text_to_image.py" \
  --model "${MODEL_PREFIX}Qwen/Qwen-Image" \
  --prompt "a cup of coffee on the table" \
  --seed 42 \
  --cfg_scale 4.0 \
  --num_images_per_prompt 1 \
  --num_inference_steps 50 \
  --height 1024 \
  --width 1024 \
  --cache_backend cache_dit \
  --output "${OUTPUT_DIR}/qwen_image_coffee_cache_dit.png"

python "${REPO_ROOT}/examples/offline_inference/text_to_image/text_to_image.py" \
  --model "${MODEL_PREFIX}Qwen/Qwen-Image" \
  --prompt "a cup of coffee on the table" \
  --seed 42 \
  --cfg_scale 4.0 \
  --num_images_per_prompt 1 \
  --num_inference_steps 50 \
  --height 1024 \
  --width 1024 \
  --cache_backend tea_cache \
  --output "${OUTPUT_DIR}/qwen_image_coffee_tea_cache.png"

python "${REPO_ROOT}/examples/offline_inference/text_to_image/text_to_image.py" \
  --model "${MODEL_PREFIX}Qwen/Qwen-Image" \
  --prompt "a cup of coffee on the table" \
  --seed 42 \
  --cfg_scale 4.0 \
  --num_images_per_prompt 1 \
  --num_inference_steps 50 \
  --height 1024 \
  --width 1024 \
  --ulysses_degree 2 \
  --output "${OUTPUT_DIR}/qwen_image_coffee_ulysses_degree.png"

python "${REPO_ROOT}/examples/offline_inference/text_to_image/text_to_image.py" \
  --model "${MODEL_PREFIX}Qwen/Qwen-Image" \
  --prompt "a cup of coffee on the table" \
  --seed 42 \
  --cfg_scale 4.0 \
  --num_images_per_prompt 1 \
  --num_inference_steps 50 \
  --height 1024 \
  --width 1024 \
  --ring_degree 2 \
  --output "${OUTPUT_DIR}/qwen_image_coffee_ring_degree.png"

python "${REPO_ROOT}/examples/offline_inference/text_to_image/text_to_image.py" \
  --model "${MODEL_PREFIX}Qwen/Qwen-Image" \
  --prompt "a cup of coffee on the table" \
  --seed 42 \
  --cfg_scale 4.0 \
  --num_images_per_prompt 1 \
  --num_inference_steps 50 \
  --height 1024 \
  --width 1024 \
  --cfg_parallel_size 2 \
  --output "${OUTPUT_DIR}/qwen_image_coffee_cfg_parallel.png"

python "${REPO_ROOT}/examples/offline_inference/text_to_image/text_to_image.py" \
  --model "${MODEL_PREFIX}Qwen/Qwen-Image" \
  --prompt "a cup of coffee on the table" \
  --seed 42 \
  --cfg_scale 4.0 \
  --num_images_per_prompt 1 \
  --num_inference_steps 50 \
  --height 1024 \
  --width 1024 \
  --tensor_parallel_size 2 \
  --output "${OUTPUT_DIR}/qwen_image_coffee_tensor_parallel.png"

python "${REPO_ROOT}/examples/offline_inference/text_to_image/text_to_image.py" \
  --model "${MODEL_PREFIX}Tongyi-MAI/Z-Image-Turbo" \
  --prompt "a cup of coffee on the table" \
  --seed 42 \
  --cfg_scale 4.0 \
  --num_images_per_prompt 1 \
  --num_inference_steps 50 \
  --height 1024 \
  --width 1024 \
  --output "${OUTPUT_DIR}/zimage_coffee.png"

python "${REPO_ROOT}/examples/offline_inference/text_to_image/text_to_image.py" \
  --model "${MODEL_PREFIX}AIDC-AI/Ovis-Image-7B" \
  --prompt "a cup of coffee on the table" \
  --seed 42 \
  --cfg_scale 4.0 \
  --num_images_per_prompt 1 \
  --num_inference_steps 50 \
  --height 1024 \
  --width 1024 \
  --output "${OUTPUT_DIR}/ovis_image_coffee.png"

python "${REPO_ROOT}/examples/offline_inference/text_to_image/text_to_image.py" \
  --model "${MODEL_PREFIX}meituan-longcat/LongCat-Image" \
  --prompt "a cup of coffee on the table" \
  --seed 42 \
  --cfg_scale 4.0 \
  --num_images_per_prompt 1 \
  --num_inference_steps 50 \
  --height 1024 \
  --width 1024 \
  --output "${OUTPUT_DIR}/longcat_image_coffee.png"

python "${REPO_ROOT}/examples/offline_inference/text_to_image/text_to_image.py" \
  --model "${MODEL_PREFIX}stabilityai/stable-diffusion-3.5-medium" \
  --prompt "a cup of coffee on the table" \
  --seed 42 \
  --cfg_scale 4.0 \
  --num_images_per_prompt 1 \
  --num_inference_steps 50 \
  --height 1024 \
  --width 1024 \
  --output "${OUTPUT_DIR}/sd3_coffee.png"

python "${REPO_ROOT}/examples/offline_inference/text_to_image/text_to_image.py" \
  --model "${MODEL_PREFIX}black-forest-labs/FLUX.1-dev" \
  --prompt "a cup of coffee on the table" \
  --seed 42 \
  --cfg_scale 4.0 \
  --num_images_per_prompt 1 \
  --num_inference_steps 50 \
  --height 1024 \
  --width 1024 \
  --output "${OUTPUT_DIR}/flux1_dev_coffee.png"

python "${REPO_ROOT}/examples/offline_inference/text_to_image/text_to_image.py" \
  --model "${MODEL_PREFIX}black-forest-labs/FLUX.2-klein-9B" \
  --prompt "a cup of coffee on the table" \
  --seed 42 \
  --cfg_scale 4.0 \
  --num_images_per_prompt 1 \
  --num_inference_steps 50 \
  --height 1024 \
  --width 1024 \
  --output "${OUTPUT_DIR}/flux2_klein_9b_coffee.png"

python "${REPO_ROOT}/examples/offline_inference/text_to_image/text_to_image.py" \
  --model "${MODEL_PREFIX}zai-org/GLM-Image" \
  --prompt "a cup of coffee on the table" \
  --seed 42 \
  --cfg_scale 4.0 \
  --num_images_per_prompt 1 \
  --num_inference_steps 50 \
  --height 1024 \
  --width 1024 \
  --output "${OUTPUT_DIR}/GLM-Image_coffee.png"

# ---------------------------
# Image-Editing models
# ---------------------------
python "${REPO_ROOT}/examples/offline_inference/image_to_image/image_edit.py" \
  --model "${MODEL_PREFIX}Qwen/Qwen-Image-Edit" \
  --image "${ASSETS_DIR}/qwen-bear.png" \
  --prompt "Let this mascot dance under the moon, surrounded by floating stars and poetic bubbles such as 'Be Kind'" \
  --output "${OUTPUT_DIR}/qwen_image_edit.png" \
  --num_inference_steps 50 \
  --cfg_scale 4.0

python "${REPO_ROOT}/examples/offline_inference/image_to_image/image_edit.py" \
  --model "${MODEL_PREFIX}Qwen/Qwen-Image-Edit-2509" \
  --image "${ASSETS_DIR}/qwen-bear.png" "${OUTPUT_DIR}/qwen_image_coffee.png" \
  --prompt "Combine these images into a single scene" \
  --output "${OUTPUT_DIR}/qwen_image_edit_2509.png" \
  --num_inference_steps 50 \
  --cfg_scale 4.0 \
  --guidance_scale 1.0

python "${REPO_ROOT}/examples/offline_inference/image_to_image/image_edit.py" \
  --model "${MODEL_PREFIX}Qwen/Qwen-Image-Layered" \
  --image "${OUTPUT_DIR}/qwen_image_edit.png" \
  --prompt "Decompose the image into layered RGBA outputs" \
  --output "${OUTPUT_DIR}/qwen_image_layered" \
  --num_inference_steps 50 \
  --cfg_scale 4.0 \
  --layers 4 \
  --color-format "RGBA"

python "${REPO_ROOT}/examples/offline_inference/image_to_image/image_edit.py" \
  --model "${MODEL_PREFIX}meituan-longcat/LongCat-Image-Edit" \
  --image "${ASSETS_DIR}/qwen-bear.png" \
  --prompt "Let this mascot dance under the moon, surrounded by floating stars and poetic bubbles such as 'Be Kind'" \
  --output "${OUTPUT_DIR}/longcat_image_edit.png" \
  --num_inference_steps 50 \
  --cfg_scale 4.0

python "${REPO_ROOT}/examples/offline_inference/image_to_image/image_edit.py" \
  --model "${MODEL_PREFIX}black-forest-labs/FLUX.2-klein-9B" \
  --image "${ASSETS_DIR}/qwen-bear.png" \
  --prompt "Let this mascot dance under the moon, surrounded by floating stars and poetic bubbles such as 'Be Kind'" \
  --output "${OUTPUT_DIR}/flux2_klein_9b_edit.png" \
  --num_inference_steps 50 \
  --cfg_scale 4.0

python "${REPO_ROOT}/examples/offline_inference/image_to_image/image_edit.py" \
  --model "${MODEL_PREFIX}zai-org/GLM-Image" \
  --image "${ASSETS_DIR}/qwen-bear.png" \
  --prompt "Let this mascot dance under the moon, surrounded by floating stars and poetic bubbles such as 'Be Kind'" \
  --output "${OUTPUT_DIR}/GLM-Image_edit.png" \
  --num_inference_steps 50 \
  --cfg_scale 4.0

# ---------------------------
# Text-to-Video (Wan2.2 T2V)
# ---------------------------
python "${REPO_ROOT}/examples/offline_inference/text_to_video/text_to_video.py" \
  --model "${MODEL_PREFIX}Wan-AI/Wan2.2-T2V-A14B-Diffusers" \
  --prompt "Two anthropomorphic cats in comfy boxing gear fight intensely on a spotlighted stage." \
  --negative_prompt "" \
  --height 480 \
  --width 640 \
  --num_frames 32 \
  --guidance_scale 4.0 \
  --guidance_scale_high 3.0 \
  --num_inference_steps 40 \
  --enable-cpu-offload \
  --fps 16 \
  --output "${OUTPUT_DIR}/wan22_t2v.mp4"

# ---------------------------
# Image-to-Video (Wan2.2 I2V / TI2V)
# ---------------------------
python "${REPO_ROOT}/examples/offline_inference/image_to_video/image_to_video.py" \
  --model "${MODEL_PREFIX}Wan-AI/Wan2.2-TI2V-5B-Diffusers" \
  --image "${ASSETS_DIR}/qwen-bear.png" \
  --prompt "A bear playing with yarn, smooth motion" \
  --negative_prompt "" \
  --height 480 \
  --width 832 \
  --num_frames 48 \
  --guidance_scale 4.0 \
  --num_inference_steps 40 \
  --flow_shift 12.0 \
  --enable-cpu-offload \
  --fps 16 \
  --output "${OUTPUT_DIR}/wan22_ti2v.mp4"

# ---------------------------
# Text-to-Audio (Stable Audio Open)
# ---------------------------
python "${REPO_ROOT}/examples/offline_inference/text_to_audio/text_to_audio.py" \
  --model "${MODEL_PREFIX}stabilityai/stable-audio-open-1.0" \
  --prompt "The sound of a hammer hitting a wooden surface." \
  --negative_prompt "Low quality." \
  --seed 42 \
  --guidance_scale 7.0 \
  --audio_length 10.0 \
  --num_inference_steps 100 \
  --num_waveforms 1 \
  --output "${OUTPUT_DIR}/stable_audio_open.wav"

# ---------------------------
# Bagel 2-stage
# ---------------------------
python3 "${REPO_ROOT}/examples/offline_inference/bagel/end2end.py" \
  --model "${MODEL_PREFIX}ByteDance-Seed/BAGEL-7B-MoT" \
  --prompts "A cute cat" \
  --modality text2img

python3 "${REPO_ROOT}/examples/offline_inference/bagel/end2end.py" \
  --model "${MODEL_PREFIX}ByteDance-Seed/BAGEL-7B-MoT" \
  --prompts "Let this mascot dance under the moon, surrounded by floating stars and poetic bubbles such as 'Be Kind'" \
  --modality img2img \
  --image-path "${ASSETS_DIR}/qwen-bear.png"

# ---------------------------
# Qwen3-TTS
# ---------------------------
python "${REPO_ROOT}/examples/offline_inference/qwen3_tts/end2end.py" --query-type CustomVoice

python "${REPO_ROOT}/examples/offline_inference/qwen3_tts/end2end.py" --query-type VoiceDesign --use-batch-sample

# ---------------------------
# Online serving tests (non-hanging)
# ---------------------------

ZIMAGE_PORT=8091
ZIMAGE_MODEL="${MODEL_PREFIX}Tongyi-MAI/Z-Image-Turbo"
ZIMAGE_LOG="${SERVER_LOG_DIR}/zimage_server.log"
ZIMAGE_OUTPUT="${OUTPUT_DIR}/zimage_coffee_online.png"
ZIMAGE_OPENAI_OUTPUT="${OUTPUT_DIR}/zimage_coffee_online_openai.png"

echo "Starting online serving test: Z-Image-Turbo (port ${ZIMAGE_PORT})"
ZIMAGE_PID=$(start_server "${ZIMAGE_MODEL}" "${ZIMAGE_PORT}" "${ZIMAGE_LOG}")
cleanup_pids+=("${ZIMAGE_PID}")
if ! wait_for_server "${ZIMAGE_PORT}" "${ZIMAGE_PID}"; then
  echo "Z-Image-Turbo server failed to start. See ${ZIMAGE_LOG}"
  stop_server "${ZIMAGE_PID}"
  exit 1
fi

curl -s --max-time 600 "http://localhost:${ZIMAGE_PORT}/v1/images/generations" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a dragon laying over the spine of the Green Mountains of Vermont",
    "size": "1024x1024",
    "seed": 42
  }' | jq -r '.data[0].b64_json' | base64 -d > "${ZIMAGE_OPENAI_OUTPUT}"

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
QWEN_EDIT_MODEL="${MODEL_PREFIX}Qwen/Qwen-Image-Edit"
QWEN_EDIT_LOG="${SERVER_LOG_DIR}/qwen_image_edit_server.log"
QWEN_EDIT_OUTPUT="${OUTPUT_DIR}/qwen_image_edit_online.png"
QWEN_EDIT_REQUEST_JSON="${OUTPUT_DIR}/qwen_image_edit_request.json"
QWEN_EDIT_PROMPT="Let this mascot dance under the moon, surrounded by floating stars and poetic bubbles such as Be Kind"

echo "Starting online serving test: Qwen-Image-Edit (port ${QWEN_EDIT_PORT})"
QWEN_EDIT_PID=$(start_server "${QWEN_EDIT_MODEL}" "${QWEN_EDIT_PORT}" "${QWEN_EDIT_LOG}")
cleanup_pids+=("${QWEN_EDIT_PID}")
if ! wait_for_server "${QWEN_EDIT_PORT}" "${QWEN_EDIT_PID}"; then
  echo "Qwen-Image-Edit server failed to start. See ${QWEN_EDIT_LOG}"
  stop_server "${QWEN_EDIT_PID}"
  exit 1
fi

QWEN_EDIT_IMG_B64=$(base64 -w0 "${ASSETS_DIR}/qwen-bear.png")
cat <<EOF > "${QWEN_EDIT_REQUEST_JSON}"
{
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "${QWEN_EDIT_PROMPT}"},
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,${QWEN_EDIT_IMG_B64}"}}
    ]
  }],
  "extra_body": {
    "num_inference_steps": 50,
    "guidance_scale": 1,
    "seed": 42
  }
}
EOF

curl -s --max-time 600 "http://localhost:${QWEN_EDIT_PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d @"${QWEN_EDIT_REQUEST_JSON}" \
  | jq -r '.choices[0].message.content[0].image_url.url' \
  | cut -d',' -f2- | base64 -d > "${QWEN_EDIT_OUTPUT}"

stop_server "${QWEN_EDIT_PID}"

# ---------------------------
# Bagel online serving (text2img)
# ---------------------------

BAGEL_PORT=8091
BAGEL_MODEL="${MODEL_PREFIX}ByteDance-Seed/BAGEL-7B-MoT"
BAGEL_LOG="${SERVER_LOG_DIR}/bagel_server.log"
BAGEL_OUTPUT="${OUTPUT_DIR}/bagel_cute_cat_online.png"

echo "Starting online serving test: BAGEL-7B-MoT (port ${BAGEL_PORT})"
BAGEL_PID=$(start_server "${BAGEL_MODEL}" "${BAGEL_PORT}" "${BAGEL_LOG}")
cleanup_pids+=("${BAGEL_PID}")
if ! wait_for_server "${BAGEL_PORT}" "${BAGEL_PID}"; then
  echo "BAGEL-7B-MoT server failed to start. See ${BAGEL_LOG}"
  stop_server "${BAGEL_PID}"
  exit 1
fi

python "${REPO_ROOT}/examples/online_serving/bagel/openai_chat_client.py" \
  --prompt "A cute cat" \
  --modality text2img \
  --output "${BAGEL_OUTPUT}"

stop_server "${BAGEL_PID}"

# ---------------------------
# Qwen3-TTS online serving
# ---------------------------

QWEN3_TTS_PORT=8093
QWEN3_TTS_LOG="${SERVER_LOG_DIR}/qwen3_tts_server.log"
QWEN3_TTS_STAGE_CONFIG="${REPO_ROOT}/vllm_omni/model_executor/stage_configs/qwen3_tts.yaml"

QWEN3_CUSTOMVOICE_MODEL="${MODEL_PREFIX}Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
QWEN3_CUSTOMVOICE_OUTPUT="${OUTPUT_DIR}/qwen3_customvoice_chinese.wav"
QWEN3_CUSTOMVOICE_OUTPUT_EN="${OUTPUT_DIR}/qwen3_customvoice_english.wav"

echo "Starting online serving test: Qwen3-TTS CustomVoice (port ${QWEN3_TTS_PORT})"
QWEN3_CUSTOMVOICE_PID=$(start_server "${QWEN3_CUSTOMVOICE_MODEL}" "${QWEN3_TTS_PORT}" "${QWEN3_TTS_LOG}" \
  --stage-configs-path "${QWEN3_TTS_STAGE_CONFIG}" \
  --trust-remote-code \
  --enforce-eager)
cleanup_pids+=("${QWEN3_CUSTOMVOICE_PID}")
if ! wait_for_server "${QWEN3_TTS_PORT}" "${QWEN3_CUSTOMVOICE_PID}"; then
  echo "Qwen3-TTS CustomVoice server failed to start. See ${QWEN3_TTS_LOG}"
  stop_server "${QWEN3_CUSTOMVOICE_PID}"
  exit 1
fi

curl -s --max-time 600 -X POST "http://localhost:${QWEN3_TTS_PORT}/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "其实我真的有发现，我是一个特别善于观察别人情绪的人。",
    "voice": "Vivian",
    "language": "Chinese",
    "instructions": "用特别愤怒的语气说"
  }' --output "${QWEN3_CUSTOMVOICE_OUTPUT}"

curl -s --max-time 600 -X POST "http://localhost:${QWEN3_TTS_PORT}/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "She said she would be here by noon.",
    "voice": "Ryan",
    "language": "English",
    "instructions": "Very happy."
  }' --output "${QWEN3_CUSTOMVOICE_OUTPUT_EN}"

stop_server "${QWEN3_CUSTOMVOICE_PID}"

QWEN3_TTS_PORT=8094
QWEN3_VOICEDESIGN_MODEL="${MODEL_PREFIX}Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
QWEN3_VOICEDESIGN_OUTPUT="${OUTPUT_DIR}/qwen3_voicedesign.wav"
QWEN3_VOICEDESIGN_LOG="${SERVER_LOG_DIR}/qwen3_tts_voicedesign_server.log"

echo "Starting online serving test: Qwen3-TTS VoiceDesign (port ${QWEN3_TTS_PORT})"
QWEN3_VOICEDESIGN_PID=$(start_server "${QWEN3_VOICEDESIGN_MODEL}" "${QWEN3_TTS_PORT}" "${QWEN3_VOICEDESIGN_LOG}" \
  --stage-configs-path "${QWEN3_TTS_STAGE_CONFIG}" \
  --trust-remote-code \
  --enforce-eager)
cleanup_pids+=("${QWEN3_VOICEDESIGN_PID}")
if ! wait_for_server "${QWEN3_TTS_PORT}" "${QWEN3_VOICEDESIGN_PID}"; then
  echo "Qwen3-TTS VoiceDesign server failed to start. See ${QWEN3_VOICEDESIGN_LOG}"
  stop_server "${QWEN3_VOICEDESIGN_PID}"
  exit 1
fi

curl -s --max-time 600 -X POST "http://localhost:${QWEN3_TTS_PORT}/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "哥哥，你回来啦，人家等了你好久好久了，要抱抱！",
    "task_type": "VoiceDesign",
    "language": "Chinese",
    "instructions": "体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显，营造出黏人、做作又刻意卖萌的听觉效果。"
  }' --output "${QWEN3_VOICEDESIGN_OUTPUT}"

stop_server "${QWEN3_VOICEDESIGN_PID}"

QWEN3_TTS_PORT=8095
QWEN3_BASE_MODEL="${MODEL_PREFIX}Qwen/Qwen3-TTS-12Hz-1.7B-Base"
QWEN3_BASE_OUTPUT="${OUTPUT_DIR}/qwen3_base_clone.wav"
QWEN3_BASE_LOG="${SERVER_LOG_DIR}/qwen3_tts_base_server.log"

echo "Starting online serving test: Qwen3-TTS Base (port ${QWEN3_TTS_PORT})"
QWEN3_BASE_PID=$(start_server "${QWEN3_BASE_MODEL}" "${QWEN3_TTS_PORT}" "${QWEN3_BASE_LOG}" \
  --stage-configs-path "${QWEN3_TTS_STAGE_CONFIG}" \
  --trust-remote-code \
  --enforce-eager)
cleanup_pids+=("${QWEN3_BASE_PID}")
if ! wait_for_server "${QWEN3_TTS_PORT}" "${QWEN3_BASE_PID}"; then
  echo "Qwen3-TTS Base server failed to start. See ${QWEN3_BASE_LOG}"
  stop_server "${QWEN3_BASE_PID}"
  exit 1
fi

curl -s --max-time 600 -X POST "http://localhost:${QWEN3_TTS_PORT}/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Good one. Okay, fine, I am just gonna leave this sock monkey here. Goodbye.",
    "task_type": "Base",
    "language": "Auto",
    "ref_audio": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_2.wav",
    "ref_text": "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you.",
    "x_vector_only_mode": false
  }' --output "${QWEN3_BASE_OUTPUT}"

stop_server "${QWEN3_BASE_PID}"
