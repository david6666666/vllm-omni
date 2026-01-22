#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASSETS_DIR="${ROOT_DIR}/assets"
OUTPUT_DIR="${ROOT_DIR}/outputs"

mkdir -p "${ASSETS_DIR}" "${OUTPUT_DIR}"

# Example image assets for image editing / image-to-video
if [ ! -f "${ASSETS_DIR}/qwen-bear.png" ]; then
  curl -L -o "${ASSETS_DIR}/qwen-bear.png" \
    https://vllm-public-assets.s3.us-west-2.amazonaws.com/omni-assets/qwen-bear.png
fi

# Duplicate for multi-image edit example
cp -f "${ASSETS_DIR}/qwen-bear.png" "${ASSETS_DIR}/qwen-bear-2.png"

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
  --model /workspace/models/Qwen/Qwen-Image-2512 \
  --prompt "a cup of coffee on the table" \
  --seed 42 \
  --cfg_scale 4.0 \
  --num_images_per_prompt 1 \
  --num_inference_steps 50 \
  --height 1024 \
  --width 1024 \
  --output "${OUTPUT_DIR}/qwen_image_2512_coffee.png"

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
  --model /workspace/models/OvisAI/Ovis-Image \
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
  --model /workspace/models/black-forest-labs/FLUX.2-klein-4B \
  --prompt "a cup of coffee on the table" \
  --seed 42 \
  --cfg_scale 4.0 \
  --num_images_per_prompt 1 \
  --num_inference_steps 50 \
  --height 1024 \
  --width 1024 \
  --output "${OUTPUT_DIR}/flux2_klein_4b_coffee.png"

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
  --image "${ASSETS_DIR}/qwen-bear.png" "${ASSETS_DIR}/qwen-bear-2.png" \
  --prompt "Combine these images into a single scene" \
  --output "${OUTPUT_DIR}/qwen_image_edit_2509.png" \
  --num_inference_steps 50 \
  --cfg_scale 4.0 \
  --guidance_scale 1.0

python examples/offline_inference/image_to_image/image_edit.py \
  --model /workspace/models/Qwen/Qwen-Image-Layered \
  --image "${ASSETS_DIR}/qwen-bear.png" \
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
