# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

# Set before CUDA context is created for deterministic matmul behavior.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

import argparse
from pathlib import Path

import numpy as np
import torch


def _configure_determinism(deterministic: bool, disable_tf32: bool, force_math_sdp: bool) -> None:
    if disable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    torch.backends.cudnn.benchmark = False

    if force_math_sdp:
        # Prefer math SDPA for reproducibility.
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)

    if deterministic:
        torch.use_deterministic_algorithms(True)


def _parse_dtype(value: str) -> torch.dtype:
    value = value.lower()
    if value in ("bf16", "bfloat16"):
        return torch.bfloat16
    if value in ("fp16", "float16", "half"):
        return torch.float16
    if value in ("fp32", "float32", "float"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diffusers LTX2 alignment script for pixel-level comparisons.")
    parser.add_argument("--model", required=True, help="Diffusers LTX2 model ID or local path.")
    parser.add_argument("--prompt", required=True, help="Text prompt.")
    parser.add_argument("--negative_prompt", default="", help="Negative prompt.")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--num_frames", type=int, default=121)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--guidance_scale", type=float, default=4.0)
    parser.add_argument("--frame_rate", type=float, default=24.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bf16", help="bf16|fp16|fp32")
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--no_deterministic", dest="deterministic", action="store_false")
    parser.add_argument("--disable_tf32", action="store_true", default=True)
    parser.add_argument("--allow_tf32", dest="disable_tf32", action="store_false")
    parser.add_argument("--force_math_sdp", action="store_true", default=True)
    parser.add_argument("--allow_flash_sdp", dest="force_math_sdp", action="store_false")
    parser.add_argument("--output", default="ltx2_align.mp4", help="Optional MP4 output path.")
    parser.add_argument("--save_frames_npy", default="ltx2_align_frames.npy")
    parser.add_argument("--save_audio_npy", default="ltx2_align_audio.npy")
    parser.add_argument("--skip_export", action="store_true", help="Skip MP4 export (useful for pure tensor diff).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = _parse_dtype(args.dtype)

    _configure_determinism(
        deterministic=args.deterministic,
        disable_tf32=args.disable_tf32,
        force_math_sdp=args.force_math_sdp,
    )

    try:
        from diffusers.pipelines.ltx2 import LTX2Pipeline
        from diffusers.pipelines.ltx2.export_utils import encode_video
    except ImportError as exc:
        raise ImportError("diffusers is required for LTX2 alignment.") from exc

    pipe = LTX2Pipeline.from_pretrained(args.model, torch_dtype=dtype)
    pipe.to(args.device)

    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    video, audio = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        output_type="np",
        return_dict=False,
    )

    frames = np.asarray(video)
    Path(args.save_frames_npy).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.save_frames_npy, frames)

    if audio is not None:
        audio_array = audio.detach().cpu().float().numpy()
        Path(args.save_audio_npy).parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save_audio_npy, audio_array)

    if args.skip_export:
        return

    video_u8 = (frames * 255).round().astype("uint8")
    video_u8 = torch.from_numpy(video_u8)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if audio is not None:
        audio_out = audio[0].float().cpu()
        audio_sample_rate = pipe.vocoder.config.output_sampling_rate
    else:
        audio_out = None
        audio_sample_rate = None

    encode_video(
        video_u8[0],
        fps=args.frame_rate,
        audio=audio_out,
        audio_sample_rate=audio_sample_rate,
        output_path=str(output_path),
    )


if __name__ == "__main__":
    main()

'''
python ltx2_diffusers_align.py \
  --model "/workspace/models/Lightricks/LTX-2" \
  --prompt "A cinematic close-up of ocean waves at golden hour." \
  --negative_prompt "worst quality, inconsistent motion, blurry, jittery, distorted" \
  --height 512 --width 768 --num_frames 121 \
  --num_inference_steps 40 --guidance_scale 4.0 \
  --frame_rate 24 --seed 42 \
  --output ltx2_align.mp4 \
'''