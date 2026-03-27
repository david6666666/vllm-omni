from __future__ import annotations

import argparse
import base64
import json
import os
from io import BytesIO
from pathlib import Path

import requests
import torch
import torch.distributed as dist
from diffusers import (
    AutoencoderKLWan,
    ContextParallelConfig,
    FlowMatchEulerDiscreteScheduler,
    WanImageToVideoPipeline,
)
from diffusers.utils import export_to_video
from PIL import Image
from transformers import CLIPVisionModel

from tests.e2e.accuracy.wan22_i2v.wan22_i2v_video_similarity_common import BOUNDARY_RATIO


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Wan2.2 I2V diffusers context-parallel offline generation.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--image-source", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", required=True)
    parser.add_argument("--size", required=True)
    parser.add_argument("--fps", type=int, required=True)
    parser.add_argument("--num-frames", type=int, required=True)
    parser.add_argument("--guidance-scale", type=float, required=True)
    parser.add_argument("--guidance-scale-2", type=float, required=True)
    parser.add_argument("--flow-shift", type=float, required=True)
    parser.add_argument("--num-inference-steps", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata-output", required=True)
    return parser.parse_args()


def _parse_size(size: str) -> tuple[int, int]:
    width_str, height_str = size.lower().split("x", 1)
    return int(width_str), int(height_str)


def _load_image(source: str) -> Image.Image:
    if source.startswith("data:image"):
        _, encoded = source.split(",", 1)
        image = Image.open(BytesIO(base64.b64decode(encoded)))
        image.load()
        return image.convert("RGB")

    source_path = Path(source)
    if source_path.exists():
        image = Image.open(source_path)
        image.load()
        return image.convert("RGB")

    response = requests.get(source, timeout=60)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content))
    image.load()
    return image.convert("RGB")


def _register_boundary_ratio(pipe: WanImageToVideoPipeline) -> None:
    if getattr(pipe.config, "boundary_ratio", None) != BOUNDARY_RATIO:
        pipe.register_to_config(boundary_ratio=BOUNDARY_RATIO)


def _apply_context_parallel(pipe: WanImageToVideoPipeline) -> None:
    cp_config = ContextParallelConfig(ulysses_degree=dist.get_world_size())
    if pipe.transformer is not None:
        pipe.transformer.enable_parallelism(config=cp_config)
    if getattr(pipe, "transformer_2", None) is not None:
        pipe.transformer_2.enable_parallelism(config=cp_config)


def _write_metadata(path: Path, *, args: argparse.Namespace, frame_count: int) -> None:
    width, height = _parse_size(args.size)
    payload = {
        "model": args.model,
        "image_source": args.image_source,
        "size": args.size,
        "width": width,
        "height": height,
        "fps": args.fps,
        "num_frames": args.num_frames,
        "actual_frame_count": frame_count,
        "guidance_scale": args.guidance_scale,
        "guidance_scale_2": args.guidance_scale_2,
        "boundary_ratio": BOUNDARY_RATIO,
        "flow_shift": args.flow_shift,
        "num_inference_steps": args.num_inference_steps,
        "seed": args.seed,
        "world_size": dist.get_world_size(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = _parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl")

    pipe: WanImageToVideoPipeline | None = None
    try:
        image_encoder = CLIPVisionModel.from_pretrained(
            args.model,
            subfolder="image_encoder",
            torch_dtype=torch.float32,
        )
        vae = AutoencoderKLWan.from_pretrained(
            args.model,
            subfolder="vae",
            torch_dtype=torch.float32,
        )
        pipe = WanImageToVideoPipeline.from_pretrained(
            args.model,
            vae=vae,
            image_encoder=image_encoder,
            torch_dtype=torch.bfloat16,
        )
        _register_boundary_ratio(pipe)
        pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            pipe.scheduler.config,
            shift=args.flow_shift,
        )
        _apply_context_parallel(pipe)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=local_rank != 0)

        width, height = _parse_size(args.size)
        input_image = _load_image(args.image_source)
        generator = torch.Generator(device=device.type).manual_seed(args.seed)

        result = pipe(
            image=input_image,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            width=width,
            height=height,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            guidance_scale_2=args.guidance_scale_2,
            generator=generator,
            output_type="np",
        )
        frames = result.frames[0]

        dist.barrier()
        if dist.get_rank() == 0:
            output_path = Path(args.output)
            metadata_path = Path(args.metadata_output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            export_to_video(frames, str(output_path), fps=args.fps)
            _write_metadata(metadata_path, args=args, frame_count=len(frames))
        dist.barrier()
        return 0
    finally:
        if pipe is not None and hasattr(pipe, "maybe_free_model_hooks"):
            pipe.maybe_free_model_hooks()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
