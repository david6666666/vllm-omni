from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import QwenImageEditPlusPipeline

from tests.e2e.accuracy.qwen_image_edit_2511 import (
    GUIDANCE_SCALE,
    HEIGHT,
    MODEL_NAME,
    NEGATIVE_PROMPT,
    NUM_IMAGES_PER_PROMPT,
    NUM_INFERENCE_STEPS,
    OUTPUT_FORMAT,
    PROMPT,
    SEED,
    TRUE_CFG_SCALE,
    WIDTH,
    load_input_image,
    resolve_image_sources,
    write_json,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen-Image-Edit-2511 diffusers baseline generation.")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--image-source", action="append", default=None)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--negative-prompt", default=NEGATIVE_PROMPT)
    parser.add_argument("--true-cfg-scale", type=float, default=TRUE_CFG_SCALE)
    parser.add_argument("--guidance-scale", type=float, default=GUIDANCE_SCALE)
    parser.add_argument("--num-inference-steps", type=int, default=NUM_INFERENCE_STEPS)
    parser.add_argument("--num-images-per-prompt", type=int, default=NUM_IMAGES_PER_PROMPT)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata-output", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen-Image-Edit-2511 diffusers baseline requires CUDA.")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    pipeline = QwenImageEditPlusPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    pipeline.to(device)
    pipeline.set_progress_bar_config(disable=False)
    pipeline.vae.use_tiling = True
    pipeline.vae.use_slicing = True

    image_sources = resolve_image_sources(args.image_source)
    images = [load_input_image(source) for source in image_sources]
    image_input = images if len(images) > 1 else images[0]

    output = pipeline(
        image=image_input,
        prompt=args.prompt,
        generator=torch.Generator(device=device.type).manual_seed(args.seed),
        true_cfg_scale=args.true_cfg_scale,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_images_per_prompt=args.num_images_per_prompt,
        width=args.width,
        height=args.height,
    )

    output_path = Path(args.output)
    metadata_path = Path(args.metadata_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_image = output.images[0].convert("RGB")
    output_image.save(output_path, format=OUTPUT_FORMAT.upper())

    write_json(
        metadata_path,
        {
            "model": args.model,
            "image_sources": image_sources,
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "true_cfg_scale": args.true_cfg_scale,
            "guidance_scale": args.guidance_scale,
            "num_inference_steps": args.num_inference_steps,
            "num_images_per_prompt": args.num_images_per_prompt,
            "seed": args.seed,
            "width": output_image.width,
            "height": output_image.height,
            "device": str(device),
            "output_path": str(output_path),
        },
    )

    if hasattr(pipeline, "maybe_free_model_hooks"):
        pipeline.maybe_free_model_hooks()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
