from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from diffusers import DiffusionPipeline

MODEL_NAME = "Qwen/Qwen-Image-2512"
PROMPT = (
    "A 20-year-old East Asian girl with delicate, charming features and large, bright brown eyes, "
    "expressive and lively, with a cheerful or subtly smiling expression. Her naturally wavy long "
    "hair is either loose or tied in twin ponytails. She has fair skin and light makeup accentuating "
    "her youthful freshness. She wears a modern, cute dress or relaxed outfit in bright, soft colors "
    "with lightweight fabric and a minimalist cut. She stands indoors at an anime convention, "
    "surrounded by banners, posters, or stalls. Lighting is typical indoor illumination with no staged "
    "lighting, and the image resembles a casual iPhone snapshot: unpretentious composition, yet "
    "brimming with vivid, fresh, youthful charm."
)
NEGATIVE_PROMPT = (
    "low resolution, low quality, deformed limbs, malformed fingers, oversaturated image, waxy look, "
    "face without details, overly smooth skin, obvious AI artifacts, chaotic composition, blurry text, "
    "distorted text"
)
WIDTH = 1664
HEIGHT = 928
NUM_INFERENCE_STEPS = 50
TRUE_CFG_SCALE = 4.0
SEED = 42


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen-Image-2512 diffusers baseline generation.")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--negative-prompt", default=NEGATIVE_PROMPT)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--num-inference-steps", type=int, default=NUM_INFERENCE_STEPS)
    parser.add_argument("--true-cfg-scale", type=float, default=TRUE_CFG_SCALE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata-output", required=True)
    return parser.parse_args()


def _build_generator(device: str, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


def main() -> None:
    args = _parse_args()
    output_path = Path(args.output)
    metadata_path = Path(args.metadata_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16
        device = "cuda"
    else:
        torch_dtype = torch.float32
        device = "cpu"

    pipe = DiffusionPipeline.from_pretrained(args.model, torch_dtype=torch_dtype).to(device)
    image = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        num_inference_steps=args.num_inference_steps,
        true_cfg_scale=args.true_cfg_scale,
        generator=_build_generator(device, args.seed),
    ).images[0]
    image.save(output_path)

    metadata = {
        "model": args.model,
        "device": device,
        "torch_dtype": str(torch_dtype),
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "width": args.width,
        "height": args.height,
        "num_inference_steps": args.num_inference_steps,
        "true_cfg_scale": args.true_cfg_scale,
        "seed": args.seed,
        "output_path": str(output_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
