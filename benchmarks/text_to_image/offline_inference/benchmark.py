# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Simple offline benchmark for Qwen-Image comparing:
1) vLLM-Omni diffusion path (Omni).
2) Diffusers pipeline baseline.

Example:
python benchmark.py \
  --model /path/to/Qwen-Image \
  --diffusers-model /path/to/Qwen-Image \
  --num-runs 3 --warmup 1 \
  --prompt-file prompts.txt \
  --height 1024 --width 1024 \
  # or: --resolutions "1024x1024,768x512" \
  --num-inference-steps 50 \
  --cfg-scale 4.0
"""

from __future__ import annotations

import argparse
import json
import inspect
import os
import statistics
import time
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
from diffusers import DiffusionPipeline

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.utils.platform_utils import detect_device_type, is_npu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline Qwen-Image benchmark (Omni vs diffusers).")
    parser.add_argument("--model", type=str, default="Qwen/Qwen-Image", help="Model path/ID for Omni.")
    parser.add_argument(
        "--diffusers-model",
        type=str,
        default=None,
        help="Model path/ID for diffusers baseline (defaults to --model).",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="File with one prompt per line. If omitted, a few defaults are used.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base seed for reproducibility.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations (not timed).")
    parser.add_argument("--num-runs", type=int, default=3, help="Timed iterations.")
    parser.add_argument("--num-images-per-prompt", type=int, default=1, help="Images per prompt.")
    parser.add_argument("--num-inference-steps", type=int, default=50, help="Denoising steps.")
    parser.add_argument("--cfg-scale", type=float, default=4.0, help="True CFG scale.")
    parser.add_argument("--height", type=int, default=1024, help="Output height.")
    parser.add_argument("--width", type=int, default=1024, help="Output width.")
    parser.add_argument(
        "--resolutions",
        type=str,
        default=None,
        help="Optional list of HEIGHTxWIDTH resolutions, comma-separated, e.g. '1024x1024,768x512'. "
        "If provided, overrides --height/--width and runs benchmarks for each resolution.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Compute dtype.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Optional directory to save the first image from each backend (for sanity check).",
    )
    parser.add_argument(
        "--results-json",
        type=str,
        default="benchmark_results.json",
        help="Path to save benchmark results as JSON (set empty to disable).",
    )
    parser.add_argument(
        "--skip-diffusers",
        action="store_true",
        help="Run Omni only (useful if diffusers is unavailable on the device).",
    )
    return parser.parse_args()


def load_prompts(prompt_file: str | None) -> list[str]:
    if prompt_file and Path(prompt_file).is_file():
        return [line.strip() for line in Path(prompt_file).read_text().splitlines() if line.strip()]

    return [
        "a cup of coffee on the table",
        "a futuristic cityscape at sunset with flying cars",
        "a cozy living room with a dog sleeping on the couch",
    ]


def parse_resolutions(res_str: str | None, fallback_height: int, fallback_width: int) -> list[tuple[int, int]]:
    if not res_str:
        return [(fallback_height, fallback_width)]
    resolutions: list[tuple[int, int]] = []
    for item in res_str.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if "x" not in item:
            raise ValueError(f"Invalid resolution format: {item}. Expected HEIGHTxWIDTH, e.g. 1024x1024")
        h_str, w_str = item.split("x", maxsplit=1)
        try:
            h, w = int(h_str), int(w_str)
        except ValueError:
            raise ValueError(f"Invalid resolution numbers: {item}")
        resolutions.append((h, w))
    if not resolutions:
        resolutions.append((fallback_height, fallback_width))
    return resolutions


def sync_device(device_type: str) -> None:
    if device_type == "cuda":
        torch.cuda.synchronize()
    elif device_type == "npu" and hasattr(torch, "npu"):
        torch.npu.synchronize()  # type: ignore[attr-defined]


def reset_peak_memory(device_type: str) -> None:
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    elif device_type == "npu" and hasattr(torch, "npu") and torch.npu.is_available():  # type: ignore[attr-defined]
        torch.npu.reset_peak_memory_stats()  # type: ignore[attr-defined]


def get_peak_memory_gb(device_type: str) -> float | None:
    if device_type == "cuda" and torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated()) / (1024**3)
    if device_type == "npu" and hasattr(torch, "npu") and torch.npu.is_available():  # type: ignore[attr-defined]
        return float(torch.npu.max_memory_allocated()) / (1024**3)  # type: ignore[attr-defined]
    return None


def pick_cfg_key(pipe: DiffusionPipeline) -> str | None:
    params = set(inspect.signature(pipe.__call__).parameters.keys())
    for key in ("true_guidance_scale", "true_cfg_scale", "cfg_scale", "guidance_scale"):
        if key in params:
            return key
    return None


def dtype_from_str(dtype_str: str) -> torch.dtype:
    if dtype_str == "bfloat16":
        return torch.bfloat16
    if dtype_str == "float16":
        return torch.float16
    return torch.float32


def benchmark_runner(
    name: str,
    run_once: Callable[[str, torch.Generator], Iterable],
    prompts: list[str],
    device_type: str,
    num_runs: int,
    warmup: int,
    seed: int,
    images_per_prompt: int,
) -> dict[str, float]:
    # Warmup
    generator = torch.Generator(device=device_type).manual_seed(seed)
    for _ in range(warmup):
        reset_peak_memory(device_type)
        _ = list(run_once(prompts[0], generator))
        sync_device(device_type)

    times: list[float] = []
    mem_peaks: list[float] = []
    for run_idx in range(num_runs):
        for prompt_idx, prompt in enumerate(prompts):
            generator = torch.Generator(device=device_type).manual_seed(seed + run_idx * len(prompts) + prompt_idx)
            reset_peak_memory(device_type)
            start = time.perf_counter()
            _ = list(run_once(prompt, generator))
            sync_device(device_type)
            times.append(time.perf_counter() - start)
            peak_gb = get_peak_memory_gb(device_type)
            if peak_gb is not None:
                mem_peaks.append(peak_gb)

    num_calls = len(times)
    total_images = num_calls * images_per_prompt
    total_time = sum(times)
    stats = {
        "p50_s": float(np.percentile(times, 50)),
        "p90_s": float(np.percentile(times, 90)),
        "p95_s": float(np.percentile(times, 95)),
        "mean_s": float(statistics.mean(times)),
        "total_time_s": float(total_time),
        "throughput_img_per_s": total_images / total_time if total_time > 0 else 0.0,
        "num_calls": num_calls,
        "images_per_call": images_per_prompt,
        "total_images": total_images,
        "latencies_s": [float(t) for t in times],
    }
    if mem_peaks:
        stats.update(
            {
                "peak_mem_gb_max": float(max(mem_peaks)),
                "peak_mem_gb_mean": float(statistics.mean(mem_peaks)),
                "peak_mem_gb_p50": float(np.percentile(mem_peaks, 50)),
                "peak_mem_gb_p90": float(np.percentile(mem_peaks, 90)),
                "peak_mem_gb_series": [float(m) for m in mem_peaks],
            }
        )

    print(f"\n[{name}] calls={len(times)} (num_runs={num_runs}, prompts/run={len(prompts)})")
    print(
        f"mean={stats['mean_s']:.3f}s, p50={stats['p50_s']:.3f}s, "
        f"p90={stats['p90_s']:.3f}s, p95={stats['p95_s']:.3f}s, "
        f"throughput={stats['throughput_img_per_s']:.2f} img/s"
    )
    return stats


def maybe_save_image(images, save_dir: str | None, name: str) -> None:
    if not save_dir or not images:
        return
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(save_dir) / f"{name}.png"
    try:
        images[0].save(out_path)
        print(f"Saved sample to {out_path}")
    except Exception as exc:
        print(f"Failed to save image for {name}: {exc}")


def main() -> None:
    args = parse_args()
    prompts = load_prompts(args.prompt_file)
    device_type = detect_device_type()
    dtype = dtype_from_str(args.dtype)
    diffusers_model = args.diffusers_model or args.model
    local_files_only = os.path.exists(diffusers_model) or bool(os.environ.get("HF_HUB_OFFLINE"))
    resolutions = parse_resolutions(args.resolutions, args.height, args.width)

    print(f"Device: {device_type}, dtype: {dtype}, prompts: {len(prompts)}, resolutions: {resolutions}")
    print(f"Omni model: {args.model}")
    print(f"Diffusers model: {diffusers_model} (local_only={local_files_only})")

    # Omni runner
    vae_opt = is_npu()
    omni = Omni(
        model=args.model,
        vae_use_slicing=vae_opt,
        vae_use_tiling=vae_opt,
        dtype=dtype,
    )
    omni_saved: set[str] = set()

    # Diffusers runner
    pipe = None
    cfg_key = None
    diffusers_saved: set[str] = set()
    if not args.skip_diffusers:
        pipe = DiffusionPipeline.from_pretrained(
            diffusers_model,
            torch_dtype=dtype,
            local_files_only=local_files_only,
        )
        pipe.to(device_type)
        cfg_key = pick_cfg_key(pipe)
        if cfg_key is None:
            print("Warning: could not find CFG scale parameter in diffusers pipeline signature; skipping CFG.")

    all_results: dict[str, dict[str, object]] = {}

    for height, width in resolutions:
        res_tag = f"{height}x{width}"

        def omni_run(prompt: str, generator: torch.Generator, h: int = height, w: int = width):
            images = omni.generate(
                prompt,
                height=h,
                width=w,
                generator=generator,
                true_cfg_scale=args.cfg_scale,
                num_inference_steps=args.num_inference_steps,
                num_images_per_prompt=args.num_images_per_prompt,
                num_outputs_per_prompt=args.num_images_per_prompt,
            )
            if args.save_dir and res_tag not in omni_saved:
                maybe_save_image(images, args.save_dir, f"omni_{res_tag}")
                omni_saved.add(res_tag)
            return images

        omni_stats = benchmark_runner(
            f"Omni[{res_tag}]",
            omni_run,
            prompts,
            device_type,
            num_runs=args.num_runs,
            warmup=args.warmup,
            seed=args.seed,
            images_per_prompt=args.num_images_per_prompt,
        )

        diff_stats = {}
        if pipe is not None:

            def diffusers_run(prompt: str, generator: torch.Generator, h: int = height, w: int = width):
                call_kwargs = {
                    "prompt": prompt,
                    "height": h,
                    "width": w,
                    "num_inference_steps": args.num_inference_steps,
                    "num_images_per_prompt": args.num_images_per_prompt,
                    "generator": generator,
                }
                if cfg_key:
                    call_kwargs[cfg_key] = args.cfg_scale
                outputs = pipe(**call_kwargs)
                images = outputs.images if hasattr(outputs, "images") else outputs
                if args.save_dir and res_tag not in diffusers_saved:
                    maybe_save_image(images, args.save_dir, f"diffusers_{res_tag}")
                    diffusers_saved.add(res_tag)
                return images

            diff_stats = benchmark_runner(
                f"Diffusers[{res_tag}]",
                diffusers_run,
                prompts,
                device_type,
                num_runs=args.num_runs,
                warmup=args.warmup,
                seed=args.seed,
                images_per_prompt=args.num_images_per_prompt,
            )

        speedup = diff_stats["mean_s"] / omni_stats["mean_s"] if diff_stats and omni_stats["mean_s"] > 0 else None

        print("\n--- Summary:", res_tag, "---")
        print(f"Omni[{res_tag}]: mean {omni_stats['mean_s']:.3f}s, throughput {omni_stats['throughput_img_per_s']:.2f} img/s")
        if diff_stats:
            print(
                f"Diffusers[{res_tag}]: mean {diff_stats['mean_s']:.3f}s, throughput "
                f"{diff_stats['throughput_img_per_s']:.2f} img/s"
            )
            if speedup is not None:
                print(f"Speedup (diffusers/omni): {speedup:.2f}x")

        all_results[res_tag] = {
            "config": {
                "height": height,
                "width": width,
                "num_inference_steps": args.num_inference_steps,
                "cfg_scale": args.cfg_scale,
                "num_images_per_prompt": args.num_images_per_prompt,
                "num_runs": args.num_runs,
                "warmup": args.warmup,
                "seed": args.seed,
                "prompts_count": len(prompts),
                "prompt_file": args.prompt_file,
            },
            "omni": omni_stats,
            "diffusers": diff_stats if diff_stats else None,
            "speedup_diffusers_over_omni": speedup,
        }

    # Save JSON results
    if args.results_json:
        results = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "device": device_type,
            "dtype": str(dtype),
            "base_config": {
                "model": args.model,
                "diffusers_model": diffusers_model if not args.skip_diffusers else None,
                "num_inference_steps": args.num_inference_steps,
                "cfg_scale": args.cfg_scale,
                "num_images_per_prompt": args.num_images_per_prompt,
                "num_runs": args.num_runs,
                "warmup": args.warmup,
                "seed": args.seed,
                "prompts_count": len(prompts),
                "prompt_file": args.prompt_file,
                "resolutions": resolutions,
            },
            "runs": all_results,
        }
        try:
            Path(args.results_json).parent.mkdir(parents=True, exist_ok=True)
            with open(args.results_json, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            print(f"Saved results to {args.results_json}")
        except Exception as exc:
            print(f"Failed to save JSON results to {args.results_json}: {exc}")


if __name__ == "__main__":
    main()
