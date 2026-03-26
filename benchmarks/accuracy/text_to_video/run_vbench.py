from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks.accuracy.common import (
    VBENCH_DIMENSION_DISPLAY,
    VllmOmniVideoClient,
    build_t2v_video_request,
    build_vbench_generation_summary,
    ensure_dir,
    load_json,
    select_balanced_vbench_entries,
    summarize_vbench_results,
    vbench_import_context,
    write_json,
)

DEFAULT_NAME = "vbench"
DEFAULT_DIMENSIONS = list(VBENCH_DIMENSION_DISPLAY)
GENERATION_MANIFEST = "generation_manifest.json"
EVALUATION_MANIFEST = "evaluation_manifest.json"
SUMMARY_FILE = "summary.json"
SELECTED_FULL_INFO = "selected_vbench_full_info.json"


def _videos_dir(output_root: Path) -> Path:
    return ensure_dir(output_root / "videos")


def _metadata_dir(output_root: Path) -> Path:
    return ensure_dir(output_root / "metadata")


def _evaluation_dir(output_root: Path) -> Path:
    return ensure_dir(output_root / "evaluation" / "official")


def _full_info_path(args: argparse.Namespace) -> Path:
    if args.full_info_json is not None:
        return args.full_info_json
    if args.vbench_root is None:
        raise ValueError("Expected --vbench-root or --full-info-json.")
    return args.vbench_root / "vbench" / "VBench_full_info.json"


def _selected_full_info_path(output_root: Path) -> Path:
    return _metadata_dir(output_root) / SELECTED_FULL_INFO


def _generation_manifest_path(output_root: Path) -> Path:
    return output_root / GENERATION_MANIFEST


def _evaluation_manifest_path(output_root: Path) -> Path:
    return output_root / EVALUATION_MANIFEST


def _build_output_filename(prompt: str, sample_index: int) -> str:
    return f"{prompt}-{sample_index}.mp4"


def _load_generation_manifest(output_root: Path) -> dict[str, Any]:
    manifest_path = _generation_manifest_path(output_root)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Generation manifest not found: {manifest_path}")
    return load_json(manifest_path)


def _load_evaluation_manifest(output_root: Path) -> dict[str, Any]:
    manifest_path = _evaluation_manifest_path(output_root)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Evaluation manifest not found: {manifest_path}")
    return load_json(manifest_path)


def _generate(args: argparse.Namespace) -> int:
    full_info = load_json(_full_info_path(args))
    selected_entries = select_balanced_vbench_entries(
        full_info,
        dimensions=args.dimension,
        prompts_per_dimension=args.prompts_per_dimension,
    )
    selected_full_info_path = _selected_full_info_path(args.output_root)
    write_json(selected_full_info_path, selected_entries)

    client = VllmOmniVideoClient(base_url=args.base_url, api_key=args.api_key, timeout=args.timeout)
    records: list[dict[str, Any]] = []
    for entry in selected_entries:
        prompt = str(entry["prompt_en"])
        for sample_index in range(args.samples_per_prompt):
            output_path = _videos_dir(args.output_root) / _build_output_filename(prompt, sample_index)
            completed = client.generate_video(
                output_path=output_path,
                form_fields=build_t2v_video_request(
                    prompt=prompt,
                    width=args.width,
                    height=args.height,
                    num_frames=args.num_frames,
                    fps=args.fps,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    guidance_scale_2=args.guidance_scale_2,
                    boundary_ratio=args.boundary_ratio,
                    flow_shift=args.flow_shift,
                    seed=None if args.seed is None else args.seed + sample_index,
                ),
            )
            records.append(
                {
                    "prompt": prompt,
                    "dimensions": list(entry["dimension"]),
                    "sample_index": sample_index,
                    "output_path": str(output_path),
                    "video_id": completed["id"],
                    "status": completed["status"],
                }
            )

    manifest = {
        "bench": "vbench",
        "mode": "vbench_standard",
        "model": args.model,
        "videos_path": str(_videos_dir(args.output_root)),
        "selected_full_info_path": str(selected_full_info_path),
        "dimensions": list(args.dimension),
        "samples_per_prompt": args.samples_per_prompt,
        "records": records,
        "summary": build_vbench_generation_summary(records),
    }
    write_json(_generation_manifest_path(args.output_root), manifest)
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    manifest = _load_generation_manifest(args.output_root)
    selected_full_info_path = Path(manifest["selected_full_info_path"])
    dimensions = list(manifest["dimensions"])
    official_output_dir = _evaluation_dir(args.output_root)

    with vbench_import_context(args.vbench_root):
        import torch
        from vbench import VBench

        bench = VBench(torch.device(args.device), str(selected_full_info_path), str(official_output_dir))
        bench.evaluate(
            videos_path=str(_videos_dir(args.output_root)),
            name=args.name,
            dimension_list=dimensions,
            local=args.local_ckpt,
            read_frame=args.read_frame,
            mode="vbench_standard",
            imaging_quality_preprocessing_mode=args.imaging_quality_preprocessing_mode,
        )

    raw_results_path = official_output_dir / f"{args.name}_eval_results.json"
    write_json(
        _evaluation_manifest_path(args.output_root),
        {
            "bench": "vbench",
            "raw_results_path": str(raw_results_path),
            "official_output_dir": str(official_output_dir),
            "selected_full_info_path": str(selected_full_info_path),
            "dimensions": dimensions,
        },
    )
    return 0


def _summarize(args: argparse.Namespace) -> int:
    generation_manifest = _load_generation_manifest(args.output_root)
    evaluation_manifest = _load_evaluation_manifest(args.output_root)
    raw_results = load_json(Path(evaluation_manifest["raw_results_path"]))
    payload = {
        "generation": generation_manifest["summary"],
        "evaluation": summarize_vbench_results(raw_results),
    }
    write_json(args.output_root / SUMMARY_FILE, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run VBench text-to-video evaluation against vLLM-Omni.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate")
    generate.add_argument("--output-root", type=Path, required=True)
    generate.add_argument("--base-url", type=str, required=True)
    generate.add_argument("--model", type=str, required=True)
    generate.add_argument("--vbench-root", type=Path, default=None)
    generate.add_argument("--full-info-json", type=Path, default=None)
    generate.add_argument("--dimension", nargs="+", default=DEFAULT_DIMENSIONS)
    generate.add_argument("--prompts-per-dimension", type=int, default=None)
    generate.add_argument("--samples-per-prompt", type=int, default=1)
    generate.add_argument("--api-key", type=str, default="EMPTY")
    generate.add_argument("--width", type=int, default=640)
    generate.add_argument("--height", type=int, default=480)
    generate.add_argument("--num-frames", type=int, default=5)
    generate.add_argument("--fps", type=int, default=8)
    generate.add_argument("--num-inference-steps", type=int, default=2)
    generate.add_argument("--guidance-scale", type=float, default=1.0)
    generate.add_argument("--guidance-scale-2", type=float, default=None)
    generate.add_argument("--boundary-ratio", type=float, default=None)
    generate.add_argument("--flow-shift", type=float, default=None)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--timeout", type=int, default=900)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--output-root", type=Path, required=True)
    evaluate.add_argument("--vbench-root", type=Path, default=None)
    evaluate.add_argument("--device", type=str, default="cuda")
    evaluate.add_argument("--name", type=str, default=DEFAULT_NAME)
    evaluate.add_argument("--local-ckpt", action="store_true")
    evaluate.add_argument("--read-frame", action="store_true")
    evaluate.add_argument("--imaging-quality-preprocessing-mode", type=str, default="longer")

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--output-root", type=Path, required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    ensure_dir(args.output_root)

    if args.command == "generate":
        return _generate(args)
    if args.command == "evaluate":
        return _evaluate(args)
    if args.command == "summarize":
        return _summarize(args)
    parser.error(f"Unknown command: {args.command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
