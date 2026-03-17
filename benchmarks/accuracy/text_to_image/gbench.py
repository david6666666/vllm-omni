from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
from PIL import Image

from benchmarks.accuracy.common import (
    VllmOmniImageClient,
    build_openai_url,
    ensure_dir,
    extract_json_object,
    find_first_image,
    load_json,
    pil_to_data_url,
    save_image,
    write_json,
)

TYPE_TO_FOLDER = {
    "type1": "01_single_step",
    "type2": "02_multi_step",
    "type3": "03_trajectory_text_fictionalapp",
    "type4": "04_trajectory_text_realapp",
    "type5": "05_grounding_data",
}
SCORE_KEYS = ("goal", "logic", "cons", "ui", "qual")
DEFAULT_SAMPLES_PER_TYPE = 10


def summarize_generated_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_type[record["data_type"]].append(record)

    return {
        "count": len(records),
        "by_type": {
            data_type: {
                "count": len(rows),
                "samples": sorted(row["sample_name"] for row in rows),
            }
            for data_type, rows in sorted(by_type.items())
        },
    }


def summarize_gebench_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_type[result["data_type"]].append(result)

    summary: dict[str, Any] = {
        "count": len(results),
        "overall_mean": statistics.fmean(r["overall"] for r in results) if results else 0.0,
        "by_type": {},
    }
    for data_type, rows in by_type.items():
        score_means: dict[str, float] = {}
        all_score_keys = {key for row in rows for key in row.get("scores", {}).keys()}
        for score_key in all_score_keys:
            values = [row["scores"][score_key] for row in rows if score_key in row.get("scores", {})]
            score_means[score_key] = statistics.fmean(values) if values else 0.0
        overall_mean = statistics.fmean(row["overall"] for row in rows)
        summary["by_type"][data_type] = {
            "count": len(rows),
            "overall_mean": overall_mean,
            "overall_mean_100": overall_mean * 100.0,
            "score_means": score_means,
        }
    return summary


def select_balanced_gebench_samples(
    sample_paths_by_type: dict[str, list[Path]],
    *,
    samples_per_type: int | None,
) -> dict[str, list[Path]]:
    if samples_per_type is None:
        return {data_type: list(paths) for data_type, paths in sample_paths_by_type.items()}
    return {data_type: list(paths)[:samples_per_type] for data_type, paths in sample_paths_by_type.items()}


def collect_gebench_generation_summary(output_root: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for data_type, folder_name in TYPE_TO_FOLDER.items():
        type_root = output_root / folder_name
        if not type_root.exists():
            continue
        for lang_dir in sorted(path for path in type_root.iterdir() if path.is_dir()):
            for sample_dir in sorted(path for path in lang_dir.iterdir() if path.is_dir()):
                expected = sample_dir / "frame5.png" if data_type in {"type2", "type3", "type4"} else None
                if expected is None:
                    expected = find_first_image(sample_dir)
                elif not expected.exists():
                    expected = None
                if expected is None:
                    continue
                records.append(
                    {
                        "data_type": data_type,
                        "sample_name": f"{lang_dir.name}/{sample_dir.name}",
                        "output_path": str(expected),
                    }
                )
    return summarize_generated_records(records)


def _normalize_score_key(key: str) -> str:
    mapping = {
        "goal": "goal",
        "logic": "logic",
        "cons": "cons",
        "consistency": "cons",
        "ui": "ui",
        "qual": "qual",
        "quality": "qual",
    }
    return mapping.get(key.lower(), key.lower())


def _normalize_scores(raw_scores: dict[str, Any]) -> dict[str, int]:
    scores: dict[str, int] = {}
    for key, value in raw_scores.items():
        normalized = _normalize_score_key(key)
        if normalized not in SCORE_KEYS:
            continue
        scalar = value.get("s", 0) if isinstance(value, dict) else value
        try:
            scores[normalized] = int(scalar)
        except (TypeError, ValueError):
            scores[normalized] = 0
    for key in SCORE_KEYS:
        scores.setdefault(key, 0)
    return scores


def _compute_overall(scores: dict[str, int]) -> float:
    return sum(scores.values()) / (len(SCORE_KEYS) * 5.0)


def _iter_sample_paths(dataset_root: Path, data_type: str) -> list[Path]:
    data_dir = dataset_root / TYPE_TO_FOLDER[data_type]
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset path not found: {data_dir}")

    samples: list[Path] = []
    for lang_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        for child in sorted(lang_dir.iterdir()):
            if child.is_dir():
                samples.append(child)
            elif child.suffix.lower() == ".json":
                samples.append(child)
    return samples


def _load_metadata(sample_path: Path) -> dict[str, Any]:
    if sample_path.is_file():
        return load_json(sample_path)
    for candidate in ("meta_data.json", "metadata.json"):
        meta_path = sample_path / candidate
        if meta_path.exists():
            return load_json(meta_path)
    raise FileNotFoundError(f"Metadata not found for sample: {sample_path}")


def _sample_name(sample_path: Path) -> str:
    return sample_path.stem if sample_path.is_file() else sample_path.name


def _lang_device(sample_path: Path, metadata: dict[str, Any]) -> str:
    return str(metadata.get("lang_device") or sample_path.parent.name)


def _resolve_referenced_image(
    *,
    metadata: dict[str, Any],
    sample_path: Path,
    dataset_root: Path,
    data_type: str,
) -> Image.Image | None:
    for key in ("image", "input_image", "initial_image", "reference_image"):
        image_ref = metadata.get(key)
        if not image_ref:
            continue
        candidate = dataset_root / TYPE_TO_FOLDER[data_type] / str(image_ref)
        if candidate.exists():
            image = Image.open(candidate)
            image.load()
            return image.convert("RGB")
    if sample_path.is_dir():
        local_image = find_first_image(sample_path)
        if local_image:
            image = Image.open(local_image)
            image.load()
            return image.convert("RGB")
    return None


def _trajectory_steps(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("trajectory", "steps", "frames"):
        value = metadata.get(key)
        if isinstance(value, list):
            return [step for step in value if isinstance(step, dict)]
    extracted: list[dict[str, Any]] = []
    for index in range(1, 6):
        value = metadata.get(f"step{index}") or metadata.get(str(index))
        if isinstance(value, dict):
            extracted.append(value)
    return extracted


def _text_or_default(value: Any, default: str = "") -> str:
    return str(value).strip() if value is not None else default


def _type1_prompt(metadata: dict[str, Any]) -> str:
    caption = _text_or_default(metadata.get("caption") or metadata.get("instruction"), "Transform the reference GUI.")
    return (
        "Using the reference GUI screenshot, generate the next GUI state after the requested interaction.\n\n"
        f"Requested change:\n{caption}\n\n"
        "Requirements:\n"
        "- Preserve layout, visual identity, and unrelated regions.\n"
        "- Only apply the requested state change.\n"
        "- Keep all text and controls readable.\n"
    )


def _type2_prompt(goal: str, step_num: int) -> str:
    return (
        "Generate the next GUI state for a multi-step task.\n\n"
        f"Overall goal: {goal}\n"
        f"Current progress step: {step_num}/5\n\n"
        "Requirements:\n"
        "- The change should be incremental and plausible.\n"
        "- Preserve layout and visual identity.\n"
        "- Make text/buttons readable.\n"
    )


def _type34_initial_prompt(metadata: dict[str, Any], first_step: dict[str, Any]) -> str:
    app_name = _text_or_default(metadata.get("app_name"), "App")
    final_goal = _text_or_default(metadata.get("final_goal") or metadata.get("instruction"), "Complete the task.")
    visual_description = _text_or_default(
        metadata.get("visual_description")
        or first_step.get("visual_description")
        or first_step.get("description"),
        "A clean product-quality app home screen.",
    )
    return (
        "Generate the first GUI frame for a task trajectory.\n\n"
        f"App name: {app_name}\n"
        f"Final goal: {final_goal}\n"
        f"Visual description:\n{visual_description}\n\n"
        "Requirements:\n"
        "- Generate a production-looking UI screenshot only.\n"
        "- Keep the layout coherent and readable.\n"
    )


def _type34_next_prompt(step_num: int, step_info: dict[str, Any]) -> str:
    action = _text_or_default(step_info.get("action") or step_info.get("instruction"), "Continue the task.")
    visual_description = _text_or_default(
        step_info.get("visual_description") or step_info.get("description"),
        "Reflect the expected next GUI state.",
    )
    return (
        "Using the previous frame as reference, generate the next GUI frame.\n\n"
        f"Step {step_num} action: {action}\n"
        f"Expected visual state:\n{visual_description}\n\n"
        "Requirements:\n"
        "- Only change UI regions affected by this action.\n"
        "- Preserve persistent bars, layout, and style.\n"
        "- Keep text and icons readable.\n"
    )


def _type5_prompt(metadata: dict[str, Any]) -> str:
    grounding = metadata.get("grounding") or {}
    explanation = _text_or_default(
        metadata.get("grounding_explanation") or grounding.get("effect") or grounding.get("description"),
        "Predict the immediate GUI reaction to the indicated target.",
    )
    return (
        "Using the reference GUI screenshot, predict the immediate GUI state after the grounded interaction.\n\n"
        f"Expected effect: {explanation}\n"
        f"Grounding metadata: {json.dumps(grounding, ensure_ascii=False)}\n\n"
        "Requirements:\n"
        "- Apply only the interaction-triggered change.\n"
        "- Preserve unrelated regions.\n"
        "- Keep the UI realistic and readable.\n"
    )


class LocalJudgeClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 600):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def evaluate(self, *, prompt: str, images: list[Image.Image]) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image in images:
            content.append({"type": "image_url", "image_url": {"url": pil_to_data_url(image)}})

        response = requests.post(
            build_openai_url(self.base_url, "/chat/completions"),
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": content}],
                "temperature": 0,
            },
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        message_content = response.json()["choices"][0]["message"]["content"]
        if isinstance(message_content, list):
            text = "\n".join(part.get("text", "") for part in message_content if part.get("type") == "text")
        else:
            text = str(message_content)
        return extract_json_object(text)


class GEBenchRunner:
    def __init__(
        self,
        *,
        dataset_root: Path,
        output_root: Path,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 8,
        guidance_scale: float | None = None,
        seed: int | None = 42,
    ):
        self.dataset_root = dataset_root
        self.output_root = output_root
        self.model = model
        self.width = width
        self.height = height
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.seed = seed
        self.client = VllmOmniImageClient(base_url=base_url, api_key=api_key)

    def generate(
        self,
        *,
        data_type: str,
        workers: int = 1,
        max_samples: int | None = None,
        samples_per_type: int | None = None,
    ) -> list[dict[str, Any]]:
        sample_paths = select_balanced_gebench_samples(
            {data_type: _iter_sample_paths(self.dataset_root, data_type)},
            samples_per_type=samples_per_type,
        )[data_type]
        if max_samples is not None:
            sample_paths = sample_paths[:max_samples]

        results: list[dict[str, Any]] = []
        if workers <= 1:
            for sample_path in sample_paths:
                result = self._generate_one(data_type, sample_path)
                if result:
                    results.append(result)
            return results

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(self._generate_one, data_type, sample_path) for sample_path in sample_paths]
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
        return results

    def _generate_one(self, data_type: str, sample_path: Path) -> dict[str, Any] | None:
        metadata = _load_metadata(sample_path)
        lang_device = _lang_device(sample_path, metadata)
        sample_name = _sample_name(sample_path)
        output_dir = ensure_dir(self.output_root / TYPE_TO_FOLDER[data_type] / lang_device / sample_name)

        if data_type == "type1":
            output_path = output_dir / "generated.png"
            if output_path.exists():
                return {"data_type": data_type, "sample_name": f"{lang_device}/{sample_name}", "output_path": str(output_path)}
            source = _resolve_referenced_image(
                metadata=metadata, sample_path=sample_path, dataset_root=self.dataset_root, data_type=data_type
            )
            if source is None:
                return None
            generated = self.client.generate_image_edit(
                model=self.model,
                prompt=_type1_prompt(metadata),
                images=source,
                width=self.width,
                height=self.height,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                seed=self.seed,
            )
            save_image(output_path, generated)
            return {"data_type": data_type, "sample_name": f"{lang_device}/{sample_name}", "output_path": str(output_path)}

        if data_type == "type2":
            goal = _text_or_default(metadata.get("question") or metadata.get("caption"), "Complete the task.")
            source = _resolve_referenced_image(
                metadata=metadata, sample_path=sample_path, dataset_root=self.dataset_root, data_type=data_type
            )
            if source is None:
                return None
            frame0_path = output_dir / "frame0.png"
            if not frame0_path.exists():
                save_image(frame0_path, source)
            previous = source
            for step_num in range(1, 6):
                frame_path = output_dir / f"frame{step_num}.png"
                if frame_path.exists():
                    previous = Image.open(frame_path).convert("RGB")
                    continue
                generated = self.client.generate_image_edit(
                    model=self.model,
                    prompt=_type2_prompt(goal, step_num),
                    images=previous,
                    width=self.width,
                    height=self.height,
                    num_inference_steps=self.num_inference_steps,
                    guidance_scale=self.guidance_scale,
                    seed=self.seed,
                )
                save_image(frame_path, generated)
                previous = generated
            output_path = output_dir / "frame5.png"
            return {"data_type": data_type, "sample_name": f"{lang_device}/{sample_name}", "output_path": str(output_path)}

        if data_type in {"type3", "type4"}:
            steps = _trajectory_steps(metadata)
            frame0_path = output_dir / "frame0.png"
            if frame0_path.exists():
                previous = Image.open(frame0_path).convert("RGB")
            else:
                previous = self.client.generate_text_to_image(
                    model=self.model,
                    prompt=_type34_initial_prompt(metadata, steps[0] if steps else {}),
                    width=self.width,
                    height=self.height,
                    num_inference_steps=self.num_inference_steps,
                    guidance_scale=self.guidance_scale,
                    seed=self.seed,
                )
                save_image(frame0_path, previous)

            for step_num in range(1, 6):
                frame_path = output_dir / f"frame{step_num}.png"
                if frame_path.exists():
                    previous = Image.open(frame_path).convert("RGB")
                    continue
                step_info = steps[step_num - 1] if step_num - 1 < len(steps) else {}
                generated = self.client.generate_image_edit(
                    model=self.model,
                    prompt=_type34_next_prompt(step_num, step_info),
                    images=previous,
                    width=self.width,
                    height=self.height,
                    num_inference_steps=self.num_inference_steps,
                    guidance_scale=self.guidance_scale,
                    seed=self.seed,
                )
                save_image(frame_path, generated)
                previous = generated
            output_path = output_dir / "frame5.png"
            return {"data_type": data_type, "sample_name": f"{lang_device}/{sample_name}", "output_path": str(output_path)}

        if data_type == "type5":
            output_path = output_dir / "generated.png"
            if output_path.exists():
                return {"data_type": data_type, "sample_name": f"{lang_device}/{sample_name}", "output_path": str(output_path)}
            source = _resolve_referenced_image(
                metadata=metadata, sample_path=sample_path, dataset_root=self.dataset_root, data_type=data_type
            )
            if source is None:
                return None
            generated = self.client.generate_image_edit(
                model=self.model,
                prompt=_type5_prompt(metadata),
                images=source,
                width=self.width,
                height=self.height,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                seed=self.seed,
            )
            save_image(output_path, generated)
            return {"data_type": data_type, "sample_name": f"{lang_device}/{sample_name}", "output_path": str(output_path)}

        raise ValueError(f"Unsupported data type: {data_type}")


class GEBenchEvaluator:
    def __init__(self, *, dataset_root: Path, output_root: Path, judge: LocalJudgeClient):
        self.dataset_root = dataset_root
        self.output_root = output_root
        self.judge = judge

    def evaluate(
        self,
        *,
        data_type: str,
        workers: int = 1,
        max_samples: int | None = None,
        samples_per_type: int | None = None,
    ) -> dict[str, Any]:
        output_type_dir = self.output_root / TYPE_TO_FOLDER[data_type]
        sample_dirs = [
            sample_dir
            for lang_dir in sorted(path for path in output_type_dir.iterdir() if path.is_dir())
            for sample_dir in sorted(path for path in lang_dir.iterdir() if path.is_dir())
        ]
        sample_dirs = select_balanced_gebench_samples(
            {data_type: sample_dirs},
            samples_per_type=samples_per_type,
        )[data_type]
        if max_samples is not None:
            sample_dirs = sample_dirs[:max_samples]
        results: list[dict[str, Any]] = []
        if workers <= 1:
            for sample_dir in sample_dirs:
                result = self._evaluate_one(data_type, sample_dir)
                if result:
                    results.append(result)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(self._evaluate_one, data_type, sample_dir) for sample_dir in sample_dirs]
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        results.append(result)

        payload = {"data_type": data_type, "results": results, "summary": summarize_gebench_results(results)}
        write_json(self.output_root / "evaluations" / f"{data_type}.json", payload)
        return payload

    def _evaluate_one(self, data_type: str, sample_dir: Path) -> dict[str, Any] | None:
        lang_device = sample_dir.parent.name
        sample_name = sample_dir.name
        dataset_sample = self.dataset_root / TYPE_TO_FOLDER[data_type] / lang_device / sample_name
        if not dataset_sample.exists() and data_type in {"type3", "type4"}:
            json_candidate = dataset_sample.with_suffix(".json")
            if json_candidate.exists():
                dataset_sample = json_candidate
        metadata = _load_metadata(dataset_sample)

        if data_type == "type1":
            source = _resolve_referenced_image(
                metadata=metadata, sample_path=dataset_sample, dataset_root=self.dataset_root, data_type=data_type
            )
            generated_path = find_first_image(sample_dir)
            if source is None or generated_path is None:
                return None
            generated = Image.open(generated_path).convert("RGB")
            raw_scores = self.judge.evaluate(prompt=_type1_prompt(metadata), images=[source, generated])
        elif data_type == "type2":
            frames = [Image.open(sample_dir / f"frame{i}.png").convert("RGB") for i in range(6)]
            goal = _text_or_default(metadata.get("question") or metadata.get("caption"), "Complete the task.")
            raw_scores = self.judge.evaluate(
                prompt=f"Evaluate a six-frame GUI trajectory.\nTask: {goal}",
                images=frames,
            )
        elif data_type in {"type3", "type4"}:
            frames = [Image.open(sample_dir / f"frame{i}.png").convert("RGB") for i in range(6)]
            instruction = _text_or_default(metadata.get("instruction") or metadata.get("caption"), "Complete the task.")
            raw_scores = self.judge.evaluate(
                prompt=f"Evaluate a six-frame GUI trajectory.\nInstruction: {instruction}",
                images=frames,
            )
        elif data_type == "type5":
            source = _resolve_referenced_image(
                metadata=metadata, sample_path=dataset_sample, dataset_root=self.dataset_root, data_type=data_type
            )
            generated_path = find_first_image(sample_dir)
            if source is None or generated_path is None:
                return None
            generated = Image.open(generated_path).convert("RGB")
            raw_scores = self.judge.evaluate(prompt=_type5_prompt(metadata), images=[source, generated])
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

        scores = _normalize_scores(raw_scores)
        return {
            "sample_name": f"{lang_device}/{sample_name}",
            "data_type": data_type,
            "scores": scores,
            "overall": _compute_overall(scores),
            "raw_scores": raw_scores,
        }


def _data_types_arg(value: str) -> list[str]:
    return list(TYPE_TO_FOLDER.keys()) if value == "all" else [value]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local GEBench generation and scoring against vLLM-Omni.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate")
    generate.add_argument("--dataset-root", type=Path, required=True)
    generate.add_argument("--output-root", type=Path, required=True)
    generate.add_argument("--base-url", type=str, required=True)
    generate.add_argument("--model", type=str, required=True)
    generate.add_argument("--data-type", choices=["all", *TYPE_TO_FOLDER.keys()], default="all")
    generate.add_argument("--api-key", type=str, default="EMPTY")
    generate.add_argument("--width", type=int, default=512)
    generate.add_argument("--height", type=int, default=512)
    generate.add_argument("--num-inference-steps", type=int, default=8)
    generate.add_argument("--guidance-scale", type=float, default=None)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--workers", type=int, default=1)
    generate.add_argument("--max-samples", type=int, default=None)
    generate.add_argument("--samples-per-type", type=int, default=None)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--dataset-root", type=Path, required=True)
    evaluate.add_argument("--output-root", type=Path, required=True)
    evaluate.add_argument("--data-type", choices=["all", *TYPE_TO_FOLDER.keys()], default="all")
    evaluate.add_argument("--judge-base-url", type=str, required=True)
    evaluate.add_argument("--judge-model", type=str, required=True)
    evaluate.add_argument("--judge-api-key", type=str, default="EMPTY")
    evaluate.add_argument("--workers", type=int, default=1)
    evaluate.add_argument("--max-samples", type=int, default=None)
    evaluate.add_argument("--samples-per-type", type=int, default=None)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--output-root", type=Path, required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "generate":
        runner = GEBenchRunner(
            dataset_root=args.dataset_root,
            output_root=args.output_root,
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
            width=args.width,
            height=args.height,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
        )
        records: list[dict[str, Any]] = []
        for data_type in _data_types_arg(args.data_type):
            records.extend(
                runner.generate(
                    data_type=data_type,
                    workers=args.workers,
                    max_samples=args.max_samples,
                    samples_per_type=args.samples_per_type,
                )
            )
        payload = {"records": records, "summary": summarize_generated_records(records)}
        write_json(args.output_root / "generation_manifest.json", payload)
        return 0

    if args.command == "evaluate":
        judge = LocalJudgeClient(
            base_url=args.judge_base_url,
            api_key=args.judge_api_key,
            model=args.judge_model,
        )
        evaluator = GEBenchEvaluator(dataset_root=args.dataset_root, output_root=args.output_root, judge=judge)
        combined_results: list[dict[str, Any]] = []
        for data_type in _data_types_arg(args.data_type):
            payload = evaluator.evaluate(
                data_type=data_type,
                workers=args.workers,
                max_samples=args.max_samples,
                samples_per_type=args.samples_per_type,
            )
            combined_results.extend(payload["results"])
        write_json(
            args.output_root / "evaluations" / "summary.json",
            {"summary": summarize_gebench_results(combined_results)},
        )
        return 0

    if args.command == "summarize":
        generation_summary = collect_gebench_generation_summary(args.output_root)
        evaluation_dir = args.output_root / "evaluations"
        result_records: list[dict[str, Any]] = []
        if evaluation_dir.exists():
            for file_path in sorted(evaluation_dir.glob("type*.json")):
                payload = load_json(file_path)
                result_records.extend(payload.get("results", []))
        payload: dict[str, Any] = {"generation": generation_summary}
        if result_records:
            payload["evaluation"] = summarize_gebench_results(result_records)
        write_json(args.output_root / "summary.json", payload)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 1
