# GEdit-Bench on vLLM-Omni

This integration adapts the upstream `stepfun-ai/Step1X-Edit/GEdit-Bench`
evaluation flow into `vllm-omni/benchmarks/accuracy/image_to_image`.

Upstream mapping:

- `run_gedit_score.py` -> `run_gedit_bench.py evaluate`
- `calculate_statistics.py` -> `run_gedit_bench.py summarize`
- upstream output layout under `results/<method>/fullset/...` is preserved

What changed:

- The upstream repo mainly ships evaluation scripts. This integration adds a
  generation runner that uses the local `vllm-omni` OpenAI-compatible
  `/v1/images/edits` endpoint to produce benchmark outputs in the expected
  directory structure.
- The evaluator keeps the same VIEScore-style decomposition:
  - `sementics_score`
  - `quality_score`
  - `overall_score = sqrt(sementics_score * quality_score)`
- Judge calls are routed to a local OpenAI-compatible model served by
  `vllm-omni`, not a remote provider.

Dataset:

- Default `--dataset-ref` is `stepfun-ai/GEdit-Bench`
- You can also pass a local dataset directory previously saved with
  Hugging Face `datasets`

Example usage:

```bash
python benchmarks/accuracy/image_to_image/run_gedit_bench.py generate \
  --output-root benchmarks/accuracy/image_to_image/results \
  --base-url http://127.0.0.1:8000 \
  --model Qwen/Qwen-Image-Edit \
  --model-name qwen_image_edit \
  --dataset-ref stepfun-ai/GEdit-Bench \
  --task-type all \
  --instruction-language en
```

```bash
python benchmarks/accuracy/image_to_image/run_gedit_bench.py evaluate \
  --output-root benchmarks/accuracy/image_to_image/results \
  --model-name qwen_image_edit \
  --save-dir benchmarks/accuracy/image_to_image/scores \
  --dataset-ref stepfun-ai/GEdit-Bench \
  --judge-base-url http://127.0.0.1:8000 \
  --judge-model Qwen/Qwen2.5-VL-7B-Instruct \
  --judge-api-key EMPTY
```

```bash
python benchmarks/accuracy/image_to_image/run_gedit_bench.py summarize \
  --csv-path benchmarks/accuracy/image_to_image/scores/qwen_image_edit_all_all_vie_score.csv \
  --language en
```

Notes:

- This flow requires the optional Hugging Face `datasets` package.
- `generate` writes `generation_manifest.json` with local output coverage.
- The current repo marker set exposes `L4` but not `L5`, so if you promote an
  end-to-end smoke test into CI, use the existing `advanced_model`, `benchmark`,
  and `L4` markers or introduce a new repo-wide marker explicitly first.
