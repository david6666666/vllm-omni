# GEdit-Bench on vLLM-Omni

This integration adapts the upstream `stepfun-ai/Step1X-Edit/GEdit-Bench`
evaluation flow into `vllm-omni/benchmarks/accuracy/image_to_image`.

Upstream mapping:

- `run_gedit_score.py` -> `run_gedit_bench.py evaluate`
- `calculate_statistics.py` -> `run_gedit_bench.py summarize`
- upstream output layout under `results/<method>/fullset/...` is preserved

What changed:

- The upstream repo only ships evaluation scripts. This integration adds a
  generation runner that uses the `vllm-omni` OpenAI-compatible edit endpoint
  to generate benchmark outputs in the expected directory structure.
- The evaluator keeps the same two-part scoring idea as VIEScore:
  instruction-following / content preservation + perceptual quality.
- Heavy optional local judge backends from upstream are not pulled into the
  main repo. The default path is an OpenAI-compatible judge endpoint.

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
  --judge-base-url https://api.openai.com/v1 \
  --judge-model gpt-4.1 \
  --judge-api-key "$OPENAI_API_KEY"
```

```bash
python benchmarks/accuracy/image_to_image/run_gedit_bench.py summarize \
  --csv-path benchmarks/accuracy/image_to_image/scores/qwen_image_edit_all_all_vie_score.csv \
  --language en
```

Notes:

- This flow requires the optional Hugging Face `datasets` package.
- The current repo marker set exposes `L4` but not `L5`, so if you promote an
  end-to-end smoke test into CI, use the existing `advanced_model`, `benchmark`,
  and `L4` markers or introduce a new repo-wide marker explicitly first.
