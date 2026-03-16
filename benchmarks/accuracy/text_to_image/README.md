# GEBench on vLLM-Omni

This integration adapts the upstream `stepfun-ai/GEBench` scripts into
`vllm-omni/benchmarks/accuracy/text_to_image`.

Upstream mapping:

- `scripts/generate.py` -> `run_gebench.py generate`
- `scripts/evaluate.py` -> `run_gebench.py evaluate`
- upstream `gui_agent` prompt / judge logic -> `gbench.py`

What changed:

- Generation no longer calls Gemini directly. It now calls the
  OpenAI-compatible `vllm-omni` endpoints:
  - `/v1/images/generations` for text-only frame generation
  - `/v1/chat/completions` for image-conditioned GUI transition generation
- Evaluation no longer depends on the upstream repo layout. It reads the
  generated artifacts from the local benchmark output tree and reuses the same
  GEBench-style scoring dimensions: `goal`, `logic`, `cons`, `ui`, `qual`.

Expected dataset layout:

- Clone the benchmark dataset from Hugging Face into a local directory:

```bash
git clone https://huggingface.co/datasets/stepfun-ai/GEBench /path/to/GEBench
```

Example usage:

```bash
python benchmarks/accuracy/text_to_image/run_gebench.py generate \
  --dataset-root /path/to/GEBench \
  --output-root benchmarks/accuracy/text_to_image/outputs \
  --base-url http://127.0.0.1:8000 \
  --model Tongyi-MAI/Z-Image-Turbo \
  --data-type type3
```

```bash
python benchmarks/accuracy/text_to_image/run_gebench.py evaluate \
  --dataset-root /path/to/GEBench \
  --output-root benchmarks/accuracy/text_to_image/outputs \
  --data-type type3 \
  --judge-base-url https://api.openai.com/v1 \
  --judge-model gpt-4.1 \
  --judge-api-key "$OPENAI_API_KEY"
```

```bash
python benchmarks/accuracy/text_to_image/run_gebench.py summarize \
  --output-root benchmarks/accuracy/text_to_image/outputs
```

Notes:

- GEBench upstream leaves type3/type4 generation unfinished. This integration
  fills that gap with a trajectory runner that generates `frame0.png` followed
  by `frame1.png` ... `frame5.png`.
- Type1/2/5 require an image-edit capable model exposed through
  `vllm-omni serve`.
