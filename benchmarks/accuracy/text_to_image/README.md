# GEBench on vLLM-Omni

This integration adapts the upstream `stepfun-ai/GEBench` scripts into
`vllm-omni/benchmarks/accuracy/text_to_image`.

Upstream mapping:

- `scripts/generate.py` -> `run_gebench.py generate`
- `scripts/evaluate.py` -> `run_gebench.py evaluate`
- upstream prompt / judge logic -> `gbench.py`

What changed:

- Generation calls the local OpenAI-compatible `vllm-omni` endpoints:
  - `/v1/images/generations` for text-only frame generation
  - `/v1/chat/completions` for image-conditioned GUI transition generation
- Evaluation still keeps the GEBench scoring dimensions:
  - `goal`
  - `logic`
  - `cons`
  - `ui`
  - `qual`
- Judge calls are also routed to a local OpenAI-compatible model served by
  `vllm-omni` instead of a remote service.

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
  --judge-base-url http://127.0.0.1:8000 \
  --judge-model Qwen/Qwen2.5-VL-7B-Instruct \
  --judge-api-key EMPTY
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
- `summarize` will report both generated coverage and any existing evaluation
  summary files.
