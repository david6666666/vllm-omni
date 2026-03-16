# Accuracy Benchmarks

This directory hosts accuracy benchmark integrations that adapt external
benchmark repos to the vLLM-Omni serving interface instead of keeping them as
standalone repositories.

Current integrations:

- `text_to_image/`: GEBench generation + judge/evaluation flow.
- `image_to_image/`: GEdit-Bench generation + VIEScore-style evaluation flow.

Design notes:

- Generation is executed through the OpenAI-compatible endpoints exposed by
  `vllm-omni serve`.
- Evaluation is kept separate from serving so the same generated artifacts can
  be re-scored with different judges.
- Output directory layout intentionally stays close to the upstream repos to
  keep downstream result comparison simple.

Test guidance:

- Local static/self-checks live in `tests/benchmarks/test_accuracy_bench_utils.py`.
- End-to-end generation/evaluation should be validated in a remote GPU
  environment. In the current repo marker system there is `L4` but no `L5`
  marker, so benchmark smoke tests should be wired as `advanced_model +
  benchmark + L4` when GPU capacity and credentials are available.
