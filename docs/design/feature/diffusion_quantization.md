# vLLM-Omni Diffusion Quantization Feature Design Doc

Time context: this design is updated on 2026-02-06.

# 1 Overview
Add production-ready diffusion quantization by separating `base repo` and `quantized transformer weights source`, while keeping Omni and Stage quantization arguments consistent.

## 1.1 Motivation
Current diffusion quantization can parse GGUF and FP8 settings, but real-world deployment still has a key gap: many quantized repos (for example `unsloth/Qwen-Image-2512-GGUF:Q8_0`) only provide transformer weights and do not provide full diffusion assets (`model_index.json`, scheduler, tokenizer, VAE, text encoder).  
If users pass such a repo directly as `--model`, initialization can fail before or during pipeline construction.

vLLM GGUF documentation already gives two important practices we should reuse:
- GGUF should be treated as a weight format, not a full model package.
- Keep tokenizer/config from base model, and use `hf-config-path` when metadata/config is missing.

So for diffusion in vLLM-Omni, the robust model is:
- `--model`: full base diffusion repo (control-plane assets + non-transformer components).
- `--quantized-weights`: quantized transformer weights source (GGUF or FP8 repo/file).

This also satisfies the consistency goal across Omni and Stage abstractions: one set of quantization arguments, one behavior contract.

## 1.2 Target

### Feature
In scope:
- Quantization methods for diffusion transformer: `fp8`, `gguf`.
- Native FP8 and FP8 online fallback.
- GGUF loading aligned with vLLM behavior:
  - support local `.gguf`, URL `.gguf`, `<repo>/<file>.gguf`, `<repo>:<quant_type>`.
  - enforce format coupling (`gguf` requires `gguf` load format).
  - keep base tokenizer/config/non-transformer components from base repo.
- New split-loading contract:
  - base model source (`model`).
  - quantized transformer source (`quantized_weights`, optional).
- Keep Omni and Stage input semantics identical.

Out of scope:
- Quantization for text encoder/VAE in this phase.
- New kernel algorithm work.
- Multi-file GGUF merge logic inside vLLM-Omni (follow vLLM guidance to provide single-file input).

### Accuracy
Correctness criteria:
- Base pipeline assets must always come from base repo unless user explicitly overrides them.
- Quantized weights are applied only to target scope (`transformer_only` by default).
- Native FP8 must use checkpoint quantization metadata when present.
- FP8 online fallback is only used when explicit FP8 is requested but serialized FP8 metadata is absent.
- GGUF path must fail fast on invalid input or mapping mismatch.
- Existing non-quantized behavior remains unchanged.

### Performance
Targets and tradeoffs:
- GGUF / FP8 should reduce memory footprint vs BF16/FP16 transformer weights.
- FP8 online fallback may increase load latency and peak memory.
- Split-loading must not regress non-quantized load path.
- Keep load observability logs and clear quantization mode logs.

# 2 Design

## 2.1 Overview of Design
Core design: split diffusion loading into two planes.

- Base plane:
  - `model` points to base diffusion repo.
  - Load config, scheduler, tokenizer, text encoder, VAE, and pipeline skeleton from base repo.
- Quantized transformer plane:
  - `quantized_weights` points to quantized transformer source.
  - Apply quantization only to transformer weights based on `quantization` + `load_format`.

High-level flow:
```text
User args (Omni kwargs / stage engine_args)
  -> OmniDiffusionConfig
      model = base repo
      quantized_weights = optional quant source
      quantization/load_format/quantization_config_*
  -> DiffusionModelRunner
      resolve method + quant config
  -> DiffusersPipelineLoader
      init pipeline components from base repo
      load transformer weights from:
         quantized_weights (if set) else model
      post-process quant modules
  -> ready for generation
```

Why this aligns with vLLM GGUF practice:
- GGUF is treated as weight payload.
- Base tokenizer/config stay in base repo.
- `hf-config-path` remains available for edge cases, but split-loading removes most config-missing failures by design.

## 2.2 Use Cases
1. Online serving with GGUF transformer (primary)
- Scenario: serve Qwen-Image with GGUF transformer from Unsloth.
- Config:
  - `model=Qwen/Qwen-Image-2512`
  - `quantized_weights=unsloth/Qwen-Image-2512-GGUF:Q8_0`
  - `quantization=gguf`, `load_format=gguf`
- Benefit: reliable startup because base assets and quantized weights come from the right sources.

2. Offline inference with native FP8 transformer (primary)
- Scenario: run offline text-to-image with FP8 safetensors transformer.
- Config:
  - `model=Qwen/Qwen-Image-2512`
  - `quantized_weights=unsloth/Qwen-Image-2512-FP8`
  - `quantization` can be `auto` or `fp8`
- Benefit: preserve native FP8 behavior while keeping base pipeline stable.

3. Backward compatibility (secondary)
- Scenario: user keeps existing non-quantized command.
- Config:
  - only `model`, no `quantized_weights`, no `quantization`.
- Benefit: no behavior change.

## 2.3 API Design

### Current Component Changes
`vllm_omni/diffusion/data.py` (`OmniDiffusionConfig`)
- Add:
  - `quantized_weights: str | None = None`
- Keep existing:
  - `quantization`, `load_format`, `quantization_config_file`,
    `quantization_config_dict_json`, `quantization_scope`.
- Why:
  - explicit source split between base model and quantized transformer weights.
- Impact:
  - one argument contract works for both Omni kwargs and stage YAML.

`vllm_omni/entrypoints/cli/serve.py` and offline example scripts
- Add CLI flag:
  - serve: `--quantized-weights`
  - offline script: `--quantized_weights`
- Why:
  - expose split-loading contract in user-facing entrypoints.
- Impact:
  - current commands remain valid; new recommended commands become stable for quantized repos.

`vllm_omni/entrypoints/omni_stage.py` (`_build_od_config`)
- Ensure `quantized_weights` is propagated from engine args into diffusion config.
- Why:
  - keep Stage and Omni behavior identical.

`vllm_omni/diffusion/model_loader/diffusers_loader.py`
- Change loading policy:
  - always initialize non-transformer components from base `model`.
  - for transformer `weights_sources`, replace `model_or_path` with `quantized_weights` when provided.
  - GGUF parsing/mapping remains on transformer source.
- Why:
  - fix current "quantized repo lacks scheduler/tokenizer/vae" failure mode.
- Impact:
  - split-loading works for GGUF and FP8 repos.

`vllm_omni/model_executor/model_loader/quant_utils.py`
- Method inference source priority update:
  1. explicit `quantization` (if not `auto`)
  2. explicit quant config file/json
  3. `quantized_weights` repo config (if present)
  4. base `model` repo config
- Why:
  - when quantized weights are external, auto-detect should read the correct repo first.

### New APIs
`OmniDiffusionConfig.quantized_weights: str | None`
- Meaning:
  - source for quantized transformer weights.
- Supported values:
  - local file/path
  - remote GGUF (`<repo>:<quant_type>` or `<repo>/<file>.gguf`)
  - remote FP8 repo/path

CLI/script args:
- serve: `--quantized-weights <source>`
- offline script: `--quantized_weights <source>`
- shared semantics for:
  - `vllm serve ... --omni`
  - `examples/offline_inference/text_to_image/text_to_image.py`

Stage YAML (`engine_args`) new field:
- `quantized_weights: "<source>"`

### Recommended command examples
Online GGUF:
```bash
vllm serve Qwen/Qwen-Image-2512 --omni --port 8091 \
  --quantized-weights "unsloth/Qwen-Image-2512-GGUF:Q8_0" \
  --quantization gguf \
  --load-format gguf
```

Offline GGUF:
```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model Qwen/Qwen-Image-2512 \
  --quantized_weights "unsloth/Qwen-Image-2512-GGUF:Q8_0" \
  --quantization gguf \
  --load_format gguf \
  --prompt "a watercolor painting of tokyo in the rain" \
  --height 1024 \
  --width 1024 \
  --num_inference_steps 28 \
  --output outputs/qwen_image_gguf.png
```

Offline FP8:
```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model Qwen/Qwen-Image-2512 \
  --quantized_weights "unsloth/Qwen-Image-2512-FP8" \
  --quantization fp8 \
  --prompt "cinematic photo of an arctic fox under aurora" \
  --height 1328 \
  --width 1328 \
  --num_inference_steps 30 \
  --output outputs/qwen_image_fp8.png
```

Online FP8:
```bash
vllm serve Qwen/Qwen-Image-2512 --omni --port 8091 \
  --quantized-weights "unsloth/Qwen-Image-2512-FP8" \
  --quantization fp8
```

FP8 notes:
- Native FP8 auto-detect requires `quantization_config` to be available from
  quantization config inputs or model repo config files.
- If a quantized repo only contains safetensors weights (no config metadata),
  `--quantization fp8` uses online FP8 fallback.

## 2.4 API call dependency
Main flow:
1. User passes `model` + optional `quantized_weights` and quantization args.
2. Args are propagated to `OmniDiffusionConfig`.
3. Quant method is resolved (`fp8`/`gguf`/none) with quant-source-aware priority.
4. Pipeline components are initialized from base `model`.
5. Transformer weight source is selected:
   - `quantized_weights` if set.
   - otherwise base `model`.
6. Loader executes HF or GGUF transformer load path.
7. Quant post-processing (`process_weights_after_loading`) finalizes modules.

Error paths:
- `quantization=gguf` with non-`gguf` load format -> fail fast.
- `quantized_weights` is GGUF but cannot resolve file or quant type -> fail fast.
- GGUF tensor name mapping is empty -> fail fast with explicit hint.
- user sets `model` as quant-only repo without full assets and does not set proper base model -> fail fast with migration hint.

Migration rule:
- Old style:
  - `--model unsloth/Qwen-Image-2512-GGUF:Q8_0`
- New recommended style:
  - `--model Qwen/Qwen-Image-2512`
  - `--quantized-weights unsloth/Qwen-Image-2512-GGUF:Q8_0`

# 3 Test cases

## 3.1 Unit Test (UT) design
`tests/diffusion/test_quant_utils.py`
- Add:
  1. `test_infer_method_prefers_quantized_weights_repo_config`
  2. `test_infer_method_falls_back_to_base_model_config`
  3. `test_resolve_quant_config_gguf_requires_gguf_load_format`

`tests/diffusion/test_diffusers_loader.py`
- Add:
  1. `test_transformer_source_switch_to_quantized_weights`
  2. `test_non_transformer_components_still_from_base_model`
  3. `test_gguf_reference_parse_repo_quant_type`
  4. `test_gguf_name_mapping_empty_raises`

`tests/entrypoints/test_omni_stage_diffusion_config.py`
- Add:
  1. `test_build_od_config_propagates_quantized_weights`

## 3.2 Smoke Test (ST) design
1. Online GGUF split-loading smoke
- Command:
  - `vllm serve Qwen/Qwen-Image-2512 --omni --quantized-weights "unsloth/Qwen-Image-2512-GGUF:Q8_0" --quantization gguf --load-format gguf`
- Expectation:
  - service starts and image generation endpoint works.

2. Offline GGUF split-loading smoke
- Command:
  - use Offline GGUF example command above.
- Expectation:
  - image generated successfully; logs show GGUF quant path.

3. Offline native FP8 split-loading smoke
- Command:
  - use Offline FP8 example command above.
- Expectation:
  - image generated successfully; logs show native FP8 detection or FP8 path.

4. Negative smoke
- Command:
  - same as GGUF command but `--load-format hf`.
- Expectation:
  - fail fast with explicit GGUF/load-format mismatch message.
