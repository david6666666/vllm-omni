# vLLM-Omni Diffusion Quantization Feature Design Doc

# 1 Overview
Enable diffusion models in vLLM-Omni to support FP8 (native checkpoint + online fallback) and GGUF with unified stage-level configuration and consistent runtime behavior.

## 1.1 Motivation
Before this feature, diffusion quantization support was incomplete and not aligned with how users publish diffusion checkpoints in the ecosystem. In practice, users frequently need:
- Native FP8 checkpoint loading (for example: `unsloth/Qwen-Image-2512-FP8`)
- GGUF checkpoint loading (for example: `unsloth/Qwen-Image-2512-GGUF:<quant_type>`)

Two gaps existed:
- Diffusion path lacked a complete native FP8 auto-detection flow from model `quantization_config`.
- Diffusion path lacked a complete GGUF loading flow (reference parsing, tensor mapping, and post-load handling).

This feature closes those gaps and aligns diffusion behavior with Omni stage-level quantization semantics.

## 1.2 Target
### Feature
In scope:
- Diffusion quantization methods: `fp8`, `gguf`.
- Native FP8 auto-detection from model config when `quantization` is unset/`auto`.
- FP8 online fallback when user requests FP8 but no FP8 serialized config is present.
- GGUF loading with enforced `load_format=gguf`.
- Unified diffusion quantization inputs in `OmniDiffusionConfig`.
- Quantization config propagation into diffusion transformer linear layers.

Out of scope:
- AWQ/GPTQ/bitsandbytes/INT8 for diffusion.
- Quantizing non-transformer diffusion modules (text encoders, VAE) in this phase.
- New quantization kernels or algorithm changes.

### Accuracy
Correctness criteria:
- Native FP8 checkpoint path must preserve checkpoint quantization semantics (`Fp8Config.from_config`).
- FP8 online fallback must only happen when explicit FP8 is requested and no serialized FP8 config exists.
- GGUF path must fail fast for invalid format coupling (`quantization=gguf` with non-gguf `load_format`).
- GGUF tensor names must map to model parameter names (or prefixed names), otherwise fail with clear error.
- Existing non-quantized diffusion loading behavior must remain unchanged when no quantization is requested/detected.

### Performance
Expected behavior and trade-offs:
- Native FP8: lower memory footprint than BF16/FP16 checkpoints with limited post-load cost.
- FP8 online fallback: additional peak memory and load-time overhead (high-precision load + quantize after loading).
- GGUF: reduced memory footprint; compatibility and speed depend on model tensor naming/layout and backend support.

Performance constraints:
- No regression to non-quantized load path.
- Quantized path must keep load-time observability (`Loading weights took ... seconds`).

# 2 Design
## 2.1 Overview of Design
Design principles:
- Single resolver for diffusion quantization method + quant config materialization.
- Keep runtime behavior explicit: detect -> validate -> load -> post-process.
- Inject quant config at model-construction boundary (linear layers) via forward context.

High-level data/control flow:
1. User provides `engine_args` / `OmniDiffusion` kwargs.
2. `OmniDiffusionConfig` carries quantization fields.
3. `DiffusionModelRunner` resolves quantization method and quant config.
4. `DiffusersPipelineLoader` chooses load path (`hf` or `gguf`) and loads weights.
5. `process_weights_after_loading` finalizes quantized modules.
6. During model construction, transformer linear layers consume `od_config.quant_config`.

Conceptual diagram:
```text
User/Stage Args
    -> OmniDiffusionConfig(quantization, load_format, quantization_config_*)
        -> DiffusionModelRunner.load_model()
            -> infer_diffusion_quantization_method()
            -> resolve_diffusion_quant_config()
            -> DiffusersPipelineLoader(load_format)
                -> HF safetensors iterator  OR  GGUF iterator
            -> process_weights_after_loading()
            -> Ready pipeline
```

Rationale:
- Native FP8 detection is based on authoritative model metadata (`quantization_config`) to avoid accidental online quantization.
- GGUF path is explicit and validated to reduce ambiguous behavior.
- Quant config injection via forward context minimizes invasive constructor signature changes across many transformer implementations.

## 2.2 Use Cases
1. Native FP8 checkpoint serving (primary)
- Scenario: user serves `unsloth/Qwen-Image-2512-FP8`.
- Config: no explicit `--quantization` required.
- Benefit: auto-detect FP8 serialized checkpoint and run native FP8 path directly.

2. Memory-constrained GGUF deployment (secondary)
- Scenario: user serves GGUF variant in low-memory environment.
- Config: `--quantization gguf --load-format gguf`, model reference as `<repo_id>:<quant_type>` or `<repo_id>/<file>.gguf`.
- Benefit: lower memory footprint with explicit and validated GGUF loading behavior.

## 2.3 API Design
### Current Component Changes
`vllm_omni/diffusion/data.py`
- Change: add quantization-related fields in `OmniDiffusionConfig`:
  - `quantization`, `quantization_config_file`, `quantization_config_dict_json`,
    `quantization_scope`, `load_format`, `quant_config`.
- Why: unify stage config inputs and runtime state carrier.
- Impact: diffusion stages can accept and propagate quantization intent consistently.

`vllm_omni/model_executor/model_loader/quant_utils.py`
- Change: centralized method inference and quant config resolution for diffusion.
- Why: avoid duplicated logic and make native FP8 auto-detect explicit.
- Impact: deterministic selection of `None` / `fp8` / `gguf` and corresponding config object.

`vllm_omni/diffusion/worker/diffusion_model_runner.py`
- Change: resolve quantization method before load, auto-set `load_format` for GGUF, store `quant_config` in both `od_config` and `vllm_config`.
- Why: runner is the correct boundary between config and concrete loader behavior.
- Impact: consistent runtime behavior across offline inference and online serving.

`vllm_omni/diffusion/model_loader/diffusers_loader.py`
- Change: add GGUF load path:
  - parse GGUF references (local file / URL / `<repo>/<file>.gguf` / `<repo>:<quant_type>`)
  - iterate GGUF tensors and map to model state dict keys
  - run `process_weights_after_loading` after weight load
- Why: diffusion loader previously only handled HF-style weights.
- Impact: diffusion models can now load GGUF checkpoints with validation and post-processing.

`vllm_omni/diffusion/utils/quant_utils.py` + `vllm_omni/diffusion/models/*_transformer.py`
- Change: add `get_diffusion_quant_config()` and inject it into vLLM linear layers (`ColumnParallelLinear`, `RowParallelLinear`, `QKVParallelLinear`, `MergedColumnParallelLinear`, `ReplicatedLinear`) in diffusion transformers.
- Why: quant method attachment happens at layer creation time.
- Impact: quantization method is applied consistently across supported diffusion transformer variants.

### New APIs
`vllm_omni.model_executor.model_loader.quant_utils.infer_diffusion_quantization_method(...) -> str | None`
- Purpose: infer effective quantization method from explicit args and model quantization metadata.
- Inputs:
  - `quantization`, `quantization_config_file`, `quantization_config_dict_json`, `model`.
- Output:
  - `None`, `"fp8"`, `"gguf"` (or explicit supported method).

`vllm_omni.model_executor.model_loader.quant_utils.resolve_diffusion_quant_config(...) -> object | None`
- Purpose: materialize runtime quant config object.
- Outputs:
  - `Fp8Config` for FP8 path.
  - `GGUFConfig` for GGUF path.
  - `None` when quantization is not used.

`vllm_omni.diffusion.utils.quant_utils.get_diffusion_quant_config()`
- Purpose: access current diffusion quant config via forward context.
- Output: `od_config.quant_config` or `None`.

`OmniDiffusionConfig` new public fields
- `quantization`, `quantization_config_file`, `quantization_config_dict_json`,
  `quantization_scope`, `load_format`, `quant_config`.

## 2.4 API call dependency
Main call graph (offline/online share this path):
1. User creates `Omni(...)` / starts `vllm serve ... --omni`.
2. Stage config fields are collected into `OmniDiffusionConfig` (`_build_od_config` path).
3. `DiffusionModelRunner.load_model()`:
   - calls `infer_diffusion_quantization_method(...)`.
   - applies GGUF coupling rule (`gguf` -> `load_format=gguf` when auto).
   - calls `resolve_diffusion_quant_config(...)`.
   - writes `od_config.quant_config` and `vllm_config.quant_config`.
4. `DiffusersPipelineLoader.load_model()`:
   - initializes model with diffusion registry.
   - `load_weights()` chooses HF or GGUF path.
   - calls `process_weights_after_loading(...)`.
5. During module construction, diffusion transformer layers call
   `get_diffusion_quant_config()` and bind quant method.

Error paths:
- `quantization="gguf"` and `load_format != "gguf"` -> `ValueError`.
- Unsupported quantization method -> `ValueError`.
- GGUF reference parse failure -> `ValueError`.
- GGUF tensor name mapping failure -> `ValueError`.

# 3 Test cases
## 3.1 Unit Test (UT) design
Existing UT in this feature:
- File: `tests/diffusion/test_quant_utils.py`
1. `test_infer_diffusion_quantization_method_auto_detect_fp8`
   - Verifies native FP8 is auto-detected from mocked HF quant config.
2. `test_infer_diffusion_quantization_method_no_quant_config`
   - Verifies method inference returns `None` when quant config is absent.
3. `test_resolve_diffusion_quant_config_native_fp8`
   - Verifies native FP8 path builds `Fp8Config` with serialized checkpoint semantics.
4. `test_resolve_diffusion_quant_config_online_fp8_fallback`
   - Verifies explicit `quantization="fp8"` with no metadata falls back to online FP8 config.

Recommended additional UT:
- GGUF format coupling validation (`gguf` + non-gguf load format should fail).
- GGUF name mapping fail-fast path in loader.

## 3.2 Smoke Test (ST) design
1. Offline native FP8 smoke
- Command:
  - `python examples/offline_inference/text_to_image/text_to_image.py --model unsloth/Qwen-Image-2512-FP8 ...`
- Expectation:
  - model loads successfully;
  - log indicates auto-detected FP8;
  - generated image is produced.

2. Online GGUF smoke
- Command:
  - `vllm serve "unsloth/Qwen-Image-2512-GGUF:Q8_0" --omni --quantization gguf --load-format gguf ...`
- Expectation:
  - server boots without quantization-format mismatch errors;
  - `/v1/images/generations` returns valid base64 image payload.

3. Negative smoke for validation
- Command:
  - `vllm serve "unsloth/Qwen-Image-2512-GGUF:Q8_0" --omni --quantization gguf --load-format hf`
- Expectation:
  - fail-fast with clear `GGUF requires load_format='gguf'` error.
