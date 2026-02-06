# Diffusion Quantization (FP8 + GGUF)
Date: 2026-02-06
Status: Design

## Summary
This document defines quantization support for diffusion models in vLLM-Omni,
limited to FP8 and GGUF. It also standardizes quantization inputs and behavior
across Omni (AR) and Diffusion stages.

## Goals
- Support FP8 and GGUF for diffusion models.
- Keep quantization inputs and behavior consistent across Omni and Stage.
- FP8 supports both native FP8 checkpoints and online quantization fallback.
- GGUF supports load_format=gguf.
- Default diffusion scope is transformer-only.

## Non-Goals
- No AWQ/GPTQ/bitsandbytes/INT8 in this phase.
- No quantization for diffusion text encoders or VAE.
- No new quantization algorithms.

## Unified Quantization Inputs
All stages (Omni/AR and Diffusion) accept the same set of inputs:
- quantization: "fp8" | "gguf" | "auto" | None
- quantization_config_file: str | None
- quantization_config_dict_json: str | None
- quantization_scope: str = "transformer_only"
- load_format: str = "auto"

Entry points:
- Stage config: engine_args.*
- Python API: OmniLLM(...) / OmniDiffusion(...)

## Consistency Rules
| Input | Omni (AR) | Diffusion |
|---|---|---|
| quantization=None + no quant config | no quant | no quant |
| quantization=None + fp8 quant config | auto-detect fp8 | auto-detect fp8 |
| quantization="fp8" | fp8 (native or online fallback) | fp8 (native or online fallback) |
| quantization="gguf" | gguf loader | gguf loader |
| quantization="gguf" + load_format!=gguf | error | error |

## Architecture Changes
### 1) Unified Quantization Resolver
Add a small utility to resolve quantization settings consistently:

Path (new):
- vllm_omni/model_executor/model_loader/quant_utils.py

Behavior:
- fp8: resolve to Fp8Config (from config file/json or HF config).
- gguf: validate load_format=gguf, return GGUFConfig.
- auto-detect: when quantization is None or "auto", inspect
  `quantization_config` from model config and infer fp8 if available.

### 2) Diffusion Loader
File:
- vllm_omni/diffusion/model_loader/diffusers_loader.py

Changes:
- Add GGUF loading branch when load_format=="gguf".
- After load_weights, call process_weights_after_loading for FP8 online quant.

### 3) Diffusion Transformer Construction
Files:
- vllm_omni/diffusion/models/*/*transformer*.py

Changes:
- Pass quant_config to vLLM linear layers:
  - ColumnParallelLinear
  - RowParallelLinear
  - QKVParallelLinear
- Provide correct prefix names for quant skip logic.

### 4) Forward Context
Use set_current_vllm_config during diffusion model init and forward
to enable vLLM CustomOp dispatch with quant_config.

## FP8 Flow (Diffusion)
### Native FP8 checkpoint (preferred)
1) quantization is None/"auto" or "fp8".
2) Resolver reads `quantization_config` from config file/json or HF config.
3) Build diffusion transformer with quant_config injected.
4) load_weights loads FP8-serialized weights and scales.
5) process_weights_after_loading runs post-processing/repacking only.

### Online FP8 fallback
1) quantization="fp8" and no FP8 quantization_config exists.
2) Resolver returns default Fp8Config (non-serialized checkpoint).
3) load_weights loads high precision weights.
4) process_weights_after_loading triggers online FP8 conversion.

Notes:
- Requires device capability support for FP8 to gain speedup.
- Online fallback needs enough memory to load FP16/BF16 weights first.

## GGUF Flow (Diffusion)
1) Resolve quantization="gguf" and enforce load_format="gguf".
2) Loader uses gguf download + gguf iterator.
3) GGUFConfig is used to enable gguf-aware weight handling.

Notes:
- GGUF is a model weight format, not a quantization algorithm.
- Works as weight-only quantization in practice.

### GGUF Variant Selection (e.g., Q8_0 vs Q4_K_M)
- `Q8_0`: 8-bit quantization; usually best quality, larger model size, higher memory use.
- `Q4_K_M`: 4-bit K-quant mixed variant; smaller model size and memory footprint, larger quality loss than Q8_0.
- Name hints:
  - `Q4` / `Q8`: approximate bit width level.
  - `_K`: K-quant family.
  - `_M`: mixed/medium variant, typically keeps key tensors at relatively higher precision.
- Practical guidance:
  - Prefer `Q8_0` when memory allows and quality is priority.
  - Use `Q4_K_M` when memory is tight and some quality drop is acceptable.

## Testing Plan
Unit:
- quantization=None + fp8 quantization_config -> auto-detect fp8.
- quantization="fp8" ensures quant_method exists on linear layers.
- quantization="gguf" ensures gguf loader path is chosen.

Integration:
- Run a small diffusion pipeline (or mock weights) to ensure:
  - load succeeds
  - native FP8 and online FP8 fallback both work

## Risks and Mitigations
- FP8 online quant may increase peak memory.
  - Mitigation: document requirement, allow CPU offload.
- GGUF compatibility depends on model structure and GGUF file layout.
  - Mitigation: clear error messages and validation.
- Incorrect prefixes may skip quantization or mis-apply it.
  - Mitigation: consistent prefix construction and tests.

## Open Questions
- Future: allow quantization_scope to include text encoders or VAE?
- Should GGUF be allowed for non-diffuser pipelines beyond current set?

## Example Commands
Offline inference (native FP8 checkpoint):
```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model unsloth/Qwen-Image-2512-FP8 \
  --prompt "cinematic photo of an arctic fox under aurora" \
  --height 1328 \
  --width 1328 \
  --num_inference_steps 30 \
  --output outputs/qwen_image_fp8.png
```

Offline inference (GGUF checkpoint):
```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model "unsloth/Qwen-Image-2512-GGUF:Q8_0" \
  --quantization gguf \
  --load_format gguf \
  --prompt "a watercolor painting of tokyo in the rain" \
  --height 1024 \
  --width 1024 \
  --num_inference_steps 28 \
  --output outputs/qwen_image_gguf.png
```

Online serving (native FP8 checkpoint):
```bash
vllm serve unsloth/Qwen-Image-2512-FP8 --omni --port 8091
```

Online serving (GGUF checkpoint):
```bash
vllm serve "unsloth/Qwen-Image-2512-GGUF:Q8_0" --omni --port 8091 \
  --quantization gguf \
  --load-format gguf
```
