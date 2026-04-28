# Quantization

vLLM-Omni exposes quantization through the unified `quantization_config`
path. The same configuration entrypoint is used across diffusion-only models,
multi-stage omni/TTS models, and multi-stage diffusion models, but each model
type has a different quantization scope.

## Model Type Support

### Diffusion Model (Qwen-Image, Wan2.2)

These models run a diffusion transformer as the primary inference module. The
default quantization target is the transformer; tokenizer, scheduler, text
encoder, and VAE stay on the base checkpoint unless a method guide says
otherwise.

| Method | Guide | Mode | Example models | Status |
|--------|-------|------|----------------|--------|
| FP8 | [FP8](fp8.md) | Load-time W8A8 | Qwen-Image; Wan2.2 is not validated | Validated for Qwen-Image family and other DiT models |
| Int8 | [Int8](int8.md) | Load-time or serialized W8A8 | Qwen-Image; Wan2.2 is not validated | Validated for Qwen-Image and Z-Image |
| GGUF | [GGUF](gguf.md) | Pre-quantized transformer weights | Qwen-Image | Validated where a model-specific GGUF adapter exists |
| AutoRound | [AutoRound](autoround.md) | Pre-quantized W4A16 checkpoints | FLUX.1-dev; Qwen-Image/Wan2.2 not validated | Checkpoint-driven |
| msModelSlim | [msModelSlim](msmodelslim.md) | Pre-quantized Ascend checkpoints | Wan2.2 recipe; HunyuanImage-3.0 inference target | Ascend/NPU path |

### Multi-Stage Omni/TTS Model (Qwen3-Omni, Qwen3-TTS)

These models combine an AR language model with audio, vision, talker, or TTS
stages. Quantization is scoped to the AR language-model stage when the
checkpoint contains a supported `quantization_config`; the non-AR stages stay
in BF16 unless the model guide explicitly adds support.

| Method | Guide | Scope | Example models | Status |
|--------|-------|-------|----------------|--------|
| FP8 | [FP8](fp8.md) | Thinker or language-model checkpoint config | Qwen3-Omni thinker | ModelOpt checkpoint path |
| Int8 | [Int8](int8.md) | Not currently validated for omni/TTS stages | Qwen3-Omni, Qwen3-TTS | Not validated |
| GGUF | [GGUF](gguf.md) | Not currently validated for omni/TTS stages | Qwen3-Omni, Qwen3-TTS | Not validated |
| AutoRound | [AutoRound](autoround.md) | Thinker or language-model checkpoint config | Qwen2.5-Omni, Qwen3-Omni | Supported through AutoRound checkpoints |
| msModelSlim | [msModelSlim](msmodelslim.md) | Not currently validated for omni/TTS stages | Qwen3-Omni, Qwen3-TTS | Not validated |

### Multi-Stage Diffusion Model (BAGEL, GLM-Image)

These models split generation across multiple stages. Quantization must be
attached to the intended stage rather than applied globally.

| Method | Guide | Scope | Example models | Status |
|--------|-------|-------|----------------|--------|
| FP8 | [FP8](fp8.md) | Stage-specific DiT or transformer module | BAGEL, GLM-Image | Requires model-specific validation |
| Int8 | [Int8](int8.md) | Stage-specific DiT or transformer module | BAGEL, GLM-Image | Requires model-specific validation |
| GGUF | [GGUF](gguf.md) | Stage-specific transformer weights | BAGEL, GLM-Image | No validated adapter listed |
| AutoRound | [AutoRound](autoround.md) | Checkpoint-defined stage | BAGEL, GLM-Image | No validated checkpoint listed |
| msModelSlim | [msModelSlim](msmodelslim.md) | Ascend-generated stage weights | GLM-Image | Requires model-specific adaptation |

!!! note
    "Dynamic" means vLLM-Omni computes the quantization data at load time.
    "Static" means the checkpoint or external quantizer provides the required
    quantized weights and scales.

## Quantization Scope

### Diffusion Model (Qwen-Image, Wan2.2)

The default target is the diffusion transformer. Component routing is available
through `build_quant_config()`:

```python
from vllm_omni.quantization import build_quant_config

config = build_quant_config({
    "transformer": {"method": "fp8"},
    "vae": None,
})
```

| Component | Default quantized? | Notes |
|-----------|--------------------|-------|
| Diffusion transformer | Yes | Primary target for FP8, Int8, GGUF, AutoRound, and msModelSlim |
| Text encoder | No | Keep BF16 unless a method-specific guide documents support |
| VAE | No | Keep BF16; storage-only paths are method-specific |
| Scheduler/tokenizer | No | Loaded from the base model repository |

### Multi-Stage Omni/TTS Model (Qwen3-Omni, Qwen3-TTS)

| Component | Default quantized? | Notes |
|-----------|--------------------|-------|
| Thinker or AR language model | Yes, when checkpoint config is supported | ModelOpt FP8/NVFP4 or AutoRound checkpoint config |
| Audio encoder | No | BF16 |
| Vision encoder | No | BF16 |
| Talker or TTS stage | No | BF16 unless model-specific support is documented |
| Code2Wav | No | BF16 |

### Multi-Stage Diffusion Model (BAGEL, GLM-Image)

| Component | Default quantized? | Notes |
|-----------|--------------------|-------|
| Selected diffusion or transformer stage | Method-specific | Must be routed to the intended stage |
| Other generation stages | No | Keep BF16 unless separately validated |
| VAE, tokenizer, scheduler | No | Loaded from the base checkpoint |

## Hardware Support

| Device | FP8 | Int8 | GGUF | AutoRound | msModelSlim |
|--------|-----|------|------|-----------|-------------|
| NVIDIA Blackwell GPU (SM 100+) | Yes | Yes | Yes | Yes | No |
| NVIDIA Ada/Hopper GPU (SM 89+) | Yes | Yes | Yes | Yes | No |
| NVIDIA Ampere GPU (SM 80+) | Weight-only FP8 where available | Yes | Yes | Yes | No |
| AMD ROCm | Not validated | Not validated | Not validated | Not validated | No |
| Intel XPU | Not validated | Not validated | Not validated | Yes, AutoRound checkpoints | No |
| Ascend NPU | No | Yes | No | No | Yes |

## Python API

`build_quant_config()` accepts strings, dictionaries, per-component
dictionaries, existing `QuantizationConfig` objects, or `None`.

```python
from vllm_omni.quantization import build_quant_config

build_quant_config("fp8")
build_quant_config({"method": "fp8", "activation_scheme": "static"})
build_quant_config("auto-round", bits=4, group_size=128)
build_quant_config({"method": "gguf", "gguf_model": "/path/to/model.gguf"})
build_quant_config({"transformer": {"method": "fp8"}, "vae": None})
build_quant_config(None)
```

## Migration Guide

### Before v0.16.0

```python
from vllm_omni.diffusion.quantization import get_diffusion_quant_config

config = get_diffusion_quant_config("fp8", activation_scheme="static")
```

### v0.16.0 and later

```python
from vllm_omni.quantization import build_quant_config

config = build_quant_config({"method": "fp8", "activation_scheme": "static"})
config = build_quant_config({
    "transformer": {"method": "fp8"},
    "vae": None,
})
```
