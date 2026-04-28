# Quantization

vLLM-Omni exposes quantization through the unified `quantization_config`
path. Diffusion models can use load-time quantization, pre-quantized
checkpoints, or method-specific weight formats. Multi-stage omni models can
also load pre-quantized language-model checkpoints while leaving the other
stages in BF16.

## Supported Methods

### Diffusion Models

| Method | Guide | Mode | Dynamic / Static | Supported models | Hardware |
|--------|-------|------|------------------|------------------|----------|
| FP8 | [FP8](fp8.md) | Load-time W8A8 | Dynamic and static | Z-Image, Qwen-Image, FLUX.1/FLUX.2, HunyuanImage-3.0, HunyuanVideo-1.5 | CUDA SM 89+ for native W8A8; older CUDA uses weight-only kernels |
| Int8 | [Int8](int8.md) | Load-time or serialized W8A8 | Dynamic only | Z-Image, Qwen-Image | CUDA SM 80+ or Ascend NPU |
| GGUF | [GGUF](gguf.md) | Pre-quantized transformer weights | Static only | Qwen-Image family, Z-Image, FLUX.2-klein | CUDA |
| AutoRound | [AutoRound](autoround.md) | Pre-quantized W4A16 checkpoints | Static only | FLUX.1-dev; Qwen2.5-Omni and Qwen3-Omni language stages | CUDA SM 80+ |
| msModelSlim | [msModelSlim](msmodelslim.md) | Pre-quantized Ascend checkpoints | Static only | HunyuanImage-3.0 | Ascend NPU |

!!! note
    "Dynamic" means vLLM-Omni computes the quantization data at load time.
    "Static" means the checkpoint or external quantizer provides the required
    quantized weights and scales.

### Multi-stage Omni Models

For multi-stage models such as Qwen3-Omni, checkpoint-level quantization is
detected from `quantization_config` in the Hugging Face config. The quantized
config is scoped to the thinker's `language_model`; audio encoder, vision
encoder, talker, and code2wav stay in BF16 unless a model-specific guide says
otherwise.

| Method | Format | Tested models | Hardware | Status |
|--------|--------|---------------|----------|--------|
| ModelOpt FP8 | `quant_algo=FP8` | Qwen3-Omni thinker | Ada/Hopper (SM 89+) | Tested; about 47% thinker memory reduction |
| ModelOpt NVFP4 | `quant_algo=NVFP4` | Qwen3-Omni thinker | Blackwell (SM 100+) | Experimental; loads but quality is not acceptable |
| AutoRound W4A16 | `quant_method=auto-round` | Qwen2.5-Omni, Qwen3-Omni thinker | Ampere+ (SM 80+) | Pre-quantized checkpoints |

AWQ, GPTQ, and BitsAndBytes are available through vLLM's upstream
quantization registry, but they are not validated for vLLM-Omni pipelines.

## Quantization Scope

### Diffusion Models

The default target is the diffusion transformer. Some methods also support
component routing through `build_quant_config()`:

```python
from vllm_omni.quantization import build_quant_config

config = build_quant_config({
    "transformer": {"method": "fp8"},
    "vae": None,
})
```

Per-method scope:

| Method | Quantized components | Notes |
|--------|----------------------|-------|
| FP8 | Linear layers in the transformer; optional text encoder and VAE storage paths | Use `ignored_layers` for quality-sensitive MLPs |
| Int8 | Linear layers in the transformer | Dynamic activation scheme only |
| GGUF | Transformer weights from a GGUF file | Base model repo still supplies tokenizer, scheduler, text encoder, and VAE |
| AutoRound | Checkpoint-defined transformer blocks | No `--quantization` flag is required when the checkpoint has `quantization_config` |
| msModelSlim | Ascend quantized weights produced offline | Use the Ascend/NPU runtime and `--quantization ascend` |

### Multi-stage Omni Models

| Component | Quantized? | Notes |
|-----------|------------|-------|
| Thinker `language_model` | Yes | ModelOpt or AutoRound checkpoint config |
| Audio encoder | No | BF16 |
| Vision encoder | No | BF16 |
| Talker | No | BF16 |
| Code2Wav | No | BF16 |

## Device Compatibility

| Device | FP8 | Int8 | GGUF | AutoRound | msModelSlim |
|--------|-----|------|------|-----------|-------------|
| Blackwell GPU (SM 100+) | Yes | Yes | Yes | Yes | No |
| Ada/Hopper GPU (SM 89+) | Yes | Yes | Yes | Yes | No |
| Ampere GPU (SM 80+) | Weight-only FP8 where available | Yes | Yes | Yes | No |
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
