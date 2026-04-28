# FP8 Quantization

## Overview

FP8 quantization converts BF16/FP16 weights to FP8 at model load time.
Dynamic activation scaling is the default and does not require calibration.
Static activation scaling is supported when calibrated scale information is
available.

Some architectures can quantize all linear layers. Others have quality-sensitive
layers that should stay in BF16 through `ignored_layers`. Image-stream MLPs
(`img_mlp`) are a common sensitive target because denoising latent ranges shift
across timesteps and small per-layer errors can compound in deep DiT blocks.

## Configuration

Python API:

```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

omni = Omni(model="<your-model>", quantization="fp8")

omni_with_skips = Omni(
    model="<your-model>",
    quantization_config={
        "method": "fp8",
        "ignored_layers": ["img_mlp"],
    },
)

outputs = omni.generate(
    "A cat sitting on a windowsill",
    OmniDiffusionSamplingParams(num_inference_steps=50),
)
```

CLI:

```bash
python text_to_image.py --model <your-model> --quantization fp8
python text_to_image.py --model <your-model> --quantization fp8 --ignored-layers "img_mlp"
vllm serve <your-model> --omni --quantization fp8
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | str | - | Quantization method (`"fp8"`) |
| `ignored_layers` | list[str] | `[]` | Layer name patterns to keep in BF16 |
| `activation_scheme` | str | `"dynamic"` | `"dynamic"` for load-time scaling, or `"static"` when scales are available |
| `weight_block_size` | list[int] \| None | `None` | Block size for block-wise weight quantization |

The available `ignored_layers` names depend on the model architecture, for
example `to_qkv`, `to_out`, `img_mlp`, or `txt_mlp`.

## Supported Models

| Model | HF Models | Dynamic | Static | Recommendation | `ignored_layers` |
|-------|-----------|:-------:|:------:|----------------|------------------|
| Z-Image | `Tongyi-MAI/Z-Image-Turbo` | Yes | Yes | All layers | None |
| Qwen-Image | `Qwen/Qwen-Image`, `Qwen/Qwen-Image-2512` | Yes | Yes | Skip sensitive image-stream MLPs when quality regresses | `img_mlp` |
| FLUX.1 | `black-forest-labs/FLUX.1-dev`, `black-forest-labs/FLUX.1-schnell` | Yes | Yes | All layers | None |
| FLUX.2-klein | `black-forest-labs/FLUX.2-klein-4B` | Yes | Yes | All layers | None |
| HunyuanImage-3.0 | `tencent/HunyuanImage-3.0`, `tencent/HunyuanImage-3.0-Instruct` | Yes | Yes | All layers; use the Hunyuan stage config for multi-stage runs | None |
| HunyuanVideo-1.5 | `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v`, `720p_t2v`, `480p_i2v` | Yes | Yes | All layers | None |

GLM-Image and Helios are not listed as FP8-supported diffusion models until
they have method-specific validation.

## Combining with Other Features

FP8 quantization can be combined with cache acceleration:

```python
omni = Omni(
    model="<your-model>",
    quantization="fp8",
    cache_backend="tea_cache",
    cache_config={"rel_l1_thresh": 0.2},
)
```
