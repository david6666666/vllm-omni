# Int8 Quantization

## Overview

Int8 quantization supports W8A8 diffusion transformer inference on CUDA and
Ascend NPU. It can quantize BF16/FP16 weights at load time, or load serialized
Int8 checkpoints that already contain quantized weights and scales.

Only the dynamic activation scheme is currently supported.

## Configuration

Python API:

```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

omni = Omni(model="<your-model>", quantization="int8")

omni_with_skips = Omni(
    model="<your-model>",
    quantization_config={
        "method": "int8",
        "ignored_layers": ["<layer-name>"],
    },
)

outputs = omni.generate(
    "A cat sitting on a windowsill",
    OmniDiffusionSamplingParams(num_inference_steps=50),
)
```

CLI:

```bash
python text_to_image.py --model <your-model> --quantization int8
python text_to_image.py --model <your-model> --quantization int8 --ignored-layers "img_mlp"
vllm serve <your-model> --omni --quantization int8
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | str | - | Quantization method (`"int8"`) |
| `activation_scheme` | str | `"dynamic"` | Dynamic activation quantization; static is not supported |
| `ignored_layers` | list[str] | `[]` | Layer name patterns to keep in BF16/FP16 |
| `is_checkpoint_int8_serialized` | bool | `False` | Set by checkpoint config when loading serialized Int8 weights |

## Supported Models

| Model | HF Models | CUDA | Ascend NPU | Mode | Recommendation |
|-------|-----------|:----:|:----------:|------|----------------|
| Z-Image | `Tongyi-MAI/Z-Image-Turbo` | Yes | Yes | Dynamic load-time W8A8 | All layers |
| Qwen-Image | `Qwen/Qwen-Image`, `Qwen/Qwen-Image-2512` | Yes | Yes | Dynamic load-time W8A8 | All layers |

Other diffusion models may work if their transformer uses supported linear
layers, but they are not validated in this guide.

## Combining with Other Features

Int8 quantization can be combined with cache acceleration:

```python
omni = Omni(
    model="<your-model>",
    quantization="int8",
    cache_backend="tea_cache",
    cache_config={"rel_l1_thresh": 0.2},
)
```
