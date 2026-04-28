# AutoRound Quantization

## Overview

[AutoRound](https://github.com/intel/auto-round) produces pre-quantized
checkpoints for LLMs, VLMs, and diffusion models. vLLM-Omni reads the
checkpoint's `config.json` and auto-detects
`quantization_config.quant_method = "auto-round"`.

AutoRound is static quantization: no `--quantization` flag is needed at
inference time when the checkpoint already contains the quantization config.

## Supported Schemes

| Scheme | Bits | Status |
|--------|------|--------|
| W4A16 | 4 | Supported |
| W8A16 | 8 | Planned |

## Supported Models

| Model | Checkpoint | Scope | Scheme | Backend |
|-------|------------|-------|--------|---------|
| FLUX.1-dev | `vllm-project-org/FLUX.1-dev-AutoRound-w4a16` | Diffusion transformer | W4A16 | GPTQ-Marlin |
| Qwen2.5-Omni-7B | `Intel/Qwen2.5-Omni-7B-int4-AutoRound` | Language model stage | W4A16 | GPTQ-Marlin |
| Qwen3-Omni-30B-A3B-Instruct | `Intel/Qwen3-Omni-30B-A3B-Instruct-int4-AutoRound` | Thinker language model stage | W4A16 | GPTQ-Marlin |

AutoRound support is checkpoint-driven. A model is supported when its
checkpoint uses a compatible INC/AutoRound config and the target stage maps to
vLLM-Omni's runtime module names.

## Configuration

Python API:

```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

omni = Omni(model="vllm-project-org/FLUX.1-dev-AutoRound-w4a16")

outputs = omni.generate(
    "A cat sitting on a windowsill",
    OmniDiffusionSamplingParams(num_inference_steps=28),
)
outputs[0].save_images("output.png")
```

CLI:

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model vllm-project-org/FLUX.1-dev-AutoRound-w4a16 \
  --prompt "A cat sitting on a windowsill" \
  --num-inference-steps 28 \
  --output outputs/flux_w4a16.png
```

## Checkpoint Config

The checkpoint should contain a config like:

```json
{
  "quantization_config": {
    "quant_method": "auto-round",
    "bits": 4,
    "group_size": 128,
    "sym": true,
    "packing_format": "auto_round:auto_gptq",
    "block_name_to_quantize": "transformer_blocks,single_transformer_blocks"
  }
}
```

At load time, vLLM-Omni builds an `OmniINCConfig`, remaps checkpoint block names
to runtime module names, and selects the matching vLLM compute backend.

## Creating a Quantized Checkpoint

```bash
auto-round \
  --model black-forest-labs/FLUX.1-dev \
  --scheme W4A16 \
  --batch_size 1 \
  --disable_opt_rtn \
  --dataset coco2014 \
  --iters 0
```

Use the generated output directory directly as the `model` argument. See the
[AutoRound documentation](https://github.com/intel/auto-round) for all
available schemes and options.
