# msModelSlim Quantization

## Overview

[msModelSlim](https://github.com/Ascend/msmodelslim) is an Ascend compression
toolkit for producing pre-quantized model checkpoints. In vLLM-Omni, these
checkpoints run through the Ascend/NPU path with `--quantization ascend`.

msModelSlim is static quantization: quantized weights are generated offline
before vLLM-Omni inference starts.

## Supported Schemes

| Scheme | Bits | Status |
|--------|------|--------|
| W8A8 | 8 | Supported |
| W4A4 | 4 | Planned |

## Supported Models

| Model | Base model | Scope | Hardware | Notes |
|-------|------------|-------|----------|-------|
| HunyuanImage-3.0 | `tencent/HunyuanImage-3.0`, `tencent/HunyuanImage-3.0-Instruct` | DiT / diffusion stage | Ascend A2/A3 NPU | Generate quantized weights with the HunyuanImage-3.0 msModelSlim adaptation |

Public Hugging Face quantized weights are not available yet. Use the
[HunyuanImage-3.0 msModelSlim adaptation](https://gitcode.com/betta18/msmodelslim/tree/hyimage3_mxfp8)
to generate the checkpoint manually.

The upstream msModelSlim repository also provides a Wan2.2 quantization
example. That recipe is useful for producing weights, but Wan2.2 is not listed
as a validated vLLM-Omni msModelSlim inference target in this guide.

## Model Quantization

Example msModelSlim command for a Wan2.2 W8A8 checkpoint:

```bash
msmodelslim quant \
  --model_path /path/to/wan2_2_t2v_float_weights \
  --save_path /path/to/wan2_2_t2v_quantized_weights \
  --device npu \
  --model_type Wan2_2 \
  --config_path /path/to/wan2_2_w8a8f8_mxfp_t2v.yaml \
  --trust_remote_code True
```

For HunyuanImage-3.0, use the Hunyuan-specific adaptation linked above.

## Configuration

Offline inference:

```bash
python text_to_image.py --model <quantized-model-path> --quantization ascend
```

Online serving:

```bash
vllm serve <quantized-model-path> --omni --quantization ascend
```

## Notes

1. Run with the Ascend/NPU installation and environment.
2. The `ascend` quantization method expects weights produced by the Ascend
   tooling; it is not a load-time CUDA quantizer.
3. Keep the quantized checkpoint aligned with the same model architecture and
   stage config used for BF16 inference.
