# GGUF Quantization

## Overview

GGUF support loads pre-quantized diffusion transformer weights while keeping
the rest of the pipeline on the base Hugging Face checkpoint. Use the base
model for tokenizer, text encoder, scheduler, and VAE, then pass the GGUF file
for the transformer.

GGUF is static quantization: the quantized weights are produced before serving.

## Supported Models

| Model | HF base model | GGUF input | Scope | Adapter |
|-------|---------------|------------|-------|---------|
| Qwen-Image family | `Qwen/Qwen-Image`, `Qwen/Qwen-Image-2512`, edit and layered Qwen-Image pipelines | Local `.gguf`, `repo/file.gguf`, or `repo:quant_type` | Transformer only | `QwenImageGGUFAdapter` |
| Z-Image | `Tongyi-MAI/Z-Image-Turbo` | Local `.gguf`, `repo/file.gguf`, or `repo:quant_type` | Transformer only | `ZImageGGUFAdapter` |
| FLUX.2-klein | `black-forest-labs/FLUX.2-klein-4B` | Local `.gguf`, `repo/file.gguf`, or `repo:quant_type` | Transformer only | `Flux2KleinGGUFAdapter` |

Generic FLUX.1 GGUF checkpoints are not listed here; the implemented adapter is
for the FLUX.2-klein path.

## Configuration

Offline:

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model Qwen/Qwen-Image \
  --gguf-model QuantStack/Qwen-Image-GGUF/Qwen_Image-Q4_K_M.gguf \
  --quantization gguf \
  --prompt "a red paper kite hanging from a pine tree in a winter courtyard" \
  --height 1024 \
  --width 1024 \
  --seed 42 \
  --num_inference_steps 20 \
  --output outputs/qwen_image_q4km.png
```

Online:

```bash
vllm serve Qwen/Qwen-Image \
  --omni \
  --port 8000 \
  --quantization-config '{"method":"gguf","gguf_model":"QuantStack/Qwen-Image-GGUF/Qwen_Image-Q4_K_M.gguf"}'
```

`gguf_model` accepts:

| Form | Example |
|------|---------|
| Local file | `/models/z-image-Q4_K_M.gguf` |
| Explicit HF file | `QuantStack/Qwen-Image-GGUF/Qwen_Image-Q4_K_M.gguf` |
| HF repo plus quant type | `owner/repo:Q4_K_M` |

## How It Works

1. `OmniDiffusionConfig` receives `{"method": "gguf", "gguf_model": ...}`.
2. `DiffusersPipelineLoader` resolves the GGUF file.
3. A model-specific adapter remaps GGUF tensor names to vLLM-Omni transformer
   names.
4. Only transformer weights are loaded from GGUF. Missing non-transformer
   weights are loaded from the base model repository.
5. vLLM's GGUF linear method performs dequantization and GEMM at runtime.

Unsupported models fail fast with a clear "No GGUF adapter matched" error
instead of falling back to a generic mapper.

## Notes

1. Many GGUF repositories do not include `model_index.json`; always pass the
   normal base model through `--model`.
2. GGUF quantization does not use `activation_scheme`; it is a static
   pre-quantized format.
3. Keep GGUF files aligned with the exact base model family. Tensor names and
   fused projections differ across Qwen-Image, Z-Image, and FLUX.2-klein.
