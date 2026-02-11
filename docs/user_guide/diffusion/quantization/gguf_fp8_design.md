# Diffusion Quantization Design: Native GGUF + FP8 (Online and Native)

Date: 2026-02-10

## Goals
1. Reuse vLLM quantization configs and weight loaders as much as possible.
2. Add native GGUF and FP8 support to diffusion transformers without changing model definitions.
3. Keep user-facing knobs minimal and consistent across offline and online flows.

## Scope
1. Models: Qwen-Image and Flux2-klein are first-class targets.
2. Components: diffusion transformer weights, loader paths, and quantization configs.
3. Modes: native GGUF, online FP8, native FP8 (pre-serialized FP8 checkpoint).

## Architecture Overview
1. `OmniDiffusionConfig` accepts `quantization` or `quantization_config`.
2. Diffusion quantization wrappers (`DiffusionGgufConfig`, `DiffusionFp8Config`) produce vLLM `QuantizationConfig` objects for linear layers.
3. `DiffusersPipelineLoader` branches on quantization method and loads either HF weights or GGUF weights for the transformer.
4. vLLM GGUF path uses `GGUFConfig` and `GGUFLinearMethod` for matmul; FP8 uses `Fp8Config` (online) or `is_checkpoint_fp8_serialized` for native FP8.

## Call Chain (Offline)
```
CLI (examples/offline_inference/text_to_image/text_to_image.py)
  |
  v
Omni (vllm_omni/entrypoints/omni.py)
  |
  v
OmniStage (diffusion)
  |
  v
DiffusionWorker
  |
  v
DiffusionModelRunner
  |
  v
DiffusersPipelineLoader
  |
  v
Pipeline.forward (Flux2/Qwen)
  |
  v
DiffusionEngine
  |
  v
OmniRequestOutput
  |
  v
Client (saved PNG)
```

## Call Chain (Online)
```
Client
  |
  | POST /v1/images/generations
  v
APIServer (vllm_omni/entrypoints/openai/api_server.py)
  |
  v
_generate_with_async_omni
  |
  v
AsyncOmni
  |
  v
DiffusionEngine
  |
  v
OmniRequestOutput
  |
  v
encode_image_base64
  |
  v
ImageGenerationResponse
  |
  v
Client
```

## Call Chain (GGUF Operator Path)
```
Pipeline.forward (Flux2/Qwen)
  |
  v
Transformer blocks
  |
  v
Flux2Attention / Flux2ParallelSelfAttention
  |
  v
QKVParallelLinear / ColumnParallelLinear / RowParallelLinear
  |
  v
LinearBase.forward
  |
  v
QuantMethod.apply (GGUFLinearMethod.apply)
  |
  v
fused_mul_mat_gguf
  |
  v
_fused_mul_mat_gguf (custom op)
  |
  v
ops.ggml_dequantize
  |
  v
x @ weight.T
```

Notes:
1. GGUF linear inputs are flattened to 2D inside `GGUFLinearMethod.apply` and reshaped back.
2. As of 2026-02-10 in this branch, `_fused_mul_mat_gguf` is forced to the dequantize path.

## Call Chain (FP8 Operator Path)
```
Pipeline.forward (Flux2/Qwen)
  |
  v
Transformer blocks
  |
  v
QKVParallelLinear / ColumnParallelLinear / RowParallelLinear
  |
  v
LinearBase.forward
  |
  v
QuantMethod.apply (Fp8LinearMethod.apply or Fp8OnlineLinearMethod.apply)
  |
  +--> apply_fp8_marlin_linear (weight-only path on older GPUs)
  |
  +--> W8A8BlockFp8LinearOp.apply (block quant path)
  |
  +--> fp8_linear.apply_weights
          |
          v
          init_fp8_linear_kernel
            |
            v
          FlashInferFP8ScaledMMLinearKernel / CutlassFP8ScaledMMLinearKernel /
          Torch FP8 ScaledMM kernels
```

Notes:
1. Online FP8 differs at load time; runtime operator path matches native FP8.
2. The kernel selection is platform and capability dependent.

## GGUF Weight Loading Path (Transformer-Only)
1. `DiffusersPipelineLoader.load_model` detects `quantization_config.method == "gguf"`.
2. `gguf_model` is resolved as one of: local file, URL, `repo/file.gguf`, or `repo:quant_type`.
3. Name mapping is applied per-architecture (Qwen-Image, Flux2-klein).
4. GGUF weights are loaded into transformer modules, remaining non-transformer weights come from the HF checkpoint.

## FP8 Loading Path
1. Online FP8: `quantization="fp8"` or `quantization_config={"method":"fp8", "ignored_layers": [...]}`.
2. Native FP8: `quantization_config={"method":"fp8", "is_checkpoint_fp8_serialized": True}` to load an FP8-serialized checkpoint.

## User Usage (Offline)

### Baseline BF16
```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model /workspace/models/black-forest-labs/FLUX.2-klein-4B \
  --prompt "a photo of a forest with mist swirling around the tree trunks. The word 'FLUX.2' is painted over it in big, red brush strokes with visible texture" \
  --height 768 \
  --width 1360 \
  --seed 42 \
  --cfg_scale 4.0 \
  --num_images_per_prompt 1 \
  --num_inference_steps 4 \
  --output outputs/flux2_klein_4b.png
```

### Native GGUF (Transformer Only)
```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model /workspace/models/black-forest-labs/FLUX.2-klein-4B \
  --gguf-model "/workspace/models/unsloth/FLUX.2-klein-4B-GGUF/flux-2-klein-4b-Q8_0.gguf" \
  --quantization gguf \
  --prompt "a photo of a forest with mist swirling around the tree trunks. The word 'FLUX.2' is painted over it in big, red brush strokes with visible texture" \
  --height 768 \
  --width 1360 \
  --seed 42 \
  --cfg_scale 4.0 \
  --num_images_per_prompt 1 \
  --num_inference_steps 4 \
  --output outputs/flux2_klein_4b_gguf.png
```

Notes for GGUF:
1. Many GGUF repos do not ship `model_index.json` and configs. Use the base repo for `--model` and only pass the GGUF file via `--gguf-model`.
2. `gguf_model` supports local path, URL, `repo/file.gguf`, or `repo:quant_type`.

### Online FP8 (Runtime Quantization)
```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model Qwen/Qwen-Image \
  --quantization fp8 \
  --prompt "a cup of coffee on the table" \
  --height 1024 \
  --width 1024
```

### Native FP8 (Serialized Checkpoint)
Use the Python API to pass `is_checkpoint_fp8_serialized`.
```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

omni = Omni(
    model="/path/to/fp8-checkpoint",
    quantization_config={
        "method": "fp8",
        "is_checkpoint_fp8_serialized": True,
    },
)

outputs = omni.generate(
    "a cup of coffee on the table",
    OmniDiffusionSamplingParams(num_inference_steps=4),
)
```

## User Usage (Online)

### Start Server (Online FP8)
```bash
vllm serve Qwen/Qwen-Image --omni --port 8000 --quantization fp8
```

### Start Server (Native GGUF via CLI)
```bash
vllm serve /workspace/models/black-forest-labs/FLUX.2-klein-4B \
  --omni \
  --port 8000 \
  --quantization-config '{"method":"gguf","gguf_model":"/workspace/models/unsloth/FLUX.2-klein-4B-GGUF/flux-2-klein-4b-Q8_0.gguf"}'
```

### Start Server (Native FP8 via CLI)
```bash
vllm serve /path/to/fp8-checkpoint \
  --omni \
  --port 8000 \
  --quantization-config '{"method":"fp8","is_checkpoint_fp8_serialized":true}'
```

### Online Request (Images API)
```bash
curl -X POST http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a dragon laying over the spine of the Green Mountains of Vermont",
    "size": "1024x1024",
    "seed": 42,
    "num_inference_steps": 4
  }'
```

## Validation Checklist
1. Fix the date in logs and docs for comparisons.
2. Use the same prompt, size, steps, and seed for BF16 vs GGUF/FP8 comparisons.
3. Expect accuracy differences for Q8_0 GGUF; verify mapping with F16/BF16 GGUF if needed.
