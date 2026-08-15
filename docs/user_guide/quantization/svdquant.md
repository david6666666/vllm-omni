# SVDQuant W4A4

## Overview

[SVDQuant](https://arxiv.org/abs/2411.05007) combines four-bit weights
and activations with a small low-rank branch that corrects part of the
quantization error. vLLM-Omni consumes an offline-quantized checkpoint;
it does not calibrate the model while loading.

Phase 1 uses vLLM's existing NVFP4 linear kernel for the four-bit GEMM.
The rank correction and optional per-output scale run as ordinary BF16
operations. No unreleased FlashInfer SVDQuant API is required.

## Support

| Model      | Scope             | Format                          | Hardware        |
| ---------- | ----------------- | ------------------------------- | --------------- |
| MiniMax-H3 | FL2VA transformer | NVFP4 W4A4 + rank-32 correction | SM100 and SM103 |

The MiniMax-H3 recipe quantizes 208 attention and MLP linears. AdaLN,
the auxiliary text encoder, and the VAEs remain in their source dtype.
Ref2VA conversion is not supported.

SM110 and consumer Blackwell GPUs are outside the Phase 1 support
contract. They can be added later without changing the canonical
checkpoint layout.

## Convert a MiniMax-H3 checkpoint

The converter imports the published Nunchaku SVDQuant tensor layout,
restores non-SVDQuant tensors from the official base pipeline, and
writes a complete vLLM-Omni FL2VA folder:

```bash
python -m vllm_omni.quantization.tools.convert_nunchaku_to_svdquant \\
  --nunchaku-checkpoint ./svdq-fp4_r32-minimax-h3-fl2va.safetensors \\
  --base-pipeline MiniMaxAI/MiniMax-H3 \\
  --output-dir ./MiniMax-H3-SVDQuant-NVFP4-r32
```

Inputs can be local paths or Hugging Face repository identifiers. By
default, unchanged base-pipeline files are hard-linked to avoid another
large local copy. Pass `--copy` to create a physically independent
folder.

The generated transformer configuration contains
`quant_method: svdquant`, so inference does not need a separate
`--quantization` argument.

## Run inference

```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

omni = Omni(
    model="./MiniMax-H3-SVDQuant-NVFP4-r32/FL2VA",
    trust_remote_code=True,
)
outputs = omni.generate(
    "integrated_multimodal_description: [Shot 1] A golden retriever "
    "runs through shallow ocean waves at sunset.\n\n"
    "overall_soundscape: Natural ambient sound.\n\n"
    "non_diegetic_music: N/A",
    OmniDiffusionSamplingParams(
        height=768,
        width=1344,
        fps=24,
        num_inference_steps=50,
        seed=12345,
        extra_args={
            "task": "t2va",
            "duration": 5.0,
            "aspect_ratio": "16:9",
            "flow_shift": 12.0,
            "audio_flow_shift": 3.0,
        },
    ),
)
```

## Execution path

For each quantized linear, Phase 1 computes:

```text
base = NVFP4_linear(x / smooth, W4)
output = base * output_scale + (x @ down) @ up.T + bias
```

The explicit `output_scale` supports MiniMax-H3's distinct Q, K, and V
outer scales without a vector-alpha GEMM epilogue. Native rank-correction,
SwiGLU preprocessing, and shape-specific kernel fusion are follow-up
optimizations rather than checkpoint-loading requirements.
