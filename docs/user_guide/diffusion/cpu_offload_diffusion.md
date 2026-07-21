# CPU Offloading for Diffusion Models

## Overview

vLLM-Omni provides three offloading strategies to reduce GPU memory usage for diffusion models:

1. **Model-level (Sequential) Offloading**: Mutual exclusion between DiT model and encoder - only one is on GPU at a time.
2. **Layerwise (Blockwise) Offloading**: Keeps only one transformer block on GPU at a time with compute-memory overlap.
3. **Distributed Layerwise Offloading**: Stores one CPU weight shard per rank and reconstructs the next block with AllGather.

All strategies use pinned memory for faster CPU-GPU transfers. Distributed
layerwise offload is **mutually exclusive** with the other two modes and rejects
invalid combinations during configuration. For backward compatibility, when
the two legacy flags are both set, layerwise offload still takes priority over
model-level offload.


## Model-level (Sequential) Offloading

### How It Works

Model-level offloading implements mutual exclusion between DiT transformer and encoder modules using pre forward hooks:

- **When encoders run**: DiT transformer is offloaded to CPU
- **When DiT runs**: Encoders are offloaded to CPU, if more than one dit models, only one loaded on GPU, others get offloaded to CPU.
- **VAE**: Stays resident on GPU

Before each module's forward pass, the hook automatically moves it to GPU while offloading the other module group to CPU. Transfers use pinned memory for speed.

### Usage

**Python API:**
```python
from vllm_omni import Omni

m = Omni(model="Wan-AI/Wan2.2-T2V-A14B-Diffusers", enable_cpu_offload=True)
```

**CLI:**
```bash
vllm serve Wan-AI/Wan2.2-T2V-A14B-Diffusers --omni --enable-cpu-offload
```

### To Support a Model

Implement the `SupportsComponentDiscovery` protocol to declare which
submodules serve as pipeline components (used by offloading, HSDP
sharding, and other framework features):

```python
from typing import ClassVar
from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery

class MyPipeline(nn.Module, SupportsComponentDiscovery):
    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder", "vision_model"]
    _vae_modules: ClassVar[list[str]] = ["vae"]
    _resident_modules: ClassVar[list[str]] = []  # optional

    def __init__(self):
        super().__init__()
        self.transformer = ...     # DiT — stays on GPU during denoising
        self.text_encoder = ...    # Encoder — offloaded to CPU during denoising
        self.vision_model = ...    # Encoder — offloaded to CPU during denoising
        self.vae = ...             # VAE — always on GPU
```

- `_dit_modules`: attribute names of denoising submodules (kept on GPU
  during the diffusion loop).
- `_encoder_modules`: attribute names of encoder/vision submodules
  (offloaded to CPU during the diffusion loop).
- `_vae_modules`: attribute names of VAE(s) (always kept on GPU, not
  part of the mutual exclusion hooks).
- `_resident_modules`: attribute names of small submodules that must
  stay on GPU during layerwise offloading (e.g. embedders, connectors).
  Optional — defaults to `[]`.

All attribute names support dotted paths for nested submodules
(e.g. `"pipe.transformer"`, `"bagel.time_embedder"`).

Both DiT and encoder lists are needed because the offload hooks use
mutual exclusion: when one group runs, the other moves to CPU.

### Limitations
- Cold start latency increases
- Adds overhead from CPU-GPU transfers between encoder and denoising phases
- Support single GPU only for now


### Component offloading for split models (e.g. Cosmos3)

Some models split their transformer into mutually-exclusive *components* that run
in different phases of a single forward pass rather than as separate pipeline
components -- e.g. Cosmos3's understanding (reasoner) component runs once per
generation while the generation (generator) component runs every denoising step.
Such models have no separate text encoder to swap against, so the transformer
owns a small model-local offload path and wraps each phase with
`with self._offload_context(name):`

```python
class Cosmos3VFMTransformer(nn.Module):
    def forward(self, ...):
        with self._offload_context("reasoner"):
            ...  # understanding pass, runs once
        with self._offload_context("generator"):
            ...  # denoising pass, runs every step
```

Model-level offloading then keeps exactly one component GPU-resident at a time
(the other on CPU), reusing the same `SequentialOffloadHook` `.to()` movers. The
pipeline opts in by exposing `enable_omni_model_cpu_offload` (which drives the
transformer's `enable_model_cpu_offload` and pins the VAE). Layerwise offloading
works for these models too -- each component declares its own block container via
`_layerwise_offload_blocks_attrs`.


## Layerwise (Blockwise) Offloading

### How It Works

Layerwise offloading keeps only one transformer block on GPU at a time.

As each block completes, the next block is prefetched to GPU while the current block is freed. The pre and forward hooks utilized by layerwise offloading apply a separate CUDA stream (`copy_stream`) to overlap weight transfer with computation, and retain flattened tensors in pinned CPU memory for block parameters re-materialization. Encoders, VAE, and non-block DiT modules (embeddings, norms) always stay on GPU.

**Execution Flow:**

| Block | Pre-forward Hook | Forward | Post-forward Hook |
|-------|------------------|---------|-------------------|
| block-0 | Prefetch block-1 (async) | Compute block-0 | Free block-0 |
| block-1 | Prefetch block-2 (async) | Compute block-1 | Free block-1 |
| ... | ... | ... | ... |
| block-(n-1) | **Prefetch block-0** (async) | Compute block-(n-1) | Free block-(n-1) |

Each transformer block has a `LayerwiseOffloadHook` that prefetches the next block before forward and frees the current block after forward.

Layerwise offloading is primarily recommended for large **video generation models** where the compute cost per block is high enough to effectively overlap with memory prefetch operations. For example, Wan2.2 T2V and I2V pipelines.

### Usage

**Python API:**
```python
from vllm_omni import Omni

# Text-to-video
m = Omni(model="Wan-AI/Wan2.2-T2V-A14B-Diffusers", enable_layerwise_offload=True)

# Or image-to-video
m = Omni(model="Wan-AI/Wan2.2-I2V-A14B-Diffusers", enable_layerwise_offload=True)
```

**CLI:**
```bash
# Text-to-video
vllm serve Wan-AI/Wan2.2-T2V-A14B-Diffusers --omni --enable-layerwise-offload

# Or image-to-video
vllm serve Wan-AI/Wan2.2-I2V-A14B-Diffusers --omni --enable-layerwise-offload
```

### To Support a Model

Models must define the blocks attribute name for layerwise offloading:

```python
class WanTransformer3DModel(nn.Module):
    _layerwise_offload_blocks_attrs = ["blocks"]  # Attribute names containing transformer blocks

    def __init__(self):
        self.blocks = nn.ModuleList([...])  # Transformer blocks
```

For models with multiple block types:

```python
class Flux2Transformer2DModel(nn.Module):
    _layerwise_offload_blocks_attrs = ["transformer_blocks", "single_transformer_blocks"]
```

### Limitations
- Cold start latency increases because offloaded components must be moved to CPU
  during setup; layerwise offload may add extra weight consolidation and pinning
  work.
- Performance depends on compute cost and H2D bandwidth as well
- Support single GPU only for now


## Distributed Layerwise Offloading

Distributed layerwise offloading shards each block's flattened weights across a
multi-rank group. Each rank retains only its pinned CPU shard. While block N is
computed, the backend copies the local shard for block N+1 to the device and
AllGathers the complete block into the next fixed device-buffer slot.

The mode currently supports exactly one of these layouts (four ranks are shown
as an example):

| Layout | Required configuration | Request behavior |
|---|---|---|
| Pure Ulysses | `ulysses_degree=4`, all other parallel degrees `1` | All ranks cooperate on one sequence-parallel request |
| Pure DP | `data_parallel_size=4`, all other parallel degrees `1` | All four weight-shard ranks replicate one logical request in lockstep |

In the pure-DP layout, `data_parallel_size` is repurposed as the weight-shard
group. It does not provide four independent request replicas: the scheduler
admits one logical request at a time, every rank executes that same request,
and only rank zero returns the generated media. Use an outer replica axis with
four additional copies of the shard group for independent request throughput;
for example, four independent `DP4` shard groups would require 16 GPUs.

TP, PP, Ring, CFG parallelism, VAE patch parallelism, expert parallelism, mixed
DP+USP, HSDP, model-level CPU offload, and legacy layerwise offload cannot be
combined with this mode yet.

**Python API:**

```python
from vllm_omni import Omni

omni = Omni(
    model="nvidia/Cosmos3-Super",
    enable_distributed_layerwise_offload=True,
    ulysses_degree=4,
    enforce_eager=True,
)
```

**CLI:**

```bash
vllm serve nvidia/Cosmos3-Super --omni \
  --usp 4 \
  --enable-distributed-layerwise-offload \
  --enforce-eager
```

Next-block prefetch is enabled by default. Use
`--disable-distributed-layerwise-offload-prefetch` only for a synchronous
correctness or performance baseline with the same sharded weight layout.
This switch serializes preparation and block compute; it is an ablation knob,
not a recommended serving configuration. Prefetch removes preparation from the
critical path only when the available block compute (or launch-ahead across
blocks) covers the H2D plus AllGather service time. Short-compute towers can
remain transfer-bound even with prefetch enabled.

Models use the same `_layerwise_offload_blocks_attrs` declaration as legacy
layerwise offload. The distributed backend additionally requires every rank to
discover blocks and flattened parameter metadata in the same order.

The current loader first constructs the full CPU model on every worker and only
then commits the local pinned shard. Consequently, steady-state host weight
storage is approximately `W/F` per rank, but startup can still peak near one
full model plus its local shard per rank. True rank-local checkpoint loading is
not implemented yet.

Only ordinary dense strided parameters are accepted today. Quantized tensor
layouts and tensor subclasses other than `nn.Parameter` are rejected. Cache
backends are also rejected because rank-local block skipping can diverge the
collective order. The stream and event path uses the platform abstraction, but
the current acceptance evidence is CUDA/NCCL only; HCCL/NPU still requires a
real multi-rank correctness and timeline run. The validated serving recipe uses
`--enforce-eager`; `torch.compile` and graph capture with parameter-storage
rebinding have not yet passed the same acceptance matrix.


### Implementation Notes

**Module Discovery**

The offloader discovers pipeline components in two ways:

1. **Protocol-based** (preferred): If the pipeline implements
    `SupportsComponentDiscovery`, its `_dit_modules`, `_encoder_modules`,
    `_vae_modules`, and `_resident_modules` class variables are used
    directly.  All attribute names support dotted paths (e.g.
    `"pipe.transformer"`, `"bagel.time_embedder"`) for nested submodules.

2. **Fallback attribute scan**: Otherwise, the offloader scans for
    well-known attribute names:
    - **DiT modules**: `transformer`, `transformer_2`, `dit`, `sr_dit`, `language_model`, `transformer_blocks`, `model`
    - **Encoders**: `text_encoder`, `text_encoder_2`, `text_encoder_3`, `image_encoder`
    - **VAE**: `vae`, `audio_vae`

**Hook System**

Both strategies use vLLM-Omni's hook registry system (`HookRegistry` and `ModelHook`) to register pre/post forward callbacks on modules, enabling automatic swapping without modifying model code.

**Backend Architecture**

```
OffloadBackend (base class)
├── ModelLevelOffloadBackend → uses SequentialOffloadHook (.to() swap)
│                              (delegates to a pipeline's enable_omni_model_cpu_offload
│                               for split models like Cosmos3)
├── LayerWiseOffloadBackend → uses LayerwiseOffloadHook
└── DistributedLayerwiseOffloadBackend → H2D shard + AllGather double buffer
```

Factory function `get_offload_backend()` selects the appropriate backend based on
configuration.

For split models, `ModelLevelOffloadBackend.enable()` detects a pipeline's
`enable_omni_model_cpu_offload` hook and delegates to it; Cosmos3 then swaps its
reasoner/generator components inside the model forward pass.


## Supported Models

| Architecture | Example Models | DiT Class | Model-Level Offload | Layerwise Offload | Blocks Attrs (Layerwise specific) |
|--------------|----------------|-----------|---------------------|-------------------|-----------------------------------|
| LongCatImagePipeline | `meituan-longcat/LongCat-Image` | `LongCatImageTransformer2DModel` | - | ✓ | `"transformer_blocks"`, `"single_transformer_blocks"` |
| NextStep11Pipeline | `stepfun-ai/NextStep-1.1` | `NextStepModel` | - | ✓ | `"layers"` |
| OvisImagePipeline | `AIDC-AI/Ovis-Image-7B` | `OvisImageTransformer2DModel` | - | ✓ | `"transformer"` |
| QwenImagePipeline | `Qwen/Qwen-Image` | `QwenImageTransformer2DModel` | ✓ | ✓ | `"transformer_blocks"` |
| StableDiffusionXLPipeline | `stabilityai/stable-diffusion-xl-base-1.0` | `SDXLUNet2DConditionModel` | ✓ | ✓ | `"down_blocks"`, `"up_blocks"` |
| StableDiffusion3Pipeline | `stabilityai/stable-diffusion-3.5-medium` | `SD3Transformer2DModel` | - | ✓ | `"transformer_blocks"` |
| Wan22I2VPipeline | `Wan-AI/Wan2.2-I2V-A14B-Diffusers` | `WanTransformer3DModel` | ✓ | ✓ | `"blocks"` |
| Wan22Pipeline | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | `WanTransformer3DModel` | ✓ | ✓ | `"blocks"` |
| SoulXSingerPipeline / SoulXSingerSVCPipeline | `Soul-AILab/SoulX-Singer` | `DiffLlama` (`cfm_decoder.model.diff_estimator`) | ✓ | ✓ | `"layers"` |
| BagelPipeline | `ByteDance-Seed/BAGEL-7B-MoT` | `Qwen2MoTModel` | - | ✓ | `"layers"`, `"customized modules"` |

**Notes:**
- Model-Level Offloading is expected to be supported by all common diffusion models (DiT and encoders) naturally
- Layerwise Offloading requires DiT class to define `_layerwise_offload_blocks_attrs` pointing to transformer blocks
