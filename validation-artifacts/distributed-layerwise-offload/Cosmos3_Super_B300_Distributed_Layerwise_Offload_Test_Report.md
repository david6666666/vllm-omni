# Cosmos3-Super Distributed Layerwise Offload Test Report — NVIDIA B300

## Test conclusion

1. **Correctness: pass.** HSDP4+USP4, legacy layerwise+USP4, DLO+USP4, and DLO+DP4 produced byte-identical T2I PNG and T2V MP4 files for the same seed. All 42 matrix operations completed successfully.
2. **HBM: pass for the intended offload goal.** DLO+USP4 used at most 25.3 GiB per GPU and DLO+DP4 used 22.2 GiB, versus 53.6 GiB for HSDP4+USP4: a 52.9%–58.6% reduction.
3. **Overlap performance: mixed.** Prefetch improved T2I by 1.418× on USP4 and 1.504× on DP4. T2V improved by only 1.123× and 1.034×, below the RFC target of 1.3×.
4. **Host memory: steady-state pass, startup fail.** DLO retained about 151–152 GiB tree RSS after sharding, but startup peaked at 603–604 GiB because every worker still constructs a full CPU model before taking its W/F shard.
5. **Overall:** the eager CUDA/NCCL dense-weight subset is functional and reviewable, but this is not yet full RFC acceptance. Direct checkpoint-to-local-shard loading and better UND/T2V overlap remain required.

## 1. Correctness

Model: `nvidia/Cosmos3-Super` (`/mnt/minecraft/model/Cosmos3-Super`)

Hardware: 4× NVIDIA B300 SXM6 AC, physical GPUs 4–7

Software: NVIDIA driver 580.95.05, CUDA 13.0, PyTorch 2.11.0+cu130, vLLM 0.25.0, and vLLM-Omni 0.24.1.dev32. The vLLM/vLLM-Omni minor-version mismatch is retained as a validation warning.

Benchmark provenance: upstream base `dacb8e6f10f0` plus feature patch SHA-256 `497ba4f02d915e14bfcbe923ffaffa6a4e754f8ef302f3785bf0e67f9e548475`. The draft branch was subsequently rebased onto `2f33df4678b2`; the feature diff remained byte-for-byte identical and the CPU/unit suite was rerun. A post-rebase GPU smoke was not available because GPUs 4–7 were occupied by an unrelated service.

| Comparison | T2I SHA-256 | T2V SHA-256 | Result |
|---|---|---|---|
| DLO overlap + USP4 vs HSDP4 + USP4 | `6e7d2a8c…` | `870c8056…` | Byte-identical |
| DLO sync vs overlap + USP4 | `6e7d2a8c…` | `870c8056…` | Byte-identical |
| DLO sync vs overlap + DP4 | `6e7d2a8c…` | `870c8056…` | Byte-identical |

All generated media decoded successfully. Exact file hashes are stronger than a feature-similarity-only comparison for these deterministic runs.

## 2. Performance benchmark

### Workloads

- T2I: 1024×1024, 50 steps, guidance 7, flow shift 3, seed 42.
- T2V: 1280×720, 189 frames, 24 FPS, 35 steps, guidance 6, flow shift 10, seed 17.
- One warmup operation followed by two measured full operations per modality and mode.

### Latency and HBM

| Strategy | Concurrency | T2I wall / HBM peak | T2V wall / HBM peak |
|---|---:|---:|---:|
| HSDP4 + USP4 baseline | 1 | 9.876 s / 53.6 GiB | 119.741 s / 53.6 GiB |
| Legacy layerwise + USP4 | 1 | 110.217 s / 26.3 GiB | 122.553 s / 26.3 GiB |
| DLO sync + USP4 | 1 | 42.296 s / 25.3 GiB | 135.325 s / 25.3 GiB |
| **DLO overlap + USP4** | **1** | **29.822 s / 25.3 GiB** | **120.515 s / 25.3 GiB** |
| Legacy layerwise + four DP replicas | 4 | 108.229 s / 22.6 GiB | 400.857 s / 22.6 GiB |
| DLO sync + DP4 shard group | 1 | 44.289 s / 22.2 GiB | 416.365 s / 22.2 GiB |
| **DLO overlap + DP4 shard group** | **1** | **29.452 s / 22.2 GiB** | **402.843 s / 22.2 GiB** |

`D0` processes four independent requests per wave. DLO DP4 processes one logical request collectively across four weight-shard ranks. These are capacity and memory comparisons, not equivalent single-request latency configurations.

### Pooled artifact-ready throughput

| Strategy | Concurrency | T2I outputs/s | T2V outputs/s |
|---|---:|---:|---:|
| HSDP4 + USP4 | 1 | 0.101257 | 0.008351 |
| Legacy layerwise + USP4 | 1 | 0.009073 | 0.008160 |
| DLO sync + USP4 | 1 | 0.023643 | 0.007390 |
| DLO overlap + USP4 | 1 | 0.033532 | 0.008298 |
| Legacy layerwise + four DP replicas | 4 | 0.036959 | 0.009979 |
| DLO sync + DP4 shard group | 1 | 0.022579 | 0.002402 |
| DLO overlap + DP4 shard group | 1 | 0.033953 | 0.002482 |

Unlike the original `je#2` prototype, the final DP4 path does not run four unrelated requests through one weight-shard collective group. Every rank must execute the same block sequence or the AllGather ordering can diverge and deadlock.

### Causal overlap speedup

| Comparison | T2I | T2V | RFC minimum |
|---|---:|---:|---:|
| DLO sync → overlap, USP4 | **1.418×** | **1.123×** | 1.1× / 1.3× |
| DLO sync → overlap, DP4 | **1.504×** | **1.034×** | 1.1× / 1.3× |

## 3. CPU memory (tree RSS)

| Strategy | Startup peak | Steady median |
|---|---:|---:|
| HSDP4 + USP4 | 480.9 GiB | 20.9 GiB |
| Legacy layerwise + USP4 | 538.4 GiB | 533.7 GiB |
| DLO overlap + USP4 | 603.4 GiB | 151.4 GiB |
| Legacy layerwise + four DP replicas | 561.7 GiB | 541.0 GiB |
| DLO overlap + DP4 | 603.4 GiB | 151.7 GiB |

DLO reaches the intended W/F steady-state host ownership, but it does not yet avoid the full-weight startup peak.

## 4. Double-buffer trace analysis

| Layer family | Compute C | H2D + AllGather P | Service hidden | Exposed stall | Verdict |
|---|---:|---:|---:|---:|---|
| UND blocks 0–63 | 1.121 ms | 8.059 ms | 13.2% | 4.110 ms | Not hidden |
| GEN blocks 64–127 | 23.939 ms | 6.679 ms | 99.7% | 0.046 ms | RFC steady-state match |

GEN satisfies the expected `max(C, P)` steady state. UND compute is shorter than transfer plus collective service, so double buffering cannot fully hide preparation without reducing transfer/collective cost or changing the prefetch window.

## 5. Validation summary and scope

- E2E matrix: 42/42 successful operations.
- CPU/unit suite: 362 passed, 5 skipped, 9 deselected.
- Static checks: Ruff check/format and `git diff --check` passed.
- Supported scope: eager CUDA/NCCL, dense contiguous unquantized parameters, pure USP or pure DP.
- Explicitly rejected or unvalidated: HSDP/DTensor, hybrid DP×SP, CFG parallel, quantization, `torch.compile`, step/streaming execution, NPU/HCCL.

The complete receipt-backed HTML report contains the raw evidence index, charts, output media, resource data, and profiler analysis.
