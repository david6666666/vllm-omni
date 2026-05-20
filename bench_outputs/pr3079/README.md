# PR3079 B300 Video Comparison

Generated on PR head `4c370def57862bb4b734d4e23f975e724087a377`.

Environment:
- GPU: NVIDIA B300 SXM6 AC (`sm_103`)
- Device scope: `CUDA_VISIBLE_DEVICES=4`
- Model: `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v`
- Shape/config: `832x480`, `33` frames, `50` steps, BF16, `guidance_scale=6.0`, `flow_shift=5.0`, seed `42`

Videos:

| File | Backend | Run |
|---|---|---:|
| `hv15_cudnn.mp4` | `CUDNN_ATTN` | 1 |
| `hv15_torch_sdpa.mp4` | `TORCH_SDPA` | 1 |
| `hv15_cudnn_r2.mp4` | `CUDNN_ATTN` | 2 |
| `hv15_torch_sdpa_r2.mp4` | `TORCH_SDPA` | 2 |

Performance from `Total generation time`:

| Backend | Run 1 | Run 2 | Mean |
|---|---:|---:|---:|
| `CUDNN_ATTN` | 28.2873 s | 27.6473 s | 27.9673 s |
| `TORCH_SDPA` | 99.6494 s | 102.0925 s | 100.8710 s |

Mean speedup on this B300 setup: `3.6067x`.
