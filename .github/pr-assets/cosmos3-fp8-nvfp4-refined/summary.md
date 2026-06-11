# Cosmos3-Nano FP8/NVFP4 Refined Search

Prompt: `A robot arm is cleaning a plate in the kitchen`

Settings: 1280x720, 25 frames, 24 FPS, 35 diffusion steps, guidance 6.0, flow shift 10.0, seed 123.

Hardware: GPU4, Blackwell, CUDNN_ATTN. E2E includes request/response encode. Diffuse time is the pipeline profiler's transformer denoise section.

| Candidate | Scope / recipe | Transformer GiB | E2E s | Diffuse s | Speedup vs BF16 E2E | SSIM | PSNR dB | Video |
|---|---|---:|---:|---:|---:|---:|---:|---|
| BF16 | original checkpoint | 28.26 | 9.680 | 7.901 | 1.00x | 1.0000 | inf | [video](bf16_gpu4_repeat.mp4) |
| FP8 static | all transformer linear | 14.17 | 9.757 | 7.701 | 0.99x | 0.7494 | 19.376 | [video](fp8_static_all_linear_gpu4.mp4) |
| FP8 static | Q/K BF16, V/O/MLP FP8 | 15.57 | 6.390 | 4.904 | 1.51x | 0.7583 | 19.595 | [video](fp8_static_qk_bf16_gpu4.mp4) |
| FP8 PCPT | all transformer linear, per-channel weight/per-token activation | 14.18 | 13.203 | 10.755 | 0.73x | 0.7430 | 19.413 | [video](fp8_pcpt_all_linear.mp4) |
| FP8 PBWO | all transformer linear, blockwise weight-only | 14.17 | 6.745 | 5.249 | 1.44x | 0.0016 | 6.528 | [video](fp8_pbwo_all_linear_bad.mp4); output collapsed |
| NVFP4 default | all transformer linear | 8.51 | 10.206 | 7.917 | 0.95x | 0.6453 | 16.550 | [video](nvfp4_default_gpu4.mp4) |
| NVFP4 AWQ_FULL | all transformer linear | 8.51 | 4.866 | 3.369 | 1.99x | 0.0016 | 6.528 | [video](nvfp4_awq_full_gpu4_bad.mp4); output collapsed |
| NVFP4 attention-only | attention projections only | 23.06 | 13.300 | 11.007 | 0.73x | 0.7238 | 18.837 | [video](nvfp4_attention_only_gpu4.mp4) |
| NVFP4 self_q L2 | one self-attention Q projection | 28.24 | 17.805 | 12.147 | 0.54x | 0.8278 | 22.304 | [video](nvfp4_self_q_L2_gpu4.mp4) |

Conclusion: with the current ModelOpt export/runtime path, FP8 did not reach the 25 dB PSNR target under any useful performance candidate tested. NVFP4 only reached the 22 dB target when quantizing a single `self_q` layer, which has no practical memory or latency benefit. The best practical FP8 speed/quality point is `qk_bf16`, but quality remains around 19.6 dB. The best NVFP4 memory-saving candidates remain below 19 dB or collapse output.
