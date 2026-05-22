# HunyuanVideo 1.5 480p Single-GPU Artifacts

This directory contains 33-frame text-to-video artifacts for reviewing the
single-GPU HunyuanVideo 1.5 480p T2V optimizations.

- `baseline_33f.mp4`: generated from the unmodified baseline worktree.
- `optimized_33f.mp4`: generated from this optimization branch.
- `similarity_33f.json`: pixel-space metrics computed on decoded RGB frames,
  using the same metric style as
  `vllm_omni/quantization/tools/compare_diffusion_trajectory_similarity.py`.

Both videos use the same prompt, seed, fixed CPU latents, 480x832 resolution,
33 frames, 50 inference steps, guidance scale 6.0, and flow shift 5.0.

## Performance

Feature-level microbenchmarks use representative 33-frame 480p T2V tensor
shapes on a single GPU with `CUDA_VISIBLE_DEVICES=3`. They isolate local
allocation or invariant-work costs and are not additive end-to-end speedups.
The raw measurements are in `performance_features_33f.json`.

| Optimization feature | Baseline mean ms | Optimized mean ms | Speedup |
| --- | ---: | ---: | ---: |
| RoPE cos/sin cache | 0.1979 | 0.0054 | 36.34x |
| Reuse T2V latent model input buffer | 0.0184 | 0.0114 | 1.61x |
| Explicit T2V `image_embeds_mask` | 0.1260 | 0.0802 | 1.57x |
| T2V zero image projection fast path | 0.0802 | 0.0432 | 1.86x |
| Scheduler timesteps/sigmas cache | 0.0322 | 0.0050 | 6.45x |

Full 33-frame generation timing from the committed review run:

| Run | Generation time ms |
| --- | ---: |
| Baseline | 99277.80 |
| Optimized | 99179.13 |
| Delta | -98.67 |
