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
