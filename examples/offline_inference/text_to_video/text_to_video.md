# Text-To-Video

The `Wan-AI/Wan2.2-T2V-A14B-Diffusers` pipeline generates short videos from text prompts. This script can also be used
for `Lightricks/LTX-2` to generate video+audio and optionally save the audio stream.

## Local CLI Usage

```bash
python text_to_video.py \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  --negative_prompt "<optional quality filter>" \
  --height 480 \
  --width 640 \
  --num_frames 32 \
  --guidance_scale 4.0 \
  --guidance_scale_high 3.0 \
  --num_inference_steps 40 \
  --fps 16 \
  --output t2v_out.mp4
```

LTX2 example (with optional audio output):

```bash
python text_to_video.py \
  --model "Lightricks/LTX-2" \
  --prompt "A cinematic close-up of ocean waves at golden hour." \
  --negative_prompt "worst quality, inconsistent motion, blurry, jittery, distorted" \
  --height 512 \
  --width 768 \
  --num_frames 121 \
  --num_inference_steps 40 \
  --guidance_scale 4.0 \
  --frame_rate 24 \
  --fps 24 \
  --output ltx2_out.mp4 \
  --audio_output ltx2_out.wav
```

Key arguments:

- `--prompt`: text description (string).
- `--height/--width`: output resolution (defaults 720x1280). Dimensions should align with Wan VAE downsampling (multiples of 8).
- `--num_frames`: Number of frames (Wan default is 81).
- `--guidance_scale` and `--guidance_scale_high`: CFG scale (applied to low/high)..
- `--negative_prompt`: optional list of artifacts to suppress (the PR demo used a long Chinese string).
- `--boundary_ratio`: Boundary split ratio for low/high DiT.
- `--fps`: frames per second for the saved MP4 (requires `diffusers` export_to_video).
- `--output`: path to save the generated video.
- `--frame_rate`: generation FPS for pipelines that require it (e.g., LTX2).
- `--audio_output`: optional WAV output path for pipelines that return audio.
- `--audio_sample_rate`: WAV sample rate for saved audio when `--audio_output` is set.
