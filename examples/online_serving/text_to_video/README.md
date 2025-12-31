# Text-To-Video

This example demonstrates how to deploy Wan2.2 video models for online video generation
using vLLM-Omni. The API base is `v1/chat/completions`.

## Start Server

### Text-to-Video (T2V)

```bash
vllm serve Wan-AI/Wan2.2-T2V-A14B-Diffusers --omni --port 8093 \
  --boundary-ratio 0.875 \
  --flow-shift 5.0
```

### Image-to-Video (I2V)

```bash
vllm serve Wan-AI/Wan2.2-I2V-A14B-Diffusers --omni --port 8094 \
  --boundary-ratio 0.875 \
  --flow-shift 5.0
```

Or use the startup script:

```bash
bash run_server.sh
```

## API Calls

### Method 1: Using curl (Text-to-Video)

```bash
bash run_curl_text_to_video.sh
```

### Method 2: Using curl (Image-to-Video)

```bash
bash run_curl_image_to_video.sh input.png "A cinematic slow zoom into the scene"
```

## Request Format

### Text-to-Video

```json
{
  "messages": [
    {"role": "user", "content": "A serene lakeside sunrise with mist over the water."}
  ],
  "extra_body": {
    "height": 720,
    "width": 1280,
    "num_frames": 81,
    "num_inference_steps": 40,
    "guidance_scale": 4.0,
    "guidance_scale_2": 4.0,
    "seed": 42,
    "fps": 24
  }
}
```

### Image-to-Video

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Make the scene come alive with gentle motion"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..." }}
      ]
    }
  ],
  "extra_body": {
    "height": 720,
    "width": 1280,
    "num_frames": 81,
    "num_inference_steps": 40,
    "guidance_scale": 4.0,
    "seed": 42,
    "fps": 24
  }
}
```

## Generation Parameters (extra_body)

| Parameter                | Type  | Default | Description                                    |
| ------------------------ | ----- | ------- | ---------------------------------------------- |
| `height`                 | int   | None    | Video height in pixels                         |
| `width`                  | int   | None    | Video width in pixels                          |
| `num_frames`             | int   | None    | Number of frames to generate                   |
| `num_inference_steps`    | int   | 50      | Number of denoising steps                      |
| `guidance_scale`         | float | None    | CFG scale                                      |
| `guidance_scale_2`        | float | None    | Optional high-noise CFG (Wan2.2)               |
| `seed`                   | int   | None    | Random seed (reproducible)                     |
| `negative_prompt`        | str   | None    | Negative prompt                                |
| `num_outputs_per_prompt` | int   | 1       | Number of videos to generate                   |
| `fps`                    | int   | 24      | Output video FPS (used for MP4 encoding only)  |

## Response Format

```json
{
  "id": "chatcmpl-xxx",
  "created": 1234567890,
  "model": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": [{
        "type": "video_url",
        "video_url": {
          "url": "data:video/mp4;base64,..."
        }
      }]
    },
    "finish_reason": "stop"
  }],
  "usage": {...}
}
```

## Extract Video

```bash
cat response.json | jq -r '.choices[0].message.content[0].video_url.url' \
  | sed 's/^data:video[^,]*,\s*//' | base64 -d > output.mp4
```

## File Description

| File                         | Description                    |
| ---------------------------- | ------------------------------ |
| `run_server.sh`              | Server startup script          |
| `run_curl_text_to_video.sh`  | Text-to-video curl example     |
| `run_curl_image_to_video.sh` | Image-to-video curl example    |
