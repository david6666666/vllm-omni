# Text-To-Image

This example demonstrates how to deploy Qwen-Image model for online image generation service using vLLM-Omni.

## Start Server

### Basic Start

```bash
vllm serve Qwen/Qwen-Image --omni --port 8091
```

### Native FP8 checkpoint (recommended)

```bash
vllm serve unsloth/Qwen-Image-2512-FP8 --omni --port 8091
```

Notes:
- Keep `--quantization` unset to use native FP8 checkpoint metadata.
- You can force it with `--quantization fp8` if needed.

### GGUF checkpoint

```bash
vllm serve "unsloth/Qwen-Image-2512-GGUF:Q8_0" --omni --port 8091 \
  --quantization gguf \
  --load-format gguf
```

Notes:
- Replace `Q8_0` with the quant type available in the repo.
- If you know the exact file, you can use `<repo_id>/<filename>.gguf`.

!!! note
    If you encounter Out-of-Memory (OOM) issues or have limited GPU memory, you can enable VAE slicing and tiling to reduce memory usage, --vae-use-slicing --vae-use-tiling

### Start with Parameters

Or use the startup script:

```bash
bash run_server.sh
```

For quantized checkpoints with the script:

```bash
# Native FP8
MODEL=unsloth/Qwen-Image-2512-FP8 bash run_server.sh

# GGUF
MODEL="unsloth/Qwen-Image-2512-GGUF:Q8_0" QUANTIZATION=gguf LOAD_FORMAT=gguf bash run_server.sh
```

## API Calls

### Method 1: Using curl

```bash
# Basic text-to-image generation
bash run_curl_text_to_image.sh

# Or execute directly
curl -s http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "A beautiful landscape painting"}
    ],
    "extra_body": {
      "height": 1024,
      "width": 1024,
      "num_inference_steps": 50,
      "true_cfg_scale": 4.0,
      "seed": 42
    }
  }' | jq -r '.choices[0].message.content[0].image_url.url' | cut -d',' -f2- | base64 -d > output.png
```

### Method 2: Using Python Client

```bash
python openai_chat_client.py --prompt "A beautiful landscape painting" --output output.png
```

### Method 3: Using Gradio Demo

```bash
python gradio_demo.py
# Visit http://localhost:7860
```

## Request Format

### Simple Text Generation

```json
{
  "messages": [
    {"role": "user", "content": "A beautiful landscape painting"}
  ]
}
```

### Generation with Parameters

Use `extra_body` to pass generation parameters:

```json
{
  "messages": [
    {"role": "user", "content": "A beautiful landscape painting"}
  ],
  "extra_body": {
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 50,
    "true_cfg_scale": 4.0,
    "seed": 42
  }
}
```

### Multimodal Input (Text + Structured Content)

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "A beautiful landscape painting"}
      ]
    }
  ]
}
```

## Generation Parameters (extra_body)

| Parameter                | Type  | Default | Description                    |
| ------------------------ | ----- | ------- | ------------------------------ |
| `height`                 | int   | None    | Image height in pixels         |
| `width`                  | int   | None    | Image width in pixels          |
| `size`                   | str   | None    | Image size (e.g., "1024x1024") |
| `num_inference_steps`    | int   | 50      | Number of denoising steps      |
| `true_cfg_scale`         | float | 4.0     | Qwen-Image CFG scale           |
| `seed`                   | int   | None    | Random seed (reproducible)     |
| `negative_prompt`        | str   | None    | Negative prompt                |
| `num_outputs_per_prompt` | int   | 1       | Number of images to generate   |
| `--cfg-parallel-size`.   | int   | 1       | Number of GPUs for CFG parallelism |

## Response Format

```json
{
  "id": "chatcmpl-xxx",
  "created": 1234567890,
  "model": "Qwen/Qwen-Image",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": [{
        "type": "image_url",
        "image_url": {
          "url": "data:image/png;base64,..."
        }
      }]
    },
    "finish_reason": "stop"
  }],
  "usage": {...}
}
```

## Extract Image

```bash
# Extract base64 from response and decode to image
cat response.json | jq -r '.choices[0].message.content[0].image_url.url' | cut -d',' -f2- | base64 -d > output.png
```

## File Description

| File                        | Description                  |
| --------------------------- | ---------------------------- |
| `run_server.sh`             | Server startup script        |
| `run_curl_text_to_image.sh` | curl example                 |
| `openai_chat_client.py`     | Python client                |
| `gradio_demo.py`            | Gradio interactive interface |
