# Text-To-Image

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/text_to_image>.


This folder provides several entrypoints for experimenting with `Qwen/Qwen-Image` `Qwen/Qwen-Image-2512` `Tongyi-MAI/Z-Image-Turbo` and quantized checkpoints (FP8/GGUF) using vLLM-Omni:

- `text_to_image.py`: command-line script for single image generation with advanced options.
- `web_demo.py`: lightweight Gradio UI for interactive prompt/seed/CFG exploration.

Note that when you pass in multiple independent prompts, they will be processed sequentially. Batching requests is currently not supported.

## Basic Usage

```python
from vllm_omni.entrypoints.omni import Omni

if __name__ == "__main__":
    omni = Omni(model="Qwen/Qwen-Image")
    prompt = "a cup of coffee on the table"
    outputs = omni.generate(prompt)
    images = outputs[0].request_output[0].images
    images[0].save("coffee.png")
```

Or put more than one prompt in a request.

```python
from vllm_omni.entrypoints.omni import Omni

if __name__ == "__main__":
    omni = Omni(model="Qwen/Qwen-Image")
    prompts = [
      "a cup of coffee on a table",
      "a toy dinosaur on a sandy beach",
      "a fox waking up in bed and yawning",
    ]
    outputs = omni.generate(prompts)
    for i, output in enumerate(outputs):
      image = output.request_output[0].images[0].save(f"{i}.jpg")
```

!!! info

    However, it is not currently recommended to do so
    because not all models support batch inference,
    and batch requesting mostly does not provide significant performance improvement (despite the impression that it does).
    This feature is primarily for the sake of interface compatibility with vLLM and to allow for future improvements.

!!! info

    For diffusion pipelines, the stage config field `stage_args.[].runtime.max_batch_size` is 1 by default, and the input
    list is sliced into single-item requests before feeding into the diffusion pipeline. For models that do internally support
    batched inputs, you can [modify this configuration](https://github.com/vllm-project/vllm-omni/tree/main/configuration/stage_configs.md) to let the model accept a longer batch of prompts.

Apart from string prompt, vLLM-Omni also supports dictionary prompts in the same style as vLLM.
This is useful for models that support negative prompts.

```python
from vllm_omni.entrypoints.omni import Omni

if __name__ == "__main__":
    omni = Omni(model="Qwen/Qwen-Image")
    outputs = omni.generate([
      {
        "prompt": "a cup of coffee on a table"，
        "negative_prompt": "low resolution"
      },
      {
        "prompt": "a toy dinosaur on a sandy beach"，
        "negative_prompt": "cinematic, realistic"
      }
    ])
    for i, output in enumerate(outputs):
      image = output.request_output[0].images[0].save(f"{i}.jpg")
```

## Local CLI Usage

```bash
python text_to_image.py \
  --model Tongyi-MAI/Z-Image-Turbo \
  --prompt "a cup of coffee on the table" \
  --seed 42 \
  --cfg_scale 4.0 \
  --num_images_per_prompt 1 \
  --num_inference_steps 50 \
  --height 1024 \
  --width 1024 \
  --output outputs/coffee.png
```

### Native FP8 checkpoint example (recommended)

```bash
python text_to_image.py \
  --model Qwen/Qwen-Image-2512 \
  --quantized_weights unsloth/Qwen-Image-2512-FP8 \
  --quantization fp8 \
  --prompt "cinematic photo of an arctic fox under aurora" \
  --height 1328 \
  --width 1328 \
  --num_inference_steps 30 \
  --output outputs/qwen_image_fp8.png
```

Notes:
- Native FP8 auto-detection requires `quantization_config` metadata in the source repo
  (or an explicit quantization config override).
- For `unsloth/Qwen-Image-2512-FP8` (weights-only repo), use `--quantization fp8`
  to select FP8 online fallback.

### GGUF checkpoint example

```bash
python text_to_image.py \
  --model Qwen/Qwen-Image-2512 \
  --quantized_weights unsloth/Qwen-Image-2512-GGUF:Q8_0 \
  --quantization gguf \
  --load_format gguf \
  --prompt "a watercolor painting of tokyo in the rain" \
  --height 1024 \
  --width 1024 \
  --num_inference_steps 28 \
  --output outputs/qwen_image_gguf.png
```

Notes:
- Replace `Q8_0` with the quant type that exists in the GGUF repository.
- If you already know the exact filename, you can also use `<repo_id>/<filename>.gguf`.
- Keep `--model` on the base diffusion repo and pass quantized transformer weights through `--quantized_weights`.

### Online serving FP8 example

```bash
vllm serve Qwen/Qwen-Image-2512 --omni --port 8091 \
  --quantized-weights unsloth/Qwen-Image-2512-FP8 \
  --quantization fp8
```

### Online serving GGUF example

```bash
vllm serve Qwen/Qwen-Image-2512 --omni --port 8091 \
  --quantized-weights unsloth/Qwen-Image-2512-GGUF:Q8_0 \
  --quantization gguf \
  --load-format gguf
```

Key arguments:

- `--prompt`: text description (string).
- `--seed`: integer seed for deterministic sampling.
- `--cfg_scale`: true CFG scale (model-specific guidance strength).
- `--num_images_per_prompt`: number of images to generate per prompt (saves as `output`, `output_1`, ...).
- `--num_inference_steps`: diffusion sampling steps (more steps = higher quality, slower).
- `--height/--width`: output resolution (defaults 1024x1024).
- `--output`: path to save the generated PNG.
- `--quantization`: `fp8`, `gguf`, or `auto` (default behavior is auto-detect when unset).
- `--load_format`: `auto`, `hf`, `gguf` (use `gguf` for GGUF checkpoints).
- `--vae_use_slicing`: enable VAE slicing for memory optimization.
- `--vae_use_tiling`: enable VAE tiling for memory optimization.
- `--cfg_parallel_size`: set it to 2 to enable CFG Parallel. See more examples in [`user_guide`](https://github.com/vllm-project/vllm-omni/tree/main/docs/user_guide/diffusion/parallelism_acceleration.md#cfg-parallel).
- `--enable-cpu-offload`: enable CPU offloading for diffusion models.

> ℹ️ If you encounter OOM errors, try using `--vae_use_slicing` and `--vae_use_tiling` to reduce memory usage.

> ℹ️ Qwen-Image currently publishes best-effort presets at `1328x1328`, `1664x928`, `928x1664`, `1472x1140`, `1140x1472`, `1584x1056`, and `1056x1584`. Adjust `--height/--width` accordingly for the most reliable outcomes.

## Web UI Demo

Launch the gradio demo:

```bash
python gradio_demo.py --port 7862
```

Then open `http://localhost:7862/` on your local browser to interact with the web UI.

## Example materials

??? abstract "gradio_demo.py"
    ``````py
    --8<-- "examples/offline_inference/text_to_image/gradio_demo.py"
    ``````
??? abstract "text_to_image.py"
    ``````py
    --8<-- "examples/offline_inference/text_to_image/text_to_image.py"
    ``````
