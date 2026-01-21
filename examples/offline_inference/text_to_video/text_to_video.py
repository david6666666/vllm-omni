# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.utils.platform_utils import detect_device_type, is_npu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a video with Wan2.2 T2V.")
    parser.add_argument(
        "--model",
        default="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        help="Diffusers Wan2.2 model ID or local path.",
    )
    parser.add_argument("--prompt", default="A serene lakeside sunrise with mist over the water.", help="Text prompt.")
    parser.add_argument("--negative_prompt", default="", help="Negative prompt.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--guidance_scale", type=float, default=4.0, help="CFG scale (applied to low/high).")
    parser.add_argument("--guidance_scale_high", type=float, default=None, help="Optional separate CFG for high-noise.")
    parser.add_argument("--height", type=int, default=720, help="Video height.")
    parser.add_argument("--width", type=int, default=1280, help="Video width.")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames (Wan default is 81).")
    parser.add_argument(
        "--frame_rate",
        type=float,
        default=None,
        help="Optional generation frame rate (used by models like LTX2). Defaults to --fps.",
    )
    parser.add_argument("--num_inference_steps", type=int, default=40, help="Sampling steps.")
    parser.add_argument("--boundary_ratio", type=float, default=0.875, help="Boundary split ratio for low/high DiT.")
    parser.add_argument(
        "--flow_shift", type=float, default=5.0, help="Scheduler flow_shift (5.0 for 720p, 12.0 for 480p)."
    )
    parser.add_argument("--output", type=str, default="wan22_output.mp4", help="Path to save the video (mp4).")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second for the output video.")
    parser.add_argument(
        "--enable-cpu-offload",
        action="store_true",
        help="Enable CPU offloading for diffusion models.",
    )
    parser.add_argument(
        "--ulysses_degree",
        type=int,
        default=1,
        help="Number of GPUs used for ulysses sequence parallelism.",
    )
    parser.add_argument(
        "--ring_degree",
        type=int,
        default=1,
        help="Number of GPUs used for ring sequence parallelism.",
    )
    parser.add_argument(
        "--enforce_eager",
        action="store_true",
        help="Disable torch.compile and force eager execution.",
    )
    parser.add_argument(
        "--audio_output",
        type=str,
        default=None,
        help="Optional path to save audio (wav) when the pipeline returns audio (e.g., LTX2).",
    )
    parser.add_argument(
        "--audio_sample_rate",
        type=int,
        default=24000,
        help="Sample rate for audio output when saved (default: 24000 for LTX2).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = detect_device_type()
    generator = torch.Generator(device=device).manual_seed(args.seed)
    frame_rate = args.frame_rate if args.frame_rate is not None else float(args.fps)

    # Enable VAE memory optimizations on NPU
    vae_use_slicing = is_npu()
    vae_use_tiling = is_npu()

    # Configure parallel settings (only SP is supported for Wan)
    # Note: cfg_parallel and tensor_parallel are not implemented for Wan models
    parallel_config = DiffusionParallelConfig(
        ulysses_degree=args.ulysses_degree,
        ring_degree=args.ring_degree,
    )

    # Check if profiling is requested via environment variable
    profiler_enabled = bool(os.getenv("VLLM_TORCH_PROFILER_DIR"))

    omni = Omni(
        model=args.model,
        vae_use_slicing=vae_use_slicing,
        vae_use_tiling=vae_use_tiling,
        boundary_ratio=args.boundary_ratio,
        flow_shift=args.flow_shift,
        enable_cpu_offload=args.enable_cpu_offload,
        parallel_config=parallel_config,
        enforce_eager=args.enforce_eager,
    )

    if profiler_enabled:
        print("[Profiler] Starting profiling...")
        omni.start_profile()

    # Print generation configuration
    print(f"\n{'=' * 60}")
    print("Generation Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Inference steps: {args.num_inference_steps}")
    print(f"  Frames: {args.num_frames}")
    print(f"  Parallel configuration: ulysses_degree={args.ulysses_degree}, ring_degree={args.ring_degree}")
    print(f"  Video size: {args.width}x{args.height}")
    print(f"{'=' * 60}\n")

    generation_start = time.perf_counter()
    frames = omni.generate(
        args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        generator=generator,
        guidance_scale=args.guidance_scale,
        guidance_scale_2=args.guidance_scale_high,
        num_inference_steps=args.num_inference_steps,
        num_frames=args.num_frames,
        frame_rate=frame_rate,
        enable_cpu_offload=True,
    )
    generation_end = time.perf_counter()
    generation_time = generation_end - generation_start

    # Print profiling results
    print(f"Total generation time: {generation_time:.4f} seconds ({generation_time * 1000:.2f} ms)")

    audio = None
    if isinstance(frames, list):
        frames = frames[0] if frames else None

    if isinstance(frames, OmniRequestOutput):
        if frames.final_output_type != "image":
            raise ValueError(
                f"Unexpected output type '{frames.final_output_type}', expected 'image' for video generation."
            )
        if frames.is_pipeline_output and frames.request_output is not None:
            inner_output = frames.request_output
            if isinstance(inner_output, list):
                inner_output = inner_output[0] if inner_output else None
            if isinstance(inner_output, OmniRequestOutput):
                frames = inner_output
        if isinstance(frames, OmniRequestOutput):
            if frames.images:
                if len(frames.images) == 1 and isinstance(frames.images[0], tuple) and len(frames.images[0]) == 2:
                    frames, audio = frames.images[0]
                elif len(frames.images) == 1 and isinstance(frames.images[0], dict):
                    audio = frames.images[0].get("audio")
                    frames = frames.images[0].get("frames") or frames.images[0].get("video")
                else:
                    frames = frames.images
            else:
                raise ValueError("No video frames found in OmniRequestOutput.")

    if isinstance(frames, list) and frames:
        first_item = frames[0]
        if isinstance(first_item, tuple) and len(first_item) == 2:
            frames, audio = first_item
        elif isinstance(first_item, dict):
            audio = first_item.get("audio")
            frames = first_item.get("frames") or first_item.get("video")
        elif isinstance(first_item, list):
            frames = first_item

    if isinstance(frames, tuple) and len(frames) == 2:
        frames, audio = frames
    elif isinstance(frames, dict):
        audio = frames.get("audio")
        frames = frames.get("frames") or frames.get("video")

    if frames is None:
        raise ValueError("No video frames found in output.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from diffusers.utils import export_to_video
    except ImportError:
        raise ImportError("diffusers is required for export_to_video.")

    def _normalize_frame(frame):
        if isinstance(frame, torch.Tensor):
            frame_tensor = frame.detach().cpu()
            if frame_tensor.dim() == 4 and frame_tensor.shape[0] == 1:
                frame_tensor = frame_tensor[0]
            if frame_tensor.dim() == 3 and frame_tensor.shape[0] in (3, 4):
                frame_tensor = frame_tensor.permute(1, 2, 0)
            if frame_tensor.is_floating_point():
                frame_tensor = frame_tensor.clamp(-1, 1) * 0.5 + 0.5
            return frame_tensor.float().numpy()
        if isinstance(frame, np.ndarray):
            frame_array = frame
            if frame_array.ndim == 4 and frame_array.shape[0] == 1:
                frame_array = frame_array[0]
            if np.issubdtype(frame_array.dtype, np.integer):
                frame_array = frame_array.astype(np.float32) / 255.0
            return frame_array
        try:
            from PIL import Image
        except ImportError:
            Image = None
        if Image is not None and isinstance(frame, Image.Image):
            return np.asarray(frame).astype(np.float32) / 255.0
        return frame

    def _ensure_frame_list(video_array):
        if isinstance(video_array, list):
            if len(video_array) == 0:
                return video_array
            first_item = video_array[0]
            if isinstance(first_item, np.ndarray):
                if first_item.ndim == 5:
                    return list(first_item[0])
                if first_item.ndim == 4:
                    if len(video_array) == 1:
                        return list(first_item)
                    return list(first_item)
                if first_item.ndim == 3:
                    return video_array
            return video_array
        if isinstance(video_array, np.ndarray):
            if video_array.ndim == 5:
                return list(video_array[0])
            if video_array.ndim == 4:
                return list(video_array)
            if video_array.ndim == 3:
                return [video_array]
        return video_array

    # frames may be np.ndarray, torch.Tensor, or list of tensors/arrays/images
    # export_to_video expects a list of frames with values in [0, 1]
    if isinstance(frames, torch.Tensor):
        video_tensor = frames.detach().cpu()
        if video_tensor.dim() == 5:
            if video_tensor.shape[1] in (3, 4):
                video_tensor = video_tensor[0].permute(1, 2, 3, 0)
            else:
                video_tensor = video_tensor[0]
        elif video_tensor.dim() == 4 and video_tensor.shape[0] in (3, 4):
            video_tensor = video_tensor.permute(1, 2, 3, 0)
        if video_tensor.is_floating_point():
            video_tensor = video_tensor.clamp(-1, 1) * 0.5 + 0.5
        video_array = video_tensor.float().numpy()
    elif isinstance(frames, np.ndarray):
        video_array = frames
        if video_array.ndim == 5:
            video_array = video_array[0]
        if np.issubdtype(video_array.dtype, np.integer):
            video_array = video_array.astype(np.float32) / 255.0
    elif isinstance(frames, list):
        if len(frames) == 0:
            raise ValueError("No video frames found in output.")
        video_array = [_normalize_frame(frame) for frame in frames]
    else:
        video_array = frames

    video_array = _ensure_frame_list(video_array)

    use_ltx2_export = False
    if args.model and "ltx" in str(args.model).lower():
        use_ltx2_export = True
    if audio is not None:
        use_ltx2_export = True

    if use_ltx2_export:
        try:
            from diffusers.pipelines.ltx2.export_utils import encode_video
        except ImportError:
            raise ImportError("diffusers is required for LTX2 encode_video.")

        if isinstance(video_array, list):
            frames_np = np.stack(video_array, axis=0)
        elif isinstance(video_array, np.ndarray):
            frames_np = video_array
        else:
            frames_np = np.asarray(video_array)

        frames_u8 = (frames_np * 255).round().clip(0, 255).astype("uint8")
        video_tensor = torch.from_numpy(frames_u8)

        audio_out = None
        if audio is not None:
            if isinstance(audio, list):
                audio = audio[0] if audio else None
            if isinstance(audio, np.ndarray):
                audio = torch.from_numpy(audio)
            if isinstance(audio, torch.Tensor):
                audio_out = audio
                if audio_out.dim() > 1:
                    audio_out = audio_out[0]
                audio_out = audio_out.float().cpu()

        encode_video(
            video_tensor,
            fps=args.fps,
            audio=audio_out,
            audio_sample_rate=args.audio_sample_rate if audio_out is not None else None,
            output_path=str(output_path),
        )
    else:
        export_to_video(video_array, str(output_path), fps=args.fps)
    print(f"Saved generated video to {output_path}")

    if audio is not None and args.audio_output:
        audio_path = Path(args.audio_output)
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().float().numpy()
        if isinstance(audio, np.ndarray) and audio.ndim == 3:
            audio_data = audio[0].T
        elif isinstance(audio, np.ndarray) and audio.ndim == 2:
            audio_data = audio.T
        else:
            audio_data = audio
        try:
            import soundfile as sf

            sf.write(str(audio_path), audio_data, args.audio_sample_rate)
        except ImportError:
            try:
                import scipy.io.wavfile as wav

                if isinstance(audio_data, np.ndarray) and np.issubdtype(audio_data.dtype, np.floating):
                    audio_data = np.clip(audio_data, -1.0, 1.0)
                    audio_data = (audio_data * 32767).astype(np.int16)
                wav.write(str(audio_path), args.audio_sample_rate, audio_data)
            except ImportError:
                raise ImportError(
                    "Either 'soundfile' or 'scipy' is required to save audio files. "
                    "Install with: pip install soundfile or pip install scipy"
                )
        print(f"Saved generated audio to {audio_path}")

    if profiler_enabled:
        print("\n[Profiler] Stopping profiler and collecting results...")
        profile_results = omni.stop_profile()
        if profile_results and isinstance(profile_results, dict):
            traces = profile_results.get("traces", [])
            print("\n" + "=" * 60)
            print("PROFILING RESULTS:")
            for rank, trace in enumerate(traces):
                print(f"\nRank {rank}:")
                if trace:
                    print(f"  • Trace: {trace}")
            if not traces:
                print("  No traces collected.")
            print("=" * 60)
        else:
            print("[Profiler] No valid profiling data returned.")


if __name__ == "__main__":
    main()
