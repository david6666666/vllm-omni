# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 Ref2VA reference-video preparation."""

from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

MINIMAX_H3_FPS = 24.0
MINIMAX_H3_QWEN_VIDEO_SAMPLE_FPS = 2.0
MINIMAX_H3_QWEN_TEMPORAL_PATCH = 2
MINIMAX_H3_BASE_SHORT_EDGE = 768
MINIMAX_H3_MAX_PIXELS = 768 * 1344
MINIMAX_H3_CANVAS_MULTIPLE = 32


def _nearest_multiple(value: float, multiple: int) -> int:
    return max(multiple, int(round(float(value) / multiple)) * multiple)


def _probe_video(path: str) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            (
                "stream=width,height,r_frame_rate,nb_read_frames,nb_frames,duration,sample_aspect_ratio:"
                "stream_tags=rotate:format=duration"
            ),
            "-of",
            "json",
            path,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(result.stdout).get("streams") or []
    if not streams:
        raise ValueError(f"media has no video stream: {path}")
    stream = streams[0]
    numerator, denominator = str(stream["r_frame_rate"]).split("/")
    raw_count = stream.get("nb_read_frames") or stream.get("nb_frames")
    if raw_count in (None, "", "N/A"):
        raise ValueError(f"cannot determine video frame count: {path}")
    raw_duration = stream.get("duration")
    if raw_duration in (None, "", "N/A"):
        raw_duration = json.loads(result.stdout).get("format", {}).get("duration")
    duration = (
        float(raw_duration)
        if raw_duration not in (None, "", "N/A")
        else int(raw_count) / (float(numerator) / float(denominator))
    )
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": float(numerator) / float(denominator),
        "frame_count": int(raw_count),
        "duration": duration,
    }


def _has_audio(path: str) -> bool:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=index",
            "-of",
            "json",
            path,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(json.loads(result.stdout).get("streams") or [])


def _reference_video_shape(width: int, height: int) -> tuple[int, int]:
    ratio = float(width) / float(height)
    if not 0.4 <= ratio <= 2.5:
        raise ValueError(f"reference video aspect ratio must be in [0.4, 2.5], got {width}x{height}")
    if ratio >= 1.0:
        target_width = MINIMAX_H3_BASE_SHORT_EDGE * ratio
        target_height = float(MINIMAX_H3_BASE_SHORT_EDGE)
    else:
        target_width = float(MINIMAX_H3_BASE_SHORT_EDGE)
        target_height = MINIMAX_H3_BASE_SHORT_EDGE / ratio
    area = target_width * target_height
    if area > MINIMAX_H3_MAX_PIXELS:
        scale = math.sqrt(MINIMAX_H3_MAX_PIXELS / area)
        target_width *= scale
        target_height *= scale
    return (
        _nearest_multiple(target_width, MINIMAX_H3_CANVAS_MULTIPLE),
        _nearest_multiple(target_height, MINIMAX_H3_CANVAS_MULTIPLE),
    )


def _transcode_reference_video(
    source: str,
    *,
    target_width: int,
    target_height: int,
    workdir: str,
    start_time_seconds: float = 0.0,
    duration_seconds: float | None = None,
) -> str:
    output = str(Path(workdir) / "prepared.mp4")
    duration_args = ["-t", f"{float(duration_seconds):.6f}"] if duration_seconds is not None else []
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-ss",
            f"{float(start_time_seconds):.6f}",
            "-i",
            source,
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            (f"fps={MINIMAX_H3_FPS:g},scale={target_width}:{target_height}:flags=lanczos,setsar=1"),
            *duration_args,
            "-metadata:s:v:0",
            "rotate=0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            output,
        ],
        check=True,
    )
    return output


def prepare_reference_videos(
    values: Any,
    *,
    target_frame_count: int,
    workdir: str,
    start_time_seconds: Any = None,
) -> list[dict[str, Any]]:
    # Keep this argument for compatibility with existing pipeline callers;
    # reference inputs are now bounded by their source/segment durations, not
    # clipped to the generated frame count.
    del target_frame_count
    if isinstance(values, (str, os.PathLike)):
        values = [values]
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("MiniMax H3 Ref2VA video input must be a path or a non-empty list of paths")
    if isinstance(start_time_seconds, (list, tuple)):
        start_times = list(start_time_seconds)
    else:
        start_times = [start_time_seconds] * len(values)
    if len(start_times) != len(values):
        raise ValueError("start_time_seconds must contain one value per reference video")

    prepared: list[dict[str, Any]] = []
    total_duration = 0.0
    for index, value in enumerate(values):
        item_start = start_times[index]
        if isinstance(value, dict):
            item_start = value.get("start_time_seconds", item_start)
            value = value.get("path", value.get("video_path", value.get("video_url")))
        if not isinstance(value, (str, os.PathLike)):
            raise TypeError(f"MiniMax H3 Ref2VA video references require file paths, got item {index}: {type(value)!r}")
        source = str(value)
        meta = _probe_video(source)
        start = float(item_start or 0.0)
        if start < 0 or start >= float(meta["duration"]):
            raise ValueError(f"reference video {index} start_time_seconds is outside the source duration")
        duration = min(15.0 - total_duration, float(meta["duration"]) - start)
        if duration < 2.0:
            raise ValueError(
                f"reference video {index} must provide at least 2 seconds after start_time_seconds, got {duration:.3f}"
            )
        total_duration += duration
        width, height = _reference_video_shape(meta["width"], meta["height"])
        item_workdir = Path(workdir) / f"video_{index}"
        item_workdir.mkdir(parents=True)
        prepared_path = _transcode_reference_video(
            source,
            target_width=width,
            target_height=height,
            workdir=str(item_workdir),
            start_time_seconds=start,
            duration_seconds=duration,
        )
        prepared.append(
            {
                "original_path": source,
                "prepared_path": prepared_path,
                "input_has_audio": _has_audio(source),
                "width": width,
                "height": height,
                "start_time_seconds": start,
                "duration_seconds": duration,
            }
        )
    return prepared


def load_video_frames(path: str) -> np.ndarray:
    try:
        import decord
    except ImportError:
        import av

        with av.open(path) as container:
            frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
        if not frames:
            raise ValueError(f"video has no frames: {path}")
        return np.stack(frames)

    reader = decord.VideoReader(path)
    if len(reader) <= 0:
        raise ValueError(f"video has no frames: {path}")
    frames = reader.get_batch(list(range(len(reader))))
    return frames.asnumpy() if hasattr(frames, "asnumpy") else np.asarray(frames)


def sample_reference_video_frames(
    prepared_path: str,
    *,
    workdir: str,
) -> dict[str, Any]:
    from PIL import Image

    meta = _probe_video(prepared_path)
    ratio = MINIMAX_H3_FPS / MINIMAX_H3_QWEN_VIDEO_SAMPLE_FPS
    indices: list[int] = []
    cursor = 0.0
    while True:
        frame_index = int(round(cursor))
        if frame_index >= meta["frame_count"]:
            break
        if not indices or frame_index > indices[-1]:
            indices.append(frame_index)
        cursor += ratio
    if not indices:
        raise ValueError(f"no frames sampled from {prepared_path}")

    frame_dir = Path(workdir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for output_index, source_index in enumerate(indices, start=1):
        output = frame_dir / f"frame_{output_index:06d}.png"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                prepared_path,
                "-vf",
                f"select=eq(n\\,{source_index})",
                "-vsync",
                "vfr",
                "-frames:v",
                "1",
                str(output),
            ],
            check=True,
        )
        frames.append(np.asarray(Image.open(output).convert("RGB")))

    timestamps = [index / MINIMAX_H3_QWEN_VIDEO_SAMPLE_FPS for index in range(len(indices))]
    timestamps += [timestamps[-1]] * ((-len(timestamps)) % MINIMAX_H3_QWEN_TEMPORAL_PATCH)
    block_timestamps = [
        (timestamps[index] + timestamps[index + MINIMAX_H3_QWEN_TEMPORAL_PATCH - 1]) / 2
        for index in range(
            0,
            len(timestamps),
            MINIMAX_H3_QWEN_TEMPORAL_PATCH,
        )
    ]
    return {
        "frames": frames,
        "block_timestamps": block_timestamps,
    }


def _soundfile_to_waveform(path: str) -> tuple[torch.Tensor, int]:
    import soundfile as sf

    data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    # soundfile is (samples, channels); torchaudio.load is (channels, samples).
    waveform = torch.from_numpy(data).t().contiguous()
    return waveform, int(sample_rate)


def load_audio_file(path: str) -> tuple[torch.Tensor, int]:
    """Load an audio file as ``(waveform[C, T] float32, sample_rate)``.

    torchaudio 2.6+ routes ``load`` through TorchCodec, whose wheels do not load
    on aarch64 with a CPU-only torch (they link CUDA libraries that are absent).
    Fall back to soundfile; if the container is not libsndfile-readable (e.g.
    mp3/m4a/mp4), demux to wav with ffmpeg first so any ffmpeg-supported input
    works at its native sample rate.
    """
    try:
        import torchaudio

        return torchaudio.load(path)
    except (ImportError, RuntimeError, OSError):
        pass
    try:
        return _soundfile_to_waveform(path)
    except Exception:
        import tempfile

        with tempfile.TemporaryDirectory(prefix="minimax_h3_audio_") as tmpdir:
            wav = str(Path(tmpdir) / "audio.wav")
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    path,
                    "-vn",
                    "-f",
                    "wav",
                    wav,
                ],
                check=True,
            )
            return _soundfile_to_waveform(wav)


def load_video_audio(
    path: str,
    *,
    start_time_seconds: float = 0.0,
    duration_seconds: float | None = None,
) -> tuple[torch.Tensor, int]:
    import tempfile

    with tempfile.TemporaryDirectory(prefix="minimax_h3_video_audio_") as tmpdir:
        output = str(Path(tmpdir) / "audio.wav")
        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-ss",
            f"{float(start_time_seconds):.6f}",
            "-i",
            path,
            "-vn",
            "-ac",
            "2",
            "-ar",
            "44100",
            "-f",
            "wav",
        ]
        if duration_seconds is not None:
            command.extend(["-t", f"{float(duration_seconds):.6f}"])
        command.append(output)
        subprocess.run(command, check=True)
        return load_audio_file(output)


__all__ = [
    "load_audio_file",
    "load_video_audio",
    "load_video_frames",
    "prepare_reference_videos",
    "sample_reference_video_frames",
]
