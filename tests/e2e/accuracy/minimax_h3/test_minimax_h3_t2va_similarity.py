# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest
import requests
import torch
from huggingface_hub import snapshot_download

from tests.e2e.accuracy.helpers import (
    assert_video_metadata,
    assert_video_similarity_metrics,
    probe_binary,
    probe_video,
    reset_artifact_dir,
)
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServer

pytestmark = [pytest.mark.benchmark, pytest.mark.diffusion, pytest.mark.full_model]

MODEL_REPO_ID = "MiniMaxAI/MiniMax-H3"
# The previously pinned revision was removed from the Hugging Face repository.
MODEL_REVISION = "main"
MODEL_ENV_VAR = "VLLM_TEST_MINIMAX_H3_FL2VA_MODEL"
REF2VA_MODEL_ENV_VAR = "VLLM_TEST_MINIMAX_H3_REF2VA_MODEL"
# Keep the official asset link visible in the test report. Hugging Face's
# ``blob`` URL is an HTML page, so resolve it to the raw asset for download.
REFERENCE_VIDEO_URL = "https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/assets/t2va.mp4"
REFERENCE_VIDEO_SHA256 = "d66903241362e224085cc93f7a5e70fba6ab378d0ac6ff6af83acf2559849a42"
I2VA_REFERENCE_VIDEO_URL = "https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/assets/i2va.mp4"
I2VA_REFERENCE_VIDEO_SHA256 = "5a5d6e0fcdda7e6a5d98cb92d9e5cb4b6b0b44d6e1dfd5a370d16dfaa79b9ae6"
I2VA_IMAGE_URL = (
    "https://cdn.hailuoai.com/prod/hailuo_demo/testsets/H3_AA_I2VA/gallery/"
    "sr_v17_variants_seed42_43_20260724/inputs/4a3a90bf9100_KDmcbkhzYo5sjjxr9FqcVmWVnzb.png"
)
I2VA_IMAGE_SHA256 = "352e959e08b886406d3766cee25ebe7bbf679e6d53eeffbd1c607b8dc93e2068"
REF2VA_REFERENCE_VIDEO_URL = "https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/assets/r2va.mp4"
REF2VA_REFERENCE_VIDEO_SHA256 = "5c9d33dfc517438c59e93da7b3d0b2a62e3cf69bc2adb7d1b2d33d7d8df7161f"
REF2VA_INPUT_VIDEO_URL = (
    "https://cdn.hailuoai.com/prod/hailuo_demo/testsets/h3_promo_eval_ref2va/"
    "gallery/sr_v2p26_trio_seed42_20260724/inputs/"
    "297573323635_00_%E8%A7%86%E9%A2%911_YnyRbxEwio_video_20260525_163755_1927e9d3.mp4"
)
REF2VA_INPUT_VIDEO_SHA256 = "d6edc20ac82aa980c9cbd0b05b371f60d12dc8d018492e571b42de0086819c41"
REF2VA_INPUT_AUDIO_URL = (
    "https://cdn.hailuoai.com/prod/hailuo_demo/testsets/h3_promo_eval_ref2va/"
    "gallery/sr_v2p26_trio_seed42_20260724/inputs/"
    "f463d523c5ce_01_%E9%9F%B3%E9%A2%911_RSLcbpzJPo_6%E6%9C%885%E6%97%A5(1).mp3"
)
REF2VA_INPUT_AUDIO_SHA256 = "2a48d447b343b38a2c504d5281a0c97c688db0ba87931ed9282c598bd6752454"

I2VA_PROMPT = """For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description: [Shot 1] This is a live-action, cinematic shot with a shallow depth of field. The camera holds a perfectly static shot throughout the entire eight-second duration, capturing a cozy family gathering in a traditional Japanese dining room. The scene opens with a large, intricately patterned blue and white ceramic bowl of ramen in the immediate foreground, rendered in crisp, sharp focus. The bowl sits on a smooth, polished long wooden table. Inside the bowl, a rich, oily golden-brown broth surrounds yellow wavy noodles, topped with two thick, round slices of chashu pork featuring visible fat marbling and a distinct spiral meat pattern. A generous mound of freshly chopped, bright green scallions rests in the center, and a crisp, dark green rectangular sheet of nori seaweed is tucked into the right edge. To the left of the bowl, a pair of light brown wooden chopsticks rests horizontally on a small, dark rectangular chopstick rest, near a small cylindrical ceramic teacup with blue painted patterns. On the right side of the table, a spherical paper lantern with a ribbed bamboo frame sits on a black wooden base. In the background, a large family of seven is gathered around the table, initially appearing as a soft, blurred presence. Behind them, traditional Japanese sliding shoji screens with wooden lattice frames are open, revealing a bright outdoor scene with lush green trees. Early in the clip, the thick, white steam rising from the hot ramen broth immediately intensifies, billowing upwards in thick, swirling clouds that dance continuously above the bowl. As the clip progresses into the middle seconds, the camera maintains its static position while the focus begins a deliberate, smooth shift deeper into the room. The foreground ramen bowl, its vibrant ingredients, and the rising steam gradually soften into a hazy, out-of-focus blur. Simultaneously, the family members in the background come into sharp, detailed clarity. The heavy steam continues to rise from the foreground, creating a dynamic, translucent veil between the camera and the family. With the focus now firmly locked on the background, the vibrant family dinner comes alive. The man in the dark navy blue long-sleeved shirt on the left leans forward, his mouth moving animatedly in a silent exchange. The young girl in the crisp white short-sleeved t-shirt beside him smiles brightly, looking toward the center of the table. The woman on the far left, wearing a soft light blue long-sleeved blouse, turns her head slightly, smiling gently. Across the table, the woman in the light grey button-down shirt smiles broadly, her eyes crinkling, as she rests her hands near her plate. The woman in the dark grey top further back uses her wooden chopsticks to pick up a small piece of food from a central ceramic dish filled with bright red pickled vegetables. The woman in the center back in the light grey sweater smiles gently, her hands clasped softly in front of her, observing the interaction. Throughout the remainder of the clip, the family continues their lively physical interaction, their mouths moving in continuous, silent cadences of conversation, while the thick, white steam from the blurred ramen bowl in the foreground never stops rising, adding a comforting atmosphere to the warm gathering.

overall_soundscape: The soundscape begins with a quiet room tone mixed with the faint, airy rustle of the thick steam billowing from the hot ramen bowl in the foreground, accompanied by the subtle, continuous hissing and bubbling of the rich broth. As the visual focus shifts deeper into the room, the physical sounds of the bustling family dinner become dominant in the foreground. The clear, sharp clinking of ceramic bowls and wooden chopsticks touching plates is clearly heard as the family members reach for food. This is followed by the faint, muffled thud of a cup being set down on the smooth wooden table, and the subtle, rhythmic rustle of cotton and wool clothing as the family members lean forward and gesture, perfectly capturing the lively, physical atmosphere of the shared meal.

non_diegetic_music: A gentle, heartwarming acoustic guitar melody plays softly in the background, accompanied by the subtle, resonant notes of a traditional Japanese koto. The music maintains a slow, comforting tempo that enhances the cozy, nostalgic, and joyful atmosphere of the family gathering."""

REF2VA_PROMPT = """subject_definitions:
<Subject 1> is the young man with short wavy blonde hair, wearing a bright pink suit jacket, matching pink trousers, an unbuttoned white shirt, and silver rings, holding a small black lamb in his arms in <Video 1>.
<Video 1> is the source video for the editing task.
<Audio 1> is the synchronized audio track of <Video 1>, providing the background music.
<Audio 2> is the voice timbre reference for <Subject 1>'s voice, containing a spoken male voiceover.

summary:
[video editing + audio reference + audio reuse] The target video is an edited version of <Video 1>. <Subject 1>, wearing a bright pink suit and holding a black lamb, stands in a grassy field with other white lambs in the background. The edit animates <Subject 1>'s face to speak the user-provided dialogue. <Audio 1> is partially reused as the continuous background music, while the target references the calm male voice timbre of <Audio 2> for <Subject 1>'s spoken lines.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - the man retains his identity, wavy blonde hair, pink suit, white shirt, accessories, and the black lamb he holds, with his mouth newly animated to speak.
<Video 1> (source video editing): fully_preserved - the original camera framing, warm golden hour lighting, grassy hill setting, and background white lambs are maintained while the central character is edited.
<Audio 1>: partially_copy - the atmospheric background music from <Audio 1> is reused in the target video, mixed beneath the newly added spoken dialogue.
<Audio 2>: reference - the target audio references the male voice timbre from <Audio 2> to generate <Subject 1>'s spoken dialogue.

detailed_description:
The target video is in realistic photographic style.
[Shot 1] The shot begins from the source <Video 1>, showing <Subject 1>, a young man with short wavy blonde hair, wearing a bright pink suit jacket, matching pink trousers, and a casually unbuttoned white shirt. He stands confidently in a sunlit green pasture, gently holding a small black lamb securely in his arms. The warm, golden hour lighting casts soft shadows across his face and the bright pink fabric of his suit. Behind him, several white lambs stand and graze on the rolling grassy hill against a clear, pale blue sky. The atmospheric background music from <Audio 1> plays continuously throughout the scene. <Subject 1> physically speaks, his mouth movements naturally syncing to the new dialogue, with his voice timbre referencing the calm male delivery from <Audio 2>. Looking thoughtfully forward, <Subject 1> (S1) speaks softly, <d>[English] Follow the wind, live free.</d> As he delivers the line, he subtly shifts his weight, cradling the resting black lamb while the camera slowly pushes in. <Subject 1> (S1) continues his thought, <d>[English] Leave worries behind, enjoy the moment.</d> Exactly as his voice stops, his lips meet in a relaxed, peaceful smile, and his jaw ceases speaking motion. He then turns his gaze slightly away toward the horizon, gently stroking the black lamb's fleece with his fingers as the camera holds on this tranquil, sunlit state through the end of the video.

overall_soundscape:
The soundscape consists of the continuous, atmospheric background music from <Audio 1>, overlaid with the clear, calm male dialogue spoken by the main character, referencing the voice timbre of <Audio 2>.

non_diegetic_music:
The atmospheric, sustained background music from <Audio 1> is reused as the continuous score, playing quietly beneath the spoken dialogue.
"""
# This is the official reproducible 768p T2VA prompt associated with
# ``assets/t2va.mp4`` in the MiniMax H3 repository.
PROMPT = """integrated_multimodal_description: [Shot 1] Cinematic, medium wide shot, pushing in slowly. In the cavernous, dimly lit bridge of a starship, sleek metallic consoles with glowing amber displays flank a massive, curved observation window. A female captain, in her late 40s with an athletic build and short silver-streaked black hair, stands in the center midground. She wears a structured, high-collared dark navy military tunic with silver chest insignias. Her back is to the camera, silhouetted against the cool, ambient starlight pouring through the thick glass. She stands perfectly still with her hands clasped tightly behind her back. Outside the window, a massive armada of jagged, dark grey dreadnoughts hovers in tight formation against a deep purple space nebula. The fleet's massive rear thrusters begin to glow with an intense, escalating bright blue light. [Shot 2] At 00:04.500, the camera cuts to a close-up of the captain's face and shakes strongly. The brilliant blue-white light from the fleet's gathering energy reflects vividly in her dark eyes. Suddenly, a blinding white flash floods through the window, completely washing out the background as the fleet jumps to hyperspace. The sheer spatial force violently jolts the bridge, causing the captain from Shot 1 to stagger slightly forward, her shoulders tensing as she visibly braces herself against the physical tremors. As the intense white light fades abruptly, leaving only the dim, empty expanse of the purple nebula reflected on her starkly lit skin, her jaw clenches, and she slowly closes her eyes in the newly emptied space.
overall_soundscape: A low, resonant hum of the ship's ambient life support systems serves as the baseline, soon drowned out by an audible, escalating, high-pitched electronic whine as the fleet outside charges its hyperdrives. A massive, deafening, bass-heavy boom and sharp crackle erupts during the blinding flash, accompanied by the loud metallic creaking, rattling, and deep thuds of the bridge's bulkheads vibrating under immense physical stress. The intense roaring impact then cuts abruptly back to a hollow, echoing room tone, leaving only the faint, steady hum of the isolated bridge.
non_diegetic_music: Cinematic space-opera orchestral score, slow tempo, featuring a solitary, mournful French horn melody over deep, sustained string dissonances that build rapidly in volume and intensity, swelling to a massive orchestral peak before snapping immediately into silence right after the jump."""

WIDTH = 1344
HEIGHT = 768
FPS = 24
NUM_FRAMES = 243
I2VA_NUM_FRAMES = 192
REF2VA_NUM_FRAMES = 124
NUM_INFERENCE_STEPS = 50
FLOW_SHIFT = 12.0
AUDIO_FLOW_SHIFT = 3.0
DURATION_SECONDS = 10.0
I2VA_DURATION_SECONDS = 8.0
REF2VA_DURATION_SECONDS = 5.0
SEED = 0
SSIM_THRESHOLD = 0.97
PSNR_THRESHOLD = 34.0
REQUEST_TIMEOUT_SECONDS = 60 * 60


def _model_name(model_env_var: str = MODEL_ENV_VAR, component: str = "FL2VA") -> str:
    configured = os.environ.get(model_env_var)
    if configured:
        return configured

    snapshot_root = snapshot_download(
        repo_id=MODEL_REPO_ID,
        revision=MODEL_REVISION,
        allow_patterns=[f"{component}/**"],
    )
    return str(Path(snapshot_root) / component)


def _server_args() -> list[str]:
    return [
        "--trust-remote-code",
        "--num-gpus",
        "4",
        "--usp",
        "4",
        "--ring",
        "1",
        "--use-hsdp",
        "--hsdp-shard-size",
        "4",
        "--text-encoder-tp-size",
        "4",
        "--vae-patch-parallel-size",
        "4",
        "--vae-parallel-mode",
        "tile",
        "--vae-use-tiling",
        "--diffusion-attention-backend",
        "FLASH_ATTN",
        "--stage-init-timeout",
        "1800",
        "--init-timeout",
        "1800",
    ]


def _download_reference_asset(
    output_dir: Path,
    *,
    filename: str,
    url: str,
    sha256: str,
    label: str,
) -> Path:
    reference_path = output_dir / filename
    download_url = url.replace("/blob/", "/resolve/", 1)
    response = requests.get(download_url, timeout=120)
    response.raise_for_status()
    content = response.content
    digest = hashlib.sha256(content).hexdigest()
    assert digest == sha256, f"Official MiniMax H3 {label} reference changed: got {digest}, expected {sha256}"
    reference_path.write_bytes(content)
    return reference_path


def _download_reference_video(output_dir: Path) -> Path:
    return _download_reference_asset(
        output_dir,
        filename="reference.mp4",
        url=REFERENCE_VIDEO_URL,
        sha256=REFERENCE_VIDEO_SHA256,
        label="T2VA",
    )


def _probe_audio(path: Path) -> dict[str, str | int]:
    probe_binary("ffprobe")
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name,sample_rate,channels",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
    assert len(streams) == 1, f"Expected one audio stream in {path}, got {streams}"
    stream = streams[0]
    return {
        "codec_name": str(stream["codec_name"]),
        "sample_rate": int(stream["sample_rate"]),
        "channels": int(stream["channels"]),
    }


def _assert_official_video(
    generated_path: Path,
    reference_path: Path,
    *,
    frame_count: int,
    label: str,
    ssim_threshold: float = SSIM_THRESHOLD,
    psnr_threshold: float = PSNR_THRESHOLD,
) -> None:
    reference_metadata = probe_video(reference_path)
    generated_metadata = probe_video(generated_path)
    assert reference_metadata == generated_metadata
    assert_video_metadata(
        generated_metadata,
        width=WIDTH,
        height=HEIGHT,
        fps=FPS,
        frame_count=frame_count,
    )

    expected_audio = {"codec_name": "aac", "sample_rate": 32000, "channels": 2}
    assert _probe_audio(reference_path) == expected_audio
    assert _probe_audio(generated_path) == expected_audio

    assert_video_similarity_metrics(
        label=label,
        online_path=generated_path,
        offline_path=reference_path,
        ssim_threshold=ssim_threshold,
        psnr_threshold=psnr_threshold,
    )


@hardware_test(res={"cuda": "H100"}, num_cards=4)
def test_minimax_h3_t2va_matches_official_reference(
    accuracy_artifact_root: Path,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("MiniMax H3 T2VA accuracy test requires CUDA.")

    probe_binary("ffmpeg")
    probe_binary("ffprobe")
    output_dir = reset_artifact_dir(accuracy_artifact_root / "minimax_h3_t2va")
    reference_path = _download_reference_video(output_dir)
    generated_path = output_dir / "vllm_omni.mp4"

    server_env = {
        "FLASHINFER_DISABLE_VERSION_CHECK": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_OMNI_VIDEO_SYNC_TIMEOUT": str(REQUEST_TIMEOUT_SECONDS),
        "VLLM_OMNI_STORAGE_PATH": str(output_dir / "storage"),
    }
    request_data = {
        "prompt": PROMPT,
        "seed": str(SEED),
        "extra_params": json.dumps(
            {
                "task": "t2va",
                "conditions": [],
                "target": {
                    "short_edge": HEIGHT,
                    "aspect_ratio": "16:9",
                    "duration_seconds": DURATION_SECONDS,
                },
            }
        ),
    }

    with OmniServer(
        _model_name(),
        _server_args(),
        env_dict=server_env,
        use_omni=True,
    ) as server:
        response = requests.post(
            f"http://{server.host}:{server.port}/v1/videos/sync",
            data=request_data,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

    response.raise_for_status()
    assert response.headers["content-type"].startswith("video/mp4")
    generated_path.write_bytes(response.content)

    _assert_official_video(
        generated_path,
        reference_path,
        frame_count=NUM_FRAMES,
        label="minimax_h3_t2va_official_reference",
    )


@hardware_test(res={"cuda": "H100"}, num_cards=4)
def test_minimax_h3_i2va_matches_official_reference(
    accuracy_artifact_root: Path,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("MiniMax H3 I2VA accuracy test requires CUDA.")

    probe_binary("ffmpeg")
    probe_binary("ffprobe")
    output_dir = reset_artifact_dir(accuracy_artifact_root / "minimax_h3_i2va")
    reference_path = _download_reference_asset(
        output_dir,
        filename="reference.mp4",
        url=I2VA_REFERENCE_VIDEO_URL,
        sha256=I2VA_REFERENCE_VIDEO_SHA256,
        label="I2VA",
    )
    image_path = _download_reference_asset(
        output_dir,
        filename="input.png",
        url=I2VA_IMAGE_URL,
        sha256=I2VA_IMAGE_SHA256,
        label="I2VA input image",
    )
    generated_path = output_dir / "vllm_omni.mp4"

    server_env = {
        "FLASHINFER_DISABLE_VERSION_CHECK": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_OMNI_VIDEO_SYNC_TIMEOUT": str(REQUEST_TIMEOUT_SECONDS),
        "VLLM_OMNI_STORAGE_PATH": str(output_dir / "storage"),
    }
    request_data = {
        "prompt": I2VA_PROMPT,
        "width": str(WIDTH),
        "height": str(HEIGHT),
        "fps": str(FPS),
        "num_inference_steps": str(NUM_INFERENCE_STEPS),
        "flow_shift": str(FLOW_SHIFT),
        "seed": str(SEED),
        "extra_params": json.dumps(
            {
                "task": "fl2va",
                "duration": I2VA_DURATION_SECONDS,
                "aspect_ratio": "auto",
                "frame_indices": [0],
                "audio_flow_shift": AUDIO_FLOW_SHIFT,
            }
        ),
    }

    with OmniServer(
        _model_name(),
        _server_args(),
        env_dict=server_env,
        use_omni=True,
    ) as server:
        with image_path.open("rb") as image_file:
            response = requests.post(
                f"http://{server.host}:{server.port}/v1/videos/sync",
                data=request_data,
                files={"input_reference": ("input.png", image_file, "image/png")},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )

    response.raise_for_status()
    assert response.headers["content-type"].startswith("video/mp4")
    generated_path.write_bytes(response.content)
    _assert_official_video(
        generated_path,
        reference_path,
        frame_count=I2VA_NUM_FRAMES,
        label="minimax_h3_i2va_official_reference",
    )


@hardware_test(res={"cuda": "H100"}, num_cards=4)
def test_minimax_h3_ref2va_matches_official_reference(
    accuracy_artifact_root: Path,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("MiniMax H3 Ref2VA accuracy test requires CUDA.")

    probe_binary("ffmpeg")
    probe_binary("ffprobe")
    output_dir = reset_artifact_dir(accuracy_artifact_root / "minimax_h3_ref2va")
    reference_path = _download_reference_asset(
        output_dir,
        filename="reference.mp4",
        url=REF2VA_REFERENCE_VIDEO_URL,
        sha256=REF2VA_REFERENCE_VIDEO_SHA256,
        label="Ref2VA",
    )
    input_video_path = _download_reference_asset(
        output_dir,
        filename="input_video.mp4",
        url=REF2VA_INPUT_VIDEO_URL,
        sha256=REF2VA_INPUT_VIDEO_SHA256,
        label="Ref2VA input video",
    )
    input_audio_path = _download_reference_asset(
        output_dir,
        filename="input_audio.mp3",
        url=REF2VA_INPUT_AUDIO_URL,
        sha256=REF2VA_INPUT_AUDIO_SHA256,
        label="Ref2VA input audio",
    )
    generated_path = output_dir / "vllm_omni.mp4"

    server_env = {
        "FLASHINFER_DISABLE_VERSION_CHECK": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_OMNI_VIDEO_SYNC_TIMEOUT": str(REQUEST_TIMEOUT_SECONDS),
        "VLLM_OMNI_STORAGE_PATH": str(output_dir / "storage"),
    }
    request_data = {
        "prompt": REF2VA_PROMPT,
        "width": str(WIDTH),
        "height": str(HEIGHT),
        "fps": str(FPS),
        "num_inference_steps": str(NUM_INFERENCE_STEPS),
        "flow_shift": str(FLOW_SHIFT),
        "seed": str(SEED),
        "extra_params": json.dumps(
            {
                "task": "ref2va",
                "duration": REF2VA_DURATION_SECONDS,
                "aspect_ratio": "auto",
                "audio_flow_shift": AUDIO_FLOW_SHIFT,
            }
        ),
    }

    with OmniServer(
        _model_name(REF2VA_MODEL_ENV_VAR, "Ref2VA"),
        _server_args(),
        env_dict=server_env,
        use_omni=True,
    ) as server:
        with input_video_path.open("rb") as video_file, input_audio_path.open("rb") as audio_file:
            response = requests.post(
                f"http://{server.host}:{server.port}/v1/videos/sync",
                data=request_data,
                files=[
                    ("input_references", ("input_video.mp4", video_file, "video/mp4")),
                    ("input_references", ("input_audio.mp3", audio_file, "audio/mpeg")),
                ],
                timeout=REQUEST_TIMEOUT_SECONDS,
            )

    response.raise_for_status()
    assert response.headers["content-type"].startswith("video/mp4")
    generated_path.write_bytes(response.content)
    _assert_official_video(
        generated_path,
        reference_path,
        frame_count=REF2VA_NUM_FRAMES,
        label="minimax_h3_ref2va_official_reference",
    )
