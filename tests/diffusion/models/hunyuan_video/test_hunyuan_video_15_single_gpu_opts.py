# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from torch import nn

from vllm_omni.diffusion.models.hunyuan_video.hunyuan_video_15_transformer import (
    HunyuanVideo15RotaryPosEmbed,
    HunyuanVideo15Transformer3DModel,
)
from vllm_omni.diffusion.models.hunyuan_video.pipeline_hunyuan_video_1_5 import HunyuanVideo15Pipeline

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_rope_cache_matches_uncached_rope_exactly() -> None:
    rope = HunyuanVideo15RotaryPosEmbed(patch_size=1, patch_size_t=1, rope_dim=[4, 4, 4])
    hidden_states = torch.zeros(1, 65, 2, 3, 4, dtype=torch.bfloat16)

    baseline_cos, baseline_sin = rope(hidden_states, use_cache=False)
    cached_cos, cached_sin = rope(hidden_states, use_cache=True)
    cached_cos_2, cached_sin_2 = rope(hidden_states, use_cache=True)

    assert cached_cos.dtype is torch.float32
    assert cached_sin.dtype is torch.float32
    assert torch.equal(cached_cos, baseline_cos)
    assert torch.equal(cached_sin, baseline_sin)
    assert cached_cos.data_ptr() == cached_cos_2.data_ptr()
    assert cached_sin.data_ptr() == cached_sin_2.data_ptr()


def test_t2v_latent_input_preallocation_matches_cat_baseline() -> None:
    pipeline = object.__new__(HunyuanVideo15Pipeline)
    latents = torch.randn(1, 3, 2, 4, 5)
    cond_latents = torch.zeros_like(latents)
    mask = torch.zeros(1, 1, 2, 4, 5)

    baseline = torch.cat([latents, cond_latents, mask], dim=1)
    actual = HunyuanVideo15Pipeline.prepare_t2v_latent_model_input(pipeline, latents)

    assert torch.equal(actual, baseline)

    next_latents = torch.randn_like(latents)
    updated = HunyuanVideo15Pipeline.update_t2v_latent_model_input(pipeline, actual, next_latents)
    next_baseline = torch.cat([next_latents, cond_latents, mask], dim=1)

    assert updated.data_ptr() == actual.data_ptr()
    assert torch.equal(updated, next_baseline)


def test_t2v_image_embeds_include_explicit_false_mask() -> None:
    pipeline = object.__new__(HunyuanVideo15Pipeline)
    pipeline.vision_num_semantic_tokens = 7
    pipeline.vision_states_dim = 5

    image_embeds, image_embeds_mask = HunyuanVideo15Pipeline.prepare_t2v_image_embeds(
        pipeline,
        batch_size=2,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )

    assert image_embeds.shape == (2, 7, 5)
    assert image_embeds_mask.shape == (2, 7)
    assert image_embeds.dtype is torch.bfloat16
    assert image_embeds_mask.dtype is torch.bfloat16
    assert not image_embeds.bool().any()
    assert not image_embeds_mask.bool().any()


class _CountingImageEmbedder(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.calls = 0

    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return torch.ones(image_embeds.shape[0], image_embeds.shape[1], self.output_dim, dtype=image_embeds.dtype)


def _make_image_context_model() -> HunyuanVideo15Transformer3DModel:
    model = object.__new__(HunyuanVideo15Transformer3DModel)
    nn.Module.__init__(model)
    model.inner_dim = 4
    model.image_embedder = _CountingImageEmbedder(model.inner_dim)
    model.cond_type_embed = nn.Embedding(3, model.inner_dim)
    with torch.no_grad():
        model.cond_type_embed.weight.copy_(torch.arange(12, dtype=torch.float32).view(3, 4))
    return model


def test_explicit_image_mask_bypasses_t2v_zero_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _make_image_context_model()
    image_embeds = torch.randn(1, 3, 5)
    image_embeds_mask = torch.tensor([[1, 0, 1]], dtype=torch.float32)
    encoder_attention_mask = torch.ones(1, 2)

    def _raise_if_called(*args, **kwargs):
        del args, kwargs
        raise AssertionError("torch.all should not run when image_embeds_mask is explicit")

    monkeypatch.setattr(torch, "all", _raise_if_called)

    _states, actual_mask = model.prepare_image_encoder_states(
        image_embeds,
        image_embeds_mask,
        encoder_attention_mask,
        image_embeds_are_zero=False,
    )

    assert torch.equal(actual_mask, image_embeds_mask)
    assert model.image_embedder.calls == 1


def test_zero_image_projection_fast_path_matches_fallback_baseline() -> None:
    model = _make_image_context_model()
    image_embeds = torch.zeros(2, 3, 5)
    encoder_attention_mask = torch.ones(2, 4)

    baseline_states, baseline_mask = model.prepare_image_encoder_states(
        image_embeds,
        image_embeds_mask=None,
        encoder_attention_mask=encoder_attention_mask,
        image_embeds_are_zero=False,
    )
    baseline_calls = model.image_embedder.calls

    fast_states, fast_mask = model.prepare_image_encoder_states(
        image_embeds,
        image_embeds_mask=torch.zeros(2, 3),
        encoder_attention_mask=encoder_attention_mask,
        image_embeds_are_zero=True,
    )

    assert baseline_calls == 1
    assert model.image_embedder.calls == baseline_calls
    assert torch.equal(fast_states, baseline_states)
    assert torch.equal(fast_mask, baseline_mask)


def test_scheduler_timestep_cache_matches_set_timesteps_baseline() -> None:
    pipeline = object.__new__(HunyuanVideo15Pipeline)
    pipeline.scheduler = FlowMatchEulerDiscreteScheduler(shift=5.0)
    pipeline._scheduler_timesteps_cache = {}
    device = torch.device("cpu")
    num_steps = 5

    baseline_scheduler = FlowMatchEulerDiscreteScheduler(shift=5.0)
    baseline_sigmas = np.linspace(1.0, 0.0, num_steps + 1)[:-1]
    baseline_scheduler.set_timesteps(sigmas=baseline_sigmas, device=device)

    timesteps = HunyuanVideo15Pipeline.get_scheduler_timesteps(pipeline, num_steps, device)

    assert torch.equal(timesteps, baseline_scheduler.timesteps)
    assert torch.equal(pipeline.scheduler.sigmas, baseline_scheduler.sigmas)

    pipeline.scheduler._step_index = 3
    cached_timesteps = HunyuanVideo15Pipeline.get_scheduler_timesteps(pipeline, num_steps, device)

    assert cached_timesteps.data_ptr() == timesteps.data_ptr()
    assert pipeline.scheduler.step_index is None
    assert torch.equal(cached_timesteps, baseline_scheduler.timesteps)
    assert len(pipeline._scheduler_timesteps_cache) == 1


class _NoopProgressBar:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback

    def update(self) -> None:
        return None


def test_t2v_forward_uses_exact_preallocated_inputs_and_zero_image_fast_path() -> None:
    pipeline = object.__new__(HunyuanVideo15Pipeline)
    nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    dtype = torch.float32
    pipeline.transformer = SimpleNamespace(
        transformer_blocks=[
            SimpleNamespace(norm1=SimpleNamespace(linear=SimpleNamespace(weight=torch.empty(1, dtype=dtype))))
        ]
    )
    pipeline.use_meanflow = False
    pipeline.vision_num_semantic_tokens = 3
    pipeline.vision_states_dim = 5
    pipeline._guidance_scale = None
    pipeline._num_timesteps = None
    pipeline._current_timestep = None
    pipeline.progress_bar = lambda total: _NoopProgressBar()
    timesteps = torch.tensor([10.0, 5.0], dtype=dtype)
    pipeline.get_scheduler_timesteps = lambda num_steps, device: timesteps

    prompt_embeds = torch.ones(1, 4, dtype=dtype)
    prompt_mask = torch.ones(1, 4, dtype=dtype)
    prompt_embeds_2 = torch.ones(1, 2, dtype=dtype)
    prompt_mask_2 = torch.ones(1, 2, dtype=dtype)
    pipeline.encode_prompt = lambda **kwargs: (
        prompt_embeds,
        prompt_mask,
        prompt_embeds_2,
        prompt_mask_2,
        None,
        None,
        None,
        None,
    )

    initial_latents = torch.randn(1, 2, 1, 2, 2, dtype=dtype)
    pipeline.prepare_latents = lambda **kwargs: initial_latents.clone()

    hidden_state_ptrs = []
    expected_cond = torch.zeros_like(initial_latents)
    expected_mask = torch.zeros(1, 1, 1, 2, 2, dtype=dtype)

    def _predict_noise_maybe_with_cfg(**kwargs):
        positive_kwargs = kwargs["positive_kwargs"]
        current_latents = initial_latents + len(hidden_state_ptrs)
        expected_hidden_states = torch.cat([current_latents, expected_cond, expected_mask], dim=1)

        assert torch.equal(positive_kwargs["hidden_states"], expected_hidden_states)
        assert positive_kwargs["image_embeds_are_zero"] is True
        assert torch.equal(positive_kwargs["image_embeds_mask"], torch.zeros(1, 3, dtype=dtype))
        hidden_state_ptrs.append(positive_kwargs["hidden_states"].data_ptr())
        return torch.ones_like(initial_latents)

    pipeline.predict_noise_maybe_with_cfg = _predict_noise_maybe_with_cfg
    pipeline.scheduler_step_maybe_with_cfg = lambda noise_pred, t, latents, do_true_cfg: latents + noise_pred

    req = SimpleNamespace(
        prompts=["prompt"],
        sampling_params=SimpleNamespace(
            height=None,
            width=None,
            num_frames=1,
            num_inference_steps=2,
            guidance_scale_provided=False,
            guidance_scale=None,
            generator=None,
            seed=None,
            latents=None,
            cfg_normalize=True,
        ),
    )

    output = pipeline.forward(req, output_type="latent", guidance_scale=1.0)

    expected_output = initial_latents.clone()
    expected_output = expected_output + torch.ones_like(initial_latents)
    expected_output = expected_output + torch.ones_like(initial_latents)
    assert torch.equal(output.output, expected_output)
    assert len(hidden_state_ptrs) == 2
    assert hidden_state_ptrs[0] == hidden_state_ptrs[1]
