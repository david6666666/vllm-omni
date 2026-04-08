import importlib
import sys
import types
from contextlib import nullcontext

import torch


def _load_npu_platform_module(monkeypatch):
    fake_vllm_ascend = types.ModuleType("vllm_ascend")
    fake_vllm_ascend_platform = types.ModuleType("vllm_ascend.platform")

    class NPUPlatform:
        pass

    fake_vllm_ascend_platform.NPUPlatform = NPUPlatform
    monkeypatch.setitem(sys.modules, "vllm_ascend", fake_vllm_ascend)
    monkeypatch.setitem(sys.modules, "vllm_ascend.platform", fake_vllm_ascend_platform)
    sys.modules.pop("vllm_omni.platforms.npu.platform", None)
    sys.modules.pop("vllm_omni.platforms.npu", None)
    return importlib.import_module("vllm_omni.platforms.npu.platform")


def test_npu_autocast_uses_base_context_before_fallback(monkeypatch, mocker):
    npu_platform_module = _load_npu_platform_module(monkeypatch)

    sentinel = object()
    base_autocast = mocker.patch.object(
        npu_platform_module.OmniPlatform,
        "create_autocast_context",
        return_value=sentinel,
    )
    fake_npu = types.SimpleNamespace(
        amp=types.SimpleNamespace(autocast=mocker.Mock(return_value=object())),
    )
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)

    ctx = npu_platform_module.NPUOmniPlatform.create_autocast_context(
        device_type="npu",
        dtype=torch.bfloat16,
        enabled=True,
    )

    assert ctx is sentinel
    base_autocast.assert_called_once_with(
        device_type="npu",
        dtype=torch.bfloat16,
        enabled=True,
    )
    fake_npu.amp.autocast.assert_not_called()


def test_npu_autocast_uses_npu_amp_when_base_context_unavailable(monkeypatch, mocker):
    npu_platform_module = _load_npu_platform_module(monkeypatch)

    fallback = object()
    mocker.patch.object(
        npu_platform_module.OmniPlatform,
        "create_autocast_context",
        return_value=nullcontext(),
    )
    fake_npu = types.SimpleNamespace(
        amp=types.SimpleNamespace(autocast=mocker.Mock(return_value=fallback)),
    )
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)

    ctx = npu_platform_module.NPUOmniPlatform.create_autocast_context(
        device_type="npu",
        dtype=torch.bfloat16,
        enabled=True,
    )

    assert ctx is fallback
    fake_npu.amp.autocast.assert_called_once_with(dtype=torch.bfloat16)
