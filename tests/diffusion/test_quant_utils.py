# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.model_executor.layers.quantization.fp8 import Fp8Config

from vllm_omni.model_executor.model_loader import quant_utils


def test_infer_diffusion_quantization_method_auto_detect_fp8(monkeypatch):
    fp8_cfg = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
    }
    monkeypatch.setattr(
        quant_utils,
        "_maybe_get_hf_quant_config",
        lambda _model: fp8_cfg,
    )

    method = quant_utils.infer_diffusion_quantization_method(
        quantization=None,
        quantization_config_file=None,
        quantization_config_dict_json=None,
        model="dummy-model",
        quantized_weights=None,
    )

    assert method == "fp8"


def test_infer_diffusion_quantization_method_no_quant_config(monkeypatch):
    monkeypatch.setattr(
        quant_utils,
        "_maybe_get_hf_quant_config",
        lambda _model: None,
    )

    method = quant_utils.infer_diffusion_quantization_method(
        quantization=None,
        quantization_config_file=None,
        quantization_config_dict_json=None,
        model="dummy-model",
        quantized_weights=None,
    )

    assert method is None


def test_resolve_diffusion_quant_config_native_fp8(monkeypatch):
    fp8_cfg = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
    }
    monkeypatch.setattr(
        quant_utils,
        "_maybe_get_hf_quant_config",
        lambda _model: fp8_cfg,
    )

    quant_config = quant_utils.resolve_diffusion_quant_config(
        quantization=None,
        quantization_config_file=None,
        quantization_config_dict_json=None,
        model="dummy-model",
        load_format="auto",
        quantized_weights=None,
    )

    assert isinstance(quant_config, Fp8Config)
    assert quant_config.is_checkpoint_fp8_serialized is True


def test_resolve_diffusion_quant_config_online_fp8_fallback(monkeypatch):
    monkeypatch.setattr(
        quant_utils,
        "_maybe_get_hf_quant_config",
        lambda _model: None,
    )

    quant_config = quant_utils.resolve_diffusion_quant_config(
        quantization="fp8",
        quantization_config_file=None,
        quantization_config_dict_json=None,
        model="dummy-model",
        load_format="auto",
        quantized_weights=None,
    )

    assert isinstance(quant_config, Fp8Config)
    assert quant_config.is_checkpoint_fp8_serialized is False


def test_infer_diffusion_quantization_method_prefers_quantized_weights(monkeypatch):
    fp8_cfg = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
    }

    def _mock_get_hf_quant_config(model):
        if model == "quantized-repo":
            return fp8_cfg
        return None

    monkeypatch.setattr(
        quant_utils,
        "_maybe_get_hf_quant_config",
        _mock_get_hf_quant_config,
    )

    method = quant_utils.infer_diffusion_quantization_method(
        quantization=None,
        quantization_config_file=None,
        quantization_config_dict_json=None,
        model="base-model",
        quantized_weights="quantized-repo",
    )

    assert method == "fp8"


def test_resolve_diffusion_quant_config_prefers_quantized_weights(monkeypatch):
    fp8_cfg = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
    }

    def _mock_get_hf_quant_config(model):
        if model == "quantized-repo":
            return fp8_cfg
        return None

    monkeypatch.setattr(
        quant_utils,
        "_maybe_get_hf_quant_config",
        _mock_get_hf_quant_config,
    )

    quant_config = quant_utils.resolve_diffusion_quant_config(
        quantization=None,
        quantization_config_file=None,
        quantization_config_dict_json=None,
        model="base-model",
        load_format="auto",
        quantized_weights="quantized-repo",
    )

    assert isinstance(quant_config, Fp8Config)
    assert quant_config.is_checkpoint_fp8_serialized is True
