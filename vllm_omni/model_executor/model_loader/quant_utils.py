# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from typing import Any

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.gguf import GGUFConfig
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.transformers_utils.config import get_hf_file_to_dict

logger = init_logger(__name__)


def _is_fp8_quant_config(config: dict[str, Any] | None) -> bool:
    if not isinstance(config, dict):
        return False
    quant_method = config.get("quant_method")
    return isinstance(quant_method, str) and "fp8" in quant_method.lower()


def _load_quant_config_dict(
    quantization_config_file: str | None,
    quantization_config_dict_json: str | None,
) -> dict[str, Any] | None:
    if quantization_config_dict_json:
        return json.loads(quantization_config_dict_json)
    if quantization_config_file:
        with open(quantization_config_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _maybe_get_hf_quant_config(model: str | None) -> dict[str, Any] | None:
    if not model:
        return None
    for config_path in ("config.json", "transformer/config.json"):
        try:
            cfg = get_hf_file_to_dict(config_path, model)
        except (OSError, ValueError):
            cfg = None
        if cfg and isinstance(cfg, dict):
            qcfg = cfg.get("quantization_config")
            if isinstance(qcfg, dict):
                return qcfg
    return None


def infer_diffusion_quantization_method(
    *,
    quantization: str | None,
    quantization_config_file: str | None,
    quantization_config_dict_json: str | None,
    model: str | None,
) -> str | None:
    if quantization is not None:
        quantization = quantization.lower()
        if quantization != "auto":
            return quantization

    cfg = _load_quant_config_dict(
        quantization_config_file,
        quantization_config_dict_json,
    )
    if cfg is None:
        cfg = _maybe_get_hf_quant_config(model)
    if _is_fp8_quant_config(cfg):
        return "fp8"
    return None


def resolve_diffusion_quant_config(
    *,
    quantization: str | None,
    quantization_config_file: str | None,
    quantization_config_dict_json: str | None,
    model: str | None,
    load_format: str | None,
) -> object | None:
    quantization = infer_diffusion_quantization_method(
        quantization=quantization,
        quantization_config_file=quantization_config_file,
        quantization_config_dict_json=quantization_config_dict_json,
        model=model,
    )
    if quantization is None:
        return None

    if quantization == "gguf":
        if load_format is not None and load_format != "gguf":
            raise ValueError(
                f"GGUF requires load_format='gguf', got {load_format!r}"
            )
        return GGUFConfig()

    if quantization == "fp8":
        cfg = _load_quant_config_dict(
            quantization_config_file,
            quantization_config_dict_json,
        )
        if cfg is None:
            cfg = _maybe_get_hf_quant_config(model)
        if cfg is not None:
            return Fp8Config.from_config(cfg)
        return Fp8Config()

    raise ValueError(f"Unsupported diffusion quantization method: {quantization}")
