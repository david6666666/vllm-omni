# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from vllm_omni.diffusion.forward_context import (
    get_forward_context,
    is_forward_context_available,
)


def get_diffusion_quant_config():
    if not is_forward_context_available():
        return None
    od_config = get_forward_context().omni_diffusion_config
    if od_config is None:
        return None
    return getattr(od_config, "quant_config", None)
