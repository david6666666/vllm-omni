# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import torch


class QwenImage21KVCache(list[dict[str, dict[str, torch.Tensor]]]):
    """Request-owned prefix cache with weak-reference support for geometry reuse."""
