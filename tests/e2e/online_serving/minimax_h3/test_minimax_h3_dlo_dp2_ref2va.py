# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""MiniMax-H3 Ref2VA DLO + DP2 L3 online-serving case."""

from __future__ import annotations

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServer, OpenAIClientHandler

from ._common import assert_h3_video, dlo_params, run_ref2va

pytestmark = [pytest.mark.core_model, pytest.mark.advanced_model, pytest.mark.diffusion, pytest.mark.slow]
H100_TWO_CARD_MARKS = hardware_marks(res={"cuda": "H100"}, num_cards=2)


@pytest.mark.parametrize(
    "omni_server",
    [pytest.param(dlo_params("ref2va"), id="minimax_h3_dlo_dp2_ref2va", marks=H100_TWO_CARD_MARKS)],
    indirect=True,
)
def test_minimax_h3_dlo_dp2_ref2va(omni_server: OmniServer, openai_client: OpenAIClientHandler) -> None:
    """Validate Ref2VA DLO/DP2 without coupling two different references.

    With the 6556 pure-DP request workaround reverted, WORLD-group broadcasts
    intentionally see the same request on every rank. A single Ref2VA request
    keeps that collective contract deterministic while still exercising the
    two-card DLO/DP2 path; the FL2VA case above retains the concurrent wave.
    """
    video = run_ref2va(openai_client, 2101)
    assert_h3_video(video, width=1344, height=768)
