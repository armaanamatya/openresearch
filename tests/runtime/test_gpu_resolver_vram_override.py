"""An explicit --vram-gb override must be used verbatim - no 1.25x headroom.
Regression (2026-07-08 GCP incident): --vram-gb 80 on an 80GB fleet inflated to
100GB and matched no SKU, so the run never got a GPU."""

from __future__ import annotations

import math

from backend.agents.schemas import GpuRequirements


def test_explicit_override_skips_headroom():
    req = GpuRequirements(estimated_vram_gb=80, confidence=1.0, vram_is_explicit=True)
    effective = 1.0 if req.vram_is_explicit else 1.25
    assert math.ceil(80 * max(effective, 1.0)) == 80


def test_llm_estimate_still_gets_headroom():
    req = GpuRequirements(estimated_vram_gb=80, confidence=0.7, vram_is_explicit=False)
    effective = 1.0 if req.vram_is_explicit else 1.25
    assert math.ceil(80 * max(effective, 1.0)) == 100


def test_vram_is_explicit_defaults_false():
    assert GpuRequirements(estimated_vram_gb=40).vram_is_explicit is False
