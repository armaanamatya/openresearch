"""The shared campaign money record.

``CampaignSpend`` used to exist twice: a full-featured copy in
``campaign_policy`` (add / to_dict / strict from_mapping) and a minimal
duplicate in ``attempt_assessment`` (created only because ``campaign_policy``
imports ``attempt_assessment``, so the reverse import would have cycled).
Both modules now import the single canonical definition from here instead,
so the two copies cannot drift apart again.

Tiny stdlib-only leaf: no imports beyond the standard library, and nothing
in this module imports any other ``backend.*`` module.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CampaignSpend:
    """Cumulative campaign spend so far, one field per money/wall meter."""

    llm_usd: float = 0.0
    gpu_usd: float = 0.0
    gpu_hours: float = 0.0
    wall_s: float = 0.0

    def add(self, other: CampaignSpend) -> CampaignSpend:
        return CampaignSpend(
            llm_usd=self.llm_usd + other.llm_usd,
            gpu_usd=self.gpu_usd + other.gpu_usd,
            gpu_hours=self.gpu_hours + other.gpu_hours,
            wall_s=self.wall_s + other.wall_s,
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "llm_usd": self.llm_usd,
            "gpu_usd": self.gpu_usd,
            "gpu_hours": self.gpu_hours,
            "wall_s": self.wall_s,
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> CampaignSpend:
        """Reconstruct from a persisted mapping. Deliberately STRICT (raises
        on a missing key, never defaults to 0.0) — fail-open here would let
        ``derive_envelope`` over-allocate off understated spend. Callers that
        want a genuine zero start should use ``CampaignSpend()`` directly."""
        return cls(
            llm_usd=float(mapping["llm_usd"]),
            gpu_usd=float(mapping["gpu_usd"]),
            gpu_hours=float(mapping["gpu_hours"]),
            wall_s=float(mapping["wall_s"]),
        )
