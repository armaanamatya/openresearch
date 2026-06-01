"""Build GEPA training examples from understand_section / extract_hyperparameters outputs."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PaperExample:
    """One training example for plan_reproduction GEPA optimization."""
    method_spec: dict
    env_spec: dict
    expected_metric_ids: list[str]


class PaperExamplesBuilder:
    """Construct a small set of PaperExample instances from run-context outputs.

    These serve as the trainset for the first GEPA call on a paper (cold start).
    Subsequent calls use the real plan_reproduction outputs appended to
    ctx.gepa_example_buffer["plan_reproduction"].
    """

    def __init__(
        self,
        *,
        understand_output: dict | None,
        extract_output: dict | None,
        env_spec: dict | None,
        expected_metric_ids: list[str],
    ) -> None:
        self._understand = understand_output or {}
        self._extract = extract_output or {}
        self._env_spec = env_spec or {}
        self._expected = expected_metric_ids

    def _make_method_spec(self) -> dict:
        return {
            "datasets": self._understand.get("datasets", []),
            "metrics": self._understand.get("metrics", []),
            "training_recipe": self._understand.get("training_recipe", {}),
            "hardware_clues": self._understand.get("hardware_clues", []),
            "ambiguities": self._understand.get("ambiguities", []),
            "hyperparameters": self._extract,
        }

    def build(self, n: int = 3) -> list[PaperExample]:
        """Return up to n PaperExample instances.

        Returns n copies of the same base example (the GEPA optimizer uses varied
        candidate prompts, not varied inputs, for the first call). On subsequent
        calls the real outputs in gepa_example_buffer provide genuine variation.
        """
        method_spec = self._make_method_spec()
        base = PaperExample(
            method_spec=method_spec,
            env_spec=self._env_spec,
            expected_metric_ids=list(self._expected),
        )
        return [base] * max(1, min(n, 3))
