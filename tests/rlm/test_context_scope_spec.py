"""RunContext scope_spec field tests (PR A Wave 3)."""

from __future__ import annotations

from pathlib import Path

from backend.agents.rlm.context import RunContext
from backend.agents.schemas import ScopeSpec


class TestRunContextScopeSpec:
    def _ctx(self, **overrides) -> RunContext:
        defaults = dict(
            project_id="p1",
            project_dir=Path("/tmp"),
            runs_root=Path("/tmp/runs"),
            dashboard=None,
            cost_ledger=None,
            llm_client=None,
            provider="anthropic",
            model="claude-sonnet-4-6",
        )
        defaults.update(overrides)
        return RunContext(**defaults)

    def test_default_is_none(self):
        ctx = self._ctx()
        assert ctx.scope_spec is None

    def test_can_carry_scope_spec(self):
        spec = ScopeSpec(models=["a", "b"], seeds=[42])
        ctx = self._ctx(scope_spec=spec)
        assert ctx.scope_spec is spec
        assert ctx.scope_spec.is_multi_model is True

    def test_empty_scope_spec(self):
        ctx = self._ctx(scope_spec=ScopeSpec())
        assert ctx.scope_spec is not None
        assert ctx.scope_spec.models == []
