"""Unit tests for the PAPER_HINTS table and lookup_paper_hint (PR A Wave 2)."""

from __future__ import annotations

import re

import pytest

from backend.agents.prompts.paper_hints import (
    PAPER_HINTS,
    _normalize_paper_id,
    lookup_paper_hint,
)
from backend.agents.schemas import PaperHint


SDAR_ID = "2605.15155"


class TestNormalizePaperId:
    def test_bare_id_passes_through(self):
        assert _normalize_paper_id("2605.15155") == "2605.15155"

    def test_strips_version_suffix(self):
        assert _normalize_paper_id("2605.15155v1") == "2605.15155"
        assert _normalize_paper_id("2605.15155v2") == "2605.15155"
        assert _normalize_paper_id("2605.15155V3") == "2605.15155"

    def test_strips_whitespace(self):
        assert _normalize_paper_id("  2605.15155  ") == "2605.15155"
        assert _normalize_paper_id("\t2605.15155v2\n") == "2605.15155"


class TestLookupPaperHint:
    def test_none_input(self):
        assert lookup_paper_hint(None) is None

    def test_empty_input(self):
        assert lookup_paper_hint("") is None
        assert lookup_paper_hint("   ") is None

    def test_unknown_id(self):
        assert lookup_paper_hint("9999.99999") is None
        assert lookup_paper_hint("not-a-paper") is None

    def test_known_id_returns_paper_hint(self):
        h = lookup_paper_hint(SDAR_ID)
        assert isinstance(h, PaperHint)

    def test_version_suffix_matches_bare(self):
        bare = lookup_paper_hint(SDAR_ID)
        v1 = lookup_paper_hint(f"{SDAR_ID}v1")
        v3 = lookup_paper_hint(f"{SDAR_ID}V3")
        assert bare is not None
        assert v1 is bare or v1 == bare  # same dict object or equal
        assert v3 is bare or v3 == bare


class TestSdarHintStructure:
    """SDAR's hint must be complete enough to drive a real reproduction."""

    @pytest.fixture
    def sdar(self) -> PaperHint:
        h = lookup_paper_hint(SDAR_ID)
        assert h is not None
        return h

    def test_guidance_is_substantive(self, sdar):
        assert len(sdar.guidance) >= 100
        # The key invariants must appear in the prose so the agent's prompt is informed.
        assert "sigmoid" in sdar.guidance.lower()
        assert "stop_gradient" in sdar.guidance.lower() or "stop gradient" in sdar.guidance.lower()
        assert "lambda" in sdar.guidance.lower()
        assert "beta" in sdar.guidance.lower()

    def test_default_scope_has_three_models(self, sdar):
        assert sdar.default_scope is not None
        assert len(sdar.default_scope.models) == 3
        assert sdar.default_scope.is_multi_model is True

    def test_default_scope_has_three_envs(self, sdar):
        assert sdar.default_scope is not None
        assert len(sdar.default_scope.datasets) == 3
        env_names = sdar.default_scope.dataset_ids()
        assert "ALFWorld" in env_names
        assert "WebShop" in env_names
        assert "Search-QA" in env_names

    def test_default_scope_uses_single_paper_seed(self, sdar):
        # The official SDAR scripts train with a single env.seed=0; the faithful
        # default is one seed, not the harness's generic [42, 43, 44].
        assert sdar.default_scope is not None
        assert sdar.default_scope.seeds == [0]

    def test_invariants_cover_key_paper_claims(self, sdar):
        names = {inv.name for inv in sdar.invariants}
        # Each name corresponds to a distinct rubric leaf this hint guards against.
        for required in (
            "sigmoid_gate_on_advantage",
            "stop_gradient_on_gate",
            "lambda_self_distill_weight_0p1",
            "beta_gate_sharpness_10",
            "real_qwen_weights_not_surrogate",
        ):
            assert required in names, f"SDAR invariant {required!r} missing — names: {names}"


class TestInvariantPatternsMatchRealCode:
    """Smoke-test each SDAR invariant against snippets the agent might write.

    These are not the rubric grader — they confirm the regex actually
    matches code shaped like the real paper, not just the exact prose in
    the rationale.
    """

    @pytest.fixture
    def sdar(self) -> PaperHint:
        h = lookup_paper_hint(SDAR_ID)
        assert h is not None
        return h

    def _matches_any(self, patterns: list[str], code: str) -> bool:
        return any(re.search(p, code) for p in patterns)

    def test_sigmoid_gate_matches_torch_sigmoid(self, sdar):
        inv = next(i for i in sdar.invariants if i.name == "sigmoid_gate_on_advantage")
        assert self._matches_any(inv.must_match, "g_t = torch.sigmoid(beta * delta_t)")
        assert self._matches_any(inv.must_match, "gate = torch.sigmoid(self.beta * advantage_diff)")

    def test_stop_gradient_matches_detach_or_no_grad(self, sdar):
        inv = next(i for i in sdar.invariants if i.name == "stop_gradient_on_gate")
        assert self._matches_any(inv.must_match, "gate = torch.sigmoid(beta * delta).detach()")
        assert self._matches_any(inv.must_match, "with torch.no_grad():\n    gate = ...")

    def test_lambda_matches_common_assignments(self, sdar):
        inv = next(i for i in sdar.invariants if i.name == "lambda_self_distill_weight_0p1")
        assert self._matches_any(inv.must_match, "lambda = 0.1")
        assert self._matches_any(inv.must_match, "opsd_weight: 0.1")
        assert self._matches_any(inv.must_match, "self_distill_weight = 0.1")

    def test_beta_matches_assignment(self, sdar):
        inv = next(i for i in sdar.invariants if i.name == "beta_gate_sharpness_10")
        assert self._matches_any(inv.must_match, "beta = 10")
        assert self._matches_any(inv.must_match, "beta: 10.0")

    def test_real_qwen_matches_from_pretrained(self, sdar):
        inv = next(i for i in sdar.invariants if i.name == "real_qwen_weights_not_surrogate")
        assert self._matches_any(
            inv.must_match,
            'AutoModel.from_pretrained("Qwen/Qwen3-1.7B-Instruct")',
        )

    def test_real_qwen_violates_on_surrogate(self, sdar):
        inv = next(i for i in sdar.invariants if i.name == "real_qwen_weights_not_surrogate")
        assert self._matches_any(inv.must_not_match, "class TinyLM(nn.Module):")
        assert self._matches_any(inv.must_not_match, "# surrogate model — placeholder for Qwen")


class TestPaperHintsTableShape:
    def test_table_is_dict(self):
        assert isinstance(PAPER_HINTS, dict)

    def test_all_values_are_paper_hints(self):
        for key, val in PAPER_HINTS.items():
            assert isinstance(key, str), f"non-string key: {key!r}"
            assert isinstance(val, PaperHint), f"{key} → {type(val).__name__}, want PaperHint"

    def test_sdar_present(self):
        assert SDAR_ID in PAPER_HINTS


OMNIZIP_ID = "2511.14582"


class TestOmniZipHint:
    """OmniZip (2511.14582) — training-free audio-guided token compression."""

    @pytest.fixture()
    def omnizip(self) -> PaperHint:
        hint = lookup_paper_hint(OMNIZIP_ID)
        assert hint is not None
        return hint

    def test_lookup_with_version_suffix(self):
        assert lookup_paper_hint("2511.14582v1") is lookup_paper_hint(OMNIZIP_ID)

    def test_guidance_is_substantive(self, omnizip):
        assert len(omnizip.guidance) > 500
        for needle in (
            "Qwen2.5-Omni",
            "device_map",
            "scope.gaps",
            "cells.json",
            "rho_max=0.75",
        ):
            assert needle in omnizip.guidance, f"guidance missing {needle!r}"

    def test_default_scope_models_and_datasets(self, omnizip):
        assert omnizip.default_scope is not None
        assert omnizip.default_scope.models == [
            "Qwen2.5-Omni-7B",
            "Qwen2.5-Omni-3B",
        ]
        names = [d.name for d in omnizip.default_scope.datasets]
        assert names == ["WorldSense", "ShortVid-Bench"]

    def test_blocked_resources_blacklist_official_repo(self, omnizip):
        assert any("OmniZip" in r for r in omnizip.blocked_resources)

    def test_sharded_hosting_invariant_matches_real_code(self, omnizip):
        inv = {i.name: i for i in omnizip.invariants}["sharded_multi_gpu_hosting"]
        pattern = re.compile(inv.must_match[0])
        assert pattern.search(
            'model = AutoModel.from_pretrained(mid, device_map="auto")'
        )
        assert pattern.search("model, opt = accelerator.prepare(model, opt)")
        assert not pattern.search("model = AutoModel.from_pretrained(mid).cuda()")

    def test_real_weights_invariant_matches_hf_id(self, omnizip):
        inv = {i.name: i for i in omnizip.invariants}[
            "real_qwen_omni_weights_not_surrogate"
        ]
        assert re.search(inv.must_match[0], 'MODEL_ID = "Qwen/Qwen2.5-Omni-7B"')
