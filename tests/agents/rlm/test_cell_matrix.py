"""Tests for cell_matrix — pure cell-result → harness-metrics aggregation.

The risk-H core of the 2026-05-31 OOM/GPU remediation.  Every test is pure: no
GPUs, no torch, no real network (the dataset probe is always injected).  The
suite verifies:

  * capacity_gate drops cells over a single-GPU budget, keeps fitting siblings,
    and never blocks on unknown capacity (per_gpu_vram_gb <= 0).
  * dataset_url_preflight drops ONLY confirmed-dead (probe False) envs, is
    fail-soft on None/transient, and probes each distinct url exactly once.
  * aggregate_cell_metrics produces the exact nested per_model[model][env][base]
    shape, the right top-level status transitions, and a populated scope.
  * The aggregator's output is STRUCTURALLY identical to the real on-disk SDAR
    metrics.json sample the scorer already consumes (byte-compatible nesting).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.agents.rlm.cell_matrix import (
    DEFAULT_HEADROOM,
    aggregate_cell_metrics,
    audit_aggregation_completeness,
    capacity_gate,
    dataset_url_preflight,
    default_dataset_probe,
)

# Vendored copy of a verified real-world leaf-shape sample whose nesting the leaf
# scorer + postflight guards consume. Vendored (not read live from runs/) so the
# test is stable when a live run mutates / preserves its output dirs.
REAL_SAMPLE = Path(__file__).resolve().parent / "__fixtures__" / "sample_leaf_metrics.json"


# ---------------------------------------------------------------------------
# Cell fixtures
# ---------------------------------------------------------------------------

def _cell(
    model_key: str,
    env: str,
    baseline: str,
    *,
    seed: int = 42,
    est_vram_gb: float | None = None,
    dataset_url: str | None = None,
    model_id: str | None = None,
) -> dict:
    """Build a cell dict mirroring code/cells.json entries."""
    cell: dict = {
        "id": f"{model_key}__{baseline}__{env}__s{seed}",
        "model_key": model_key,
        "baseline": baseline,
        "env": env,
        "seed": seed,
    }
    if model_id is not None:
        cell["model_id"] = model_id
    if est_vram_gb is not None:
        cell["est_vram_gb"] = est_vram_gb
    if dataset_url is not None:
        cell["dataset_url"] = dataset_url
    return cell


# ===========================================================================
# capacity_gate
# ===========================================================================

class TestCapacityGate:
    def test_24gb_drops_7b_keeps_1_7b(self):
        """A 24GB card (23.68 usable): 7B (est 28) is dropped, 1.7B (est 14) kept."""
        cells = [
            _cell("qwen3_1_7b", "search_qa", "sdar", est_vram_gb=14.0),
            _cell("qwen2_5_7b", "search_qa", "sdar", est_vram_gb=28.0),
        ]
        kept, gaps, skipped = capacity_gate(cells, per_gpu_vram_gb=23.68)

        kept_models = {c["model_key"] for c in kept}
        assert kept_models == {"qwen3_1_7b"}
        assert skipped == ["qwen2_5_7b"]
        assert len(gaps) == 1
        assert gaps[0]["item"] == "qwen2_5_7b"
        assert gaps[0]["kind"] == "capacity"
        assert "per-GPU budget" in gaps[0]["reason"]

    def test_1_7b_kept_even_with_headroom(self):
        """14GB est × 1.25 headroom = 17.5GB ≤ 23.68 → 1.7B stays."""
        cells = [_cell("qwen3_1_7b", "search_qa", "sdar", est_vram_gb=14.0)]
        kept, gaps, skipped = capacity_gate(cells, per_gpu_vram_gb=23.68)
        assert len(kept) == 1
        assert gaps == []
        assert skipped == []

    def test_headroom_is_applied(self):
        """est 19 × 1.25 = 23.75 > 23.68 → dropped purely because of headroom."""
        cells = [_cell("qwen2_5_3b", "search_qa", "sdar", est_vram_gb=19.0)]
        kept, gaps, skipped = capacity_gate(cells, per_gpu_vram_gb=23.68)
        assert kept == []
        assert skipped == ["qwen2_5_3b"]
        # Without headroom (×1.0) the same cell would fit.
        kept2, _, skipped2 = capacity_gate(cells, per_gpu_vram_gb=23.68, headroom=1.0)
        assert len(kept2) == 1
        assert skipped2 == []

    def test_80gb_keeps_the_7b(self):
        """An 80GB card admits the 7B cell (28 × 1.25 = 35 ≤ 80)."""
        cells = [
            _cell("qwen3_1_7b", "search_qa", "sdar", est_vram_gb=14.0),
            _cell("qwen2_5_7b", "search_qa", "sdar", est_vram_gb=28.0),
        ]
        kept, gaps, skipped = capacity_gate(cells, per_gpu_vram_gb=80.0)
        assert len(kept) == 2
        assert gaps == []
        assert skipped == []

    def test_zero_budget_keeps_everything(self):
        """per_gpu_vram_gb <= 0 means unknown capacity → never block."""
        cells = [
            _cell("qwen3_1_7b", "search_qa", "sdar", est_vram_gb=14.0),
            _cell("qwen2_5_7b", "search_qa", "sdar", est_vram_gb=28.0),
            _cell("huge", "search_qa", "sdar", est_vram_gb=999.0),
        ]
        kept, gaps, skipped = capacity_gate(cells, per_gpu_vram_gb=0)
        assert len(kept) == 3
        assert gaps == []
        assert skipped == []

    def test_negative_budget_keeps_everything(self):
        cells = [_cell("qwen2_5_7b", "search_qa", "sdar", est_vram_gb=28.0)]
        kept, gaps, skipped = capacity_gate(cells, per_gpu_vram_gb=-1.0)
        assert len(kept) == 1
        assert skipped == []

    def test_model_with_one_fitting_one_too_big_keeps_fitting(self):
        """A model is dropped ONLY if ALL its cells exceed budget.

        One env (search_qa, est 14) fits; another (search_qa would, but a giant
        env est 40) does not.  The model keeps the fitting env and is NOT in
        models_skipped (gate is per-cell).
        """
        cells = [
            _cell("qwen3_1_7b", "search_qa", "sdar", est_vram_gb=14.0),
            _cell("qwen3_1_7b", "bigenv", "sdar", est_vram_gb=40.0),
        ]
        kept, gaps, skipped = capacity_gate(cells, per_gpu_vram_gb=23.68)

        kept_envs = {c["env"] for c in kept}
        assert kept_envs == {"search_qa"}        # fitting env survives
        assert skipped == []                      # model NOT fully dropped
        assert gaps == []                         # no model-level gap emitted

    def test_missing_est_vram_is_kept(self):
        """A cell with no est_vram_gb has unknown footprint → keep it."""
        cells = [_cell("mystery", "search_qa", "sdar")]  # no est_vram_gb
        kept, gaps, skipped = capacity_gate(cells, per_gpu_vram_gb=10.0)
        assert len(kept) == 1
        assert skipped == []
        assert gaps == []

    def test_empty_cells(self):
        kept, gaps, skipped = capacity_gate([], per_gpu_vram_gb=24.0)
        assert kept == []
        assert gaps == []
        assert skipped == []

    def test_default_headroom_constant(self):
        assert DEFAULT_HEADROOM == 1.25


# ===========================================================================
# dataset_url_preflight
# ===========================================================================

class TestDatasetUrlPreflight:
    def test_confirmed_dead_drops_only_that_env(self):
        """probe(webshop_url)=False drops webshop cells; search_qa survives."""
        webshop_url = "https://example.com/webshop.json"
        searchqa_url = "https://example.com/searchqa.json"
        cells = [
            _cell("qwen3_1_7b", "search_qa", "sdar", dataset_url=searchqa_url),
            _cell("qwen3_1_7b", "webshop", "sdar", dataset_url=webshop_url),
            _cell("qwen2_5_3b", "webshop", "grpo", dataset_url=webshop_url),
        ]

        def fake_probe(url: str) -> bool | None:
            return False if url == webshop_url else True

        kept, gaps, skipped_envs = dataset_url_preflight(cells, probe=fake_probe)

        kept_envs = {c["env"] for c in kept}
        assert kept_envs == {"search_qa"}
        assert skipped_envs == ["webshop"]
        assert len(gaps) == 1
        assert gaps[0]["item"] == "webshop"
        assert gaps[0]["kind"] == "dataset_unavailable"
        assert webshop_url in gaps[0]["reason"]
        assert "404" in gaps[0]["reason"]

    def test_transient_none_keeps_everything(self):
        """probe returning None (transient/unknown) must NEVER drop a live env."""
        cells = [
            _cell("qwen3_1_7b", "search_qa", "sdar", dataset_url="https://x/a"),
            _cell("qwen3_1_7b", "webshop", "sdar", dataset_url="https://x/b"),
        ]

        def fake_probe(url: str) -> bool | None:
            return None  # transient — unknown

        kept, gaps, skipped_envs = dataset_url_preflight(cells, probe=fake_probe)
        assert len(kept) == 2
        assert gaps == []
        assert skipped_envs == []

    def test_distinct_urls_probed_once(self):
        """Each distinct url is probed exactly once even across many cells."""
        url_a = "https://x/a"
        url_b = "https://x/b"
        cells = [
            _cell("m1", "search_qa", "sdar", dataset_url=url_a),
            _cell("m2", "search_qa", "grpo", dataset_url=url_a),
            _cell("m3", "search_qa", "opsd", dataset_url=url_a),
            _cell("m1", "webshop", "sdar", dataset_url=url_b),
            _cell("m2", "webshop", "grpo", dataset_url=url_b),
        ]
        calls: list[str] = []

        def counting_probe(url: str) -> bool | None:
            calls.append(url)
            return True

        kept, _, _ = dataset_url_preflight(cells, probe=counting_probe)
        assert len(kept) == 5
        # 5 cells, 2 distinct urls → exactly 2 probe calls.
        assert sorted(calls) == [url_a, url_b]
        assert len(calls) == 2

    def test_cells_without_dataset_url_kept(self):
        cells = [
            _cell("m1", "search_qa", "sdar"),  # no dataset_url
            _cell("m2", "webshop", "sdar", dataset_url="https://x/dead"),
        ]

        def fake_probe(url: str) -> bool | None:
            return False  # everything probed is dead

        kept, gaps, skipped_envs = dataset_url_preflight(cells, probe=fake_probe)
        kept_envs = {c["env"] for c in kept}
        assert kept_envs == {"search_qa"}      # url-less cell survives
        assert skipped_envs == ["webshop"]

    def test_throwing_probe_is_failsoft(self):
        """A probe that raises is treated as unknown → keep the cell."""
        cells = [_cell("m1", "search_qa", "sdar", dataset_url="https://x/a")]

        def boom(url: str) -> bool | None:
            raise RuntimeError("network exploded")

        kept, gaps, skipped_envs = dataset_url_preflight(cells, probe=boom)
        assert len(kept) == 1
        assert gaps == []
        assert skipped_envs == []

    def test_mixed_dead_and_live_for_same_env_across_urls(self):
        """Two webshop cells on different urls: only the dead-url cell drops."""
        live = "https://x/live"
        dead = "https://x/dead"
        cells = [
            _cell("m1", "webshop", "sdar", dataset_url=live),
            _cell("m2", "webshop", "grpo", dataset_url=dead),
        ]

        def fake_probe(url: str) -> bool | None:
            return url != dead

        kept, gaps, skipped_envs = dataset_url_preflight(cells, probe=fake_probe)
        kept_ids = {c["id"] for c in kept}
        assert kept_ids == {"m1__sdar__webshop__s42"}
        # webshop is still reported skipped because a dead-url cell dropped.
        assert skipped_envs == ["webshop"]
        assert len(gaps) == 1

    def test_empty_cells(self):
        kept, gaps, skipped_envs = dataset_url_preflight([], probe=lambda u: True)
        assert kept == []
        assert gaps == []
        assert skipped_envs == []

    def test_default_probe_is_callable_without_network(self):
        """default_dataset_probe returns None (fail-soft) on an unroutable host.

        No assertion on a live network — only that an unreachable endpoint yields
        the fail-soft None verdict and never raises.
        """
        result = default_dataset_probe(
            "http://127.0.0.1:0/definitely-not-listening", timeout_s=0.05
        )
        assert result is None
        # An empty/None url is also None.
        assert default_dataset_probe("") is None

    def test_huggingface_urls_are_never_confirmed_dead(self):
        """HF/Kaggle dataset URLs resolve via a client library, not a raw GET — a
        HEAD 404 on the page is not authoritative, so the probe must return None
        (keep) WITHOUT any network call (the 2026-05-31 nq_open false-drop)."""
        for url in (
            "https://huggingface.co/datasets/nq_open",
            "https://hf.co/datasets/google-research-datasets/nq_open",
            "https://www.kaggle.com/datasets/whatever",
        ):
            assert default_dataset_probe(url, timeout_s=0.01) is None, url

    def test_huggingface_cell_survives_preflight(self):
        """A cell whose dataset_url is a HF page must NOT be dropped to a gap."""
        cells = [{"id": "c1", "env": "search_qa", "model_key": "q", "baseline": "sdar",
                  "dataset_url": "https://huggingface.co/datasets/nq_open"}]
        kept, gaps, skipped = dataset_url_preflight(cells)  # default probe, no network
        assert len(kept) == 1 and not gaps and not skipped


# ===========================================================================
# aggregate_cell_metrics
# ===========================================================================

class TestAggregateCellMetrics:
    def _matrix_2x2(self) -> tuple[list[dict], dict]:
        """A 2-model × 2-env × 1-baseline matrix; mix of ok + oom_failed."""
        cells = [
            _cell("qwen3_1_7b", "search_qa", "sdar"),
            _cell("qwen3_1_7b", "alfworld", "sdar"),
            _cell("qwen2_5_3b", "search_qa", "sdar"),
            _cell("qwen2_5_3b", "alfworld", "sdar"),
        ]
        matrix_result = {
            "qwen3_1_7b__sdar__search_qa__s42": {
                "status": "ok",
                "metrics": {"status": "ok", "metric": 0.42, "steps_run": 150,
                            "reward_mean": 1.3},
                "gpu": "0", "retries": 0, "error": None,
            },
            "qwen3_1_7b__sdar__alfworld__s42": {
                "status": "oom_failed",
                "metrics": None,
                "gpu": "1", "retries": 2,
                "error": "CUDA out of memory. Tried to allocate 1.11 GiB.",
            },
            "qwen2_5_3b__sdar__search_qa__s42": {
                "status": "ok",
                "metrics": {"status": "ok", "metric": 0.55, "steps_run": 150},
                "gpu": "2", "retries": 0, "error": None,
            },
            "qwen2_5_3b__sdar__alfworld__s42": {
                "status": "error",
                "metrics": None,
                "gpu": "3", "retries": 0,
                "error": "AttributeError: 'ALFWorldEnv' object has no attribute 'build_student_prompt'",
            },
        }
        return cells, matrix_result

    def test_exact_nesting(self):
        cells, matrix_result = self._matrix_2x2()
        out = aggregate_cell_metrics(matrix_result, cells)

        # The precise path the scorer + postflight read.
        leaf = out["per_model"]["qwen3_1_7b"]["search_qa"]["sdar"]
        assert leaf["status"] == "ok"
        assert leaf["metric"] == 0.42
        assert leaf["steps_run"] == 150
        assert leaf["reward_mean"] == 1.3

        # No per_dataset wrapper — env keyed directly under model.
        assert "per_dataset" not in out["per_model"]["qwen3_1_7b"]
        assert set(out["per_model"]["qwen3_1_7b"].keys()) == {"search_qa", "alfworld"}

    def test_failed_leaf_carries_truncated_error(self):
        cells, matrix_result = self._matrix_2x2()
        out = aggregate_cell_metrics(matrix_result, cells)

        oom_leaf = out["per_model"]["qwen3_1_7b"]["alfworld"]["sdar"]
        assert oom_leaf["status"] == "failed"
        assert oom_leaf["metric"] is None
        assert "CUDA out of memory" in oom_leaf["error"]

        err_leaf = out["per_model"]["qwen2_5_3b"]["alfworld"]["sdar"]
        assert err_leaf["status"] == "failed"
        assert "ALFWorldEnv" in err_leaf["error"]

    def test_status_partial_on_mixed(self):
        cells, matrix_result = self._matrix_2x2()
        out = aggregate_cell_metrics(matrix_result, cells)
        assert out["status"] == "partial"

    def test_status_complete_when_all_ok(self):
        cells = [
            _cell("qwen3_1_7b", "search_qa", "sdar"),
            _cell("qwen2_5_3b", "search_qa", "sdar"),
        ]
        matrix_result = {
            "qwen3_1_7b__sdar__search_qa__s42": {
                "status": "ok", "metrics": {"metric": 0.4}, "gpu": "0",
                "retries": 0, "error": None,
            },
            "qwen2_5_3b__sdar__search_qa__s42": {
                "status": "ok", "metrics": {"metric": 0.5}, "gpu": "1",
                "retries": 0, "error": None,
            },
        }
        out = aggregate_cell_metrics(matrix_result, cells)
        assert out["status"] == "complete"
        assert out["scope"]["models_run"] == ["qwen2_5_3b", "qwen3_1_7b"]

    def test_skipped_cell_aggregates_as_complete_leaf(self):
        """A resume-skipped cell (Track B) is an ok leaf and counts toward complete."""
        cells = [_cell("qwen3_1_7b", "search_qa", "sdar")]
        matrix_result = {
            "qwen3_1_7b__sdar__search_qa__s42": {
                "status": "skipped",
                "metrics": {"status": "ok", "metric": 0.42, "reward_mean": 1.3},
                "gpu": "0", "retries": 0, "error": None,
            },
        }
        out = aggregate_cell_metrics(matrix_result, cells)
        leaf = out["per_model"]["qwen3_1_7b"]["search_qa"]["sdar"]
        assert leaf["status"] == "ok"           # forced to ok in the leaf
        assert leaf["metric"] == 0.42
        assert leaf["reward_mean"] == 1.3       # prior metrics passed through
        assert out["status"] == "complete"      # a skipped cell is not a failure
        assert out["scope"]["models_run"] == ["qwen3_1_7b"]

    def test_skipped_plus_ok_is_complete(self):
        """A mix of skipped + freshly-ok cells is still 'complete' (no failures)."""
        cells = [
            _cell("qwen3_1_7b", "search_qa", "sdar"),
            _cell("qwen2_5_3b", "search_qa", "sdar"),
        ]
        matrix_result = {
            "qwen3_1_7b__sdar__search_qa__s42": {
                "status": "skipped", "metrics": {"metric": 0.4}, "gpu": "0",
                "retries": 0, "error": None,
            },
            "qwen2_5_3b__sdar__search_qa__s42": {
                "status": "ok", "metrics": {"metric": 0.5}, "gpu": "1",
                "retries": 0, "error": None,
            },
        }
        out = aggregate_cell_metrics(matrix_result, cells)
        assert out["status"] == "complete"
        assert sorted(out["scope"]["models_run"]) == ["qwen2_5_3b", "qwen3_1_7b"]

    def test_skipped_plus_failed_is_partial(self):
        """skipped (=ok) + a failed cell → partial, mirroring ok+failed."""
        cells = [
            _cell("qwen3_1_7b", "search_qa", "sdar"),
            _cell("qwen2_5_3b", "search_qa", "sdar"),
        ]
        matrix_result = {
            "qwen3_1_7b__sdar__search_qa__s42": {
                "status": "skipped", "metrics": {"metric": 0.4}, "gpu": "0",
                "retries": 0, "error": None,
            },
            "qwen2_5_3b__sdar__search_qa__s42": {
                "status": "oom_failed", "metrics": None, "gpu": "1",
                "retries": 2, "error": "CUDA out of memory",
            },
        }
        out = aggregate_cell_metrics(matrix_result, cells)
        assert out["status"] == "partial"

    def test_status_failed_when_none_ok(self):
        cells = [_cell("qwen3_1_7b", "search_qa", "sdar")]
        matrix_result = {
            "qwen3_1_7b__sdar__search_qa__s42": {
                "status": "oom_failed", "metrics": None, "gpu": "0",
                "retries": 2, "error": "CUDA out of memory",
            },
        }
        out = aggregate_cell_metrics(matrix_result, cells)
        assert out["status"] == "failed"
        assert out["scope"]["models_run"] == []

    def test_missing_result_record_becomes_failed_leaf(self):
        """A cell with no matrix_result entry → failed leaf, error 'no result'."""
        cells = [_cell("qwen3_1_7b", "search_qa", "sdar")]
        out = aggregate_cell_metrics({}, cells)  # empty matrix_result
        leaf = out["per_model"]["qwen3_1_7b"]["search_qa"]["sdar"]
        assert leaf["status"] == "failed"
        assert leaf["error"] == "no result"
        assert out["status"] == "failed"

    def test_ok_cell_missing_metric_key_gets_null(self):
        cells = [_cell("m1", "search_qa", "sdar")]
        matrix_result = {
            "m1__sdar__search_qa__s42": {
                "status": "ok",
                "metrics": {"steps_run": 10},  # NO "metric" key
                "gpu": "0", "retries": 0, "error": None,
            },
        }
        out = aggregate_cell_metrics(matrix_result, cells)
        leaf = out["per_model"]["m1"]["search_qa"]["sdar"]
        assert leaf["status"] == "ok"
        assert leaf["metric"] is None      # synthesised null
        assert leaf["steps_run"] == 10     # passthrough preserved

    def test_failed_cell_partial_metrics_preserved(self):
        """A failed cell that wrote partial metrics keeps them under the leaf."""
        cells = [_cell("m1", "search_qa", "sdar")]
        matrix_result = {
            "m1__sdar__search_qa__s42": {
                "status": "error",
                "metrics": {"steps_run": 7, "partial_reward": 0.1},
                "gpu": "0", "retries": 0, "error": "boom",
            },
        }
        out = aggregate_cell_metrics(matrix_result, cells)
        leaf = out["per_model"]["m1"]["search_qa"]["sdar"]
        assert leaf["status"] == "failed"
        assert leaf["steps_run"] == 7
        assert leaf["partial_reward"] == 0.1
        assert leaf["error"] == "boom"

    def test_scope_threads_gaps_and_skips(self):
        cells = [_cell("qwen3_1_7b", "search_qa", "sdar")]
        matrix_result = {
            "qwen3_1_7b__sdar__search_qa__s42": {
                "status": "ok", "metrics": {"metric": 0.4}, "gpu": "0",
                "retries": 0, "error": None,
            },
        }
        cap_gaps = [{"item": "qwen2_5_7b", "reason": "too big", "kind": "capacity"}]
        ds_gaps = [{"item": "webshop", "reason": "404", "kind": "dataset_unavailable"}]
        out = aggregate_cell_metrics(
            matrix_result,
            cells,
            capacity_gaps=cap_gaps,
            dataset_gaps=ds_gaps,
            models_skipped=["qwen2_5_7b", "qwen2_5_7b"],   # dup on purpose
            environments_skipped=["webshop"],
        )
        scope = out["scope"]
        assert scope["models_skipped"] == ["qwen2_5_7b"]   # deduped
        assert scope["environments_skipped"] == ["webshop"]
        assert len(scope["gaps"]) == 2
        gap_items = {g["item"] for g in scope["gaps"]}
        assert gap_items == {"qwen2_5_7b", "webshop"}

    def test_malformed_cells_derived_not_dropped(self):
        """A cell missing axes is DERIVED into the tree (never silently dropped),
        a non-dict cell and a non-dict result still don't raise.

        2026-06-09: the old contract silently skipped axis-less cells — an
        All-CNN run trained 14 cells to paper-grade accuracy and aggregated to
        ``per_model={}``. Axis-less cells now land under derived axes (the cell
        id as model_key, ``default`` env/baseline).
        """
        cells = [
            {"id": "noaxes__s42", "seed": 42},                # missing model/env/baseline
            "not-a-dict",                                      # type: ignore[list-item]
            _cell("m1", "search_qa", "sdar"),
        ]
        matrix_result = {
            "m1__sdar__search_qa__s42": {
                "status": "ok", "metrics": {"metric": 0.4}, "gpu": "0",
                "retries": 0, "error": None,
            },
            "noaxes__s42": "bad-record",                       # non-dict record
        }
        out = aggregate_cell_metrics(matrix_result, cells)
        # The well-formed cell lands under its explicit axes; the axis-less cell
        # is preserved under derived axes with a failed leaf (its record was
        # unusable), so the gap is VISIBLE instead of silently vanishing.
        assert set(out["per_model"].keys()) == {"m1", "noaxes__s42"}
        derived_leaf = out["per_model"]["noaxes__s42"]["default"]["default"]
        assert derived_leaf["status"] == "failed"
        assert out["status"] == "partial"

    def test_empty_inputs(self):
        out = aggregate_cell_metrics({}, [])
        assert out["status"] == "failed"
        assert out["per_model"] == {}
        assert out["scope"]["models_run"] == []
        assert out["scope"]["gaps"] == []

    def test_output_is_json_serialisable(self):
        cells, matrix_result = self._matrix_2x2()
        out = aggregate_cell_metrics(matrix_result, cells)
        # Round-trips through JSON without error.
        assert json.loads(json.dumps(out)) == out


# ===========================================================================
# Real-sample shape compatibility — the load-bearing assertion
# ===========================================================================

class TestRealSampleShape:
    def test_real_sample_exists_and_has_expected_nesting(self):
        """Sanity-check the verified on-disk sample is shaped as documented."""
        assert REAL_SAMPLE.is_file(), f"missing real sample: {REAL_SAMPLE}"
        sample = json.loads(REAL_SAMPLE.read_text(encoding="utf-8"))
        assert "status" in sample
        assert "per_model" in sample
        assert "scope" in sample
        # per_model -> model_key -> env -> baseline -> {status, metric}
        leaf = sample["per_model"]["qwen3_1_7b"]["search_qa"]["sdar"]
        assert "status" in leaf
        assert "metric" in leaf
        # SDAR keys env DIRECTLY under model — no per_dataset wrapper.
        assert "per_dataset" not in sample["per_model"]["qwen3_1_7b"]

    def test_aggregator_output_matches_real_sample_structure(self):
        """The aggregator's nesting is structurally identical to the real sample.

        We rebuild the sample's (model, env, baseline) triples through the
        aggregator and assert the result walks the SAME path the scorer already
        consumes: per_model -> model_key -> env -> baseline -> dict with both
        "status" and "metric".  This is the byte-compatibility guarantee.
        """
        sample = json.loads(REAL_SAMPLE.read_text(encoding="utf-8"))
        sample_per_model = sample["per_model"]

        # Reconstruct cells + a matrix_result from every triple in the sample.
        cells: list[dict] = []
        matrix_result: dict = {}
        for model_key, envs in sample_per_model.items():
            for env, baselines in envs.items():
                for baseline, sample_leaf in baselines.items():
                    cell = _cell(model_key, env, baseline)
                    cells.append(cell)
                    # Mirror the sample's own status into a matrix record.
                    if sample_leaf.get("status") == "ok":
                        matrix_result[cell["id"]] = {
                            "status": "ok",
                            "metrics": {
                                "status": "ok",
                                "metric": sample_leaf.get("metric"),
                            },
                            "gpu": "0", "retries": 0, "error": None,
                        }
                    else:
                        matrix_result[cell["id"]] = {
                            "status": "error",
                            "metrics": None,
                            "gpu": "0", "retries": 0,
                            "error": sample_leaf.get("error", "failed"),
                        }

        out = aggregate_cell_metrics(matrix_result, cells)
        out_per_model = out["per_model"]

        # Same model_keys, same envs per model, same baselines per env.
        assert set(out_per_model.keys()) == set(sample_per_model.keys())
        for model_key, envs in sample_per_model.items():
            assert set(out_per_model[model_key].keys()) == set(envs.keys())
            for env, baselines in envs.items():
                assert set(out_per_model[model_key][env].keys()) == set(baselines.keys())
                for baseline in baselines:
                    out_leaf = out_per_model[model_key][env][baseline]
                    # Every leaf the scorer reads carries BOTH keys.
                    assert "status" in out_leaf
                    assert "metric" in out_leaf
                    # No per_dataset layer anywhere (matches the real sample).
                    assert "per_dataset" not in out_per_model[model_key]

        # Top-level + scope keys present and of the right type, matching what
        # leaf_scorer._collect_gaps / _detect_data_unavailable_leaves read.
        assert out["status"] in {"complete", "partial", "failed"}
        assert isinstance(out["scope"]["models_skipped"], list)
        assert isinstance(out["scope"]["environments_skipped"], list)
        assert isinstance(out["scope"]["gaps"], list)


# ---------------------------------------------------------------------------
# audit_aggregation_completeness (L6, 2026-06-16) — declared-vs-aggregated
# reconciliation: catches cells lost BEFORE the aggregate loop (the gap the
# derive-not-drop guarantee can't cover).
# ---------------------------------------------------------------------------


def _acell(cid, mk="m", env="e", base="b", **extra):
    return {"id": cid, "model_key": mk, "env": env, "baseline": base, **extra}


class TestAuditAggregationCompleteness:
    def test_all_ok_is_complete(self):
        cells = [_acell("c1", base="adam"), _acell("c2", base="sgd")]
        agg = aggregate_cell_metrics(
            {"c1": {"status": "ok", "metrics": {"metric": 0.9}},
             "c2": {"status": "ok", "metrics": {"metric": 0.8}}},
            cells,
        )
        out = audit_aggregation_completeness(cells, agg)
        assert out["complete"] is True
        assert set(out["ok"]) == {"c1", "c2"}
        assert out["failed"] == [] and out["unaccounted"] == []

    def test_failed_cell_surfaced(self):
        cells = [_acell("c1", base="adam"), _acell("c2", base="sgd")]
        agg = aggregate_cell_metrics(
            {"c1": {"status": "ok", "metrics": {"metric": 0.9}},
             "c2": {"status": "error", "error": "boom"}},
            cells,
        )
        out = audit_aggregation_completeness(cells, agg)
        assert out["failed"] == ["c2"]
        assert out["complete"] is False
        assert any("re-run" in n.lower() for n in out["notes"])

    def test_unaccounted_cell_is_the_silent_loss_case(self):
        # c2 is declared but NEVER reached aggregation (no matrix record, no gap,
        # not skipped) — the bucket derive-not-drop cannot surface.
        declared = [_acell("c1", base="adam"), _acell("c2", base="sgd")]
        agg = aggregate_cell_metrics({"c1": {"status": "ok", "metrics": {"metric": 0.9}}},
                                     [declared[0]])  # only c1 reached the loop
        out = audit_aggregation_completeness(declared, agg)
        assert out["unaccounted"] == ["c2"]
        assert out["complete"] is False
        assert any("unaccounted" in n.lower() for n in out["notes"])

    def test_known_gap_is_accounted_not_unaccounted(self):
        declared = [_acell("c1", base="adam"), _acell("big7b", mk="qwen7b", base="sgd")]
        agg = aggregate_cell_metrics(
            {"c1": {"status": "ok", "metrics": {"metric": 0.9}}},
            [declared[0]],
            models_skipped=["qwen7b"],  # capacity gate dropped it → KNOWN
        )
        out = audit_aggregation_completeness(declared, agg)
        assert out["gapped"] == ["big7b"]
        assert out["unaccounted"] == []  # a known drop is not silent loss

    def test_tolerates_dup_disambiguation_suffix(self):
        # Two cells resolve to the same triple → normalize_cell_axes suffixes the
        # later baseline with its id; the audit must still match it as ok.
        from backend.agents.rlm.cell_matrix import normalize_cell_axes
        raw = [_acell("c1", base="b"), _acell("c2", base="b")]  # identical triple
        cells, _ = normalize_cell_axes(raw)
        agg = aggregate_cell_metrics(
            {"c1": {"status": "ok", "metrics": {"metric": 0.9}},
             "c2": {"status": "ok", "metrics": {"metric": 0.8}}},
            cells,
        )
        out = audit_aggregation_completeness(raw, agg)  # audit the RAW declared cells
        assert set(out["ok"]) == {"c1", "c2"}
        assert out["complete"] is True

    def test_never_raises_on_garbage(self):
        # Returns a well-formed dict, never raises. (An empty-dict cell with no
        # matching aggregate is legitimately "unaccounted" → complete False; the
        # contract under test is "no exception", not a particular verdict.)
        for decl, agg in ((None, None), ("x", {"per_model": 5}), ([{}, None, "s"], {})):
            out = audit_aggregation_completeness(decl, agg)
            assert isinstance(out, dict) and "complete" in out and isinstance(out["notes"], list)
