"""Harness-owned staged-search (tune-then-run) pure core.

Codex's #1 lever from the 2026-06-14 Adam-plan review: convert the unenforceable
"tune-then-run" prose into deterministic harness behavior — bounded candidate
phase, winner selection by the claim metric, budget preflight, one full cell per
group.
"""

from __future__ import annotations

from backend.agents.rlm.staged_search import (
    SearchGroup,
    affordable_full_cells,
    budget_feasible,
    candidate_rate,
    estimate_full_seconds,
    extract_select_value,
    materialize_full_cells,
    parse_search_spec,
    run_staged_search,
    select_winner,
)


def _group(**kw):
    base = dict(
        group="g", select_metric="final_train_loss", select_objective="min",
        candidates=[{"id": "g__a", "params": {"lr": 1e-3, "epochs": 3}}],
        promote={"id": "g", "params": {"epochs": 200}}, param_from_winner=["lr"],
    )
    base.update(kw)
    return SearchGroup(**base)


class TestParse:
    def test_absent_search_returns_empty(self):
        assert parse_search_spec({"cells": [{"id": "x"}]}) == []
        assert parse_search_spec(None) == []
        assert parse_search_spec({"search": []}) == []

    def test_parses_a_group(self):
        spec = {"search": [{
            "group": "mlp_adam", "select_metric": "loss", "select_objective": "min",
            "candidates": [{"id": "c1", "params": {"lr": 1e-3}}],
            "promote": {"id": "mlp_adam", "params": {"epochs": 200}},
            "param_from_winner": ["lr"]}]}
        groups = parse_search_spec(spec)
        assert len(groups) == 1
        g = groups[0]
        assert g.group == "mlp_adam" and g.select_metric == "loss" and g.select_objective == "min"
        assert g.param_from_winner == ["lr"]

    def test_caps_candidates_per_group(self):
        cands = [{"id": f"c{i}", "params": {}} for i in range(20)]
        groups = parse_search_spec({"search": [{"candidates": cands, "promote": {"id": "g"}}]})
        assert len(groups[0].candidates) == 5  # _MAX_CANDIDATES_PER_GROUP

    def test_total_candidate_cap_drops_extra_groups(self):
        # 40 groups × 5 candidates = 200 > 80 cap → truncated
        spec = {"search": [
            {"group": f"g{i}", "candidates": [{"id": f"g{i}_{j}", "params": {}} for j in range(5)],
             "promote": {"id": f"g{i}"}} for i in range(40)]}
        groups = parse_search_spec(spec)
        total = sum(len(g.candidates) for g in groups)
        assert total <= 80

    def test_bad_objective_defaults_min(self):
        groups = parse_search_spec({"search": [{"select_objective": "sideways",
                                                "candidates": [{"id": "c"}], "promote": {"id": "g"}}]})
        assert groups[0].select_objective == "min"

    def test_group_without_promote_id_skipped(self):
        assert parse_search_spec({"search": [{"candidates": [{"id": "c"}], "promote": {}}]}) == []


class TestExtractSelect:
    def test_flat_key(self):
        assert extract_select_value({"final_train_loss": 0.05}, "final_train_loss") == 0.05

    def test_dotted_path(self):
        m = {"per_model": {"mlp": {"adam": {"loss": 0.1}}}}
        assert extract_select_value(m, "per_model.mlp.adam.loss") == 0.1

    def test_missing_or_nonnumeric_is_none(self):
        assert extract_select_value({"x": "abc"}, "x") is None
        assert extract_select_value({}, "loss") is None
        assert extract_select_value(None, "loss") is None
        assert extract_select_value({"x": True}, "x") is None  # bool is not a metric


class TestSelectWinner:
    def test_picks_min(self):
        g = _group(candidates=[
            {"id": "a", "params": {"lr": 1e-3}}, {"id": "b", "params": {"lr": 3e-3}}])
        results = {"a": {"metrics": {"final_train_loss": 0.05}},
                   "b": {"metrics": {"final_train_loss": 0.01}}}
        assert select_winner(g, results)["id"] == "b"

    def test_picks_max(self):
        g = _group(select_objective="max", select_metric="acc", candidates=[
            {"id": "a"}, {"id": "b"}])
        results = {"a": {"metrics": {"acc": 0.9}}, "b": {"metrics": {"acc": 0.95}}}
        assert select_winner(g, results)["id"] == "b"

    def test_crashed_candidate_never_wins(self):
        g = _group(candidates=[{"id": "a", "params": {}}, {"id": "b", "params": {}}])
        results = {"a": {"metrics": None, "status": "error"},
                   "b": {"metrics": {"final_train_loss": 0.2}}}
        assert select_winner(g, results)["id"] == "b"

    def test_no_usable_metric_returns_none(self):
        g = _group(candidates=[{"id": "a"}])
        assert select_winner(g, {"a": {"metrics": {}}}) is None


class TestMaterialize:
    def test_copies_winner_params_into_full_cell(self):
        g = _group(promote={"id": "mlp_adam", "params": {"epochs": 200, "model_key": "mlp"}},
                   param_from_winner=["lr"])
        winners = {"g": {"id": "g__b", "params": {"lr": 3e-3, "epochs": 3}}}
        full = materialize_full_cells([g], winners)
        assert len(full) == 1
        assert full[0]["params"]["lr"] == 3e-3          # winner's tuned lr
        assert full[0]["params"]["epochs"] == 200       # promote template's full epochs
        assert full[0]["params"]["model_key"] == "mlp"
        assert full[0]["_tuned_from"] == "g__b"         # provenance

    def test_group_without_winner_is_omitted(self):
        assert materialize_full_cells([_group()], {}) == []


class TestBudget:
    def test_estimate_scales_by_epochs(self):
        # candidates: 3 cells × 3 epochs = 9 epochs took 90s → 10 s/epoch
        cand = [{"params": {"epochs": 3}} for _ in range(3)]
        full = [{"params": {"epochs": 200}}]  # 200 epochs
        est = estimate_full_seconds(90.0, cand, full)
        # 10 s/epoch × 200 × 1.25 safety = 2500
        assert est is not None and 2400 < est < 2600

    def test_estimate_none_on_bad_input(self):
        assert estimate_full_seconds(0, [{"params": {"epochs": 3}}], [{"params": {"epochs": 1}}]) is None
        assert estimate_full_seconds(90.0, [], []) is None

    def test_feasible_when_fits(self):
        ok, _ = budget_feasible(1000.0, remaining_s=5000.0, reserve_s=500.0)
        assert ok is True

    def test_infeasible_when_over(self):
        ok, reason = budget_feasible(9000.0, remaining_s=5000.0, reserve_s=500.0)
        assert ok is False and "infeasible" in reason

    def test_failsoft_when_no_estimate(self):
        # un-estimable runs are NOT blocked (fail-open, like the rest of the harness)
        ok, reason = budget_feasible(None, remaining_s=5000.0, reserve_s=500.0)
        assert ok is True and reason == "no_estimate"

    def test_candidate_rate(self):
        # 9 candidate-epochs in 90s → 10 s/epoch
        assert candidate_rate(90.0, [{"params": {"epochs": 3}} for _ in range(3)]) == 10.0
        assert candidate_rate(0, [{"params": {"epochs": 3}}]) is None


class TestAffordableReduction:
    def test_keeps_cheapest_drops_expensive(self):
        cells = [{"id": "cheap", "params": {"epochs": 10}},
                 {"id": "mid", "params": {"epochs": 100}},
                 {"id": "vae", "params": {"epochs": 6000}}]
        # rate 1s/epoch, safety 1.25 → cheap=12.5, mid=125, vae=7500; budget 200
        kept, dropped = affordable_full_cells(cells, rate=1.0, remaining_s=200.0, reserve_s=0.0)
        kept_ids = {c["id"] for c in kept}
        assert "cheap" in kept_ids and "mid" in kept_ids  # breadth preserved
        assert [c["id"] for c in dropped] == ["vae"]       # the long-pole dropped

    def test_failsoft_keeps_all_when_rate_unknown(self):
        cells = [{"id": "a", "params": {"epochs": 1}}]
        kept, dropped = affordable_full_cells(cells, rate=None, remaining_s=None, reserve_s=0.0)
        assert len(kept) == 1 and dropped == []


class TestOrchestration:
    def test_two_phase_flow_with_winner_propagation(self, monkeypatch):
        """run_staged_search: phase-1 candidates → select winner → phase-2 full
        cell carries the winner's tuned lr. run_matrix is mocked."""
        calls = []

        def fake_run_matrix(cells, cell_script, **kw):
            calls.append([c["id"] for c in cells])
            out = {}
            for c in cells:
                # candidate 'lr3e-3' is best (lowest loss); full cells echo lr
                lr = (c.get("params") or {}).get("lr")
                loss = 0.01 if lr == 3e-3 else 0.5
                out[c["id"]] = {"status": "ok", "metrics": {"final_train_loss": loss, "lr_used": lr}}
            return out

        monkeypatch.setattr("backend.agents.rlm.gpu_cell_runner.run_matrix", fake_run_matrix)
        groups = parse_search_spec({"search": [{
            "group": "mlp_adam", "select_metric": "final_train_loss", "select_objective": "min",
            "candidates": [
                {"id": "mlp_adam__lr1e-3", "params": {"lr": 1e-3, "epochs": 3}},
                {"id": "mlp_adam__lr3e-3", "params": {"lr": 3e-3, "epochs": 3}}],
            "promote": {"id": "mlp_adam", "params": {"epochs": 200, "model_key": "mlp"}},
            "param_from_winner": ["lr"]}]})
        out = run_staged_search(groups, "train_cell.py", output_root="/tmp/x", gpus=["0"])

        assert len(calls) == 2                          # two phases ran
        assert calls[0] == ["mlp_adam__lr1e-3", "mlp_adam__lr3e-3"]  # candidates first
        assert calls[1] == ["mlp_adam"]                 # one full cell
        assert out["winners"] == {"mlp_adam": "mlp_adam__lr3e-3"}
        assert "mlp_adam" in out["results"]

    def test_no_full_phase_when_all_candidates_crash(self, monkeypatch):
        def crash_run_matrix(cells, cell_script, **kw):
            return {c["id"]: {"status": "error", "metrics": None} for c in cells}
        monkeypatch.setattr("backend.agents.rlm.gpu_cell_runner.run_matrix", crash_run_matrix)
        groups = parse_search_spec({"search": [{
            "candidates": [{"id": "c1", "params": {"epochs": 3}}],
            "promote": {"id": "g1", "params": {"epochs": 100}}}]})
        out = run_staged_search(groups, "t.py", output_root="/tmp/x", gpus=["0"])
        assert out["winners"] == {} and out["results"] == {} and out["full_cells"] == []
