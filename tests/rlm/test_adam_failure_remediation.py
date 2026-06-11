"""2026-06-11 Adam failure-mode remediation: classifier + route retention + guidance.

Three fixes from the Adam forensics (14 attempts, 4 competitive scores):
A. cuda_device_assert classification (was: unknown → blind repairs); the
   classifier also reads stdout/stderr tails (the live assert lived ONLY in
   result["stdout"]).
B. Route-retention guard: a repair pass that drops cells.json gets an explicit
   contract warning + the manifest preserved at rlm_state/last_cells.json
   (the monolithic regression that lost the matrix happens loudly or not at all).
C. Multi-family isolation guidance (generic block + Adam paper hint).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from backend.agents.rlm.failure_classifier import FAILURE_CLASSES, classify_failure
from backend.agents.rlm.primitives import (
    _check_cells_manifest_retention,
    _stash_cells_manifest,
)


# ---------------------------------------------------------------------------
# A. cuda_device_assert classification
# ---------------------------------------------------------------------------


def test_device_assert_in_stdout_classifies():
    """The live failure shape: error=None, logs empty, trace only in stdout."""
    result = {
        "success": False,
        "error": None,
        "logs": "",
        "stdout": (
            "RuntimeError: CUDA error: device-side assert triggered\n"
            "CUDA kernel errors might be asynchronously reported..."
        ),
    }
    klass, fix = classify_failure(result)
    assert klass == "cuda_device_assert"
    assert "num_classes" in fix and "own process" in fix.lower() or "OWN process" in fix


def test_device_assert_in_logs_classifies():
    result = {"success": False, "error": "", "logs": "blah device-side assert blah"}
    assert classify_failure(result)[0] == "cuda_device_assert"


def test_device_assert_wins_over_secondary_memory_noise():
    """The assert poisons the context; later calls emit memory-flavoured noise."""
    result = {
        "success": False,
        "error": "",
        "logs": (
            "RuntimeError: CUDA error: device-side assert triggered\n"
            "torch._C._cuda_emptyCache() CUDA error: out of memory"
        ),
    }
    assert classify_failure(result)[0] == "cuda_device_assert"


def test_stderr_tail_reaches_classifier():
    result = {"success": False, "stderr": "ModuleNotFoundError: No module named 'einops'"}
    klass, fix = classify_failure(result)
    assert klass == "missing_module"
    assert "einops" in fix


def test_class_registered():
    assert "cuda_device_assert" in FAILURE_CLASSES


def test_plain_oom_still_classifies_oom():
    result = {"success": False, "logs": "torch.cuda.OutOfMemoryError: CUDA out of memory"}
    assert classify_failure(result)[0] == "cuda_oom"


# ---------------------------------------------------------------------------
# B. cells-manifest route retention
# ---------------------------------------------------------------------------


def _ctx(tmp_path: Path) -> SimpleNamespace:
    project_dir = tmp_path / "prj"
    project_dir.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        project_id="prj",
        project_dir=project_dir,
        emit=None,
        dashboard=None,
    )


def test_stash_preserves_manifest(tmp_path):
    ctx = _ctx(tmp_path)
    code = tmp_path / "code"
    code.mkdir()
    (code / "cells.json").write_text('[{"cell_id": "a"}]')

    assert _stash_cells_manifest(code, ctx) is True
    stashed = ctx.project_dir / "rlm_state" / "last_cells.json"
    assert json.loads(stashed.read_text()) == [{"cell_id": "a"}]


def test_stash_returns_false_without_manifest(tmp_path):
    ctx = _ctx(tmp_path)
    code = tmp_path / "code"
    code.mkdir()
    assert _stash_cells_manifest(code, ctx) is False


def test_dropped_manifest_warns_on_repair(tmp_path):
    ctx = _ctx(tmp_path)
    code = tmp_path / "code"
    code.mkdir()  # repair rewrote the tree; cells.json gone
    result = {"ok": True, "code_path": str(code), "files": ["train.py"]}

    out = _check_cells_manifest_retention(
        result, code_dir=code, had_manifest=True, is_repair=True, ctx=ctx,
    )
    warnings = out.get("contract_warnings") or []
    assert any("cells_manifest_dropped" in w for w in warnings)
    assert any("last_cells.json" in w for w in warnings)


def test_retained_manifest_stays_silent(tmp_path):
    ctx = _ctx(tmp_path)
    code = tmp_path / "code"
    code.mkdir()
    (code / "cells.json").write_text("[]")
    result = {"ok": True}
    out = _check_cells_manifest_retention(
        result, code_dir=code, had_manifest=True, is_repair=True, ctx=ctx,
    )
    assert "contract_warnings" not in out


def test_initial_implementation_never_warns(tmp_path):
    ctx = _ctx(tmp_path)
    code = tmp_path / "code"
    code.mkdir()
    result = {"ok": True}
    out = _check_cells_manifest_retention(
        result, code_dir=code, had_manifest=False, is_repair=False, ctx=ctx,
    )
    assert "contract_warnings" not in out


def test_failed_result_passes_through_unchanged(tmp_path):
    ctx = _ctx(tmp_path)
    code = tmp_path / "code"
    code.mkdir()
    result = {"ok": False, "error": "x"}
    out = _check_cells_manifest_retention(
        result, code_dir=code, had_manifest=True, is_repair=True, ctx=ctx,
    )
    assert out == {"ok": False, "error": "x"}


# ---------------------------------------------------------------------------
# C. isolation guidance present (generic block + Adam hint)
# ---------------------------------------------------------------------------


def test_generic_guidance_carries_multi_family_isolation():
    import backend.agents.baseline_implementation as bi
    blob = "".join(
        v for k, v in vars(bi).items() if isinstance(v, str) and k.startswith("_")
    )
    assert "MULTI-FAMILY ISOLATION" in blob
    assert "incrementally" in blob


def test_adam_hint_carries_failure_isolation():
    from backend.agents.prompts.paper_hints import PAPER_HINTS
    g = PAPER_HINTS["1412.6980"].guidance
    assert "FAILURE ISOLATION" in g
    assert "last_cells.json" in g
    assert "num_classes" in g


# ---------------------------------------------------------------------------
# B2. ACTIVE route retention (auto-restore when applicable)
# ---------------------------------------------------------------------------


def _repair_setup(tmp_path: Path, *, trainer: bool, stash: bool):
    ctx = _ctx(tmp_path)
    code = tmp_path / "code"
    code.mkdir()
    if trainer:
        (code / "train_cell.py").write_text("# per-cell trainer\n")
    if stash:
        state = ctx.project_dir / "rlm_state"
        state.mkdir(parents=True, exist_ok=True)
        (state / "last_cells.json").write_text('[{"cell_id": "vae_b1"}]')
    return ctx, code


def test_auto_restores_manifest_when_trainer_survives(tmp_path, monkeypatch):
    monkeypatch.delenv("REPROLAB_CELLS_ROUTE_RETENTION", raising=False)
    ctx, code = _repair_setup(tmp_path, trainer=True, stash=True)
    result = {"ok": True}

    out = _check_cells_manifest_retention(
        result, code_dir=code, had_manifest=True, is_repair=True, ctx=ctx,
        repair_failure_class="cuda_device_assert",
    )
    assert json.loads((code / "cells.json").read_text()) == [{"cell_id": "vae_b1"}]
    assert any("cells_manifest_restored" in w for w in out["contract_warnings"])


def test_no_restore_without_trainer(tmp_path, monkeypatch):
    """Manifest without a per-cell trainer is useless — warn-only."""
    monkeypatch.delenv("REPROLAB_CELLS_ROUTE_RETENTION", raising=False)
    ctx, code = _repair_setup(tmp_path, trainer=False, stash=True)
    out = _check_cells_manifest_retention(
        {"ok": True}, code_dir=code, had_manifest=True, is_repair=True, ctx=ctx,
    )
    assert not (code / "cells.json").exists()
    assert any("cells_manifest_dropped" in w for w in out["contract_warnings"])


def test_no_restore_after_cells_route_failure(tmp_path, monkeypatch):
    """The repair may be deliberately abandoning a broken route — warn-only."""
    monkeypatch.delenv("REPROLAB_CELLS_ROUTE_RETENTION", raising=False)
    ctx, code = _repair_setup(tmp_path, trainer=True, stash=True)
    out = _check_cells_manifest_retention(
        {"ok": True}, code_dir=code, had_manifest=True, is_repair=True, ctx=ctx,
        repair_failure_class="cell_execution_error",
    )
    assert not (code / "cells.json").exists()
    assert any("cells_manifest_dropped" in w for w in out["contract_warnings"])


def test_flag_disables_restore(tmp_path, monkeypatch):
    monkeypatch.setenv("REPROLAB_CELLS_ROUTE_RETENTION", "0")
    ctx, code = _repair_setup(tmp_path, trainer=True, stash=True)
    out = _check_cells_manifest_retention(
        {"ok": True}, code_dir=code, had_manifest=True, is_repair=True, ctx=ctx,
    )
    assert not (code / "cells.json").exists()
    assert any("cells_manifest_dropped" in w for w in out["contract_warnings"])


def test_missing_stash_degrades_to_warning(tmp_path, monkeypatch):
    monkeypatch.delenv("REPROLAB_CELLS_ROUTE_RETENTION", raising=False)
    ctx, code = _repair_setup(tmp_path, trainer=True, stash=False)
    out = _check_cells_manifest_retention(
        {"ok": True}, code_dir=code, had_manifest=True, is_repair=True, ctx=ctx,
    )
    assert not (code / "cells.json").exists()
    assert any("cells_manifest_dropped" in w for w in out["contract_warnings"])


# ---------------------------------------------------------------------------
# D. per_model derivation from family-shaped top-level keys (derive-not-drop)
# ---------------------------------------------------------------------------


def test_derives_per_model_from_family_keys():
    """The live 2026-06-11 shape: six families top-level, per_model empty."""
    from backend.agents.rlm.primitives import _derive_per_model_from_families

    metrics = {
        "status": "completed",
        "mnist_logreg": {"adam": {"test_accuracy": 92.63}, "sgd": {"test_accuracy": 92.65}},
        "mnist_mlp": {"adam": {"test_accuracy": 98.1}},
        "cifar10_cnn": {"adam": {"test_accuracy": 80.2}},
        "synthetic": {"adam": {"final_loss": 0.01}},
        "imdb_bow": {"adam": {"test_accuracy": 88.0}},
        "vae_lr_sweep": {"lr_0.001": {"elbo": -98.0}},
        "history": {"mnist_logreg": {"adam": {"epoch": [1, 2]}}},
        "regret": [1.0, 0.5],
        "per_model": {},
        "data_load_failures": {},
        "scope": {"gaps": []},
    }
    out, notes = _derive_per_model_from_families(metrics)
    assert notes and "per_model_derived_from_families" in notes[0]
    pm = out["per_model"]
    assert set(pm) == {
        "mnist_logreg", "mnist_mlp", "cifar10_cnn", "synthetic", "imdb_bow", "vae_lr_sweep",
    }
    assert pm["mnist_logreg"]["adam"]["test_accuracy"] == 92.63
    # Reserved blocks never masquerade as models.
    assert "history" not in pm and "scope" not in pm and "data_load_failures" not in pm


def test_derivation_satisfies_the_live_scope_check():
    """End-to-end vs the validator: the exact Adam scope now passes."""
    from backend.agents.rlm.primitives import (
        _derive_per_model_from_families,
        _validate_scope_metrics,
    )
    from backend.agents.schemas import DatasetSlice, ScopeSpec

    scope = ScopeSpec(datasets=[
        DatasetSlice(name="MNIST"), DatasetSlice(name="IMDB"), DatasetSlice(name="CIFAR-10"),
    ])
    metrics = {
        "status": "completed",
        "mnist_logreg": {"adam": {"test_accuracy": 92.63}},
        "imdb_bow": {"adam": {"test_accuracy": 88.0}},
        "cifar10_cnn": {"adam": {"test_accuracy": 80.2}},
        "per_model": {},
    }
    assert _validate_scope_metrics(scope, metrics) is not None  # refused before
    out, notes = _derive_per_model_from_families(metrics)
    assert notes
    assert _validate_scope_metrics(scope, out) is None  # passes after


def test_no_derivation_when_per_model_populated():
    from backend.agents.rlm.primitives import _derive_per_model_from_families

    metrics = {"per_model": {"m1": {"acc": 1.0}}, "mnist_logreg": {"adam": {}}}
    out, notes = _derive_per_model_from_families(metrics)
    assert notes == [] and out is metrics


def test_no_derivation_without_family_shaped_keys():
    from backend.agents.rlm.primitives import _derive_per_model_from_families

    metrics = {"status": "completed", "per_model": {}, "regret": [1.0], "wall_clock_s": 5.0}
    out, notes = _derive_per_model_from_families(metrics)
    assert notes == [] and "per_model" in out and out["per_model"] == {}
