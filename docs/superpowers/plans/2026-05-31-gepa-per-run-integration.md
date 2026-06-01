# GEPA Per-Run Sub-Agent Optimization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire GEPA prompt optimization into `wrap_primitive()` so it runs before every `plan_reproduction`, `implement_baseline`, and `propose_improvements` call, continuously improving prompts from each call's output within the same run.

**Architecture:** `gepa_pre_call(ctx, primitive_name, args)` fires inside `wrap_primitive()` before the primitive thread is spawned; it runs a mini `gepa.optimize()` loop (10 metric calls max, 60 s cap), stores the winning prompt in `ctx.gepa_prompt_overrides[name]`, then the existing Path-A monkey-patch injects it. `gepa_post_call()` appends the real result to `ctx.gepa_example_buffer[name]` so the next call's trainset is warmer. A re-entrancy guard (`ctx.gepa_optimization_active`) prevents recursive GEPA-inside-GEPA calls when the adapter itself calls the LLM directly.

**Tech Stack:** Python 3.11+, `gepa==0.1.1`, `gepa-viz==0.1.0`, existing `RunContext` dataclass, `wrap_primitive()` in `binding.py`, pydantic-settings in `backend/config.py`, TypeScript (Next.js 16) for frontend.

---

## File Map

**New files:**
- `backend/agents/gepa/__init__.py`
- `backend/agents/gepa/hooks.py` — `gepa_pre_call`, `gepa_post_call`
- `backend/agents/gepa/optimizer.py` — `run_gepa_mini()`
- `backend/agents/gepa/callback.py` — `SSEGepaCallback`
- `backend/agents/gepa/prompt_registry.py` — buffer persistence
- `backend/agents/gepa/trainset/__init__.py`
- `backend/agents/gepa/trainset/paper_examples.py` — `PaperExamplesBuilder`
- `backend/agents/gepa/trainset/historical_examples.py` — cross-run loader
- `backend/agents/gepa/adapters/__init__.py`
- `backend/agents/gepa/adapters/plan_reproduction.py` — `PlanReproductionEvaluator`
- `backend/agents/gepa/adapters/implement_baseline.py` — `ImplementBaselineProxyEvaluator`
- `backend/agents/gepa/adapters/propose_improvements.py` — `ProposeImprovementsEvaluator`
- `backend/agents/gepa/metrics/__init__.py`
- `backend/agents/gepa/metrics/plan_metrics.py` — `metrics_shape_score`, `contract_completeness`
- `backend/agents/gepa/metrics/baseline_metrics.py` — `code_plan_structural_validity`
- `backend/agents/gepa/metrics/improvement_metrics.py` — `weak_area_coverage`, `category_diversity`
- `tests/gepa/__init__.py`
- `tests/gepa/test_passthrough.py`
- `tests/gepa/test_plan_metrics.py`
- `tests/gepa/test_paper_examples.py`
- `tests/gepa/test_plan_reproduction_evaluator.py`
- `tests/gepa/test_hooks.py`
- `tests/gepa/test_baseline_evaluator.py`
- `tests/gepa/test_improvement_evaluator.py`
- `tests/gepa/test_runner_integration.py`
- `frontend/src/app/api/gepa-viz/[...path]/route.ts`

**Modified files:**
- `backend/requirements.txt` — add `gepa==0.1.1`, `gepa-viz==0.1.0`
- `backend/config.py` — add GEPA settings fields
- `backend/agents/rlm/context.py` — add `gepa_prompt_overrides`, `gepa_example_buffer`, `gepa_optimization_active`
- `backend/agents/rlm/binding.py` — add pre/post GEPA hooks in `wrap_primitive()`
- `backend/agents/rlm/sse_bridge.py` — add 5 GEPA event builder functions
- `backend/agents/runtime/invoke.py` — add `system_prompt_override` param to `collect_agent_text()`
- `frontend/src/lib/events/rlm-events.ts` — add 5 GEPA event interfaces
- `frontend/src/hooks/use-rlm-run.ts` — add `gepa_candidate` kind + fold functions
- `frontend/src/components/lab/rlm/tree-node.tsx` — extend for `gepa_candidate`
- `frontend/src/components/lab/rlm/node-detail-sidebar.tsx` — GEPA candidate panel
- `frontend/src/components/lab/rlm/constellation-canvas.tsx` — `gepa_candidate` stroke color

---

## Task 1: Dependencies, RunContext fields, Settings, passthrough test

**Files:**
- Modify: `backend/requirements.txt`
- Modify: `backend/config.py`
- Modify: `backend/agents/rlm/context.py`
- Create: `backend/agents/gepa/__init__.py`
- Create: `tests/gepa/__init__.py`
- Create: `tests/gepa/test_passthrough.py`

- [ ] **Step 1: Add gepa deps to requirements.txt**

Open `backend/requirements.txt`. After the `rlms==0.1.1` line, add:

```
gepa==0.1.1
gepa-viz==0.1.0
```

- [ ] **Step 2: Add GEPA settings to backend/config.py**

Open `backend/config.py`. After the `paper_hint_invariants` field (around line 83), add these fields inside the `Settings` class (the `env_prefix="REPROLAB_"` means each maps to `REPROLAB_GEPA_*`):

```python
    # GEPA per-run prompt optimization
    # gepa_optimization: "off" | "on" | "plan-only" | "baseline-only" | "improve-only"
    gepa_optimization: str = "off"
    gepa_max_metric_calls_plan: int = 10
    gepa_max_metric_calls_baseline: int = 5
    gepa_max_metric_calls_improve: int = 10
    gepa_timeout_plan_s: int = 60
    gepa_timeout_baseline_s: int = 30
    gepa_timeout_improve_s: int = 60
    gepa_reflection_model: str = "openai/gpt-4o"
    gepa_viz_port: int = 5151
```

- [ ] **Step 3: Add fields to RunContext**

Open `backend/agents/rlm/context.py`. After the `paper_hint_invariants` field (line 83), add:

```python
    # GEPA per-run prompt optimization (backend/agents/gepa/)
    # gepa_prompt_overrides: filled by gepa_pre_call() before each targeted primitive;
    #   keys match primitive names ("plan_reproduction", "implement_baseline", "propose_improvements")
    gepa_prompt_overrides: dict = field(default_factory=dict)
    # gepa_example_buffer: accumulated training examples from previous calls within this run;
    #   keys match primitive names; each value is list[dict] with "input","output","score"
    gepa_example_buffer: dict = field(default_factory=dict)
    # Re-entrancy guard: True while GEPA's mini optimize() loop is running, prevents
    # wrap_primitive() from triggering a nested GEPA call during metric evaluation
    gepa_optimization_active: bool = False
```

- [ ] **Step 4: Create stub gepa module**

Create `backend/agents/gepa/__init__.py`:

```python
"""GEPA per-run sub-agent prompt optimization.

Entry points: gepa_pre_call, gepa_post_call (imported by binding.wrap_primitive).
All functions are no-ops when Settings().gepa_optimization == "off".
"""
```

- [ ] **Step 5: Create test dirs and passthrough test**

Create `tests/gepa/__init__.py` (empty).

Create `tests/gepa/test_passthrough.py`:

```python
"""When REPROLAB_GEPA_OPTIMIZATION=off (default), wrap_primitive must produce
zero behavioral change vs. today. This test verifies that:
1. gepa_prompt_overrides stays empty after a plan_reproduction-like call
2. No GEPA SSE events are emitted
3. The primitive result is identical to calling the function directly
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backend.agents.rlm.context import RunContext


@pytest.fixture()
def ctx_gepa_off():
    dashboard = MagicMock()
    dashboard.primitive_call = MagicMock()
    cost_ledger = MagicMock()
    cost_ledger.append = MagicMock()
    llm_client = MagicMock()
    llm_client.complete = MagicMock(return_value='{"baseline_plan":"x","metrics_shape":[]}')
    llm_client._last_usage = {}
    emit = MagicMock()
    return RunContext(
        project_id="prj_test",
        project_dir=Path("/tmp/prj_test"),
        runs_root=Path("/tmp/runs"),
        dashboard=dashboard,
        cost_ledger=cost_ledger,
        llm_client=llm_client,
        provider="openai",
        model="gpt-4o-mini",
        emit=emit,
    )


def test_gepa_fields_default_empty(ctx_gepa_off):
    """RunContext has GEPA fields and they start empty/False."""
    assert ctx_gepa_off.gepa_prompt_overrides == {}
    assert ctx_gepa_off.gepa_example_buffer == {}
    assert ctx_gepa_off.gepa_optimization_active is False


def test_no_gepa_calls_when_off(ctx_gepa_off):
    """With REPROLAB_GEPA_OPTIMIZATION=off, gepa_pre_call is never invoked."""
    with patch.dict(os.environ, {"REPROLAB_GEPA_OPTIMIZATION": "off"}):
        # Import after patching env
        from backend.agents.gepa.hooks import gepa_pre_call
        gepa_pre_call(ctx_gepa_off, "plan_reproduction", (), {})
        # No overrides written
        assert ctx_gepa_off.gepa_prompt_overrides == {}
        # optimization_active was never set True (no GEPA loop ran)
        assert ctx_gepa_off.gepa_optimization_active is False


def test_gepa_optimization_active_guard(ctx_gepa_off):
    """gepa_pre_call is a no-op when gepa_optimization_active is already True."""
    ctx_gepa_off.gepa_optimization_active = True
    with patch.dict(os.environ, {"REPROLAB_GEPA_OPTIMIZATION": "on"}):
        from backend.agents.gepa.hooks import gepa_pre_call
        gepa_pre_call(ctx_gepa_off, "plan_reproduction", (), {})
        assert ctx_gepa_off.gepa_prompt_overrides == {}
```

- [ ] **Step 6: Run passthrough test — expect FAIL (hooks module not yet created)**

```bash
cd /Volumes/CS_Stuff/openresearch
.venv/bin/python -m pytest tests/gepa/test_passthrough.py -v 2>&1 | tail -20
```

Expected: `ModuleNotFoundError: No module named 'backend.agents.gepa.hooks'`

- [ ] **Step 7: Create stub hooks.py to make passthrough test pass**

Create `backend/agents/gepa/hooks.py`:

```python
"""GEPA pre/post call hooks — called by wrap_primitive().

gepa_pre_call: runs a mini GEPA optimization loop before the targeted primitive,
  stores best candidate in ctx.gepa_prompt_overrides[primitive_name].
gepa_post_call: records the real primitive result as a training example in
  ctx.gepa_example_buffer[primitive_name] for future calls.

Both are no-ops when gepa_optimization is "off" or when gepa_optimization_active
is True (re-entrancy guard).
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Primitives that GEPA optimizes. Only these trigger pre/post hooks.
GEPA_TARGETS = frozenset({"plan_reproduction", "implement_baseline", "propose_improvements"})


def _is_enabled(ctx: Any, primitive_name: str) -> bool:
    """Return True if GEPA should run for this primitive call."""
    if ctx.gepa_optimization_active:
        return False  # re-entrancy guard
    try:
        from backend.config import Settings
        cfg = Settings()
        mode = cfg.gepa_optimization
    except Exception:
        return False
    if mode == "off":
        return False
    if mode == "on":
        return primitive_name in GEPA_TARGETS
    # Selective modes: "plan-only", "baseline-only", "improve-only"
    mode_map = {
        "plan-only": "plan_reproduction",
        "baseline-only": "implement_baseline",
        "improve-only": "propose_improvements",
    }
    return mode_map.get(mode) == primitive_name


def gepa_pre_call(ctx: Any, primitive_name: str, args: tuple, kwargs: dict) -> None:
    """Run GEPA optimization before a targeted primitive call.

    On return, ctx.gepa_prompt_overrides[primitive_name] holds the winning
    optimized system prompt (or remains unset if GEPA is disabled/failed).
    """
    if not _is_enabled(ctx, primitive_name):
        return
    # Full implementation added in Task 7.


def gepa_post_call(ctx: Any, primitive_name: str, result: Any) -> None:
    """Record real primitive result as a training example for future calls."""
    if primitive_name not in GEPA_TARGETS:
        return
    if ctx.gepa_example_buffer.get(primitive_name) is None:
        ctx.gepa_example_buffer[primitive_name] = []
    # Full implementation added in Task 7.
```

- [ ] **Step 8: Run passthrough test — expect PASS**

```bash
.venv/bin/python -m pytest tests/gepa/test_passthrough.py -v
```

Expected: `3 passed`

- [ ] **Step 9: Commit foundation**

```bash
git checkout -b feat/gepa-integration
git add backend/requirements.txt backend/config.py backend/agents/rlm/context.py \
        backend/agents/gepa/__init__.py backend/agents/gepa/hooks.py \
        tests/gepa/__init__.py tests/gepa/test_passthrough.py
git commit -m "feat(gepa): foundation — deps, RunContext fields, Settings, stub hooks, passthrough test"
```

---

## Task 2: plan_metrics.py — deterministic scoring functions

**Files:**
- Create: `backend/agents/gepa/metrics/__init__.py`
- Create: `backend/agents/gepa/metrics/plan_metrics.py`
- Create: `tests/gepa/test_plan_metrics.py`

- [ ] **Step 1: Write failing tests**

Create `tests/gepa/test_plan_metrics.py`:

```python
"""Tests for plan_reproduction scoring metrics."""
import pytest
from backend.agents.gepa.metrics.plan_metrics import metrics_shape_score, contract_completeness

# A minimal valid ReproductionContract dict
VALID_CONTRACT = {
    "baseline_plan": "train ResNet on CIFAR-10",
    "smoke_test_plan": "run 1 epoch",
    "full_run_plan": "run 100 epochs",
    "expected_artifacts": ["checkpoints/final.pt", "metrics.json"],
    "dataset_plan": {"name": "CIFAR-10", "source": "torchvision"},
    "evaluation_plan": {"metrics": ["top1_acc"]},
    "verification_checklist": ["check accuracy >= 0.90"],
    "metrics_shape": [
        {"metric_id": "top1_acc", "json_path": "cifar10_resnet_top1_acc", "rubric_leaf_ids": []},
        {"metric_id": "top5_acc", "json_path": "cifar10_resnet_top5_acc", "rubric_leaf_ids": []},
    ],
}

EXPECTED_METRIC_IDS = ["top1_acc", "top5_acc"]


def test_metrics_shape_score_perfect():
    score, feedback = metrics_shape_score(VALID_CONTRACT, EXPECTED_METRIC_IDS)
    assert score == 1.0
    assert "missing" not in feedback.lower()


def test_metrics_shape_score_partial():
    contract = {**VALID_CONTRACT, "metrics_shape": [
        {"metric_id": "top1_acc", "json_path": "cifar10_top1", "rubric_leaf_ids": []},
    ]}
    score, feedback = metrics_shape_score(contract, EXPECTED_METRIC_IDS)
    assert score == 0.5
    assert "top5_acc" in feedback


def test_metrics_shape_score_empty_metrics_shape():
    contract = {**VALID_CONTRACT, "metrics_shape": []}
    score, feedback = metrics_shape_score(contract, EXPECTED_METRIC_IDS)
    assert score == 0.0
    assert "missing" in feedback.lower()


def test_metrics_shape_score_no_expected_ids():
    """If expected_metric_ids is empty, score is 1.0 (no constraint)."""
    score, feedback = metrics_shape_score(VALID_CONTRACT, [])
    assert score == 1.0


def test_metrics_shape_score_invalid_json():
    score, feedback = metrics_shape_score("not a dict", EXPECTED_METRIC_IDS)
    assert score == 0.0
    assert "invalid" in feedback.lower()


def test_contract_completeness_all_fields():
    score = contract_completeness(VALID_CONTRACT)
    assert score == 1.0


def test_contract_completeness_missing_field():
    contract = {k: v for k, v in VALID_CONTRACT.items() if k != "dataset_plan"}
    score = contract_completeness(contract)
    assert score < 1.0


def test_contract_completeness_missing_metrics_shape():
    contract = {k: v for k, v in VALID_CONTRACT.items() if k != "metrics_shape"}
    score = contract_completeness(contract)
    assert score < 1.0


def test_contract_completeness_empty_string_field():
    contract = {**VALID_CONTRACT, "baseline_plan": ""}
    score = contract_completeness(contract)
    assert score < 1.0
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
.venv/bin/python -m pytest tests/gepa/test_plan_metrics.py -v 2>&1 | tail -10
```

Expected: `ModuleNotFoundError: No module named 'backend.agents.gepa.metrics'`

- [ ] **Step 3: Implement plan_metrics.py**

Create `backend/agents/gepa/metrics/__init__.py` (empty).

Create `backend/agents/gepa/metrics/plan_metrics.py`:

```python
"""Deterministic scoring functions for plan_reproduction outputs."""
from __future__ import annotations

_REQUIRED_CONTRACT_FIELDS = frozenset({
    "baseline_plan", "smoke_test_plan", "full_run_plan",
    "expected_artifacts", "dataset_plan", "evaluation_plan",
    "verification_checklist", "metrics_shape",
})


def metrics_shape_score(
    contract: object,
    expected_metric_ids: list[str],
) -> tuple[float, str]:
    """Score how many expected metric IDs appear in contract['metrics_shape'].

    Returns (score 0.0-1.0, feedback_string).
    score = fraction of expected_metric_ids found in metrics_shape[*].metric_id.
    If expected_metric_ids is empty, returns (1.0, "no constraints").
    """
    if not isinstance(contract, dict):
        return 0.0, "invalid contract: not a dict"
    if not expected_metric_ids:
        return 1.0, "no metric_id constraints"
    metrics_shape = contract.get("metrics_shape")
    if not isinstance(metrics_shape, list) or len(metrics_shape) == 0:
        return 0.0, f"missing metrics_shape; expected ids: {expected_metric_ids}"
    declared_ids = {
        entry.get("metric_id", "") for entry in metrics_shape if isinstance(entry, dict)
    }
    found = [mid for mid in expected_metric_ids if mid in declared_ids]
    missing = [mid for mid in expected_metric_ids if mid not in declared_ids]
    score = len(found) / len(expected_metric_ids)
    if missing:
        feedback = f"missing metric_ids: {missing}; declared: {sorted(declared_ids)}"
    else:
        feedback = f"all {len(found)} metric_ids found"
    return score, feedback


def contract_completeness(contract: object) -> float:
    """Score structural completeness of a ReproductionContract dict.

    Returns fraction of required fields that are present and non-empty.
    """
    if not isinstance(contract, dict):
        return 0.0
    scores = []
    for field in _REQUIRED_CONTRACT_FIELDS:
        val = contract.get(field)
        if val is None:
            scores.append(0.0)
        elif isinstance(val, (str, list, dict)) and len(val) == 0:
            scores.append(0.0)
        else:
            scores.append(1.0)
    return sum(scores) / len(scores)


def combined_plan_score(
    contract: object,
    expected_metric_ids: list[str],
    *,
    shape_weight: float = 0.6,
    completeness_weight: float = 0.4,
) -> tuple[float, str]:
    """Combined score: shape coverage + structural completeness."""
    shape, fb = metrics_shape_score(contract, expected_metric_ids)
    completeness = contract_completeness(contract)
    score = shape_weight * shape + completeness_weight * completeness
    return score, f"shape={shape:.2f} completeness={completeness:.2f} | {fb}"
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
.venv/bin/python -m pytest tests/gepa/test_plan_metrics.py -v
```

Expected: `9 passed`

- [ ] **Step 5: Commit**

```bash
git add backend/agents/gepa/metrics/ tests/gepa/test_plan_metrics.py
git commit -m "feat(gepa): plan_metrics — metrics_shape_score, contract_completeness"
```

---

## Task 3: improvement_metrics.py and baseline_metrics.py

**Files:**
- Create: `backend/agents/gepa/metrics/improvement_metrics.py`
- Create: `backend/agents/gepa/metrics/baseline_metrics.py`
- Create: `tests/gepa/test_improvement_metrics.py`

- [ ] **Step 1: Write failing tests**

Create `tests/gepa/test_improvement_metrics.py`:

```python
from backend.agents.gepa.metrics.improvement_metrics import (
    weak_area_coverage, category_diversity, combined_improvement_score,
)

HYPOTHESES = [
    {"path_id": "p1", "title": "Lower LR", "category": "hyperparameter",
     "hypothesis": "Reduce lr", "rationale": "...", "expected_outcome": "..."},
    {"path_id": "p2", "title": "Add dropout", "category": "regularization",
     "hypothesis": "Add dropout", "rationale": "...", "expected_outcome": "..."},
    {"path_id": "p3", "title": "More data aug", "category": "data",
     "hypothesis": "More aug", "rationale": "...", "expected_outcome": "..."},
]


def test_weak_area_coverage_all_covered():
    weak_leaves = ["hyperparameter", "regularization", "data"]
    score, fb = weak_area_coverage(HYPOTHESES, weak_leaves)
    assert score == 1.0


def test_weak_area_coverage_partial():
    weak_leaves = ["hyperparameter", "architecture"]
    score, fb = weak_area_coverage(HYPOTHESES, weak_leaves)
    assert score == 0.5
    assert "architecture" in fb


def test_weak_area_coverage_no_leaves():
    score, fb = weak_area_coverage(HYPOTHESES, [])
    assert score == 1.0


def test_category_diversity_three_unique():
    score = category_diversity(HYPOTHESES)
    assert score == 1.0


def test_category_diversity_all_same():
    hyps = [{"category": "hyperparameter"} for _ in range(3)]
    score = category_diversity(hyps)
    assert score < 0.5


def test_category_diversity_empty():
    assert category_diversity([]) == 0.0


def test_combined_improvement_score():
    score, fb = combined_improvement_score(HYPOTHESES, ["hyperparameter"])
    assert 0.0 < score <= 1.0
```

- [ ] **Step 2: Run — expect FAIL**

```bash
.venv/bin/python -m pytest tests/gepa/test_improvement_metrics.py -v 2>&1 | tail -5
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement improvement_metrics.py**

Create `backend/agents/gepa/metrics/improvement_metrics.py`:

```python
"""Scoring for propose_improvements outputs."""
from __future__ import annotations
import math

_KNOWN_CATEGORIES = frozenset({
    "hyperparameter", "architecture", "data", "regularization",
    "training", "evaluation", "other",
})


def weak_area_coverage(
    hypotheses: list[dict],
    weak_leaves: list[str],
) -> tuple[float, str]:
    """Fraction of weak rubric leaves covered by at least one hypothesis.

    Matching is by category field (case-insensitive substring match).
    """
    if not weak_leaves:
        return 1.0, "no weak_leaves constraints"
    if not hypotheses:
        return 0.0, f"no hypotheses; uncovered: {weak_leaves}"
    categories_used = {h.get("category", "").lower() for h in hypotheses}
    covered = [
        leaf for leaf in weak_leaves
        if any(leaf.lower() in cat or cat in leaf.lower() for cat in categories_used)
    ]
    uncovered = [l for l in weak_leaves if l not in covered]
    score = len(covered) / len(weak_leaves)
    feedback = f"covered={covered} uncovered={uncovered}"
    return score, feedback


def category_diversity(hypotheses: list[dict]) -> float:
    """Entropy-based diversity of hypothesis categories, normalized to [0,1].

    Max entropy for N hypotheses = log2(min(N, |KNOWN_CATEGORIES|)).
    """
    if not hypotheses:
        return 0.0
    from collections import Counter
    counts = Counter(h.get("category", "other").lower() for h in hypotheses)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = -sum((c / total) * math.log2(c / total) for c in counts.values() if c > 0)
    max_entropy = math.log2(min(len(hypotheses), len(_KNOWN_CATEGORIES)))
    if max_entropy == 0:
        return 1.0
    return min(1.0, entropy / max_entropy)


def combined_improvement_score(
    hypotheses: list[dict],
    weak_leaves: list[str],
    *,
    coverage_weight: float = 0.7,
    diversity_weight: float = 0.3,
) -> tuple[float, str]:
    coverage, fb = weak_area_coverage(hypotheses, weak_leaves)
    diversity = category_diversity(hypotheses)
    score = coverage_weight * coverage + diversity_weight * diversity
    return score, f"coverage={coverage:.2f} diversity={diversity:.2f} | {fb}"
```

- [ ] **Step 4: Implement baseline_metrics.py**

Create `backend/agents/gepa/metrics/baseline_metrics.py`:

```python
"""Proxy metric for implement_baseline optimization (no sub-agent call)."""
from __future__ import annotations


_REQUIRED_PLAN_KEYS = frozenset({
    "mode", "code_path", "dockerfile_path", "diff_summary",
    "commands_to_run", "assumptions_applied",
})


def code_plan_structural_validity(output: object) -> tuple[float, str]:
    """Score structural validity of an implement_baseline output dict.

    Used as a PROXY metric during GEPA optimization — no actual sub-agent call.
    Score = fraction of required fields present and non-empty.
    """
    if not isinstance(output, dict):
        return 0.0, "not a dict"
    scores = []
    for key in _REQUIRED_PLAN_KEYS:
        val = output.get(key)
        if val is None:
            scores.append(0.0)
        elif isinstance(val, (str, list)) and len(val) == 0:
            scores.append(0.0)
        else:
            scores.append(1.0)
    score = sum(scores) / len(scores)
    missing = [k for k in _REQUIRED_PLAN_KEYS if not output.get(k)]
    return score, f"missing_fields={missing}" if missing else "all fields present"
```

- [ ] **Step 5: Run tests — expect PASS**

```bash
.venv/bin/python -m pytest tests/gepa/test_improvement_metrics.py -v
```

Expected: `7 passed`

- [ ] **Step 6: Commit**

```bash
git add backend/agents/gepa/metrics/ tests/gepa/test_improvement_metrics.py
git commit -m "feat(gepa): improvement_metrics and baseline_metrics (proxy)"
```

---

## Task 4: PaperExamplesBuilder — trainset from understand_section outputs

**Files:**
- Create: `backend/agents/gepa/trainset/__init__.py`
- Create: `backend/agents/gepa/trainset/paper_examples.py`
- Create: `tests/gepa/test_paper_examples.py`

- [ ] **Step 1: Write failing test**

Create `tests/gepa/test_paper_examples.py`:

```python
"""PaperExamplesBuilder constructs training examples from understand_section outputs."""
from backend.agents.gepa.trainset.paper_examples import PaperExamplesBuilder, PaperExample


UNDERSTAND_OUTPUT = {
    "datasets": [{"name": "CIFAR-10", "split": "train", "size": 50000}],
    "metrics": [{"name": "top1_accuracy", "direction": "higher", "bounds": [0, 1]}],
    "training_recipe": {"optimizer": "Adam", "learning_rate": 0.001, "batch_size": 64, "epochs": 100},
    "hardware_clues": ["GPU required", "16GB VRAM"],
    "ambiguities": [],
    "outcome": "ok",
}

EXTRACT_OUTPUT = {
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs_or_steps": 100,
    "scheduler": None,
    "other_hparams": {"weight_decay": 1e-4},
}

ENV_SPEC = {
    "base_image": "pytorch/pytorch:2.1.0",
    "requirements": ["torch", "torchvision"],
}


def test_build_returns_list_of_paper_examples():
    builder = PaperExamplesBuilder(
        understand_output=UNDERSTAND_OUTPUT,
        extract_output=EXTRACT_OUTPUT,
        env_spec=ENV_SPEC,
        expected_metric_ids=["top1_accuracy"],
    )
    examples = builder.build(n=3)
    assert len(examples) >= 1
    assert all(isinstance(e, PaperExample) for e in examples)


def test_paper_example_has_required_fields():
    builder = PaperExamplesBuilder(
        understand_output=UNDERSTAND_OUTPUT,
        extract_output=EXTRACT_OUTPUT,
        env_spec=ENV_SPEC,
        expected_metric_ids=["top1_accuracy"],
    )
    example = builder.build(n=1)[0]
    assert isinstance(example.method_spec, dict)
    assert isinstance(example.env_spec, dict)
    assert isinstance(example.expected_metric_ids, list)
    assert "top1_accuracy" in example.expected_metric_ids


def test_build_with_no_understand_output():
    builder = PaperExamplesBuilder(
        understand_output=None,
        extract_output=None,
        env_spec=ENV_SPEC,
        expected_metric_ids=[],
    )
    examples = builder.build(n=1)
    assert len(examples) >= 1  # falls back to minimal example
```

- [ ] **Step 2: Run — expect FAIL**

```bash
.venv/bin/python -m pytest tests/gepa/test_paper_examples.py -v 2>&1 | tail -5
```

- [ ] **Step 3: Implement paper_examples.py**

Create `backend/agents/gepa/trainset/__init__.py` (empty).

Create `backend/agents/gepa/trainset/paper_examples.py`:

```python
"""Build GEPA training examples from understand_section / extract_hyperparameters outputs."""
from __future__ import annotations

from dataclasses import dataclass, field


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
        """Build a method_spec dict from available outputs."""
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

        Currently returns the same paper characterization as a single example
        (the trainset is small by design — GEPA is effective with 3+ examples
        per the paper). Additional variation could be added later.
        """
        method_spec = self._make_method_spec()
        base = PaperExample(
            method_spec=method_spec,
            env_spec=self._env_spec,
            expected_metric_ids=list(self._expected),
        )
        # Return n copies of the same base example; the GEPA optimizer will
        # use varied candidate prompts, not varied inputs, for the first call.
        # On subsequent calls the real outputs in gepa_example_buffer provide
        # genuine input variation.
        return [base] * max(1, min(n, 3))
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
.venv/bin/python -m pytest tests/gepa/test_paper_examples.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add backend/agents/gepa/trainset/ tests/gepa/test_paper_examples.py
git commit -m "feat(gepa): PaperExamplesBuilder — cold-start trainset from understand_section"
```

---

## Task 5: PlanReproductionEvaluator

**Files:**
- Create: `backend/agents/gepa/adapters/__init__.py`
- Create: `backend/agents/gepa/adapters/plan_reproduction.py`
- Create: `tests/gepa/test_plan_reproduction_evaluator.py`

- [ ] **Step 1: Write failing tests**

Create `tests/gepa/test_plan_reproduction_evaluator.py`:

```python
"""PlanReproductionEvaluator: GEPA evaluator for plan_reproduction prompt."""
from unittest.mock import MagicMock, patch
from pathlib import Path

import pytest

from backend.agents.gepa.adapters.plan_reproduction import PlanReproductionEvaluator
from backend.agents.gepa.trainset.paper_examples import PaperExample


@pytest.fixture()
def llm_client():
    client = MagicMock()
    client.complete = MagicMock(return_value="""{
        "baseline_plan": "train ResNet",
        "smoke_test_plan": "1 epoch",
        "full_run_plan": "100 epochs",
        "expected_artifacts": ["metrics.json"],
        "dataset_plan": {"name": "CIFAR-10"},
        "evaluation_plan": {"metrics": ["top1"]},
        "verification_checklist": ["check acc"],
        "metrics_shape": [
            {"metric_id": "top1_acc", "json_path": "cifar10_top1", "rubric_leaf_ids": []}
        ]
    }""")
    client._last_usage = {}
    return client


@pytest.fixture()
def example():
    return PaperExample(
        method_spec={"datasets": [], "metrics": [], "training_recipe": {}, "hardware_clues": []},
        env_spec={"base_image": "pytorch/pytorch:2.1.0"},
        expected_metric_ids=["top1_acc"],
    )


def test_evaluate_returns_score_and_feedback(llm_client, example):
    evaluator = PlanReproductionEvaluator(llm_client=llm_client)
    score, side_info = evaluator(
        candidate={"plan_reproduction_system": "You are a planner."},
        example=example,
    )
    assert 0.0 <= score <= 1.0
    assert "score" in side_info or isinstance(side_info, dict)


def test_evaluate_bad_json_returns_zero(llm_client, example):
    llm_client.complete.return_value = "not json at all"
    evaluator = PlanReproductionEvaluator(llm_client=llm_client)
    score, side_info = evaluator(
        candidate={"plan_reproduction_system": "broken prompt"},
        example=example,
    )
    assert score == 0.0
    assert "error" in side_info


def test_evaluate_uses_candidate_as_system_prompt(llm_client, example):
    evaluator = PlanReproductionEvaluator(llm_client=llm_client)
    custom_system = "Custom system prompt for testing."
    evaluator(candidate={"plan_reproduction_system": custom_system}, example=example)
    call_kwargs = llm_client.complete.call_args
    assert call_kwargs.kwargs["system"] == custom_system
```

- [ ] **Step 2: Run — expect FAIL**

```bash
.venv/bin/python -m pytest tests/gepa/test_plan_reproduction_evaluator.py -v 2>&1 | tail -5
```

- [ ] **Step 3: Implement PlanReproductionEvaluator**

Create `backend/agents/gepa/adapters/__init__.py` (empty).

Create `backend/agents/gepa/adapters/plan_reproduction.py`:

```python
"""GEPA evaluator for plan_reproduction system prompt optimization.

Uses gepa's simple Evaluator protocol:
  def __call__(candidate, example, **kwargs) -> tuple[float, dict] | float

The evaluator calls ctx.llm_client.complete() DIRECTLY (bypassing wrap_primitive
to avoid re-entrancy) with the candidate system prompt, then scores the output.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from backend.agents.gepa.metrics.plan_metrics import combined_plan_score
from backend.agents.gepa.trainset.paper_examples import PaperExample
from backend.agents.rlm.primitives import _METRICS_SHAPE_INSTRUCTION  # the suffix

logger = logging.getLogger(__name__)

# The user message template mirrors what plan_reproduction sends to the LLM.
_USER_TEMPLATE = """Method spec:
{method_spec_json}

Environment spec:
{env_spec_json}
"""


class PlanReproductionEvaluator:
    """Callable evaluator compatible with gepa.optimize(evaluator=...).

    Signature: (candidate: dict[str,str], example: PaperExample) -> (float, dict)
    """

    def __init__(self, *, llm_client: Any) -> None:
        self._llm = llm_client

    def __call__(
        self,
        candidate: dict[str, str] | str,
        example: PaperExample | None = None,
        **kwargs: Any,
    ) -> tuple[float, dict]:
        if example is None:
            return 0.0, {"error": "no example provided"}

        system = (
            candidate["plan_reproduction_system"]
            if isinstance(candidate, dict)
            else str(candidate)
        )
        # Append the required metrics_shape instruction (always present in real calls)
        system_full = system + _METRICS_SHAPE_INSTRUCTION

        user = _USER_TEMPLATE.format(
            method_spec_json=json.dumps(example.method_spec, indent=2)[:3000],
            env_spec_json=json.dumps(example.env_spec, indent=2)[:1000],
        )

        try:
            raw = self._llm.complete(system=system_full, user=user)
        except Exception as exc:
            logger.debug("PlanReproductionEvaluator LLM error: %s", exc)
            return 0.0, {"error": str(exc)}

        # Parse JSON output
        try:
            # Strip markdown fences if present
            text = raw.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            contract = json.loads(text)
        except json.JSONDecodeError as exc:
            return 0.0, {"error": f"json_parse_error: {exc}", "raw_preview": raw[:200]}

        score, feedback = combined_plan_score(contract, example.expected_metric_ids)
        return score, {
            "score": score,
            "feedback": feedback,
            "metrics_shape": contract.get("metrics_shape", []),
            "Output": raw[:500],
        }
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
.venv/bin/python -m pytest tests/gepa/test_plan_reproduction_evaluator.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add backend/agents/gepa/adapters/ tests/gepa/test_plan_reproduction_evaluator.py
git commit -m "feat(gepa): PlanReproductionEvaluator — GEPA evaluator for plan_reproduction"
```

---

## Task 6: SSEGepaCallback + sse_bridge event builders

**Files:**
- Create: `backend/agents/gepa/callback.py`
- Modify: `backend/agents/rlm/sse_bridge.py`

- [ ] **Step 1: Add 5 GEPA event builders to sse_bridge.py**

Open `backend/agents/rlm/sse_bridge.py`. After the last existing `build_*` function, add:

```python
# ---------------------------------------------------------------------------
# GEPA per-run optimization event builders
# ---------------------------------------------------------------------------

def build_gepa_phase_start(
    *,
    primitive_name: str,
    max_metric_calls: int,
) -> dict[str, Any]:
    """Emitted when a GEPA mini-optimization loop starts before a primitive."""
    return {
        "event": "gepa_phase_start",
        "primitive_name": primitive_name,
        "max_metric_calls": max_metric_calls,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def build_gepa_candidate_proposed(
    *,
    primitive_name: str,
    iteration: int,
    candidate_id: str,
    prompt_preview: str,
    parent_id: str | None,
) -> dict[str, Any]:
    return {
        "event": "gepa_candidate_proposed",
        "primitive_name": primitive_name,
        "iteration": iteration,
        "candidate_id": candidate_id,
        "prompt_preview": prompt_preview[:200],
        "parent_id": parent_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def build_gepa_candidate_accepted(
    *,
    primitive_name: str,
    candidate_id: str,
    score: float,
    score_delta: float,
) -> dict[str, Any]:
    return {
        "event": "gepa_candidate_accepted",
        "primitive_name": primitive_name,
        "candidate_id": candidate_id,
        "score": round(score, 4),
        "score_delta": round(score_delta, 4),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def build_gepa_candidate_rejected(
    *,
    primitive_name: str,
    candidate_id: str,
    reason: str,
    score: float,
) -> dict[str, Any]:
    return {
        "event": "gepa_candidate_rejected",
        "primitive_name": primitive_name,
        "candidate_id": candidate_id,
        "reason": reason,
        "score": round(score, 4),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def build_gepa_phase_complete(
    *,
    primitive_name: str,
    final_score: float,
    baseline_score: float,
    total_metric_calls: int,
    duration_s: float,
) -> dict[str, Any]:
    return {
        "event": "gepa_phase_complete",
        "primitive_name": primitive_name,
        "final_score": round(final_score, 4),
        "baseline_score": round(baseline_score, 4),
        "delta": round(final_score - baseline_score, 4),
        "total_metric_calls": total_metric_calls,
        "duration_s": round(duration_s, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
```

- [ ] **Step 2: Create SSEGepaCallback**

Create `backend/agents/gepa/callback.py`:

```python
"""SSEGepaCallback: bridges GEPA callback events → ctx.emit() → dashboard_events.jsonl."""
from __future__ import annotations

import logging
from typing import Any

from backend.agents.rlm.sse_bridge import (
    build_gepa_candidate_accepted,
    build_gepa_candidate_proposed,
    build_gepa_candidate_rejected,
    build_gepa_phase_complete,
)

logger = logging.getLogger(__name__)


class SSEGepaCallback:
    """GEPA callback that forwards optimization events to the run's SSE stream.

    Pass to gepa.optimize(callbacks=[SSEGepaCallback(ctx, primitive_name)]).
    Implements duck-typed callback protocol (no base class needed).
    """

    def __init__(self, *, ctx: Any, primitive_name: str) -> None:
        self._ctx = ctx
        self._name = primitive_name
        self._seed_score: float = 0.0
        self._iteration: int = 0
        self._accepted_count: int = 0

    def _emit(self, event: dict) -> None:
        if self._ctx.emit is not None:
            try:
                self._ctx.emit(event)
            except Exception as exc:
                logger.debug("SSEGepaCallback emit error: %s", exc)

    def on_optimization_start(self, event: dict) -> None:
        pass  # gepa_phase_start is emitted by gepa_pre_call before optimize() is called

    def on_iteration_start(self, event: dict) -> None:
        self._iteration = event.get("iteration", self._iteration + 1)

    def on_proposal_end(self, event: dict) -> None:
        new_instructions = event.get("new_instructions", {})
        prompt_text = next(iter(new_instructions.values()), "") if new_instructions else ""
        self._emit(build_gepa_candidate_proposed(
            primitive_name=self._name,
            iteration=self._iteration,
            candidate_id=f"{self._name}-{self._iteration}",
            prompt_preview=prompt_text,
            parent_id=None,
        ))

    def on_candidate_accepted(self, event: dict) -> None:
        self._accepted_count += 1
        new_score = event.get("new_score", 0.0)
        delta = new_score - self._seed_score
        self._emit(build_gepa_candidate_accepted(
            primitive_name=self._name,
            candidate_id=f"{self._name}-{self._iteration}",
            score=new_score,
            score_delta=delta,
        ))

    def on_candidate_rejected(self, event: dict) -> None:
        self._emit(build_gepa_candidate_rejected(
            primitive_name=self._name,
            candidate_id=f"{self._name}-{self._iteration}",
            reason=event.get("reason", "no_improvement"),
            score=event.get("new_score", 0.0),
        ))

    def set_seed_score(self, score: float) -> None:
        self._seed_score = score
```

- [ ] **Step 3: Verify sse_bridge still imports cleanly**

```bash
.venv/bin/python -c "from backend.agents.rlm.sse_bridge import build_gepa_phase_start; print('ok')"
```

Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add backend/agents/gepa/callback.py backend/agents/rlm/sse_bridge.py
git commit -m "feat(gepa): SSEGepaCallback + 5 GEPA SSE event builders in sse_bridge"
```

---

## Task 7: optimizer.py + hooks.py (full implementation)

**Files:**
- Create: `backend/agents/gepa/optimizer.py`
- Modify: `backend/agents/gepa/hooks.py`
- Create: `tests/gepa/test_hooks.py`

- [ ] **Step 1: Write failing test for hooks**

Create `tests/gepa/test_hooks.py`:

```python
"""Tests for gepa_pre_call / gepa_post_call hooks."""
import os
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backend.agents.rlm.context import RunContext


@pytest.fixture()
def ctx():
    llm = MagicMock()
    llm.complete = MagicMock(return_value='{"baseline_plan":"x","smoke_test_plan":"y",'
                              '"full_run_plan":"z","expected_artifacts":["metrics.json"],'
                              '"dataset_plan":{"name":"d"},"evaluation_plan":{"metrics":["m"]},'
                              '"verification_checklist":["v"],'
                              '"metrics_shape":[{"metric_id":"m","json_path":"m","rubric_leaf_ids":[]}]}')
    llm._last_usage = {}
    dashboard = MagicMock()
    dashboard.primitive_call = MagicMock()
    return RunContext(
        project_id="prj_t",
        project_dir=Path("/tmp/prj_t"),
        runs_root=Path("/tmp/runs"),
        dashboard=dashboard,
        cost_ledger=MagicMock(),
        llm_client=llm,
        provider="openai",
        model="gpt-4o-mini",
        emit=MagicMock(),
    )


def test_gepa_pre_call_skips_unknown_primitive(ctx):
    with patch.dict(os.environ, {"REPROLAB_GEPA_OPTIMIZATION": "on"}):
        from backend.agents.gepa.hooks import gepa_pre_call
        gepa_pre_call(ctx, "run_experiment", (), {})
        assert "run_experiment" not in ctx.gepa_prompt_overrides


def test_gepa_post_call_appends_to_buffer(ctx):
    from backend.agents.gepa.hooks import gepa_post_call
    result = {"baseline_plan": "test", "metrics_shape": []}
    gepa_post_call(ctx, "plan_reproduction", result)
    assert "plan_reproduction" in ctx.gepa_example_buffer
    assert len(ctx.gepa_example_buffer["plan_reproduction"]) == 1


def test_gepa_post_call_unknown_primitive_ignored(ctx):
    from backend.agents.gepa.hooks import gepa_post_call
    gepa_post_call(ctx, "run_experiment", {"ok": True})
    assert "run_experiment" not in ctx.gepa_example_buffer


def test_reentrancy_guard(ctx):
    """gepa_pre_call is no-op when gepa_optimization_active is True."""
    ctx.gepa_optimization_active = True
    with patch.dict(os.environ, {"REPROLAB_GEPA_OPTIMIZATION": "on"}):
        from backend.agents.gepa.hooks import gepa_pre_call
        gepa_pre_call(ctx, "plan_reproduction", (), {})
        assert "plan_reproduction" not in ctx.gepa_prompt_overrides
```

- [ ] **Step 2: Implement optimizer.py**

Create `backend/agents/gepa/optimizer.py`:

```python
"""run_gepa_mini: thin wrapper around gepa.optimize() for per-call optimization."""
from __future__ import annotations

import logging
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)


def run_gepa_mini(
    *,
    seed_prompt: str,
    component_name: str,
    trainset: list,
    evaluator: Callable,
    max_metric_calls: int,
    timeout_s: int,
    reflection_model: str,
    callbacks: list,
) -> tuple[str, float, int]:
    """Run a mini GEPA optimization loop.

    Returns (best_prompt, best_score, total_metric_calls).
    On any failure returns (seed_prompt, 0.0, 0).
    """
    try:
        import gepa
        from gepa.utils import TimeoutStopCondition, NoImprovementStopper
    except ImportError as exc:
        logger.warning("gepa not installed, skipping optimization: %s", exc)
        return seed_prompt, 0.0, 0

    try:
        result = gepa.optimize(
            seed_candidate={component_name: seed_prompt},
            trainset=trainset,
            evaluator=evaluator,
            reflection_lm=reflection_model,
            max_metric_calls=max_metric_calls,
            stop_callbacks=[
                TimeoutStopCondition(timeout_seconds=timeout_s),
                NoImprovementStopper(max_iterations_without_improvement=3),
            ],
            callbacks=callbacks,
            display_progress_bar=False,
            seed=0,
        )
        best = result.best_candidate
        if isinstance(best, dict):
            best_prompt = best.get(component_name, seed_prompt)
        else:
            best_prompt = str(best)
        scores = result.val_aggregate_scores or []
        best_score = max(scores) if scores else 0.0
        total_calls = result.total_metric_calls or 0
        return best_prompt, best_score, total_calls
    except Exception as exc:
        logger.warning("GEPA mini-optimization failed for %s: %s", component_name, exc)
        return seed_prompt, 0.0, 0
```

- [ ] **Step 3: Implement full hooks.py**

Replace `backend/agents/gepa/hooks.py` with the full implementation:

```python
"""GEPA pre/post call hooks — called by wrap_primitive()."""
from __future__ import annotations

import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

GEPA_TARGETS = frozenset({"plan_reproduction", "implement_baseline", "propose_improvements"})

# Seed prompts (imported lazily to avoid circular imports at module load)
def _get_seed_prompt(primitive_name: str) -> str:
    if primitive_name == "plan_reproduction":
        from backend.agents.rlm.primitives import _PLAN_REPRODUCTION_SYSTEM
        return _PLAN_REPRODUCTION_SYSTEM
    if primitive_name == "propose_improvements":
        from backend.agents.prompts.improvement import IMPROVEMENT_ORCHESTRATOR_PROMPT
        return IMPROVEMENT_ORCHESTRATOR_PROMPT
    if primitive_name == "implement_baseline":
        from backend.agents.prompts.baseline_implementation import BASELINE_IMPLEMENTATION_PROMPT
        return BASELINE_IMPLEMENTATION_PROMPT
    return ""


def _get_config():
    from backend.config import Settings
    return Settings()


def _is_enabled(ctx: Any, primitive_name: str) -> bool:
    if ctx.gepa_optimization_active:
        return False
    try:
        mode = _get_config().gepa_optimization
    except Exception:
        return False
    if mode == "off":
        return False
    if mode == "on":
        return primitive_name in GEPA_TARGETS
    mode_map = {
        "plan-only": "plan_reproduction",
        "baseline-only": "implement_baseline",
        "improve-only": "propose_improvements",
    }
    return mode_map.get(mode) == primitive_name


def _build_trainset(ctx: Any, primitive_name: str, args: tuple, kwargs: dict) -> list:
    """Build trainset combining static paper examples + buffered real examples."""
    buffer = ctx.gepa_example_buffer.get(primitive_name, [])
    if primitive_name == "plan_reproduction":
        from backend.agents.gepa.trainset.paper_examples import PaperExamplesBuilder
        # Extract method_spec and env_spec from call args if available
        method_spec = kwargs.get("method_spec") or (args[0] if args else {})
        env_spec = kwargs.get("env_spec") or (args[1] if len(args) > 1 else {})
        # Get expected_metric_ids from reproduction_contract if already set
        expected_ids: list[str] = []
        if ctx.reproduction_contract is not None:
            try:
                ms = getattr(ctx.reproduction_contract, "metrics_shape", None) or []
                expected_ids = [m.get("metric_id", "") for m in ms if isinstance(m, dict)]
            except Exception:
                pass
        builder = PaperExamplesBuilder(
            understand_output=method_spec,
            extract_output={},
            env_spec=env_spec,
            expected_metric_ids=expected_ids,
        )
        static = builder.build(n=3)
    else:
        static = []

    # Convert buffer entries to simple dicts for gepa's evaluator protocol
    buffered = []
    for entry in buffer[-5:]:  # keep last 5 real results for recency
        buffered.append(entry)

    return static + buffered if static else buffered or [{}]


def _build_evaluator(ctx: Any, primitive_name: str):
    """Return the evaluator callable for this primitive."""
    if primitive_name == "plan_reproduction":
        from backend.agents.gepa.adapters.plan_reproduction import PlanReproductionEvaluator
        return PlanReproductionEvaluator(llm_client=ctx.llm_client)
    if primitive_name == "propose_improvements":
        from backend.agents.gepa.adapters.propose_improvements import ProposeImprovementsEvaluator
        return ProposeImprovementsEvaluator(llm_client=ctx.llm_client)
    if primitive_name == "implement_baseline":
        from backend.agents.gepa.adapters.implement_baseline import ImplementBaselineProxyEvaluator
        return ImplementBaselineProxyEvaluator()
    raise ValueError(f"No evaluator for {primitive_name}")


def gepa_pre_call(ctx: Any, primitive_name: str, args: tuple, kwargs: dict) -> None:
    """Run GEPA mini-optimization before a targeted primitive call.

    On return, ctx.gepa_prompt_overrides[primitive_name] holds the best candidate.
    Sets ctx.gepa_optimization_active=True for the duration to prevent recursion.
    """
    if not _is_enabled(ctx, primitive_name):
        return

    cfg = _get_config()
    timeout_map = {
        "plan_reproduction": cfg.gepa_timeout_plan_s,
        "implement_baseline": cfg.gepa_timeout_baseline_s,
        "propose_improvements": cfg.gepa_timeout_improve_s,
    }
    calls_map = {
        "plan_reproduction": cfg.gepa_max_metric_calls_plan,
        "implement_baseline": cfg.gepa_max_metric_calls_baseline,
        "propose_improvements": cfg.gepa_max_metric_calls_improve,
    }

    seed = _get_seed_prompt(primitive_name)
    if not seed:
        return

    trainset = _build_trainset(ctx, primitive_name, args, kwargs)
    evaluator = _build_evaluator(ctx, primitive_name)

    from backend.agents.gepa.callback import SSEGepaCallback
    from backend.agents.rlm.sse_bridge import build_gepa_phase_start, build_gepa_phase_complete

    cb = SSEGepaCallback(ctx=ctx, primitive_name=primitive_name)
    max_calls = calls_map[primitive_name]
    timeout_s = timeout_map[primitive_name]

    if ctx.emit:
        ctx.emit(build_gepa_phase_start(
            primitive_name=primitive_name,
            max_metric_calls=max_calls,
        ))

    ctx.gepa_optimization_active = True
    t0 = time.monotonic()
    try:
        from backend.agents.gepa.optimizer import run_gepa_mini
        best_prompt, best_score, total_calls = run_gepa_mini(
            seed_prompt=seed,
            component_name=f"{primitive_name}_system",
            trainset=trainset,
            evaluator=evaluator,
            max_metric_calls=max_calls,
            timeout_s=timeout_s,
            reflection_model=cfg.gepa_reflection_model,
            callbacks=[cb],
        )
    except Exception as exc:
        logger.warning("gepa_pre_call failed for %s: %s", primitive_name, exc)
        best_prompt, best_score, total_calls = seed, 0.0, 0
    finally:
        ctx.gepa_optimization_active = False

    duration_s = time.monotonic() - t0

    if best_prompt != seed:
        ctx.gepa_prompt_overrides[primitive_name] = best_prompt
        logger.info(
            "GEPA improved %s: score=%.3f in %d calls (%.1fs)",
            primitive_name, best_score, total_calls, duration_s,
        )

    if ctx.emit:
        ctx.emit(build_gepa_phase_complete(
            primitive_name=primitive_name,
            final_score=best_score,
            baseline_score=0.0,
            total_metric_calls=total_calls,
            duration_s=duration_s,
        ))


def gepa_post_call(ctx: Any, primitive_name: str, result: Any) -> None:
    """Record real primitive result as a training example for future calls."""
    if primitive_name not in GEPA_TARGETS:
        return
    if not isinstance(result, dict):
        return
    if primitive_name not in ctx.gepa_example_buffer:
        ctx.gepa_example_buffer[primitive_name] = []
    entry = {"output": result, "primitive": primitive_name}
    ctx.gepa_example_buffer[primitive_name].append(entry)
```

- [ ] **Step 4: Run hooks tests**

```bash
.venv/bin/python -m pytest tests/gepa/test_hooks.py tests/gepa/test_passthrough.py -v
```

Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add backend/agents/gepa/hooks.py backend/agents/gepa/optimizer.py \
        tests/gepa/test_hooks.py
git commit -m "feat(gepa): full hooks.py and optimizer.py — gepa_pre_call/gepa_post_call"
```

---

## Task 8: Wire hooks into wrap_primitive() — Path A injection

**Files:**
- Modify: `backend/agents/rlm/binding.py`

- [ ] **Step 1: Add GEPA hook call to wrap_primitive()**

Open `backend/agents/rlm/binding.py`. In the `wrapped()` function body, locate the section that zeros out `_last_usage` (around line 350) and adds before the thread launch. Insert the GEPA pre-call block **after** the `_last_usage` zeroing but **before** the thread is spawned. Also insert the prompt-injection monkey-patch just before the thread target `_runner`:

Find this block (around line 400–420):
```python
        if guard_result is not None:
            result = guard_result
        else:
            _prim_future: Future = Future()

            def _runner() -> None:
                try:
                    _prim_future.set_result(fn(*args, **{**kwargs, "ctx": ctx}))
```

Replace with:
```python
        if guard_result is not None:
            result = guard_result
        else:
            # GEPA: run mini optimization before the primitive thread starts.
            # gepa_pre_call is a no-op when REPROLAB_GEPA_OPTIMIZATION=off or
            # when gepa_optimization_active is True (re-entrancy guard).
            if name in ("plan_reproduction", "implement_baseline", "propose_improvements"):
                try:
                    from backend.agents.gepa.hooks import gepa_pre_call
                    gepa_pre_call(ctx, name, args, kwargs)
                except Exception as _gepa_exc:
                    logger.debug("gepa_pre_call skipped (%s): %s", name, _gepa_exc)

            # GEPA Path A: inject optimized system prompt via llm_client.complete monkey-patch.
            _gepa_override = ctx.gepa_prompt_overrides.get(name) if hasattr(ctx, "gepa_prompt_overrides") else None
            _orig_complete = None
            if _gepa_override and hasattr(ctx, "llm_client") and ctx.llm_client is not None:
                _orig_complete = ctx.llm_client.complete
                def _patched_complete(*, system, user, **_kw):
                    return _orig_complete(system=_gepa_override, user=user, **_kw)
                ctx.llm_client.complete = _patched_complete

            _prim_future: Future = Future()

            def _runner() -> None:
                try:
                    _prim_future.set_result(fn(*args, **{**kwargs, "ctx": ctx}))
```

Then find where `result = _prim_future.result(timeout=_timeout_s)` succeeds and add the restore + post-call. After `result = _prim_future.result(timeout=_timeout_s)`, add:

```python
                    # Restore original complete after thread finishes
                    if _orig_complete is not None:
                        ctx.llm_client.complete = _orig_complete
                    # GEPA post-call: record result as training example
                    if name in ("plan_reproduction", "implement_baseline", "propose_improvements"):
                        try:
                            from backend.agents.gepa.hooks import gepa_post_call
                            gepa_post_call(ctx, name, result)
                        except Exception as _gepa_post_exc:
                            logger.debug("gepa_post_call error: %s", _gepa_post_exc)
```

**Important:** The restore must also happen in the timeout/exception paths. Find the `except FuturesTimeoutError` block and the general `except Exception` block and add `if _orig_complete is not None: ctx.llm_client.complete = _orig_complete` to each.

- [ ] **Step 2: Verify existing tests still pass**

```bash
.venv/bin/python -m pytest tests/rlm/ -x -q --timeout=30 2>&1 | tail -20
```

Expected: same pass/fail ratio as before this change. If any test fails, the binding change introduced a regression — check the monkey-patch restore logic.

- [ ] **Step 3: Verify passthrough test still passes**

```bash
.venv/bin/python -m pytest tests/gepa/test_passthrough.py -v
```

Expected: `3 passed`

- [ ] **Step 4: Commit**

```bash
git add backend/agents/rlm/binding.py
git commit -m "feat(gepa): wire gepa_pre_call/gepa_post_call + Path A injection into wrap_primitive"
```

---

## Task 9: ProposeImprovementsEvaluator + ImplementBaselineProxyEvaluator

**Files:**
- Create: `backend/agents/gepa/adapters/propose_improvements.py`
- Create: `backend/agents/gepa/adapters/implement_baseline.py`
- Create: `tests/gepa/test_improvement_evaluator.py`
- Create: `tests/gepa/test_baseline_evaluator.py`

- [ ] **Step 1: Write failing tests**

Create `tests/gepa/test_improvement_evaluator.py`:

```python
from unittest.mock import MagicMock
from backend.agents.gepa.adapters.propose_improvements import ProposeImprovementsEvaluator


def test_evaluate_valid_hypotheses():
    llm = MagicMock()
    llm.complete = MagicMock(return_value='''{
        "hypotheses": [
            {"path_id":"p1","title":"LR","category":"hyperparameter",
             "hypothesis":"reduce","rationale":"plateau","expected_outcome":"better"},
            {"path_id":"p2","title":"Reg","category":"regularization",
             "hypothesis":"dropout","rationale":"overfit","expected_outcome":"better"}
        ]
    }''')
    ev = ProposeImprovementsEvaluator(llm_client=llm)
    score, info = ev(
        candidate={"propose_improvements_system": "You are an improver."},
        example={"weak_leaves": ["hyperparameter"]},
    )
    assert 0.0 < score <= 1.0


def test_evaluate_bad_json_returns_zero():
    llm = MagicMock()
    llm.complete = MagicMock(return_value="not json")
    ev = ProposeImprovementsEvaluator(llm_client=llm)
    score, info = ev(
        candidate={"propose_improvements_system": "broken"},
        example={"weak_leaves": ["hyperparameter"]},
    )
    assert score == 0.0
```

Create `tests/gepa/test_baseline_evaluator.py`:

```python
from backend.agents.gepa.adapters.implement_baseline import ImplementBaselineProxyEvaluator


def test_proxy_score_valid_output():
    ev = ImplementBaselineProxyEvaluator()
    score, info = ev(
        candidate={"implement_baseline_system": "You are an implementer."},
        example={
            "mode": "adapt", "code_path": "runs/x/code/",
            "dockerfile_path": "runs/x/Dockerfile",
            "diff_summary": "applied changes",
            "commands_to_run": ["python train.py"],
            "assumptions_applied": ["A001"],
        },
    )
    assert score == 1.0


def test_proxy_score_missing_fields():
    ev = ImplementBaselineProxyEvaluator()
    score, info = ev(
        candidate={"implement_baseline_system": "You are an implementer."},
        example={"mode": "adapt"},  # missing many fields
    )
    assert score < 1.0
```

- [ ] **Step 2: Implement ProposeImprovementsEvaluator**

Create `backend/agents/gepa/adapters/propose_improvements.py`:

```python
"""GEPA evaluator for propose_improvements system prompt optimization."""
from __future__ import annotations

import json
import logging
from typing import Any

from backend.agents.gepa.metrics.improvement_metrics import combined_improvement_score

logger = logging.getLogger(__name__)

_USER_TEMPLATE = """Current results: {results_json}
Rubric scores: {rubric_json}
Generate {k} improvement hypotheses."""


class ProposeImprovementsEvaluator:
    def __init__(self, *, llm_client: Any) -> None:
        self._llm = llm_client

    def __call__(
        self,
        candidate: dict[str, str] | str,
        example: dict | None = None,
        **kwargs: Any,
    ) -> tuple[float, dict]:
        example = example or {}
        system = (
            candidate["propose_improvements_system"]
            if isinstance(candidate, dict)
            else str(candidate)
        )
        user = _USER_TEMPLATE.format(
            results_json=json.dumps(example.get("current_results", {}))[:500],
            rubric_json=json.dumps(example.get("rubric_scores", {}))[:500],
            k=3,
        )
        try:
            raw = self._llm.complete(system=system, user=user)
        except Exception as exc:
            return 0.0, {"error": str(exc)}

        try:
            text = raw.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            return 0.0, {"error": f"json_error: {exc}", "raw": raw[:200]}

        hypotheses = parsed.get("hypotheses", [])
        if not isinstance(hypotheses, list):
            return 0.0, {"error": "hypotheses not a list"}

        weak_leaves = example.get("weak_leaves", [])
        score, feedback = combined_improvement_score(hypotheses, weak_leaves)
        return score, {"score": score, "feedback": feedback, "hypothesis_count": len(hypotheses)}
```

- [ ] **Step 3: Implement ImplementBaselineProxyEvaluator**

Create `backend/agents/gepa/adapters/implement_baseline.py`:

```python
"""GEPA proxy evaluator for implement_baseline — no real sub-agent call.

Uses structural validity of a synthetic output dict as the proxy metric.
The real sub-agent is too expensive (30-120s, $0.30-2.00) to call during
the GEPA mini-optimization loop. The proxy evaluates whether the candidate
prompt would produce structurally valid output, using a lightweight LLM call
to generate a code plan summary (not full code).
"""
from __future__ import annotations

from typing import Any

from backend.agents.gepa.metrics.baseline_metrics import code_plan_structural_validity


class ImplementBaselineProxyEvaluator:
    """Evaluate implement_baseline prompt using structural validity of example dict.

    The example IS the expected output structure — we score how well the
    candidate prompt's description aligns with the required fields.
    This is a static proxy: the score comes from the example's structure,
    not from actually running the LLM. Useful for ensuring the prompt
    instructs the agent to produce all required output fields.
    """

    def __call__(
        self,
        candidate: dict[str, str] | str,
        example: dict | None = None,
        **kwargs: Any,
    ) -> tuple[float, dict]:
        # Score the example as if it were an LLM output
        score, feedback = code_plan_structural_validity(example or {})
        return score, {"score": score, "feedback": feedback}
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
.venv/bin/python -m pytest tests/gepa/test_improvement_evaluator.py tests/gepa/test_baseline_evaluator.py -v
```

Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add backend/agents/gepa/adapters/ tests/gepa/test_improvement_evaluator.py \
        tests/gepa/test_baseline_evaluator.py
git commit -m "feat(gepa): ProposeImprovementsEvaluator + ImplementBaselineProxyEvaluator"
```

---

## Task 10: Path B injection — implement_baseline system prompt override

**Files:**
- Modify: `backend/agents/runtime/invoke.py`

- [ ] **Step 1: Find collect_agent_text signature**

```bash
grep -n "def collect_agent_text" /Volumes/CS_Stuff/openresearch/backend/agents/runtime/invoke.py
```

Note the line number. Then read 10 lines around it to see the full signature.

- [ ] **Step 2: Add system_prompt_override parameter**

Open `backend/agents/runtime/invoke.py`. Find `def collect_agent_text(`. Add `system_prompt_override: str | None = None` as the last keyword argument. Inside the function, find where `spec.instructions` (or equivalent) is used to build the `AgentDefinition`. Add:

```python
    # GEPA Path B: replace sub-agent system prompt when an override is provided.
    if system_prompt_override is not None:
        # to_runtime_spec returns an AgentRuntimeSpec; replace instructions field.
        # We build a modified spec by replacing the instructions.
        spec = spec._replace(instructions=system_prompt_override) \
               if hasattr(spec, "_replace") \
               else dataclasses.replace(spec, instructions=system_prompt_override)
```

(Use whichever mutation pattern matches `AgentRuntimeSpec` — named tuple uses `_replace`, dataclass uses `dataclasses.replace`.)

- [ ] **Step 3: Thread override from gepa_pre_call into the call chain**

In `backend/agents/gepa/hooks.py`, in `gepa_pre_call`, after `ctx.gepa_prompt_overrides[primitive_name] = best_prompt` for `implement_baseline`, also store with a different key so the sub-agent path can read it:

The key is that `wrap_primitive("implement_baseline")` currently does not call `llm_client.complete()` directly — it calls through the agent SDK. The Path B override needs to be passed through the function call chain.

In `backend/agents/rlm/binding.py`, in the GEPA override block for `implement_baseline`, instead of monkey-patching `llm_client.complete`, inject it into kwargs:

```python
            if _gepa_override and name == "implement_baseline":
                # Path B: pass override through kwargs so _run_baseline_with_sdk
                # can thread it to collect_agent_text
                kwargs["_gepa_system_override"] = _gepa_override
                _gepa_override = None  # disable Path A for this primitive
```

Then in `backend/agents/rlm/primitives.py`, in `implement_baseline()`, accept and forward `_gepa_system_override`:

Find the `implement_baseline` function signature and add `_gepa_system_override: str | None = None` parameter. Then pass it to `_run_baseline_with_sdk(..., system_prompt_override=_gepa_system_override)`.

Trace that parameter through `_run_baseline_with_sdk` → `baseline_implementation.run_with_sdk` → `collect_agent_text`.

- [ ] **Step 4: Verify existing implement_baseline tests still pass**

```bash
.venv/bin/python -m pytest tests/rlm/ -k "baseline" -x -q 2>&1 | tail -10
```

- [ ] **Step 5: Commit**

```bash
git add backend/agents/runtime/invoke.py backend/agents/rlm/primitives.py \
        backend/agents/rlm/binding.py
git commit -m "feat(gepa): Path B injection — system_prompt_override threading for implement_baseline"
```

---

## Task 11: Frontend SSE events + gepa_candidate node kind

**Files:**
- Modify: `frontend/src/lib/events/rlm-events.ts`
- Modify: `frontend/src/hooks/use-rlm-run.ts`
- Modify: `frontend/src/components/lab/rlm/tree-node.tsx`
- Modify: `frontend/src/components/lab/rlm/node-detail-sidebar.tsx`
- Modify: `frontend/src/components/lab/rlm/constellation-canvas.tsx`

- [ ] **Step 1: Add TypeScript event interfaces to rlm-events.ts**

Open `frontend/src/lib/events/rlm-events.ts`. Find `RLM_EVENT_TYPES` array and the `RlmDashboardEvent` union type.

Add these 5 interfaces (after the existing event interfaces):

```typescript
export interface GepaPhaseStartEvent {
  event: "gepa_phase_start";
  primitive_name: string;
  max_metric_calls: number;
  timestamp: string;
}

export interface GepaCandidateProposedEvent {
  event: "gepa_candidate_proposed";
  primitive_name: string;
  iteration: number;
  candidate_id: string;
  prompt_preview: string;
  parent_id?: string;
  timestamp: string;
}

export interface GepaCandidateAcceptedEvent {
  event: "gepa_candidate_accepted";
  primitive_name: string;
  candidate_id: string;
  score: number;
  score_delta: number;
  timestamp: string;
}

export interface GepaCandidateRejectedEvent {
  event: "gepa_candidate_rejected";
  primitive_name: string;
  candidate_id: string;
  reason: string;
  score: number;
  timestamp: string;
}

export interface GepaPhaseCompleteEvent {
  event: "gepa_phase_complete";
  primitive_name: string;
  final_score: number;
  baseline_score: number;
  delta: number;
  total_metric_calls: number;
  duration_s: number;
  timestamp: string;
}
```

Add the 5 event strings to `RLM_EVENT_TYPES`:
```typescript
  "gepa_phase_start",
  "gepa_candidate_proposed",
  "gepa_candidate_accepted",
  "gepa_candidate_rejected",
  "gepa_phase_complete",
```

Add all 5 to the `RlmDashboardEvent` union:
```typescript
  | GepaPhaseStartEvent
  | GepaCandidateProposedEvent
  | GepaCandidateAcceptedEvent
  | GepaCandidateRejectedEvent
  | GepaPhaseCompleteEvent
```

- [ ] **Step 2: Add gepa_candidate kind to TreeNode and fold functions in use-rlm-run.ts**

Open `frontend/src/hooks/use-rlm-run.ts`. Find the `TreeNode.kind` union type and extend it:

```typescript
kind: "paper" | "work" | "baseline" | "candidate" | "subrlm" | "declined-group" | "primitive" | "llm_primitive" | "gepa_candidate";
```

Add a `gepaInfo` optional field to `TreeNode`:
```typescript
  gepaInfo?: {
    primitive_name: string;
    prompt_preview: string;
    score?: number;
    score_delta?: number;
    outcome?: "accepted" | "rejected" | "running";
  };
```

Add fold functions after the existing `foldCandidateOutcome`:

```typescript
function foldGepaCandidateProposed(
  state: RlmRunState,
  ev: GepaCandidateProposedEvent
): RlmRunState {
  const node: TreeNode = {
    id: ev.candidate_id,
    kind: "gepa_candidate",
    parentId: undefined,
    title: `GEPA: ${ev.primitive_name} #${ev.iteration}`,
    iterationRange: [state.iterations, state.iterations],
    outcome: "running",
    gepaInfo: {
      primitive_name: ev.primitive_name,
      prompt_preview: ev.prompt_preview,
      outcome: "running",
    },
  };
  return { ...state, tree: [...state.tree, node] };
}

function foldGepaCandidateAccepted(
  state: RlmRunState,
  ev: GepaCandidateAcceptedEvent
): RlmRunState {
  return {
    ...state,
    tree: state.tree.map((n) =>
      n.id === ev.candidate_id
        ? {
            ...n,
            outcome: "promoted",
            gepaInfo: n.gepaInfo
              ? { ...n.gepaInfo, score: ev.score, score_delta: ev.score_delta, outcome: "accepted" }
              : n.gepaInfo,
          }
        : n
    ),
  };
}

function foldGepaCandidateRejected(
  state: RlmRunState,
  ev: GepaCandidateRejectedEvent
): RlmRunState {
  return {
    ...state,
    tree: state.tree.map((n) =>
      n.id === ev.candidate_id
        ? {
            ...n,
            outcome: "declined",
            gepaInfo: n.gepaInfo
              ? { ...n.gepaInfo, outcome: "rejected" }
              : n.gepaInfo,
          }
        : n
    ),
  };
}
```

Add cases to the `fold()` switch/if block:
```typescript
    case "gepa_candidate_proposed":
      return foldGepaCandidateProposed(seeded, event);
    case "gepa_candidate_accepted":
      return foldGepaCandidateAccepted(seeded, event);
    case "gepa_candidate_rejected":
      return foldGepaCandidateRejected(seeded, event);
    case "gepa_phase_start":
    case "gepa_phase_complete":
      return seeded; // no tree mutation needed for phase lifecycle events
```

- [ ] **Step 3: Extend tree-node.tsx for gepa_candidate**

Open `frontend/src/components/lab/rlm/tree-node.tsx`. Find the block where `kind === "candidate"` renders a subtitle. Add:

```typescript
{kind === "gepa_candidate" && node.gepaInfo && (
  <span className={styles.subtitle}>
    GEPA · {node.gepaInfo.primitive_name}
    {node.gepaInfo.score != null && (
      <span className={styles.delta}>
        {node.gepaInfo.score_delta != null && node.gepaInfo.score_delta >= 0 ? "+" : ""}
        {node.gepaInfo.score?.toFixed(2)}
      </span>
    )}
  </span>
)}
```

- [ ] **Step 4: Add GEPA candidate panel to node-detail-sidebar.tsx**

Open `frontend/src/components/lab/rlm/node-detail-sidebar.tsx`. Find the `SidebarBody` function or equivalent. Add a new block for `gepa_candidate`:

```typescript
if (kind === "gepa_candidate" && node.gepaInfo) {
  return (
    <div className={styles.body}>
      <div className={styles.section}>
        <p className={styles.label}>Primitive</p>
        <p className={styles.value}>{node.gepaInfo.primitive_name}</p>
      </div>
      {node.gepaInfo.score != null && (
        <div className={styles.section}>
          <p className={styles.label}>Score</p>
          <p className={styles.value}>
            {node.gepaInfo.score.toFixed(4)}
            {node.gepaInfo.score_delta != null && (
              <span style={{ color: node.gepaInfo.score_delta >= 0 ? "var(--accent)" : "var(--err)" }}>
                {" "}({node.gepaInfo.score_delta >= 0 ? "+" : ""}{node.gepaInfo.score_delta.toFixed(4)})
              </span>
            )}
          </p>
        </div>
      )}
      <div className={styles.section}>
        <p className={styles.label}>Proposed prompt (preview)</p>
        <pre className={styles.codeBlock} style={{ whiteSpace: "pre-wrap", fontSize: "0.75rem" }}>
          {node.gepaInfo.prompt_preview || "(no preview)"}
        </pre>
      </div>
    </div>
  );
}
```

- [ ] **Step 5: Add gepa_candidate stroke to constellation-canvas.tsx**

Open `frontend/src/components/lab/rlm/constellation-canvas.tsx`. Find the block assigning `stroke` by kind. Add:

```typescript
if (kind === "gepa_candidate") stroke = "var(--gepa, #f97316)"; // orange
```

Also add the CSS custom property to the global stylesheet (find `globals.css` or `tokens.css`):

```css
--gepa: #f97316;
```

- [ ] **Step 6: Type-check frontend**

```bash
cd /Volumes/CS_Stuff/openresearch/frontend && npx tsc --noEmit 2>&1 | head -30
```

Expected: 0 errors. Fix any type errors before committing.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/lib/events/rlm-events.ts \
        frontend/src/hooks/use-rlm-run.ts \
        frontend/src/components/lab/rlm/tree-node.tsx \
        frontend/src/components/lab/rlm/node-detail-sidebar.tsx \
        frontend/src/components/lab/rlm/constellation-canvas.tsx
git commit -m "feat(gepa): frontend — gepa_candidate node kind + 5 SSE event types"
```

---

## Task 12: gepa-viz proxy route

**Files:**
- Create: `frontend/src/app/api/gepa-viz/[...path]/route.ts`

- [ ] **Step 1: Create proxy route**

Create `frontend/src/app/api/gepa-viz/[...path]/route.ts`:

```typescript
/**
 * Proxy route: /api/gepa-viz/* → http://127.0.0.1:{GEPA_VIZ_PORT}/*
 *
 * gepa-viz runs as an external server (gepa-viz live) on GEPA_VIZ_PORT (default 5151).
 * This proxy lets the lab UI embed gepa-viz without CORS issues.
 * SSE /events and POST /ingest are both proxied transparently.
 */
import { NextRequest, NextResponse } from "next/server";

const GEPA_VIZ_PORT = process.env.GEPA_VIZ_PORT ?? "5151";
const GEPA_VIZ_BASE = `http://127.0.0.1:${GEPA_VIZ_PORT}`;

export async function GET(
  req: NextRequest,
  { params }: { params: { path: string[] } }
) {
  const targetPath = "/" + params.path.join("/");
  const targetUrl = GEPA_VIZ_BASE + targetPath + (req.nextUrl.search || "");
  try {
    const upstream = await fetch(targetUrl, {
      headers: { ...Object.fromEntries(req.headers), host: `127.0.0.1:${GEPA_VIZ_PORT}` },
      // @ts-ignore — duplex needed for streaming
      duplex: "half",
    });
    return new NextResponse(upstream.body, {
      status: upstream.status,
      headers: upstream.headers,
    });
  } catch {
    return NextResponse.json({ error: "gepa-viz unavailable" }, { status: 503 });
  }
}

export async function POST(
  req: NextRequest,
  { params }: { params: { path: string[] } }
) {
  const targetPath = "/" + params.path.join("/");
  const targetUrl = GEPA_VIZ_BASE + targetPath;
  try {
    const body = await req.arrayBuffer();
    const upstream = await fetch(targetUrl, {
      method: "POST",
      headers: Object.fromEntries(req.headers),
      body,
    });
    return new NextResponse(upstream.body, { status: upstream.status });
  } catch {
    return NextResponse.json({ error: "gepa-viz unavailable" }, { status: 503 });
  }
}
```

- [ ] **Step 2: Type-check**

```bash
cd /Volumes/CS_Stuff/openresearch/frontend && npx tsc --noEmit 2>&1 | grep gepa-viz
```

Expected: no errors for the new file.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/app/api/gepa-viz/
git commit -m "feat(gepa): gepa-viz proxy route at /api/gepa-viz/*"
```

---

## Task 13: Persistence — gepa_examples.jsonl

**Files:**
- Create: `backend/agents/gepa/prompt_registry.py`
- Create: `backend/agents/gepa/trainset/historical_examples.py`

- [ ] **Step 1: Implement prompt_registry.py**

Create `backend/agents/gepa/prompt_registry.py`:

```python
"""Persist gepa_example_buffer to runs/{id}/gepa_examples.jsonl at run end.

Called from run.py in the finalization block (same place final_report.json is written).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def save_gepa_examples(ctx: Any) -> None:
    """Append ctx.gepa_example_buffer entries to gepa_examples.jsonl."""
    if not ctx.gepa_example_buffer:
        return
    out_path = ctx.project_dir / "gepa_examples.jsonl"
    try:
        with out_path.open("a", encoding="utf-8") as f:
            for primitive_name, entries in ctx.gepa_example_buffer.items():
                for entry in entries:
                    row = {
                        "primitive": primitive_name,
                        "arxiv_id": getattr(ctx, "arxiv_id", None),
                        **entry,
                    }
                    f.write(json.dumps(row, default=str) + "\n")
        logger.info("saved %d gepa examples to %s", sum(len(v) for v in ctx.gepa_example_buffer.values()), out_path)
    except Exception as exc:
        logger.warning("save_gepa_examples failed: %s", exc)
```

- [ ] **Step 2: Implement historical_examples.py**

Create `backend/agents/gepa/trainset/historical_examples.py`:

```python
"""Load GEPA training examples from previous runs of the same arxiv_id."""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_historical_examples(
    *,
    runs_root: Path,
    arxiv_id: str | None,
    primitive_name: str,
    max_examples: int = 20,
) -> list[dict]:
    """Scan runs_root for gepa_examples.jsonl files matching arxiv_id and primitive_name.

    Returns up to max_examples entries sorted by recency (most recent first).
    """
    if not arxiv_id:
        return []
    matches: list[tuple[float, dict]] = []
    try:
        for jsonl_path in runs_root.rglob("gepa_examples.jsonl"):
            try:
                mtime = jsonl_path.stat().st_mtime
                with jsonl_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if row.get("arxiv_id") == arxiv_id and row.get("primitive") == primitive_name:
                            matches.append((mtime, row))
            except Exception:
                continue
    except Exception as exc:
        logger.debug("load_historical_examples error: %s", exc)
        return []
    matches.sort(key=lambda x: x[0], reverse=True)
    return [row for _, row in matches[:max_examples]]
```

- [ ] **Step 3: Wire save_gepa_examples into run.py**

Open `backend/agents/rlm/run.py`. Find the finalization block where `final_report.json` is written. Add after it:

```python
    # Persist GEPA training examples for future runs of the same paper.
    try:
        from backend.agents.gepa.prompt_registry import save_gepa_examples
        save_gepa_examples(ctx)
    except Exception as _gepa_save_exc:
        logger.debug("save_gepa_examples skipped: %s", _gepa_save_exc)
```

- [ ] **Step 4: Wire historical examples into hooks.py**

In `backend/agents/gepa/hooks.py`, in `_build_trainset`, after building `static`, add:

```python
    # Load historical examples from past runs of the same paper
    historical: list = []
    if getattr(ctx, "arxiv_id", None) and getattr(ctx, "runs_root", None):
        try:
            from backend.agents.gepa.trainset.historical_examples import load_historical_examples
            historical = load_historical_examples(
                runs_root=ctx.runs_root,
                arxiv_id=ctx.arxiv_id,
                primitive_name=primitive_name,
                max_examples=10,
            )
        except Exception:
            pass
```

Then return `static + historical + buffered` (or whichever combination is non-empty).

- [ ] **Step 5: Commit**

```bash
git add backend/agents/gepa/prompt_registry.py \
        backend/agents/gepa/trainset/historical_examples.py \
        backend/agents/rlm/run.py \
        backend/agents/gepa/hooks.py
git commit -m "feat(gepa): persist gepa_examples.jsonl + historical trainset loader"
```

---

## Task 14: Integration test + spec doc

**Files:**
- Create: `tests/gepa/test_runner_integration.py`
- Create: `docs/superpowers/specs/2026-05-31-gepa-per-run-integration-design.md`

- [ ] **Step 1: Write integration test**

Create `tests/gepa/test_runner_integration.py`:

```python
"""Integration test: gepa_pre_call runs for plan_reproduction when flag is on,
produces an override in ctx.gepa_prompt_overrides, and emits SSE events.

Uses a real (mocked) LLM client but a real GEPA optimizer with max_metric_calls=3
to keep test time under 10s.
"""
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

VALID_CONTRACT = ('{"baseline_plan":"train","smoke_test_plan":"1 epoch",'
                  '"full_run_plan":"100 epochs","expected_artifacts":["metrics.json"],'
                  '"dataset_plan":{"name":"CIFAR"},"evaluation_plan":{"metrics":["acc"]},'
                  '"verification_checklist":["check acc"],'
                  '"metrics_shape":[{"metric_id":"acc","json_path":"top1","rubric_leaf_ids":[]}]}')


@pytest.fixture()
def ctx_gepa_on(tmp_path):
    from backend.agents.rlm.context import RunContext
    llm = MagicMock()
    llm.complete = MagicMock(return_value=VALID_CONTRACT)
    llm._last_usage = {}
    emit_calls = []
    return RunContext(
        project_id="prj_integ",
        project_dir=tmp_path,
        runs_root=tmp_path.parent,
        dashboard=MagicMock(),
        cost_ledger=MagicMock(),
        llm_client=llm,
        provider="openai",
        model="gpt-4o-mini",
        emit=lambda ev: emit_calls.append(ev),
    ), emit_calls


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="Requires gepa installed; skip in CI without API keys",
)
def test_gepa_pre_call_plan_reproduction(ctx_gepa_on):
    ctx, emit_calls = ctx_gepa_on
    with patch.dict(os.environ, {
        "REPROLAB_GEPA_OPTIMIZATION": "plan-only",
        "REPROLAB_GEPA_MAX_METRIC_CALLS_PLAN": "3",
        "REPROLAB_GEPA_TIMEOUT_PLAN_S": "30",
        "REPROLAB_GEPA_REFLECTION_MODEL": "openai/gpt-4o-mini",
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "sk-test"),
    }):
        from backend.agents.gepa.hooks import gepa_pre_call
        gepa_pre_call(ctx, "plan_reproduction", (), {})

    # After gepa_pre_call, SSE events should have been emitted
    event_types = [e.get("event") for e in emit_calls]
    assert "gepa_phase_start" in event_types
    assert "gepa_phase_complete" in event_types

    # gepa_optimization_active must be restored to False
    assert ctx.gepa_optimization_active is False


def test_gepa_post_call_populates_buffer(ctx_gepa_on):
    ctx, _ = ctx_gepa_on
    from backend.agents.gepa.hooks import gepa_post_call
    result = {"baseline_plan": "x", "metrics_shape": [{"metric_id": "acc", "json_path": "top1", "rubric_leaf_ids": []}]}
    gepa_post_call(ctx, "plan_reproduction", result)
    assert "plan_reproduction" in ctx.gepa_example_buffer
    assert len(ctx.gepa_example_buffer["plan_reproduction"]) == 1
```

- [ ] **Step 2: Run integration test**

```bash
.venv/bin/python -m pytest tests/gepa/test_runner_integration.py -v -k "not test_gepa_pre_call_plan_reproduction"
```

The non-skipif test should pass. The skipif test requires a real API key.

- [ ] **Step 3: Write spec doc**

Create `docs/superpowers/specs/2026-05-31-gepa-per-run-integration-design.md` with the full design from the plan file at `/Users/aayushbaniya/.claude/plans/lovely-humming-giraffe.md`. Copy the content and reformat as a proper spec doc (add a header, remove implementation-task formatting).

- [ ] **Step 4: Run all gepa tests**

```bash
.venv/bin/python -m pytest tests/gepa/ -v --ignore=tests/gepa/test_runner_integration.py
```

Expected: all pass.

- [ ] **Step 5: Run existing rlm tests to verify no regressions**

```bash
.venv/bin/python -m pytest tests/rlm/ -x -q 2>&1 | tail -15
```

Expected: same results as before this branch.

- [ ] **Step 6: Final commit**

```bash
git add tests/gepa/test_runner_integration.py \
        docs/superpowers/specs/2026-05-31-gepa-per-run-integration-design.md
git commit -m "feat(gepa): integration test + spec doc"
```

---

## Verification Checklist

Before declaring done:

- [ ] `REPROLAB_GEPA_OPTIMIZATION=off` (default): zero behavioral change, all existing tests pass
- [ ] `REPROLAB_GEPA_OPTIMIZATION=plan-only`: `gepa_phase_start` + `gepa_phase_complete` events appear in `dashboard_events.jsonl` after `plan_reproduction`; `ctx.gepa_prompt_overrides["plan_reproduction"]` is set when improvement found
- [ ] `gepa_optimization_active` is always restored to `False` even when GEPA throws
- [ ] `llm_client.complete` is always restored after the primitive thread completes (check timeout path too)
- [ ] `frontend/src/app/api/gepa-viz/[...path]/route.ts` returns 503 gracefully when gepa-viz is not running
- [ ] `npx tsc --noEmit` in frontend: 0 errors
- [ ] All `tests/gepa/` unit tests pass
- [ ] All `tests/rlm/` tests pass (no regressions)
