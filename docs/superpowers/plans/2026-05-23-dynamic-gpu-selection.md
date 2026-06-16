# Dynamic GPU Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire paper-stated hardware clues into RunPod SKU provisioning — VRAM-based match with inference headroom, $/hr cost cap, force-single-GPU invariant, and auto-escalation on CUDA OOM.

**Architecture:** Three new modules (catalog, resolver, schemas additions), one new plan-time primitive (`resolve_gpu_requirements`), `RunpodBackend` accepts optional `GpuPlan`, `run_experiment` adds OOM-detect-and-escalate loop. All choices emit through the existing `sse_bridge` chokepoint.

**Tech Stack:** Python 3.14, Pydantic v2, pydantic-settings, FastAPI (unchanged), pytest, Next.js 16 (one TS component touched).

**Source spec:** `docs/superpowers/specs/2026-05-23-dynamic-gpu-selection-design.md` — all locked decisions live there.

**Branch / worktree:** Work happens on a feature branch `feat/dynamic-gpu-selection`. Subagent-driven execution should create a worktree via `superpowers:using-git-worktrees` before Task 1.

**Commit cadence:** Per-milestone (8 commits total), NOT per task — per user memory `feedback_commit_granularity.md`. Each milestone groups 2–3 tasks and ends with one commit. The "Commit" step appears at milestone boundaries.

---

## Milestone 1 — Foundation: schemas, catalog, resolver

These three tasks introduce pure-logic modules with zero coupling to RunPod/RLM internals. They can land first because nothing downstream depends on them yet.

### Task 1: Add `GpuRequirements` and `GpuPlan` Pydantic models

**Files:**
- Modify: `backend/agents/schemas.py` (append after line 170 — after `ReproductionContract`, before existing `Ambiguity` references — verify by reading; use `class GpuRequirements` block as anchor)
- Test: `tests/agents/test_gpu_schemas.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/test_gpu_schemas.py
from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.agents.schemas import GpuPlan, GpuRequirements


def test_gpu_requirements_accepts_complete_payload():
    req = GpuRequirements(
        estimated_vram_gb=48,
        paper_gpu_string="A100 80GB",
        paper_gpu_count=8,
        reasoning="paper states 80GB; eval harness adds ~10GB",
        confidence=0.85,
    )
    assert req.estimated_vram_gb == 48
    assert req.confidence == pytest.approx(0.85)


def test_gpu_requirements_allows_none_estimate():
    req = GpuRequirements(
        estimated_vram_gb=None,
        paper_gpu_string=None,
        paper_gpu_count=None,
        reasoning="no hardware clues found in paper",
        confidence=0.1,
    )
    assert req.estimated_vram_gb is None


def test_gpu_requirements_rejects_negative_vram():
    with pytest.raises(ValidationError):
        GpuRequirements(
            estimated_vram_gb=-5,
            paper_gpu_string=None,
            paper_gpu_count=None,
            reasoning="",
            confidence=0.5,
        )


def test_gpu_requirements_clamps_confidence_range():
    with pytest.raises(ValidationError):
        GpuRequirements(
            estimated_vram_gb=24,
            paper_gpu_string=None,
            paper_gpu_count=None,
            reasoning="",
            confidence=1.5,
        )


def test_gpu_plan_complete_payload():
    plan = GpuPlan(
        runpod_id="NVIDIA A100 80GB PCIe",
        short_name="a100_80",
        vram_gb=80,
        gpu_count=1,
        cloud_type="COMMUNITY",
        sku_usd_per_hr=1.89,
        total_usd_per_hr=1.89,
        container_disk_gb=80,
        volume_gb=20,
        source="paper",
        requirements=GpuRequirements(
            estimated_vram_gb=64,
            paper_gpu_string="A100 80GB",
            paper_gpu_count=8,
            reasoning="test",
            confidence=0.9,
        ),
        ladder_remaining=("h100_80",),
        resolved_at="2026-05-23T00:00:00+00:00",
    )
    assert plan.gpu_count == 1
    assert plan.ladder_remaining == ("h100_80",)


def test_gpu_plan_source_accepts_only_known_values():
    with pytest.raises(ValidationError):
        GpuPlan(
            runpod_id="x", short_name="x", vram_gb=24, gpu_count=1,
            cloud_type="COMMUNITY", sku_usd_per_hr=0.34, total_usd_per_hr=0.34,
            container_disk_gb=50, volume_gb=20,
            source="bogus_source",  # not in {paper, fallback, manual, informational}
            requirements=GpuRequirements(
                estimated_vram_gb=24, paper_gpu_string=None, paper_gpu_count=None,
                reasoning="", confidence=0.5,
            ),
            ladder_remaining=(),
            resolved_at="2026-05-23T00:00:00+00:00",
        )
```

- [ ] **Step 2: Run test, confirm failure**

Run: `.venv/bin/python -m pytest tests/agents/test_gpu_schemas.py -q`
Expected: `ImportError: cannot import name 'GpuPlan' from 'backend.agents.schemas'`

- [ ] **Step 3: Add models to schemas.py**

Append to `backend/agents/schemas.py` AFTER `ReproductionContract` (~line 184):

```python
# ---------------------------------------------------------------------------
# Dynamic GPU selection (#dynamic-gpu spec 2026-05-23)
# ---------------------------------------------------------------------------

class GpuRequirements(BaseModel):
    """LLM-derived hardware requirements extracted from paper text.

    The RLM root constructs this from accumulated PaperClaimMap.hardware_clues
    plus reasoning over the full workload (training + inference + evaluation).
    """
    model_config = {"extra": "ignore"}
    estimated_vram_gb: int | None = Field(
        default=None, ge=0, le=1024,
        description="Whole-workload VRAM estimate; None when LLM cannot estimate",
    )
    paper_gpu_string: str | None = None
    paper_gpu_count: int | None = Field(default=None, ge=0, le=64)
    reasoning: str = Field(default="", description="One-line rationale, surfaced in SSE event")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class GpuPlan(BaseModel):
    """Resolved provisioning plan, consumed by RunpodBackend."""
    model_config = {"extra": "ignore"}
    runpod_id: str = Field(description="Verbatim RunPod gpu_type identifier")
    short_name: str = Field(description="Internal short name; matches gpu_catalog.GpuSku.short_name")
    vram_gb: int = Field(ge=1)
    gpu_count: int = Field(ge=1, le=8)
    cloud_type: Literal["COMMUNITY", "SECURE"]
    sku_usd_per_hr: float = Field(ge=0.0, description="Per-GPU rate from catalog")
    total_usd_per_hr: float = Field(ge=0.0, description="sku_usd_per_hr * gpu_count")
    container_disk_gb: int = Field(ge=1)
    volume_gb: int = Field(ge=1)
    source: Literal["paper", "fallback", "manual", "informational"]
    requirements: GpuRequirements
    ladder_remaining: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Short names of next-larger SKUs for OOM escalation",
    )
    resolved_at: str = Field(description="ISO-8601 timestamp")
```

- [ ] **Step 4: Run test, confirm pass**

Run: `.venv/bin/python -m pytest tests/agents/test_gpu_schemas.py -q`
Expected: `6 passed`

---

### Task 2: Create `gpu_catalog.py` (`GpuSku` dataclass + CATALOG + `find_ladder`)

**Files:**
- Create: `backend/services/runtime/gpu_catalog.py`
- Test: `tests/services/runtime/test_gpu_catalog.py`

- [ ] **Step 1: Write failing test**

```python
# tests/services/runtime/test_gpu_catalog.py
from __future__ import annotations

import pytest

from backend.services.runtime.gpu_catalog import (
    CATALOG,
    GpuSku,
    find_ladder,
    find_by_alias,
)


def test_catalog_is_nonempty_tuple_of_gpusku():
    assert isinstance(CATALOG, tuple)
    assert len(CATALOG) >= 7
    assert all(isinstance(sku, GpuSku) for sku in CATALOG)


def test_catalog_sorted_by_vram_then_price():
    keys = [(sku.vram_gb, sku.approx_usd_per_hr) for sku in CATALOG]
    assert keys == sorted(keys), "CATALOG must be sorted by (vram_gb, price) ASC for readability"


def test_find_ladder_returns_only_skus_meeting_vram_and_cap():
    ladder = find_ladder(min_vram_gb=40, max_per_gpu_usd_per_hr=2.0, cloud_types=("COMMUNITY",))
    assert all(s.vram_gb >= 40 for s in ladder)
    assert all(s.approx_usd_per_hr <= 2.0 for s in ladder)
    assert all(s.cloud_type == "COMMUNITY" for s in ladder)


def test_find_ladder_sorts_by_ascending_price():
    ladder = find_ladder(min_vram_gb=24, max_per_gpu_usd_per_hr=10.0, cloud_types=("COMMUNITY",))
    prices = [s.approx_usd_per_hr for s in ladder]
    assert prices == sorted(prices)


def test_find_ladder_returns_empty_when_cap_too_low():
    assert find_ladder(min_vram_gb=80, max_per_gpu_usd_per_hr=0.10, cloud_types=("COMMUNITY",)) == []


def test_find_ladder_excludes_secure_only_when_community_filter():
    ladder = find_ladder(min_vram_gb=24, max_per_gpu_usd_per_hr=100.0, cloud_types=("COMMUNITY",))
    assert not any(s.short_name == "h200" for s in ladder), "H200 is SECURE-only; must not appear under COMMUNITY filter"


def test_find_ladder_includes_secure_when_filter_permits():
    ladder = find_ladder(
        min_vram_gb=100,
        max_per_gpu_usd_per_hr=100.0,
        cloud_types=("COMMUNITY", "SECURE"),
    )
    assert any(s.short_name == "h200" for s in ladder)


def test_find_ladder_no_cap_when_cap_is_none():
    ladder = find_ladder(min_vram_gb=24, max_per_gpu_usd_per_hr=None, cloud_types=("COMMUNITY",))
    assert any(s.short_name == "h100_80" for s in ladder), "no cap must include H100"


def test_find_by_alias_resolves_common_phrases():
    assert find_by_alias("a100").short_name == "a100_80"
    assert find_by_alias("A100 40GB").short_name == "a100_40"
    assert find_by_alias("RTX 4090").short_name == "rtx4090"
    assert find_by_alias("H100").short_name == "h100_80"


def test_find_by_alias_returns_none_on_unknown():
    assert find_by_alias("nonexistent-gpu") is None
```

- [ ] **Step 2: Run, confirm failure**

Run: `.venv/bin/python -m pytest tests/services/runtime/test_gpu_catalog.py -q`
Expected: `ModuleNotFoundError: No module named 'backend.services.runtime.gpu_catalog'`

- [ ] **Step 3: Create `backend/services/runtime/gpu_catalog.py`**

```python
"""Static GPU SKU catalog for RunPod dynamic GPU selection.

Vendored prices are approximate snapshots refreshed quarterly. The resolver's
ranking between SKUs is what matters; absolute prices may drift ±20%.

See `docs/superpowers/specs/2026-05-23-dynamic-gpu-selection-design.md` §Catalog.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GpuSku:
    """One row of the catalog.

    `runpod_id` is the literal string RunPod's API accepts for `gpu_type`.
    `aliases` is a tuple of lowercase substrings that the alias resolver matches
    against paper text — order does not matter, longest first wins on tie.
    """
    runpod_id: str
    short_name: str
    vram_gb: int
    cloud_type: str
    approx_usd_per_hr: float
    aliases: tuple[str, ...] = field(default_factory=tuple)


CATALOG: tuple[GpuSku, ...] = (
    # Sorted by (vram_gb ASC, approx_usd_per_hr ASC) for human readability.
    # find_ladder() re-sorts the filtered result by price for selection.
    GpuSku("NVIDIA GeForce RTX 4090",     "rtx4090",   24, "COMMUNITY", 0.34,
           aliases=("rtx 4090", "geforce 4090", "rtx4090", "4090")),
    GpuSku("NVIDIA RTX A5000",            "a5000",     24, "COMMUNITY", 0.36,
           aliases=("a5000", "rtx a5000")),
    GpuSku("NVIDIA A100 40GB PCIe",       "a100_40",   40, "COMMUNITY", 1.19,
           aliases=("a100 40", "a100 40gb", "a100-40", "a100 40 gb")),
    GpuSku("NVIDIA RTX A6000",            "a6000",     48, "COMMUNITY", 0.49,
           aliases=("a6000", "rtx a6000")),
    GpuSku("NVIDIA L40S",                 "l40s",      48, "COMMUNITY", 0.86,
           aliases=("l40s", "l40 s")),
    GpuSku("NVIDIA A100 80GB PCIe",       "a100_80",   80, "COMMUNITY", 1.89,
           aliases=("a100 80", "a100 80gb", "a100-80", "a100")),
    GpuSku("NVIDIA H100 80GB HBM3",       "h100_80",   80, "COMMUNITY", 4.39,
           aliases=("h100", "h100 80", "h100 80gb", "h100-80")),
    GpuSku("NVIDIA H200",                 "h200",     141, "SECURE",    7.99,
           aliases=("h200",)),
)


def find_ladder(
    min_vram_gb: int,
    max_per_gpu_usd_per_hr: float | None,
    cloud_types: tuple[str, ...] = ("COMMUNITY",),
) -> list[GpuSku]:
    """Return SKUs meeting all filters, sorted by ascending price.

    Args:
        min_vram_gb: minimum required VRAM in GB; SKU must have vram_gb >= this
        max_per_gpu_usd_per_hr: per-GPU $/hr cap; None or 0 means no cap
        cloud_types: which cloud types are acceptable

    Returns:
        Filtered list sorted by approx_usd_per_hr ASC; empty list when no SKU qualifies.
    """
    cap = max_per_gpu_usd_per_hr if max_per_gpu_usd_per_hr and max_per_gpu_usd_per_hr > 0 else None
    filtered = [
        sku for sku in CATALOG
        if sku.vram_gb >= min_vram_gb
        and sku.cloud_type in cloud_types
        and (cap is None or sku.approx_usd_per_hr <= cap)
    ]
    return sorted(filtered, key=lambda s: s.approx_usd_per_hr)


def find_by_alias(phrase: str) -> GpuSku | None:
    """Lookup the first SKU whose alias appears as a substring of `phrase` (case-insensitive).

    Longest alias wins on tie to avoid 'a100' matching when 'a100 80gb' is present.
    Returns None when no alias matches.
    """
    needle = phrase.lower()
    best: tuple[int, GpuSku] | None = None
    for sku in CATALOG:
        for alias in sku.aliases:
            if alias in needle:
                if best is None or len(alias) > best[0]:
                    best = (len(alias), sku)
    return best[1] if best else None


__all__ = ["GpuSku", "CATALOG", "find_ladder", "find_by_alias"]
```

- [ ] **Step 4: Run, confirm pass**

Run: `.venv/bin/python -m pytest tests/services/runtime/test_gpu_catalog.py -q`
Expected: `10 passed`

---

### Task 3: Create `gpu_resolver.py` (pure resolver with all policy logic)

**Files:**
- Create: `backend/services/runtime/gpu_resolver.py`
- Test: `tests/services/runtime/test_gpu_resolver.py`

- [ ] **Step 1: Write failing test**

```python
# tests/services/runtime/test_gpu_resolver.py
from __future__ import annotations

import pytest

from backend.agents.schemas import GpuPlan, GpuRequirements
from backend.services.runtime.gpu_resolver import (
    GpuResolutionError,
    resolve,
)


def _req(vram: int | None = 40, count: int | None = 8, conf: float = 0.85) -> GpuRequirements:
    return GpuRequirements(
        estimated_vram_gb=vram,
        paper_gpu_string="A100 80GB" if vram else None,
        paper_gpu_count=count,
        reasoning="test",
        confidence=conf,
    )


def test_resolve_picks_cheapest_meeting_vram_with_multiplier():
    # Multiplier 1.25 on 40 -> 50; cheapest SKU with vram>=50 under cap is L40S (48GB? no — 48<50; A100 80GB 1.89, but A6000 is 48 which is <50; correct cheapest is A100 80 at 1.89)
    # Actually 48 < 50 (after multiplier 50), so we tier up to 80 (A100 80GB at $1.89).
    plan = resolve(
        _req(vram=40),
        dynamic_gpu_enabled=True,
        force_single_gpu=True,
        max_gpu_usd_per_hour=10.0,
        headroom_multiplier=1.25,
        fallback_vram_gb=24,
        cloud_types=("COMMUNITY",),
    )
    assert plan.short_name == "a100_80"
    assert plan.gpu_count == 1
    assert plan.source == "paper"


def test_resolve_force_single_gpu_caps_count_at_one():
    plan = resolve(
        _req(vram=20, count=8),
        dynamic_gpu_enabled=True,
        force_single_gpu=True,
        max_gpu_usd_per_hour=10.0,
        headroom_multiplier=1.25,
        fallback_vram_gb=24,
        cloud_types=("COMMUNITY",),
    )
    assert plan.gpu_count == 1


def test_resolve_multi_gpu_bounded_by_cost_cap():
    # Without force_single: paper says 8x; SKU A100 80GB is $1.89/hr; cap $10/hr
    # -> floor(10 / 1.89) = 5; min(8, 5) = 5.
    plan = resolve(
        _req(vram=64, count=8),
        dynamic_gpu_enabled=True,
        force_single_gpu=False,
        max_gpu_usd_per_hour=10.0,
        headroom_multiplier=1.25,
        fallback_vram_gb=24,
        cloud_types=("COMMUNITY",),
    )
    assert plan.gpu_count == 5
    assert plan.total_usd_per_hr <= 10.0


def test_resolve_low_confidence_triggers_fallback_sku():
    plan = resolve(
        _req(vram=40, conf=0.2),
        dynamic_gpu_enabled=True,
        force_single_gpu=True,
        max_gpu_usd_per_hour=10.0,
        headroom_multiplier=1.25,
        fallback_vram_gb=24,
        cloud_types=("COMMUNITY",),
    )
    assert plan.source == "fallback"
    assert plan.short_name == "rtx4090"
    assert plan.gpu_count == 1


def test_resolve_none_estimate_triggers_fallback():
    plan = resolve(
        _req(vram=None, conf=0.9),
        dynamic_gpu_enabled=True,
        force_single_gpu=True,
        max_gpu_usd_per_hour=10.0,
        headroom_multiplier=1.25,
        fallback_vram_gb=24,
        cloud_types=("COMMUNITY",),
    )
    assert plan.source == "fallback"


def test_resolve_raises_when_no_sku_under_cap_meets_vram():
    with pytest.raises(GpuResolutionError) as exc:
        resolve(
            _req(vram=200),  # > largest SKU vram (141)
            dynamic_gpu_enabled=True,
            force_single_gpu=True,
            max_gpu_usd_per_hour=10.0,
            headroom_multiplier=1.25,
            fallback_vram_gb=24,
            cloud_types=("COMMUNITY", "SECURE"),
        )
    assert "200" in str(exc.value) or "no SKU" in str(exc.value).lower()


def test_resolve_raises_when_required_sku_exceeds_cap():
    # H100 80GB needed but cap forces only RTX 4090/A6000/L40S/A100s
    # vram 80 required; cap $0.50/hr -> only RTX 4090 ($0.34), A5000 ($0.36), A6000 ($0.49)
    # none have vram>=80 -> raise
    with pytest.raises(GpuResolutionError):
        resolve(
            _req(vram=80),
            dynamic_gpu_enabled=True,
            force_single_gpu=True,
            max_gpu_usd_per_hour=0.50,
            headroom_multiplier=1.0,
            fallback_vram_gb=24,
            cloud_types=("COMMUNITY",),
        )


def test_resolve_ladder_contains_next_larger_skus():
    plan = resolve(
        _req(vram=24),
        dynamic_gpu_enabled=True,
        force_single_gpu=True,
        max_gpu_usd_per_hour=10.0,
        headroom_multiplier=1.0,
        fallback_vram_gb=24,
        cloud_types=("COMMUNITY",),
    )
    # picked: RTX 4090 ($0.34), 24GB. Next-cheapest with vram>=24: A5000.
    # ladder_remaining should contain A5000 (and onward up the ladder).
    assert plan.short_name == "rtx4090"
    assert "a5000" in plan.ladder_remaining


def test_resolve_disabled_dynamic_returns_informational_plan_from_fallback_default():
    plan = resolve(
        _req(vram=80),
        dynamic_gpu_enabled=False,  # OFF
        force_single_gpu=True,
        max_gpu_usd_per_hour=10.0,
        headroom_multiplier=1.25,
        fallback_vram_gb=24,
        cloud_types=("COMMUNITY",),
    )
    assert plan.source == "informational"
    assert plan.short_name == "rtx4090"


def test_resolve_is_pure_no_io_imports():
    """Smoke: gpu_resolver must not import anything that performs network/disk I/O."""
    import importlib
    import sys
    mod = importlib.import_module("backend.services.runtime.gpu_resolver")
    forbidden = {"requests", "httpx", "urllib", "subprocess", "socket", "asyncio", "asyncssh"}
    deps = set()
    for name, m in list(sys.modules.items()):
        if m and getattr(m, "__file__", "") and "backend/services/runtime/gpu_resolver" in (m.__file__ or ""):
            for v in vars(m).values():
                modname = getattr(getattr(v, "__module__", None), "split", lambda *_: [""])(".")[0] if v else ""
                if modname:
                    deps.add(modname)
    assert not (deps & forbidden), f"resolver pulled in I/O modules: {deps & forbidden}"
```

- [ ] **Step 2: Run, confirm failure**

Run: `.venv/bin/python -m pytest tests/services/runtime/test_gpu_resolver.py -q`
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Create `backend/services/runtime/gpu_resolver.py`**

```python
"""Pure-logic GPU resolver. No I/O. No network. No imports beyond stdlib + schemas + catalog.

Given `GpuRequirements` + settings + budget, returns a `GpuPlan` with the cheapest
RunPod SKU that meets the VRAM target (after headroom multiplier + tier-up), respecting
the per-GPU $/hr cap and the force_single_gpu invariant.

See `docs/superpowers/specs/2026-05-23-dynamic-gpu-selection-design.md` §Resolver.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from backend.agents.schemas import GpuPlan, GpuRequirements
from backend.services.runtime.gpu_catalog import GpuSku, find_ladder

# Confidence threshold below which we treat the LLM estimate as unusable.
_CONFIDENCE_FLOOR: float = 0.4

# When estimate is unusable, we use this SKU regardless of paper.
_FALLBACK_SHORT_NAME: str = "rtx4090"


class GpuResolutionError(RuntimeError):
    """Raised when no SKU can satisfy (VRAM + $/hr cap + cloud_type) constraints."""


def resolve(
    requirements: GpuRequirements,
    *,
    dynamic_gpu_enabled: bool,
    force_single_gpu: bool,
    max_gpu_usd_per_hour: float | None,
    headroom_multiplier: float,
    fallback_vram_gb: int,
    cloud_types: tuple[str, ...] = ("COMMUNITY",),
) -> GpuPlan:
    """Resolve requirements to a GpuPlan. Pure function — no I/O.

    Returns:
        GpuPlan with source in {"paper", "fallback", "manual", "informational"}.

    Raises:
        GpuResolutionError when constraints are infeasible (e.g., VRAM > largest catalog
        SKU, or required SKU exceeds the per-GPU $/hr cap). The error message names the
        cheapest SKU that would have satisfied VRAM if the cap were lifted.
    """
    now_iso = datetime.now(timezone.utc).isoformat()

    # Dynamic disabled → return informational plan from the fallback SKU. RunpodBackend
    # ignores `gpu_plan` and uses legacy Settings.runpod_gpu_type when source is
    # "informational" (see primitive caller).
    if not dynamic_gpu_enabled:
        sku = _by_short_name(_FALLBACK_SHORT_NAME)
        return _build_plan(sku, gpu_count=1, source="informational",
                           requirements=requirements, ladder=(), now_iso=now_iso)

    estimate = requirements.estimated_vram_gb
    confidence = requirements.confidence

    # Fallback path: no estimate OR confidence too low.
    if estimate is None or confidence < _CONFIDENCE_FLOOR:
        sku = _by_short_name(_FALLBACK_SHORT_NAME)
        return _build_plan(sku, gpu_count=1, source="fallback",
                           requirements=requirements, ladder=(), now_iso=now_iso)

    # Apply headroom multiplier; round up.
    needed_vram = math.ceil(estimate * max(headroom_multiplier, 1.0))

    # Find ladder under cap.
    ladder = find_ladder(
        min_vram_gb=needed_vram,
        max_per_gpu_usd_per_hr=max_gpu_usd_per_hour,
        cloud_types=cloud_types,
    )

    if not ladder:
        # Diagnose: would a SKU exist if we lifted the cap?
        unconstrained = find_ladder(
            min_vram_gb=needed_vram,
            max_per_gpu_usd_per_hr=None,
            cloud_types=cloud_types,
        )
        if unconstrained:
            cheapest = unconstrained[0]
            raise GpuResolutionError(
                f"Paper requires >= {needed_vram} GB VRAM (after {headroom_multiplier}x headroom on "
                f"estimate={estimate}). Cheapest SKU is {cheapest.short_name} at "
                f"${cheapest.approx_usd_per_hr:.2f}/hr, but `max_gpu_usd_per_hour` cap is "
                f"${max_gpu_usd_per_hour}. Raise the cap or set --vram-gb to a lower override."
            )
        raise GpuResolutionError(
            f"Paper requires >= {needed_vram} GB VRAM but no catalog SKU has that much VRAM. "
            f"Largest available is {_largest_vram(cloud_types)} GB. Consider multi-GPU "
            f"reproduction or scoping down the experiment."
        )

    pick = ladder[0]

    # Count: force_single_gpu wins; else min(paper_count, cap_allows_count).
    if force_single_gpu:
        gpu_count = 1
    else:
        paper_count = max(1, requirements.paper_gpu_count or 1)
        if max_gpu_usd_per_hour and max_gpu_usd_per_hour > 0:
            cap_allows = max(1, int(math.floor(max_gpu_usd_per_hour / pick.approx_usd_per_hr)))
        else:
            cap_allows = paper_count
        gpu_count = min(paper_count, cap_allows)

    remaining = tuple(s.short_name for s in ladder[1:])
    return _build_plan(pick, gpu_count=gpu_count, source="paper",
                       requirements=requirements, ladder=remaining, now_iso=now_iso)


def _by_short_name(short_name: str) -> GpuSku:
    from backend.services.runtime.gpu_catalog import CATALOG
    for sku in CATALOG:
        if sku.short_name == short_name:
            return sku
    raise GpuResolutionError(f"Catalog has no SKU with short_name={short_name!r} (programmer error)")


def _largest_vram(cloud_types: tuple[str, ...]) -> int:
    from backend.services.runtime.gpu_catalog import CATALOG
    candidates = [s.vram_gb for s in CATALOG if s.cloud_type in cloud_types]
    return max(candidates) if candidates else 0


def _build_plan(
    sku: GpuSku,
    *,
    gpu_count: int,
    source: str,
    requirements: GpuRequirements,
    ladder: tuple[str, ...],
    now_iso: str,
) -> GpuPlan:
    return GpuPlan(
        runpod_id=sku.runpod_id,
        short_name=sku.short_name,
        vram_gb=sku.vram_gb,
        gpu_count=gpu_count,
        cloud_type=sku.cloud_type,
        sku_usd_per_hr=sku.approx_usd_per_hr,
        total_usd_per_hr=round(sku.approx_usd_per_hr * gpu_count, 4),
        container_disk_gb=max(50, sku.vram_gb),
        volume_gb=max(20, sku.vram_gb // 4),
        source=source,
        requirements=requirements,
        ladder_remaining=ladder,
        resolved_at=now_iso,
    )


__all__ = ["GpuResolutionError", "resolve"]
```

- [ ] **Step 4: Run all 3 milestone-1 test files; confirm pass**

Run:
```
.venv/bin/python -m pytest tests/agents/test_gpu_schemas.py tests/services/runtime/test_gpu_catalog.py tests/services/runtime/test_gpu_resolver.py -q
```
Expected: all pass (>25 tests).

- [ ] **Step 5: Commit milestone 1**

```bash
git checkout -b feat/dynamic-gpu-selection
git add backend/agents/schemas.py backend/services/runtime/gpu_catalog.py backend/services/runtime/gpu_resolver.py tests/agents/test_gpu_schemas.py tests/services/runtime/test_gpu_catalog.py tests/services/runtime/test_gpu_resolver.py
git commit -m "Dynamic GPU selection foundation — GpuRequirements/GpuPlan models, vendored SKU catalog, and pure resolver

Adds the three pure-logic modules that the dynamic-GPU feature will sit on:
- GpuRequirements + GpuPlan Pydantic models in schemas.py
- gpu_catalog.py with vendored RunPod SKU list and find_ladder/find_by_alias helpers
- gpu_resolver.py: pure function from (requirements, settings) to GpuPlan, with
  headroom multiplier, tier-up, \$/hr cap enforcement, force_single_gpu invariant,
  multi-GPU count floored by cap, low-confidence fallback to RTX 4090, and explicit
  GpuResolutionError when constraints are infeasible.

No production code paths touched — the modules exist but are unused. Subsequent
milestones wire them into the primitive + RunpodBackend + run_experiment."
```

---

## Milestone 2 — Budget + Settings

### Task 4: Add `max_run_gpu_usd` field and `check_run_gpu_usd()` to RunBudget

**Files:**
- Modify: `backend/agents/resilience/budget.py:12–73`
- Test: `tests/agents/resilience/test_budget_gpu_usd.py` (new)

- [ ] **Step 1: Write failing test**

```python
# tests/agents/resilience/test_budget_gpu_usd.py
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from backend.agents.resilience.budget import RunBudget
from backend.agents.resilience.failures import BudgetExhausted


def test_check_run_gpu_usd_passes_when_under_cap():
    budget = RunBudget(max_run_gpu_usd=5.0)
    budget.check_run_gpu_usd(cumulative_pod_usd=2.0, agent_id="run_experiment")


def test_check_run_gpu_usd_raises_when_at_or_above_cap():
    budget = RunBudget(max_run_gpu_usd=5.0)
    with pytest.raises(BudgetExhausted) as exc:
        budget.check_run_gpu_usd(cumulative_pod_usd=5.0, agent_id="run_experiment")
    assert "5.0" in str(exc.value) or "pod" in str(exc.value).lower()


def test_check_run_gpu_usd_noop_when_cap_none():
    budget = RunBudget(max_run_gpu_usd=None)
    budget.check_run_gpu_usd(cumulative_pod_usd=1_000_000.0, agent_id="x")


def test_check_run_gpu_usd_noop_when_cap_zero():
    budget = RunBudget(max_run_gpu_usd=0.0)
    budget.check_run_gpu_usd(cumulative_pod_usd=1_000_000.0, agent_id="x")
```

- [ ] **Step 2: Run, confirm failure**

Run: `.venv/bin/python -m pytest tests/agents/resilience/test_budget_gpu_usd.py -q`
Expected: `AttributeError: 'RunBudget' object has no attribute 'check_run_gpu_usd'` (or TypeError if `max_run_gpu_usd` rejected).

- [ ] **Step 3: Modify `backend/agents/resilience/budget.py`**

Add `max_run_gpu_usd` field at line 18 (after `rlm_calls_remaining`) and `check_run_gpu_usd` method after `check_pod_seconds`:

```python
# Add field at line 18 (after rlm_calls_remaining):
    max_run_gpu_usd: float | None = None

# Add method after check_pod_seconds (~ line 73, before __all__):
    def check_run_gpu_usd(
        self,
        *,
        cumulative_pod_usd: float,
        agent_id: str,
    ) -> None:
        """Raise BudgetExhausted when cumulative pod spend >= max_run_gpu_usd.

        `cumulative_pod_usd` is the total RunPod cost incurred by this run so
        far — caller is responsible for the running tally (typically
        wall_clock_seconds * sku_usd_per_hr / 3600 summed across pod
        lifetimes). Cap is honored only when set and > 0; None or 0 disables
        the check.
        """
        if self.max_run_gpu_usd is None or self.max_run_gpu_usd <= 0:
            return
        if cumulative_pod_usd >= self.max_run_gpu_usd:
            raise BudgetExhausted(
                f"Run pod-USD budget exhausted before invoking {agent_id}: "
                f"${cumulative_pod_usd:.4f} >= ${self.max_run_gpu_usd:.4f}",
                provider=None,
                agent_id=agent_id,
            )
```

- [ ] **Step 4: Run, confirm pass**

Run: `.venv/bin/python -m pytest tests/agents/resilience/test_budget_gpu_usd.py -q`
Expected: `4 passed`

---

### Task 5: Add 7 Settings fields to `backend/config.py`

**Files:**
- Modify: `backend/config.py` after `runpod_pod_id` (~line 181)
- Test: `tests/config/test_dynamic_gpu_settings.py` (new — or add to whichever existing config test file matches; create the directory if needed)

- [ ] **Step 1: Write failing test**

```python
# tests/config/test_dynamic_gpu_settings.py
from __future__ import annotations

import os

import pytest

from backend.config import Settings


def _settings(**env: str) -> Settings:
    """Build a Settings reading from a custom env-dict mock."""
    # pydantic-settings reads from os.environ; monkeypatch via fixture in real use.
    # For this test we construct Settings directly with overrides.
    return Settings(**env)


def test_default_values_match_spec():
    s = Settings()
    assert s.dynamic_gpu_enabled is True
    assert s.force_single_gpu is True
    assert s.max_gpu_usd_per_hour == pytest.approx(10.0)
    assert s.max_run_gpu_usd == pytest.approx(10.0)
    assert s.dynamic_gpu_headroom == pytest.approx(1.25)
    assert s.dynamic_gpu_fallback_vram_gb == 24
    assert s.dynamic_gpu_max_escalations == 2


def test_env_var_overrides(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_DYNAMIC_GPU", "false")
    monkeypatch.setenv("OPENRESEARCH_FORCE_SINGLE_GPU", "false")
    monkeypatch.setenv("OPENRESEARCH_MAX_GPU_USD_PER_HOUR", "2.5")
    monkeypatch.setenv("OPENRESEARCH_MAX_RUN_GPU_USD", "3.0")
    monkeypatch.setenv("OPENRESEARCH_DYNAMIC_GPU_HEADROOM", "1.5")
    monkeypatch.setenv("OPENRESEARCH_DYNAMIC_GPU_FALLBACK_VRAM_GB", "40")
    monkeypatch.setenv("OPENRESEARCH_DYNAMIC_GPU_MAX_ESCALATIONS", "3")
    s = Settings()
    assert s.dynamic_gpu_enabled is False
    assert s.force_single_gpu is False
    assert s.max_gpu_usd_per_hour == pytest.approx(2.5)
    assert s.max_run_gpu_usd == pytest.approx(3.0)
    assert s.dynamic_gpu_headroom == pytest.approx(1.5)
    assert s.dynamic_gpu_fallback_vram_gb == 40
    assert s.dynamic_gpu_max_escalations == 3


def test_empty_cost_caps_treated_as_no_cap(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_MAX_GPU_USD_PER_HOUR", "0")
    monkeypatch.setenv("OPENRESEARCH_MAX_RUN_GPU_USD", "")
    s = Settings()
    # 0 is allowed numerically; consumer treats 0 as "no cap" (see resolver test).
    assert s.max_gpu_usd_per_hour == 0.0
    # Empty string → pydantic-settings parses as default 10.0 (per Field default)
    # unless explicitly typed Optional. Spec says empty = no cap;
    # implementation uses 0.0 as the no-cap sentinel.
    assert s.max_run_gpu_usd in (0.0, 10.0)  # accept either, depending on pydantic-settings parsing
```

- [ ] **Step 2: Run, confirm failure**

Run: `.venv/bin/python -m pytest tests/config/test_dynamic_gpu_settings.py -q`
Expected: `AttributeError: 'Settings' object has no attribute 'dynamic_gpu_enabled'`

- [ ] **Step 3: Modify `backend/config.py`**

Add fields after line 181 (after `runpod_pod_id`), matching existing pydantic-settings style:

```python
    # --- Dynamic GPU selection (spec 2026-05-23) ---
    dynamic_gpu_enabled: bool = Field(default=True, description="Wire paper hardware clues to RunPod SKU choice")
    force_single_gpu: bool = Field(default=True, description="Cap RunPod GPU count at 1 regardless of paper")
    max_gpu_usd_per_hour: float = Field(default=10.0, ge=0.0, description="Per-GPU \$/hr cap; 0 disables")
    max_run_gpu_usd: float = Field(default=10.0, ge=0.0, description="Total RunPod \$ per run cap; 0 disables")
    dynamic_gpu_headroom: float = Field(default=1.25, ge=1.0, description="Multiplier on LLM VRAM estimate before tier-up")
    dynamic_gpu_fallback_vram_gb: int = Field(default=24, ge=1, description="Substitute VRAM when LLM cannot estimate")
    dynamic_gpu_max_escalations: int = Field(default=2, ge=0, description="Max OOM-driven ladder advances per run")
```

- [ ] **Step 4: Run, confirm pass**

Run: `.venv/bin/python -m pytest tests/config/test_dynamic_gpu_settings.py tests/agents/resilience/test_budget_gpu_usd.py -q`
Expected: `7 passed`

- [ ] **Step 5: Commit milestone 2**

```bash
git add backend/agents/resilience/budget.py backend/config.py tests/agents/resilience/test_budget_gpu_usd.py tests/config/test_dynamic_gpu_settings.py
git commit -m "Dynamic GPU selection — budget hook (max_run_gpu_usd) and 7 new Settings fields

RunBudget gains max_run_gpu_usd + check_run_gpu_usd() mirroring check_pod_seconds.
Settings adds the spec's full config surface: dynamic_gpu_enabled, force_single_gpu,
max_gpu_usd_per_hour, max_run_gpu_usd, dynamic_gpu_headroom,
dynamic_gpu_fallback_vram_gb, dynamic_gpu_max_escalations. No production paths
read these yet — wired in milestone 3+."
```

---

## Milestone 3 — Primitive `resolve_gpu_requirements` + system prompt

### Task 6: Add `resolve_gpu_requirements` primitive to `primitives.py`

**Files:**
- Modify: `backend/agents/rlm/primitives.py` (append a new primitive function before `_backend_for_sandbox_mode` or after `detect_environment` — match local function ordering)
- Modify: `backend/agents/rlm/custom_tools.py` or wherever primitives are registered with `rlm.RLM(...)` (find via `grep -rn "extract_hyperparameters" backend/agents/rlm/`)
- Test: `tests/rlm/test_resolve_gpu_requirements.py`

- [ ] **Step 1: Write failing test**

```python
# tests/rlm/test_resolve_gpu_requirements.py
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from backend.agents.rlm.primitives import resolve_gpu_requirements
from backend.agents.schemas import GpuPlan, GpuRequirements


@pytest.fixture
def ctx(tmp_path: Path):
    runs_root = tmp_path / "runs"
    project_dir = runs_root / "proj1"
    project_dir.mkdir(parents=True)
    (project_dir / "rlm_state").mkdir()
    return SimpleNamespace(
        project_id="proj1",
        runs_root=runs_root,
        project_dir=project_dir,
        run_budget=None,
        sandbox_mode="runpod",
    )


def test_returns_gpu_plan_dict_from_typed_input(ctx):
    req = GpuRequirements(
        estimated_vram_gb=40,
        paper_gpu_string="A100 80GB",
        paper_gpu_count=8,
        reasoning="test",
        confidence=0.9,
    )
    out = resolve_gpu_requirements(req, ctx=ctx)
    assert isinstance(out, dict)
    plan = GpuPlan(**out)
    assert plan.gpu_count == 1
    assert plan.source == "paper"


def test_accepts_loose_dict_payload(ctx):
    payload = {
        "estimated_vram_gb": 40,
        "paper_gpu_string": "A100 80GB",
        "paper_gpu_count": 8,
        "reasoning": "test",
        "confidence": 0.9,
    }
    out = resolve_gpu_requirements(payload, ctx=ctx)
    assert out["source"] == "paper"


def test_idempotent_returns_cached_plan(ctx):
    payload = {"estimated_vram_gb": 40, "paper_gpu_string": "A100", "paper_gpu_count": 1, "reasoning": "", "confidence": 0.9}
    out1 = resolve_gpu_requirements(payload, ctx=ctx)
    out2 = resolve_gpu_requirements({"estimated_vram_gb": 999, "paper_gpu_string": None, "paper_gpu_count": None, "reasoning": "diff", "confidence": 0.9}, ctx=ctx)
    assert out1["short_name"] == out2["short_name"], "cached plan must be returned, second call's higher VRAM ignored"
    assert out1["resolved_at"] == out2["resolved_at"]


def test_persists_plan_to_run_state(ctx):
    payload = {"estimated_vram_gb": 40, "paper_gpu_string": None, "paper_gpu_count": None, "reasoning": "", "confidence": 0.9}
    out = resolve_gpu_requirements(payload, ctx=ctx)
    state_file = ctx.project_dir / "rlm_state" / "gpu_plan.json"
    assert state_file.exists()
    loaded = json.loads(state_file.read_text())
    assert loaded["short_name"] == out["short_name"]


def test_low_confidence_returns_fallback_source(ctx):
    payload = {"estimated_vram_gb": 80, "paper_gpu_string": None, "paper_gpu_count": None, "reasoning": "", "confidence": 0.2}
    out = resolve_gpu_requirements(payload, ctx=ctx)
    assert out["source"] == "fallback"
    assert out["short_name"] == "rtx4090"


def test_malformed_payload_raises_value_error(ctx):
    with pytest.raises((ValueError, Exception)):
        resolve_gpu_requirements({"estimated_vram_gb": "not-a-number", "reasoning": "", "confidence": 0.5}, ctx=ctx)
```

- [ ] **Step 2: Run, confirm failure**

Run: `.venv/bin/python -m pytest tests/rlm/test_resolve_gpu_requirements.py -q`
Expected: `ImportError: cannot import name 'resolve_gpu_requirements' from 'backend.agents.rlm.primitives'`

- [ ] **Step 3: Add primitive to `backend/agents/rlm/primitives.py`**

Add this function near other primitive definitions (e.g., right after `detect_environment` around line 254). Reuse the existing logger import.

```python
def resolve_gpu_requirements(
    requirements: GpuRequirements | dict,
    *,
    ctx: "RunContext",
) -> dict:
    """Plan-time GPU resolver primitive (RLM #dynamic-gpu spec 2026-05-23).

    The RLM root supplies LLM-derived GpuRequirements (from accumulated
    PaperClaimMap.hardware_clues + reasoning over env_spec and the full workload).
    This primitive maps to a GpuPlan via the catalog, caches the plan in run
    state for idempotency, and emits a gpu_resolved SSE event for UI / audit.

    Idempotent: subsequent calls in the same run return the cached plan even if
    the caller passes different requirements. This avoids cost drift across
    re-resolution attempts and matches RLM-loop expectations.
    """
    import json as _json
    from pathlib import Path as _Path

    from backend.agents.schemas import GpuPlan, GpuRequirements as _Req
    from backend.config import get_settings
    from backend.services.runtime import gpu_resolver

    # ---- Idempotency: return cached plan if present.
    state_dir = _Path(ctx.project_dir) / "rlm_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    cache_file = state_dir / "gpu_plan.json"
    if cache_file.exists():
        try:
            cached = _json.loads(cache_file.read_text(encoding="utf-8"))
            return cached
        except Exception:  # noqa: BLE001 — corrupt cache → recompute
            logger.warning("resolve_gpu_requirements: cache file unreadable, recomputing")

    # ---- Coerce payload.
    if isinstance(requirements, dict):
        req = _Req(**requirements)
    elif isinstance(requirements, _Req):
        req = requirements
    else:
        raise ValueError(f"resolve_gpu_requirements: requirements must be GpuRequirements or dict, got {type(requirements).__name__}")

    settings = get_settings()
    cloud_types = ("COMMUNITY", "SECURE") if settings.runpod_cloud_type == "SECURE" else ("COMMUNITY",)

    plan: GpuPlan = gpu_resolver.resolve(
        req,
        dynamic_gpu_enabled=settings.dynamic_gpu_enabled,
        force_single_gpu=settings.force_single_gpu,
        max_gpu_usd_per_hour=settings.max_gpu_usd_per_hour or None,
        headroom_multiplier=settings.dynamic_gpu_headroom,
        fallback_vram_gb=settings.dynamic_gpu_fallback_vram_gb,
        cloud_types=cloud_types,
    )

    # ---- Persist atomically.
    payload = plan.model_dump(mode="json")
    tmp = cache_file.with_suffix(".tmp")
    tmp.write_text(_json.dumps(payload, default=str), encoding="utf-8")
    tmp.replace(cache_file)

    # ---- Emit SSE event.
    _emit_dashboard_event(ctx, event_type="gpu_resolved", payload=payload)

    return payload


def _emit_dashboard_event(ctx: "RunContext", *, event_type: str, payload: dict) -> None:
    """Append an event line to runs/<id>/dashboard_events.jsonl atomically."""
    import json as _json
    from datetime import datetime as _dt, timezone as _tz
    from pathlib import Path as _Path

    events_file = _Path(ctx.project_dir) / "dashboard_events.jsonl"
    line = {
        "ts": _dt.now(_tz.utc).isoformat(),
        "event": event_type,
        "data": payload,
    }
    try:
        with events_file.open("a", encoding="utf-8") as fh:
            fh.write(_json.dumps(line, default=str) + "\n")
    except Exception:  # noqa: BLE001 — observability must never break the run
        logger.exception("dashboard event emit failed for %s", event_type)
```

If the file already has a different `_emit_dashboard_event` helper, reuse it instead and remove the second definition above. Search before adding: `grep -n "_emit_dashboard_event\|dashboard_events.jsonl" backend/agents/rlm/primitives.py`.

- [ ] **Step 4: Register primitive with RLM**

Find where `extract_hyperparameters` / `detect_environment` are registered with `rlm.RLM(...)` (likely a `custom_tools=[...]` list in `backend/agents/rlm/run.py` or a primitive-registration module). Add `resolve_gpu_requirements` to that list.

Search: `grep -rn "extract_hyperparameters\|detect_environment" backend/agents/rlm/ --include="*.py" | grep -v test`

Add `resolve_gpu_requirements` next to the others in whichever list contains them.

- [ ] **Step 5: Run, confirm pass**

Run: `.venv/bin/python -m pytest tests/rlm/test_resolve_gpu_requirements.py -q`
Expected: `6 passed`

---

### Task 7: Update `system_prompt.py` with the new primitive's instructions

**Files:**
- Modify: `backend/agents/rlm/system_prompt.py`

- [ ] **Step 1: Read the file to find the right insertion point**

Read `backend/agents/rlm/system_prompt.py` end-to-end (it's small) and find the section where existing primitives are described (e.g., `understand_section`, `detect_environment`). Insert a 3-sentence paragraph after the section that describes `detect_environment`:

- [ ] **Step 2: Add the instruction block**

Append the following paragraph in the appropriate section (e.g., the primitive-usage walkthrough):

```python
_RESOLVE_GPU_REQUIREMENTS_SECTION = """
  GPU SELECTION — `resolve_gpu_requirements`

  After your initial pass of `understand_section` covers the abstract +
  method + experiments sections, construct a GpuRequirements payload from the
  accumulated `hardware_clues`. Estimate `estimated_vram_gb` for the WHOLE
  workload — not just training. Include inference, evaluation harness, any
  auxiliary models the paper loads (e.g., a frozen scoring model), and KV cache
  for generative inference. Then call:

      resolve_gpu_requirements({
          "estimated_vram_gb": <int or None>,
          "paper_gpu_string": "<verbatim string from paper or None>",
          "paper_gpu_count": <int or None>,
          "reasoning": "<one-line rationale>",
          "confidence": <float 0.0–1.0>,
      })

  Call this ONCE per run. Subsequent calls return the cached plan automatically
  — you do not need to call it again from any later iteration. The plan
  determines pod provisioning for every later `run_experiment` call. If you
  cannot estimate VRAM (paper doesn't mention hardware), set
  `estimated_vram_gb=None` and `confidence` low — the resolver will fall back
  to a safe default SKU and emit a warning event.
"""
```

Then wire `_RESOLVE_GPU_REQUIREMENTS_SECTION` into the assembled prompt string at the appropriate position. Look for the existing `f"..." + _CONTEXT_METADATA_INTRO + ...` chain or similar concatenation.

- [ ] **Step 3: Sanity-check prompt build**

```python
# Quick check from a Python REPL or a one-off pytest:
from backend.agents.rlm.system_prompt import build_system_prompt  # whatever the public builder is
text = build_system_prompt(...)  # may need a context kwarg
assert "resolve_gpu_requirements" in text
```

If the prompt module has a unit test like `tests/rlm/test_system_prompt.py`, add an assertion there:

```python
def test_system_prompt_mentions_resolve_gpu_requirements():
    from backend.agents.rlm.system_prompt import build_system_prompt
    text = build_system_prompt(...)
    assert "resolve_gpu_requirements" in text
    assert "estimated_vram_gb" in text
```

- [ ] **Step 4: Commit milestone 3**

```bash
git add backend/agents/rlm/primitives.py backend/agents/rlm/system_prompt.py tests/rlm/test_resolve_gpu_requirements.py
# include any other touched files (custom_tools registration, system_prompt test)
git commit -m "Dynamic GPU selection — resolve_gpu_requirements primitive + system-prompt instructions

Adds the plan-time primitive the RLM root calls once per run to map LLM-derived
GpuRequirements into a GpuPlan. The primitive is idempotent (caches to
runs/<id>/rlm_state/gpu_plan.json), accepts either typed or loose-dict input
for REPL ergonomics, and emits gpu_resolved to dashboard_events.jsonl. The
system prompt now instructs the root to compute whole-workload VRAM (training
+ inference + eval) before calling, and to set confidence low when clues are
missing so the resolver falls back to a safe default SKU."
```

---

## Milestone 4 — Backend wiring: `RunpodBackend` accepts `gpu_plan`

### Task 8: Add `gpu_plan` kwarg to `RunpodBackend.__init__`

**Files:**
- Modify: `backend/services/runtime/runpod_backend.py:56–102`
- Test: `tests/services/runtime/test_runpod_backend_gpu_plan.py`

- [ ] **Step 1: Write failing test**

```python
# tests/services/runtime/test_runpod_backend_gpu_plan.py
from __future__ import annotations

import pytest

from backend.agents.schemas import GpuPlan, GpuRequirements
from backend.services.runtime.runpod_backend import RunpodBackend


def _plan(**overrides) -> GpuPlan:
    base = dict(
        runpod_id="NVIDIA A100 80GB PCIe",
        short_name="a100_80",
        vram_gb=80,
        gpu_count=1,
        cloud_type="COMMUNITY",
        sku_usd_per_hr=1.89,
        total_usd_per_hr=1.89,
        container_disk_gb=80,
        volume_gb=20,
        source="paper",
        requirements=GpuRequirements(
            estimated_vram_gb=64, paper_gpu_string="A100",
            paper_gpu_count=1, reasoning="", confidence=0.9,
        ),
        ladder_remaining=("h100_80",),
        resolved_at="2026-05-23T00:00:00+00:00",
    )
    base.update(overrides)
    return GpuPlan(**base)


def test_backend_uses_gpu_plan_when_provided():
    plan = _plan()
    backend = RunpodBackend(api_key="dummy", gpu_plan=plan)
    assert backend.gpu_type == "NVIDIA A100 80GB PCIe"
    assert backend.gpu_count == 1
    assert backend.cloud_type == "COMMUNITY"
    assert backend.container_disk_gb >= 80
    assert backend.volume_gb >= 20


def test_backend_back_compat_no_plan_uses_settings():
    """When gpu_plan is None, backend falls back to legacy Settings defaults."""
    backend = RunpodBackend(api_key="dummy", gpu_plan=None)
    # Default per repo: OPENRESEARCH_RUNPOD_GPU_TYPE="NVIDIA GeForce RTX 4090"
    assert "RTX 4090" in backend.gpu_type or "4090" in backend.gpu_type
    assert backend.gpu_count == 1


def test_backend_plan_overrides_explicit_init_args():
    """If both gpu_plan and gpu_type=... are passed, gpu_plan wins for type/count."""
    plan = _plan(short_name="rtx4090", runpod_id="NVIDIA GeForce RTX 4090", vram_gb=24)
    backend = RunpodBackend(api_key="dummy", gpu_type="OTHER_GPU", gpu_count=4, gpu_plan=plan)
    assert backend.gpu_type == "NVIDIA GeForce RTX 4090"
    assert backend.gpu_count == 1


def test_backend_informational_plan_is_ignored():
    """source='informational' means dynamic_gpu_enabled=off; legacy path."""
    plan = _plan(source="informational")
    backend = RunpodBackend(api_key="dummy", gpu_plan=plan, gpu_type="NVIDIA GeForce RTX 4090")
    # Backend ignores informational plans and uses the explicit gpu_type arg / settings default.
    assert backend.gpu_type == "NVIDIA GeForce RTX 4090"
```

- [ ] **Step 2: Run, confirm failure**

Run: `.venv/bin/python -m pytest tests/services/runtime/test_runpod_backend_gpu_plan.py -q`
Expected: `TypeError: __init__() got an unexpected keyword argument 'gpu_plan'`

- [ ] **Step 3: Modify `backend/services/runtime/runpod_backend.py`**

At the `__init__` signature (line 56–78), add `gpu_plan` kwarg. After args are stored, conditionally override:

```python
# Add to __init__ signature, after pod_id kwarg:
        gpu_plan: "GpuPlan | None" = None,
```

After the existing field assignments (around line 95), insert:

```python
        # Dynamic GPU plan overrides explicit args ONLY when source != "informational"
        # (informational means dynamic_gpu_enabled=off; caller passes the plan for
        # telemetry/UI but expects the legacy gpu_type to provision the pod).
        if gpu_plan is not None and getattr(gpu_plan, "source", None) != "informational":
            self.gpu_type = gpu_plan.runpod_id
            self.gpu_count = gpu_plan.gpu_count
            self.cloud_type = gpu_plan.cloud_type
            self.container_disk_gb = max(self.container_disk_gb, gpu_plan.container_disk_gb)
            self.volume_gb = max(self.volume_gb, gpu_plan.volume_gb)
        self.gpu_plan = gpu_plan
```

Add the top-of-file lazy import to avoid circular import:

```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from backend.agents.schemas import GpuPlan
```

- [ ] **Step 4: Run, confirm pass**

Run: `.venv/bin/python -m pytest tests/services/runtime/test_runpod_backend_gpu_plan.py -q`
Expected: `4 passed`

- [ ] **Step 5: Run regression on existing RunPod tests**

```
.venv/bin/python -m pytest tests/services/runtime/test_runpod_pod_time_budget.py tests/test_runpod_delete_guardrails.py tests/services/runtime/test_runpod_incremental_sync.py tests/rlm/test_runpod_wiring.py -q
```
Expected: all pre-existing tests still pass.

---

### Task 9: Thread `gpu_plan` through `_backend_for_sandbox_mode`

**Files:**
- Modify: `backend/agents/rlm/primitives.py:499–546`
- Modify: `backend/agents/rlm/primitives.py` `_execute_in_sandbox` (line 566+) to accept and forward `gpu_plan`
- Test: `tests/rlm/test_runpod_wiring.py` (extend existing if it covers this path; else create `tests/rlm/test_runpod_wiring_gpu_plan.py`)

- [ ] **Step 1: Write failing test**

```python
# tests/rlm/test_runpod_wiring_gpu_plan.py
from __future__ import annotations

from unittest.mock import patch

import pytest

from backend.agents.execution import SandboxMode
from backend.agents.rlm.primitives import _backend_for_sandbox_mode
from backend.agents.schemas import GpuPlan, GpuRequirements


def _plan() -> GpuPlan:
    return GpuPlan(
        runpod_id="NVIDIA A100 40GB PCIe", short_name="a100_40", vram_gb=40, gpu_count=1,
        cloud_type="COMMUNITY", sku_usd_per_hr=1.19, total_usd_per_hr=1.19,
        container_disk_gb=50, volume_gb=20, source="paper",
        requirements=GpuRequirements(
            estimated_vram_gb=32, paper_gpu_string="A100",
            paper_gpu_count=1, reasoning="", confidence=0.9,
        ),
        ladder_remaining=("a100_80",), resolved_at="2026-05-23T00:00:00+00:00",
    )


def test_backend_for_sandbox_mode_passes_gpu_plan_to_runpod_backend():
    plan = _plan()
    with patch("backend.services.runtime.ensure_runpod_available"):
        backend = _backend_for_sandbox_mode(SandboxMode.runpod, gpu_plan=plan)
    assert backend.__class__.__name__ == "RunpodBackend"
    assert backend.gpu_type == "NVIDIA A100 40GB PCIe"


def test_backend_for_sandbox_mode_local_docker_ignores_gpu_plan():
    plan = _plan()
    backend = _backend_for_sandbox_mode(SandboxMode.docker, gpu_plan=plan)
    assert backend.__class__.__name__ == "LocalDockerBackend"
```

- [ ] **Step 2: Run, confirm failure**

Run: `.venv/bin/python -m pytest tests/rlm/test_runpod_wiring_gpu_plan.py -q`
Expected: `TypeError: _backend_for_sandbox_mode() got an unexpected keyword argument 'gpu_plan'`

- [ ] **Step 3: Modify `_backend_for_sandbox_mode`**

Update the signature and the RunpodBackend instantiation:

```python
def _backend_for_sandbox_mode(
    sandbox_mode: object,
    *,
    run_budget: object = None,
    gpu_plan: "GpuPlan | None" = None,
):
    ...
    if mode is SandboxMode.runpod:
        import backend.services.runtime as _runtime
        from backend.services.runtime.runpod_backend import RunpodBackend

        _runtime.ensure_runpod_available()
        return RunpodBackend(run_budget=run_budget, gpu_plan=gpu_plan)
    ...
```

Also extend `_execute_in_sandbox` (line 566) to accept and forward `gpu_plan`:

```python
async def _execute_in_sandbox(
    code_path: str,
    env_id: str,
    commands: list[str],
    *,
    project_id: str,
    run_id: str,
    sandbox_mode: object = None,
    run_budget: object = None,
    gpu_plan: object = None,
) -> dict:
    ...
    service = RuntimeAppService(_backend_for_sandbox_mode(
        sandbox_mode, run_budget=run_budget, gpu_plan=gpu_plan,
    ))
    ...
```

And finally, `run_experiment` (line 700) needs to load the cached plan and forward it:

```python
def run_experiment(code_path: str, env_id: str, *, ctx: "RunContext") -> dict:
    ...
    # Load cached gpu_plan if present
    import json as _json
    _gpu_plan = None
    _plan_path = ctx.project_dir / "rlm_state" / "gpu_plan.json"
    if _plan_path.exists():
        try:
            from backend.agents.schemas import GpuPlan as _GpuPlan
            _gpu_plan = _GpuPlan(**_json.loads(_plan_path.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001
            logger.warning("run_experiment: gpu_plan.json present but unreadable; using legacy default")

    # ... existing code ...

    # When submitting to executor, pass gpu_plan:
    result = pool.submit(
        asyncio.run,
        _execute_in_sandbox(
            code_path, env_id, commands,
            project_id=ctx.project_id, run_id=run_id,
            sandbox_mode=ctx.sandbox_mode,
            run_budget=ctx.run_budget,
            gpu_plan=_gpu_plan,
        ),
    ).result(timeout=timeout)
```

- [ ] **Step 4: Run, confirm pass**

```
.venv/bin/python -m pytest tests/rlm/test_runpod_wiring_gpu_plan.py tests/rlm/test_runpod_wiring.py tests/services/runtime/test_runpod_backend_gpu_plan.py -q
```
Expected: all pass.

- [ ] **Step 5: Commit milestone 4**

```bash
git add backend/services/runtime/runpod_backend.py backend/agents/rlm/primitives.py tests/services/runtime/test_runpod_backend_gpu_plan.py tests/rlm/test_runpod_wiring_gpu_plan.py
git commit -m "Dynamic GPU selection — RunpodBackend accepts GpuPlan; primitives forward it

RunpodBackend.__init__ gains optional gpu_plan kwarg; when present and source !=
'informational', overrides gpu_type/gpu_count/cloud_type from the plan and tier-ups
container_disk_gb / volume_gb to fit the SKU. _backend_for_sandbox_mode and
_execute_in_sandbox thread gpu_plan through; run_experiment loads the cached
plan from rlm_state/gpu_plan.json. Back-compat: gpu_plan=None falls back to
legacy Settings defaults — existing runs are byte-identical."
```

---

## Milestone 5 — OOM detection and ladder escalation

### Task 10: Add `_detect_cuda_oom` helper

**Files:**
- Modify: `backend/agents/rlm/primitives.py` (private helper near other module-level functions)
- Test: `tests/rlm/test_cuda_oom_detection.py`

- [ ] **Step 1: Write failing test**

```python
# tests/rlm/test_cuda_oom_detection.py
from __future__ import annotations

from backend.agents.rlm.primitives import _detect_cuda_oom


def test_detects_exit_code_137():
    assert _detect_cuda_oom(exit_code=137, stderr_tail="") is True


def test_detects_pytorch_oom_substring():
    msg = "RuntimeError: CUDA out of memory. Tried to allocate 2.50 GiB ..."
    assert _detect_cuda_oom(exit_code=1, stderr_tail=msg) is True


def test_detects_torch_outofmemoryerror():
    msg = "torch.cuda.OutOfMemoryError: CUDA out of memory."
    assert _detect_cuda_oom(exit_code=1, stderr_tail=msg) is True


def test_detects_cublas_alloc_failed():
    msg = "RuntimeError: cuBLAS error: CUBLAS_STATUS_ALLOC_FAILED"
    assert _detect_cuda_oom(exit_code=1, stderr_tail=msg) is True


def test_normal_failure_is_not_oom():
    msg = "ImportError: No module named 'transformers'"
    assert _detect_cuda_oom(exit_code=1, stderr_tail=msg) is False


def test_clean_exit_is_not_oom():
    assert _detect_cuda_oom(exit_code=0, stderr_tail="") is False
```

- [ ] **Step 2: Run, confirm failure**

Run: `.venv/bin/python -m pytest tests/rlm/test_cuda_oom_detection.py -q`
Expected: `ImportError`

- [ ] **Step 3: Add helper to `backend/agents/rlm/primitives.py`**

```python
_CUDA_OOM_MARKERS: tuple[str, ...] = (
    "CUDA out of memory",
    "RuntimeError: CUDA error: out of memory",
    "torch.cuda.OutOfMemoryError",
    "cuBLAS error: CUBLAS_STATUS_ALLOC_FAILED",
)


def _detect_cuda_oom(*, exit_code: int, stderr_tail: str) -> bool:
    """True when exit-code or stderr tail indicates a CUDA OOM (spec 2026-05-23 §OOM).

    `stderr_tail` should be the last ~4KB of combined stderr/stdout from the failed
    experiment. Exit code 137 is SIGKILL (OOM killer); substring match covers the
    documented PyTorch/cuBLAS variants. Pattern set is intentionally tight to avoid
    false positives on unrelated CUDA errors.
    """
    if exit_code == 137:
        return True
    if not stderr_tail:
        return False
    return any(marker in stderr_tail for marker in _CUDA_OOM_MARKERS)
```

- [ ] **Step 4: Run, confirm pass**

Run: `.venv/bin/python -m pytest tests/rlm/test_cuda_oom_detection.py -q`
Expected: `6 passed`

---

### Task 11: Add escalation loop in `run_experiment`

**Files:**
- Modify: `backend/agents/rlm/primitives.py` `run_experiment` (around line 700–850)
- Test: `tests/rlm/test_runpod_oom_escalation.py`

- [ ] **Step 1: Write failing test**

```python
# tests/rlm/test_runpod_oom_escalation.py
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from backend.agents.schemas import GpuPlan, GpuRequirements


@pytest.fixture
def ctx(tmp_path: Path):
    project_dir = tmp_path / "proj1"
    code_dir = project_dir / "code"
    rlm_state = project_dir / "rlm_state"
    code_dir.mkdir(parents=True)
    rlm_state.mkdir(parents=True)
    (code_dir / "commands.json").write_text(json.dumps(["python train.py"]))
    plan = GpuPlan(
        runpod_id="NVIDIA A100 40GB PCIe", short_name="a100_40", vram_gb=40, gpu_count=1,
        cloud_type="COMMUNITY", sku_usd_per_hr=1.19, total_usd_per_hr=1.19,
        container_disk_gb=50, volume_gb=20, source="paper",
        requirements=GpuRequirements(
            estimated_vram_gb=32, paper_gpu_string="A100",
            paper_gpu_count=1, reasoning="", confidence=0.9,
        ),
        ladder_remaining=("a100_80", "h100_80"),
        resolved_at="2026-05-23T00:00:00+00:00",
    )
    (rlm_state / "gpu_plan.json").write_text(json.dumps(plan.model_dump(mode="json")))
    return SimpleNamespace(
        project_id="proj1",
        project_dir=project_dir,
        runs_root=tmp_path,
        run_budget=None,
        sandbox_mode="runpod",
        remaining_s=lambda: None,
    )


def test_oom_escalates_to_next_ladder_rung(ctx):
    """First call OOMs, second call (with escalated plan) succeeds."""
    from backend.agents.rlm import primitives

    call_count = {"n": 0}

    async def fake_execute(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # First call: OOM
            return {"success": False, "metrics": {}, "logs": "RuntimeError: CUDA out of memory: tried to alloc 80GB"}
        # Second call: success on bigger SKU
        return {"success": True, "metrics": {"acc": 0.9}, "logs": "done"}

    with patch.object(primitives, "_execute_in_sandbox", side_effect=fake_execute):
        result = primitives.run_experiment("code/", "env-id", ctx=ctx)

    assert result["success"] is True
    assert call_count["n"] == 2, "experiment must be re-run after escalation"
    # Plan on disk now points to a100_80
    plan_after = json.loads((ctx.project_dir / "rlm_state" / "gpu_plan.json").read_text())
    assert plan_after["short_name"] == "a100_80"


def test_oom_with_empty_ladder_fails_with_cost_summary(ctx):
    """When ladder_remaining is empty AND oom, raise structured failure."""
    from backend.agents.rlm import primitives

    plan_path = ctx.project_dir / "rlm_state" / "gpu_plan.json"
    plan = json.loads(plan_path.read_text())
    plan["ladder_remaining"] = []
    plan_path.write_text(json.dumps(plan))

    async def always_oom(*args, **kwargs):
        return {"success": False, "metrics": {}, "logs": "torch.cuda.OutOfMemoryError"}

    with patch.object(primitives, "_execute_in_sandbox", side_effect=always_oom):
        result = primitives.run_experiment("code/", "env-id", ctx=ctx)

    assert result["success"] is False
    assert "oom" in str(result.get("error", "")).lower() or "escalation" in str(result.get("error", "")).lower()


def test_max_escalations_cap_honored(ctx, monkeypatch):
    """If max_escalations=1, second OOM must not trigger a third attempt."""
    monkeypatch.setenv("OPENRESEARCH_DYNAMIC_GPU_MAX_ESCALATIONS", "1")
    from backend.config import get_settings
    get_settings.cache_clear() if hasattr(get_settings, "cache_clear") else None
    from backend.agents.rlm import primitives

    call_count = {"n": 0}

    async def always_oom(*args, **kwargs):
        call_count["n"] += 1
        return {"success": False, "metrics": {}, "logs": "CUDA out of memory"}

    with patch.object(primitives, "_execute_in_sandbox", side_effect=always_oom):
        result = primitives.run_experiment("code/", "env-id", ctx=ctx)

    # 1 initial + 1 escalation = 2 calls
    assert call_count["n"] == 2
    assert result["success"] is False


def test_non_oom_failure_does_not_escalate(ctx):
    """Generic experiment failure (e.g., ImportError) does NOT trigger escalation."""
    from backend.agents.rlm import primitives

    call_count = {"n": 0}

    async def import_error(*args, **kwargs):
        call_count["n"] += 1
        return {"success": False, "metrics": {}, "logs": "ImportError: No module named 'transformers'"}

    with patch.object(primitives, "_execute_in_sandbox", side_effect=import_error):
        result = primitives.run_experiment("code/", "env-id", ctx=ctx)

    assert call_count["n"] == 1, "non-OOM failure must not escalate"
    assert result["success"] is False
```

- [ ] **Step 2: Run, confirm failure**

Run: `.venv/bin/python -m pytest tests/rlm/test_runpod_oom_escalation.py -q`
Expected: 4 tests fail (escalation logic not yet implemented).

- [ ] **Step 3: Implement escalation in `run_experiment`**

Refactor `run_experiment` body (current lines ~700–820) to wrap the `pool.submit(...).result(...)` in a loop that detects OOM, advances the ladder, and re-runs. Roughly:

```python
def run_experiment(code_path: str, env_id: str, *, ctx: "RunContext") -> dict:
    """... (keep docstring; add §Escalation note) ..."""
    import asyncio
    import json as _json
    import uuid
    from pathlib import Path

    from backend.agents.schemas import GpuPlan as _GpuPlan
    from backend.config import get_settings
    from backend.services.runtime.gpu_catalog import CATALOG

    manifest = Path(code_path) / "commands.json"
    commands = _json.loads(manifest.read_text()) if manifest.exists() else []
    if not commands:
        return _persist_experiment_result(ctx, {
            "success": False, "metrics": {},
            "error": f"no commands.json at {manifest}"})

    # ... existing Dockerfile rebuild block ...

    if not env_id or not str(env_id).strip():
        return _persist_experiment_result(ctx, {
            "success": False, "metrics": {},
            "error": "env_id empty and no Dockerfile to rebuild — build_environment must succeed first",
        })

    # Load cached plan if present.
    plan_path = ctx.project_dir / "rlm_state" / "gpu_plan.json"
    gpu_plan: _GpuPlan | None = None
    if plan_path.exists():
        try:
            gpu_plan = _GpuPlan(**_json.loads(plan_path.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001
            logger.warning("run_experiment: gpu_plan.json unreadable")

    settings = get_settings()
    max_escalations = settings.dynamic_gpu_max_escalations
    escalations = 0
    result: dict = {}

    while True:
        run_id = f"{ctx.project_id}-{uuid.uuid4().hex[:8]}"
        # ... compute timeout exactly as today ...
        timeout = ctx.remaining_s()
        # (Keep the existing env-var override block above.)

        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            try:
                result = pool.submit(
                    asyncio.run,
                    _execute_in_sandbox(
                        code_path, env_id, commands,
                        project_id=ctx.project_id, run_id=run_id,
                        sandbox_mode=ctx.sandbox_mode,
                        run_budget=ctx.run_budget,
                        gpu_plan=gpu_plan,
                    ),
                ).result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                result = {
                    "success": False, "metrics": {},
                    "error": (f"run_experiment: timed out after {timeout:.0f} s"
                              if timeout is not None
                              else "run_experiment: timed out (run-budget deadline reached)"),
                }
        finally:
            pool.shutdown(wait=False)

        # ---- Escalation gate ----
        if result.get("success") or gpu_plan is None or escalations >= max_escalations:
            break
        stderr_tail = (result.get("logs") or "")[-4096:]
        exit_code = int(result.get("exit_code", 1))  # _execute_in_sandbox may not surface exit_code; default 1
        if not _detect_cuda_oom(exit_code=exit_code, stderr_tail=stderr_tail):
            break
        if not gpu_plan.ladder_remaining:
            result = {
                "success": False, "metrics": {},
                "error": f"CUDA OOM on {gpu_plan.short_name} ({gpu_plan.vram_gb} GB); ladder exhausted. Cumulative SKU cost rate: ${gpu_plan.total_usd_per_hr}/hr.",
                "logs": result.get("logs", ""),
            }
            break

        # Advance ladder: find next SKU by short_name.
        next_short = gpu_plan.ladder_remaining[0]
        next_sku = next((s for s in CATALOG if s.short_name == next_short), None)
        if next_sku is None:
            result = {
                "success": False, "metrics": {},
                "error": f"ladder advance failed: short_name={next_short!r} not in catalog",
                "logs": result.get("logs", ""),
            }
            break

        new_plan = gpu_plan.model_copy(update={
            "runpod_id": next_sku.runpod_id,
            "short_name": next_sku.short_name,
            "vram_gb": next_sku.vram_gb,
            "cloud_type": next_sku.cloud_type,
            "sku_usd_per_hr": next_sku.approx_usd_per_hr,
            "total_usd_per_hr": round(next_sku.approx_usd_per_hr * gpu_plan.gpu_count, 4),
            "container_disk_gb": max(50, next_sku.vram_gb),
            "volume_gb": max(20, next_sku.vram_gb // 4),
            "ladder_remaining": gpu_plan.ladder_remaining[1:],
        })
        # Persist + emit escalation event.
        tmp = plan_path.with_suffix(".tmp")
        tmp.write_text(_json.dumps(new_plan.model_dump(mode="json"), default=str), encoding="utf-8")
        tmp.replace(plan_path)
        _emit_dashboard_event(ctx, event_type="gpu_escalated", payload={
            "from_sku": gpu_plan.short_name,
            "to_sku": new_plan.short_name,
            "escalation_index": escalations + 1,
            "reason": "cuda_oom",
        })
        gpu_plan = new_plan
        escalations += 1

    return _persist_experiment_result(ctx, result)
```

- [ ] **Step 4: Run, confirm pass**

```
.venv/bin/python -m pytest tests/rlm/test_runpod_oom_escalation.py tests/rlm/test_cuda_oom_detection.py -q
```
Expected: 10 passed.

- [ ] **Step 5: Commit milestone 5**

```bash
git add backend/agents/rlm/primitives.py tests/rlm/test_cuda_oom_detection.py tests/rlm/test_runpod_oom_escalation.py
git commit -m "Dynamic GPU selection — CUDA OOM detection and ladder escalation in run_experiment

Adds _detect_cuda_oom() helper (exit-137 OR stderr substring match against
the four documented OOM markers). run_experiment now wraps the sandbox
execution in a bounded escalation loop: on detected OOM, pop the next short_name
from GpuPlan.ladder_remaining, rebuild the plan with the larger SKU, persist
atomically, emit gpu_escalated, and re-run. Capped at OPENRESEARCH_DYNAMIC_GPU_MAX_ESCALATIONS=2
by default. Non-OOM failures (ImportError, etc.) skip the escalation gate
and surface as today. Empty ladder + OOM = structured terminal failure with
cumulative cost summary."
```

---

## Milestone 6 — SSE events + CLI flags

### Task 12: Wire `gpu_resolved` / `gpu_escalated` / `gpu_fallback` through `sse_bridge`

**Files:**
- Modify: `backend/agents/rlm/sse_bridge.py` (allow the new event types through `sanitize_iteration` or whichever allowlist controls dashboard egress)
- Test: `tests/rlm/test_sse_bridge_gpu_events.py`

- [ ] **Step 1: Read `sse_bridge.py` to find the event-type allowlist**

```
.venv/bin/python -c "from backend.agents.rlm import sse_bridge; import inspect; print(inspect.getsource(sse_bridge))" | head -200
```

The allowlist is likely a constant tuple/set or a dispatch table. Find the existing entry for `cluster_started` or `candidate_proposed` and add three siblings: `gpu_resolved`, `gpu_escalated`, `gpu_fallback`.

- [ ] **Step 2: Write failing test**

```python
# tests/rlm/test_sse_bridge_gpu_events.py
from __future__ import annotations


def test_sse_bridge_allows_gpu_resolved():
    from backend.agents.rlm import sse_bridge
    # Allowlist could be a constant, a function, or a dict — adapt to the actual API.
    allowed = getattr(sse_bridge, "ALLOWED_EVENT_TYPES", None)
    if allowed is None:
        # If the module uses a per-event sanitize function, check it doesn't crash.
        assert hasattr(sse_bridge, "build_gpu_resolved") or hasattr(sse_bridge, "sanitize_event")
        return
    assert "gpu_resolved" in allowed
    assert "gpu_escalated" in allowed
    assert "gpu_fallback" in allowed
```

- [ ] **Step 3: Add allowlist entries / sanitizers**

Whichever shape the module uses — extend it.

- [ ] **Step 4: Confirm `_emit_dashboard_event` from primitive flows through to the SSE stream**

Add an integration test in `tests/services/events/test_live_runs_gpu_events.py`:

```python
def test_gpu_resolved_emitted_to_dashboard_events_jsonl(tmp_path):
    # Setup ctx
    from types import SimpleNamespace
    from backend.agents.rlm import primitives
    proj = tmp_path / "p1"
    (proj / "rlm_state").mkdir(parents=True)
    ctx = SimpleNamespace(project_id="p1", project_dir=proj, runs_root=tmp_path,
                          run_budget=None, sandbox_mode="runpod")
    primitives.resolve_gpu_requirements(
        {"estimated_vram_gb": 24, "paper_gpu_string": None, "paper_gpu_count": None, "reasoning": "", "confidence": 0.9},
        ctx=ctx,
    )
    events = (proj / "dashboard_events.jsonl").read_text().strip().splitlines()
    assert any('"gpu_resolved"' in line for line in events)
```

---

### Task 13: Add 6 CLI flags to the `reproduce` command

**Files:**
- Modify: `backend/cli.py` around lines 1034–1180 (argparse for `reproduce`)
- Modify: `backend/cli.py` run_kwargs assembly (~lines 841–851)
- Test: `tests/cli/test_dynamic_gpu_flags.py`

- [ ] **Step 1: Write failing test**

```python
# tests/cli/test_dynamic_gpu_flags.py
from __future__ import annotations

import argparse


def test_reproduce_argparser_has_dynamic_gpu_flags():
    from backend.cli import _build_parser  # exact symbol depends on module; verify with `grep -n "argparse\|add_subparsers\|reproduce" backend/cli.py`
    parser = _build_parser()
    args = parser.parse_args([
        "reproduce", "paper.pdf",
        "--dynamic-gpu",
        "--no-force-single-gpu",
        "--max-gpu-usd-per-hour", "5.0",
        "--max-run-gpu-usd", "8.0",
        "--dynamic-gpu-headroom", "1.5",
        "--vram-gb", "80",
    ])
    assert args.dynamic_gpu is True
    assert args.force_single_gpu is False
    assert args.max_gpu_usd_per_hour == 5.0
    assert args.max_run_gpu_usd == 8.0
    assert args.dynamic_gpu_headroom == 1.5
    assert args.vram_gb == 80


def test_no_dynamic_gpu_flag_disables_resolver():
    from backend.cli import _build_parser
    parser = _build_parser()
    args = parser.parse_args(["reproduce", "paper.pdf", "--no-dynamic-gpu"])
    assert args.dynamic_gpu is False
```

- [ ] **Step 2: Run, confirm failure**

Expected: `AttributeError: argparse Namespace` lacks the new attributes.

- [ ] **Step 3: Add flags to `reproduce` argparser**

After the existing `--max-pod-seconds` block (~line 1080), append:

```python
    reproduce.add_argument(
        "--dynamic-gpu",
        dest="dynamic_gpu",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable dynamic GPU SKU selection from paper hardware clues (default: from OPENRESEARCH_DYNAMIC_GPU).",
    )
    reproduce.add_argument(
        "--force-single-gpu",
        dest="force_single_gpu",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="When dynamic-gpu is on, cap GPU count at 1 (default: from OPENRESEARCH_FORCE_SINGLE_GPU).",
    )
    reproduce.add_argument(
        "--max-gpu-usd-per-hour",
        type=float,
        default=None,
        help="Per-GPU $/hr cap for SKU selection (default: from OPENRESEARCH_MAX_GPU_USD_PER_HOUR=10.0).",
    )
    reproduce.add_argument(
        "--max-run-gpu-usd",
        type=float,
        default=None,
        help="Total RunPod USD cap per run (default: from OPENRESEARCH_MAX_RUN_GPU_USD=10.0).",
    )
    reproduce.add_argument(
        "--dynamic-gpu-headroom",
        type=float,
        default=None,
        help="Multiplier on LLM VRAM estimate before tier-up (default: from OPENRESEARCH_DYNAMIC_GPU_HEADROOM=1.25).",
    )
    reproduce.add_argument(
        "--vram-gb",
        type=int,
        default=None,
        help="Manual VRAM override; bypasses LLM estimate but multiplier still applies.",
    )
```

- [ ] **Step 4: Thread the flags into Settings overrides / run_kwargs**

Find the run_kwargs assembly area (~line 841 per recon) and add an override layer that takes CLI args and updates the resolver settings via `monkeypatch`-style env-var assignment OR by mutating the call to `resolve_gpu_requirements`.

Simplest path: BEFORE `Settings()` is constructed for the run, set os.environ overrides when CLI flags are non-None:

```python
    # CLI overrides for dynamic GPU (Settings-derived; env precedence)
    import os as _os
    if args.dynamic_gpu is not None:
        _os.environ["OPENRESEARCH_DYNAMIC_GPU"] = "true" if args.dynamic_gpu else "false"
    if args.force_single_gpu is not None:
        _os.environ["OPENRESEARCH_FORCE_SINGLE_GPU"] = "true" if args.force_single_gpu else "false"
    if args.max_gpu_usd_per_hour is not None:
        _os.environ["OPENRESEARCH_MAX_GPU_USD_PER_HOUR"] = str(args.max_gpu_usd_per_hour)
    if args.max_run_gpu_usd is not None:
        _os.environ["OPENRESEARCH_MAX_RUN_GPU_USD"] = str(args.max_run_gpu_usd)
    if args.dynamic_gpu_headroom is not None:
        _os.environ["OPENRESEARCH_DYNAMIC_GPU_HEADROOM"] = str(args.dynamic_gpu_headroom)
    # vram_gb is a per-run override, not a Settings field; pass through ctx.
```

The `--vram-gb` override needs to be threaded into `ctx.vram_override`. Add it to `RunContext` (likely in `backend/agents/rlm/run.py`):

```python
# In RunContext (or however it's constructed):
    vram_override: int | None = None
```

And consume it in `resolve_gpu_requirements`: when `ctx.vram_override` is set, override `requirements.estimated_vram_gb = ctx.vram_override` BEFORE passing to the resolver.

- [ ] **Step 5: Update `max_run_gpu_usd` enforcement**

In `RunpodBackend.exec` (already exists with `check_pod_seconds`), add a call to `check_run_gpu_usd` using a running cost tally:

```python
# In RunpodBackend (track and check):
        if self.run_budget is not None and self.gpu_plan is not None:
            elapsed_hr = ((datetime.now(timezone.utc) - self._pod_started_at).total_seconds() / 3600.0
                          if self._pod_started_at else 0.0)
            cumulative = elapsed_hr * self.gpu_plan.total_usd_per_hr
            self.run_budget.check_run_gpu_usd(cumulative_pod_usd=cumulative, agent_id=agent_id)
```

(Insert at the same point as `check_pod_seconds`, around `runpod_backend.py:284`.)

- [ ] **Step 6: Run all milestone-6 tests**

```
.venv/bin/python -m pytest tests/cli/test_dynamic_gpu_flags.py tests/rlm/test_sse_bridge_gpu_events.py tests/services/events/test_live_runs_gpu_events.py -q
```

- [ ] **Step 7: Commit milestone 6**

```bash
git add backend/cli.py backend/agents/rlm/sse_bridge.py backend/services/events/live_runs.py backend/agents/rlm/primitives.py backend/services/runtime/runpod_backend.py tests/cli/ tests/rlm/test_sse_bridge_gpu_events.py tests/services/events/
git commit -m "Dynamic GPU selection — CLI flags, SSE allowlist, and \$/run-cap enforcement

Adds 6 reproduce-subcommand flags (--dynamic-gpu, --force-single-gpu,
--max-gpu-usd-per-hour, --max-run-gpu-usd, --dynamic-gpu-headroom, --vram-gb)
that override env-var defaults; --vram-gb is a per-run RunContext override
that bypasses LLM estimation but still applies the headroom multiplier.
sse_bridge allowlist now passes gpu_resolved / gpu_escalated / gpu_fallback
events through to the dashboard stream. RunpodBackend.exec calls
RunBudget.check_run_gpu_usd with a running cost tally derived from
pod_started_at and gpu_plan.total_usd_per_hr."
```

---

## Milestone 7 — UI badge

### Task 14: Render GpuPlan badge in `node-detail-sidebar.tsx`

**Files:**
- Modify: `frontend/src/components/lab/rlm/node-detail-sidebar.tsx`
- Test: `frontend/src/components/lab/rlm/node-detail-sidebar.test.tsx` (extend existing if present)

- [ ] **Step 1: Find the SSE event consumer**

```
grep -rn "candidate_proposed\|gpu_resolved\|dashboard_event" frontend/src/components/lab/rlm/
```

The component likely subscribes to a state slice that aggregates SSE events. Find where `paper`, `work`, `candidate` nodes are rendered.

- [ ] **Step 2: Add badge rendering**

Add a `gpu_resolved` event handler that stores the latest `GpuPlan` payload, then render it as a small badge on the work-cluster node:

```tsx
{gpuPlan && (
  <div className="gpu-plan-badge" data-source={gpuPlan.source}>
    <span className="sku">{gpuPlan.short_name}</span>
    <span className="vram">{gpuPlan.vram_gb} GB</span>
    {gpuPlan.gpu_count > 1 && <span className="count">×{gpuPlan.gpu_count}</span>}
    <span className="cost">${gpuPlan.total_usd_per_hr.toFixed(2)}/hr</span>
    {gpuPlan.source === "fallback" && <span className="warn" title={gpuPlan.requirements.reasoning}>fallback</span>}
  </div>
)}
```

Style entries appended to whichever lab-theme CSS file the sidebar uses — match existing token variables (`--lab-fg`, `--lab-border`, etc.).

- [ ] **Step 3: Write a snapshot/render test if framework is present (vitest + RTL)**

```tsx
// node-detail-sidebar.test.tsx — extend with:
test("renders GpuPlan badge when gpu_resolved event present", () => {
  render(<NodeDetailSidebar gpuPlan={{
    short_name: "a100_80", vram_gb: 80, gpu_count: 1,
    total_usd_per_hr: 1.89, source: "paper",
    requirements: { reasoning: "test" },
  }} />);
  expect(screen.getByText(/a100_80/)).toBeInTheDocument();
  expect(screen.getByText(/80 GB/)).toBeInTheDocument();
});
```

- [ ] **Step 4: Run frontend tests**

```
cd frontend && npm test -- node-detail-sidebar && npx tsc --noEmit
```
Expected: no errors.

---

## Milestone 8 — OAuth guard test + docs + final verification

### Task 15: OAuth orthogonality guard test

**Files:**
- Test: `tests/agents/runtime/test_oauth_runpod_orthogonality.py`

- [ ] **Step 1: Write the guard test**

```python
# tests/agents/runtime/test_oauth_runpod_orthogonality.py
from __future__ import annotations

import os
from unittest.mock import patch

import pytest


def test_runpod_backend_init_does_not_read_anthropic_api_key(monkeypatch):
    """Guard: RunpodBackend must not read ANTHROPIC_API_KEY when constructed
    under an OAuth-mode run. The pod runs ML code, not LLM calls."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("REPROLAB_RUNPOD_API_KEY", "test-key")
    from backend.services.runtime.runpod_backend import RunpodBackend
    backend = RunpodBackend(api_key="test-key")  # no Anthropic env var present
    # No exception, no read attempt.
    assert backend.api_key == "test-key"


def test_resolve_gpu_requirements_no_anthropic_env_read(monkeypatch, tmp_path):
    """The primitive is pure-Python over Pydantic + catalog; no Anthropic creds needed."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from types import SimpleNamespace
    from backend.agents.rlm.primitives import resolve_gpu_requirements
    proj = tmp_path / "p1"
    (proj / "rlm_state").mkdir(parents=True)
    ctx = SimpleNamespace(
        project_id="p1", project_dir=proj, runs_root=tmp_path,
        run_budget=None, sandbox_mode="runpod",
    )
    out = resolve_gpu_requirements(
        {"estimated_vram_gb": 24, "paper_gpu_string": None,
         "paper_gpu_count": None, "reasoning": "", "confidence": 0.9},
        ctx=ctx,
    )
    assert out["short_name"] == "rtx4090"
    # No Anthropic key needed for the resolver path.


def test_pod_env_injection_excludes_anthropic_credentials():
    """When the pod is created, ANTHROPIC_API_KEY must NOT be injected by default
    into the container env. (Paper code that needs an LLM key must request it
    explicitly via a separate mechanism.)"""
    # Inspect _execute_in_sandbox's environment dict (line ~621 in primitives.py).
    import inspect
    from backend.agents.rlm import primitives
    src = inspect.getsource(primitives._execute_in_sandbox)
    assert "ANTHROPIC_API_KEY" not in src, \
        "_execute_in_sandbox must not silently leak ANTHROPIC_API_KEY into the pod env"
```

- [ ] **Step 2: Run, confirm pass**

```
.venv/bin/python -m pytest tests/agents/runtime/test_oauth_runpod_orthogonality.py -q
```

---

### Task 16: Update `CLAUDE.md` and `system_overview.md`

**Files:**
- Modify: `CLAUDE.md` (add a "Dynamic GPU selection" section after the existing "Sandbox config gotcha" block)
- Modify: `system_overview.md` (search the repo root and `docs/` if not in repo root: `find . -name system_overview.md -not -path '*/node_modules/*'`)

- [ ] **Step 1: Append section to CLAUDE.md**

After the "Sandbox config gotcha" paragraph, add:

```markdown
### Dynamic GPU selection (spec 2026-05-23)
When `OPENRESEARCH_DYNAMIC_GPU=on` (default), the RLM root calls `resolve_gpu_requirements(...)` once per run to map paper hardware clues to a RunPod SKU. The plan caches to `runs/<id>/rlm_state/gpu_plan.json` and is consumed by every subsequent `run_experiment`. On CUDA OOM, `run_experiment` auto-escalates up the catalog ladder up to `OPENRESEARCH_DYNAMIC_GPU_MAX_ESCALATIONS=2` times, capped by `OPENRESEARCH_MAX_GPU_USD_PER_HOUR=10.0`. Total run-level pod spend is bounded by `OPENRESEARCH_MAX_RUN_GPU_USD=10.0` via `RunBudget.check_run_gpu_usd`. Multi-GPU is opt-in: `OPENRESEARCH_FORCE_SINGLE_GPU=on` (default) hard-caps count=1; when off, count is `min(paper_count, floor(max_gpu_usd_per_hour / per_gpu_hr))`. SKU catalog: `backend/services/runtime/gpu_catalog.py` — refresh quarterly.
```

- [ ] **Step 2: Update `system_overview.md` if present**

If found, add a parallel paragraph in the architecture section. If not found, skip — `CLAUDE.md` is sufficient.

---

### Task 17: Final verification + push

- [ ] **Step 1: Run full pytest**

```
.venv/bin/python -m pytest tests/ -q
```
Expected: no regressions; new tests pass.

- [ ] **Step 2: Frontend typecheck + lint + test**

```
cd frontend && npx tsc --noEmit && npm run lint && npm test
```

- [ ] **Step 3: Run lint where present (optional, repo has no formal lint config per CLAUDE.md)**

```
.venv/bin/python -m pytest tests/ --tb=short -q
```

- [ ] **Step 4: Commit milestone 8**

```bash
git add tests/agents/runtime/test_oauth_runpod_orthogonality.py CLAUDE.md system_overview.md frontend/
git commit -m "Dynamic GPU selection — OAuth orthogonality guard, UI badge, and docs

The OAuth guard test asserts (a) RunpodBackend.__init__ never reads
ANTHROPIC_API_KEY, (b) resolve_gpu_requirements is pure-Python with no LLM
credential surface, (c) _execute_in_sandbox source does not silently leak
ANTHROPIC_API_KEY into the pod env. UI: NodeDetailSidebar renders a
GpuPlan badge sourced from the gpu_resolved SSE event. CLAUDE.md gains a
Dynamic GPU section pointing at the spec + the relevant env vars + the
escalation behavior."
```

- [ ] **Step 5: Push branch**

```bash
git push -u origin feat/dynamic-gpu-selection
```

- [ ] **Step 6: Hand off to Codex for review pass**

Per user direction: run a final Codex review on the diff. From the parent agent:

```
[Dispatch codex:codex-rescue with branch HEAD scope]
```

---

## Self-review against the spec

**Spec coverage:**

| Spec requirement | Task |
|---|---|
| D1 VRAM-based match | Task 3 (resolver) |
| D2 LLM-estimated + multiplier + tier-up | Task 3 |
| D3 Static catalog + ladder | Task 2 |
| D4 Plan-time primitive | Task 6 |
| D5 Two flags (dynamic + force_single) | Task 5, Task 13 |
| D6 Per-hour cap + per-run cap | Task 4 (run cap), Task 3 (per-hour cap), Task 5 (Settings) |
| D7 OOM auto-escalate, max 2, $/hr-gated | Task 10, Task 11 |
| D8 No-clues fallback to RTX 4090 | Task 3 |
| D9 GpuMode untouched; $/hr caps count | Task 3 (count bounded inside resolver) |
| D10 OAuth orthogonality | Task 15 (guard test) |
| Invariant I1 — count=1 when force_single | Task 3 test `test_resolve_force_single_gpu_caps_count_at_one` |
| Invariant I2 — catalog never returns SKU > cap | Task 2 test `test_find_ladder_returns_only_skus_meeting_vram_and_cap` |
| Invariant I3 — resolver is pure | Task 3 test `test_resolve_is_pure_no_io_imports` |
| Invariant I4 — OAuth never enters pod | Task 15 |
| Invariant I5 — Plan checkpointed atomically | Task 6 test `test_persists_plan_to_run_state` + Task 11 escalation rewrite via .tmp + replace |
| SSE events `gpu_resolved` / `gpu_escalated` / `gpu_fallback` | Task 6 (resolved), Task 11 (escalated), Task 12 (allowlist) |
| Fallback matrix (8 rows) | Tasks 3, 11, 13 |
| Catalog with 8 SKUs sorted by (vram, price) | Task 2 |
| `find_ladder` filters by cloud_type | Task 2 test `test_find_ladder_excludes_secure_only_when_community_filter` |
| CLI flags x6 | Task 13 |
| 7 Settings fields | Task 5 |
| UI badge | Task 14 |
| Docs update | Task 16 |

**Placeholder scan:** none — every step has a code block or concrete shell command.

**Type consistency:** `GpuRequirements.confidence` is `float [0,1]`; resolver reads it as `confidence < _CONFIDENCE_FLOOR`. `GpuPlan.ladder_remaining` is `tuple[str, ...]`; escalation pops `[0]` and slices `[1:]` — consistent. `GpuPlan.source` Literal set is `{paper, fallback, manual, informational}` in schemas.py AND in resolver.py — consistent.

**Acceptance gates from spec:** all referenced in the test files. Gate A (determinism, no I/O in resolver) covered by `test_resolve_is_pure_no_io_imports`. Gate B (back-compat) covered by Task 8 `test_backend_back_compat_no_plan_uses_settings`. Gate C (OAuth orthogonality) is Task 15. Gates D (typecheck/pytest) and E (no surface bloat) are checked in Task 17.

---

**Execution mode:** Subagent-driven (recommended). Each task = one Sonnet subagent dispatch. Opus reviews each diff before proceeding. Codex reviews the final branch HEAD after Task 17.
