# Grok Validator on Every Finalize Path — Implementation Plan (Plan 3 of N)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`).

**Goal:** Extract the `_finalize` adversarial-validation-panel block into one reusable run.py helper `_run_finalize_validation_panel(ctx, report, project_dir)`, then call it from ALL three ctx-bearing finalize paths (`_finalize`, `_finalize_fatal_primitive_abort`, `_hard_stop_with_report`) so the grok validator fires on every terminal path — closing the confirmed evasion where a fabrication-abort or wall-clock timeout shipped an unvalidated report.

**Architecture:** The validator panel is already a self-contained, fail-soft, gated block in `_finalize` (run.py ~3571–3648). Move it verbatim into a module-level helper; replace `_finalize`'s inline block with a call (byte-identical); add the same call to the abort path (after its regrade, before its write) and the hard-stop path (after its salvage, before its write, guarded on `ctx is not None`). The helper is gated INSIDE by `external_validator_enabled() and ctx.validator_client is not None`, so with `OPENRESEARCH_EXTERNAL_VALIDATOR` unset (default) every path is byte-identical. The hard-stop path already makes an LLM call (`regrade_for_hard_stop`), so a sync validator call there is consistent with existing behavior.

**Tech Stack:** Python (3.11 floor / 3.12 image / 3.14 dev), pytest (monkeypatch), ruff.

## Global Constraints

- **Red line:** the validator's machine checks read deterministic evidence, never the LLM grade.
- **Byte-identical when OFF:** `OPENRESEARCH_EXTERNAL_VALIDATOR` unset ⇒ the helper no-ops on every path (the existing default).
- **Fail-soft:** the helper is one big `try/except` — a panel failure NEVER breaks any finalize path (preserves the existing `_finalize` guarantee).
- **Verbatim extraction:** move the existing `_finalize` block into the helper unchanged except for (a) dedent, (b) the log-message prefix `_finalize:` → `finalize-validation:` (it is now shared). Do NOT alter the panel logic, the reuse-verdict short-circuit, or the gating.
- **Deferred imports stay deferred** inside the helper (`# noqa: PLC0415`), exactly as today.
- **Lint:** `uvx ruff@0.15.16 check backend/agents/rlm/run.py` clean.
- **Test location:** new file `tests/rlm/test_finalize_validation_panel.py`.
- **Commit policy:** standing authorization — ONE milestone commit at plan end (no push); controller commits after the inline diff review.
- **Test command:** `.venv/bin/python -m pytest tests/rlm/test_finalize_validation_panel.py -v` then `.venv/bin/python -m pytest tests/rlm/ -q`.

---

### Task 1: Extract `_run_finalize_validation_panel` + unit tests; rewire `_finalize`

**Files:**
- Modify: `backend/agents/rlm/run.py` (add the helper near the other finalize helpers, e.g. immediately before `def _finalize_fatal_primitive_abort`; replace `_finalize`'s inline validator block with a call)
- Test: `tests/rlm/test_finalize_validation_panel.py` (new)

**Interfaces:**
- Consumes (all module-level / deferred-importable, already used by the block): `external_validator_enabled`, `run_validation_panel`, `persist_verdict`, `evidence_fingerprint`, `load_verdict` (`backend.agents.rlm.external_validator`); `_validator_separation_tier` (run.py module-level); `os`, `json`, `logger` (run.py module-level).
- Produces: `_run_finalize_validation_panel(ctx: Any, report: Any, project_dir: Path) -> None` — runs the OFFLINE panel + `persist_verdict` when enabled and a `validator_client` is present and no verdict for this evidence fingerprint is already persisted; otherwise no-op. Never raises.

- [ ] **Step 1: Write the failing tests**

```python
# tests/rlm/test_finalize_validation_panel.py
"""Tests for _run_finalize_validation_panel — the shared finalize validator panel."""
from types import SimpleNamespace


def _report():
    return SimpleNamespace(baseline_metrics={}, reproduction_summary="", reported_metrics=None)


def test_panel_noop_when_disabled(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENRESEARCH_EXTERNAL_VALIDATOR", raising=False)
    from backend.agents.rlm.run import _run_finalize_validation_panel
    ctx = SimpleNamespace(validator_client=object(), role_selection=None)
    _run_finalize_validation_panel(ctx, _report(), tmp_path)  # must not raise
    assert not (tmp_path / "rlm_state" / "validation_verdict.json").exists()


def test_panel_noop_when_no_client(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENRESEARCH_EXTERNAL_VALIDATOR", "1")
    from backend.agents.rlm.run import _run_finalize_validation_panel
    ctx = SimpleNamespace(validator_client=None, role_selection=None)
    _run_finalize_validation_panel(ctx, _report(), tmp_path)  # no client -> no-op
    assert not (tmp_path / "rlm_state" / "validation_verdict.json").exists()


def test_panel_runs_when_enabled_with_client(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENRESEARCH_EXTERNAL_VALIDATOR", "1")
    from backend.agents.rlm.run import _run_finalize_validation_panel
    panel_calls, persist_calls = [], []
    fake_verdict = SimpleNamespace(status="clean", veto_set=[], separation="independent")
    monkeypatch.setattr(
        "backend.agents.rlm.external_validator.run_validation_panel",
        lambda **k: panel_calls.append(k) or fake_verdict,
    )
    monkeypatch.setattr(
        "backend.agents.rlm.external_validator.persist_verdict",
        lambda pd, v: persist_calls.append((pd, v)),
    )
    monkeypatch.setattr(
        "backend.agents.rlm.external_validator.load_verdict",
        lambda pd, expect_fingerprint=None: None,  # not already validated
    )
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "metrics.json").write_text(
        '{"per_model": {"m": {"e": {"b": {"status": "ok", "accuracy": 0.8}}}}}'
    )
    ctx = SimpleNamespace(validator_client=object(), role_selection=None)
    _run_finalize_validation_panel(ctx, _report(), tmp_path)
    assert panel_calls, "panel should have run"
    assert persist_calls, "verdict should have been persisted"


def test_panel_reuses_existing_verdict(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENRESEARCH_EXTERNAL_VALIDATOR", "1")
    from backend.agents.rlm.run import _run_finalize_validation_panel
    panel_calls = []
    monkeypatch.setattr(
        "backend.agents.rlm.external_validator.run_validation_panel",
        lambda **k: panel_calls.append(k),
    )
    monkeypatch.setattr(
        "backend.agents.rlm.external_validator.load_verdict",
        lambda pd, expect_fingerprint=None: SimpleNamespace(status="clean"),  # already validated
    )
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "metrics.json").write_text('{"per_model": {}}')
    ctx = SimpleNamespace(validator_client=object(), role_selection=None)
    _run_finalize_validation_panel(ctx, _report(), tmp_path)
    assert not panel_calls, "should reuse the persisted verdict, not re-run the panel"


def test_panel_failsoft_on_panel_error(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENRESEARCH_EXTERNAL_VALIDATOR", "1")
    from backend.agents.rlm.run import _run_finalize_validation_panel

    def boom(**k):
        raise RuntimeError("panel exploded")

    monkeypatch.setattr("backend.agents.rlm.external_validator.run_validation_panel", boom)
    monkeypatch.setattr(
        "backend.agents.rlm.external_validator.load_verdict",
        lambda pd, expect_fingerprint=None: None,
    )
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "metrics.json").write_text('{"per_model": {}}')
    ctx = SimpleNamespace(validator_client=object(), role_selection=None)
    _run_finalize_validation_panel(ctx, _report(), tmp_path)  # must not raise
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/rlm/test_finalize_validation_panel.py -v`
Expected: FAIL — `ImportError: cannot import name '_run_finalize_validation_panel'`.

- [ ] **Step 3: Add the helper**

In `backend/agents/rlm/run.py`, add this module-level function immediately BEFORE `def _finalize_fatal_primitive_abort(`. Its body is the EXACT validation-panel block currently in `_finalize` (the `try:` … `except Exception:` at ~run.py:3576–3648), dedented to function-body level, with the two `logger` message prefixes changed from `_finalize:` to `finalize-validation:`:

```python
def _run_finalize_validation_panel(ctx: Any, report: Any, project_dir: Path) -> None:
    """OFFLINE adversarial validation panel (report-stamping only), shared by every
    ctx-bearing finalize path so the grok validator runs on the normal, fatal-abort,
    AND hard-stop paths — not just the happy one. Gated by OPENRESEARCH_EXTERNAL_VALIDATOR
    + ctx.validator_client (unset/None -> no-op -> byte-identical). Reuses a verdict the
    P3 FINAL_VAR gate already persisted for this evidence (no duplicate panel). Fail-soft:
    a panel failure must NEVER break finalize."""
    try:
        from backend.agents.rlm.external_validator import (  # noqa: PLC0415
            external_validator_enabled,
            run_validation_panel,
            persist_verdict,
        )
        _val_client = getattr(ctx, "validator_client", None)
        if external_validator_enabled() and _val_client is not None:
            _val_metrics: dict = dict(report.baseline_metrics) if report.baseline_metrics else {}
            if not _val_metrics:
                _mpath = project_dir / "code" / "metrics.json"
                if _mpath.exists():
                    try:
                        _val_metrics = json.loads(_mpath.read_text(encoding="utf-8"))
                    except Exception:  # noqa: BLE001
                        _val_metrics = {}
            from backend.agents.rlm.external_validator import (  # noqa: PLC0415
                evidence_fingerprint as _val_efp,
                load_verdict as _load_verdict,
            )
            _already_validated = (
                _load_verdict(project_dir, expect_fingerprint=_val_efp(_val_metrics)) is not None
            )
            if _already_validated:
                logger.info("finalize-validation: reusing the validator verdict from the FINAL_VAR gate (no re-run)")
            else:
                _leaf_records: list[dict] = []
                _eval_p = project_dir / "rubric_evaluation.json"
                if _eval_p.exists():
                    try:
                        _eval_data = json.loads(_eval_p.read_text(encoding="utf-8"))
                        _leaf_records = list(_eval_data.get("leaf_scores", {}).values())
                    except Exception:  # noqa: BLE001
                        _leaf_records = []
                _val_tier = _validator_separation_tier(getattr(ctx, "role_selection", None))
                _val_label = os.environ.get("OPENRESEARCH_VALIDATOR_MODEL", "").strip() or \
                             os.environ.get("OPENRESEARCH_VALIDATOR_BACKEND", "").strip() or "validator"
                _report_claims_for_panel = None
                if os.environ.get("OPENRESEARCH_VALIDATOR_CHECK_REPORT", "").strip() == "1":
                    try:
                        from backend.agents.rlm.claim_grounding import extract_result_claims as _erc  # noqa: PLC0415
                        _summary = getattr(report, "reproduction_summary", "") or ""
                        _rm = getattr(report, "reported_metrics", None)
                        _rm_text = (
                            _rm if isinstance(_rm, str) else
                            (__import__("json").dumps(_rm, default=str) if _rm else "")
                        )
                        _report_claims_for_panel = _erc(_summary + "\n" + _rm_text)
                    except Exception:  # noqa: BLE001
                        _report_claims_for_panel = None
                _verdict = run_validation_panel(
                    validator_client=_val_client,
                    panel_models=[_val_label],
                    metrics=_val_metrics,
                    project_dir=project_dir,
                    leaf_records=_leaf_records,
                    separation=_val_tier,
                    report_claims=_report_claims_for_panel,
                )
                persist_verdict(project_dir, _verdict)
                logger.info(
                    "finalize-validation: validation panel complete — status=%s veto_set=%r separation=%s",
                    _verdict.status, _verdict.veto_set, _verdict.separation,
                )
    except Exception:  # noqa: BLE001 — panel failure must never break finalize
        logger.warning("finalize-validation: external validation panel failed (non-fatal)", exc_info=True)
```

- [ ] **Step 4: Replace `_finalize`'s inline block with a call**

In `_finalize`, DELETE the entire inline validation-panel section — from the comment line `# P2.3 — OFFLINE adversarial validation panel (report-stamping only).` through the end of its `except Exception:` block (`logger.warning("_finalize: external validation panel failed (non-fatal)", exc_info=True)`), i.e. the block now living in the helper — and REPLACE it with:

```python
    # P2.3 — OFFLINE adversarial validation panel (report-stamping only). Extracted to
    # _run_finalize_validation_panel so the fatal-abort + hard-stop paths run it too.
    # Runs BEFORE write_final_report_rlm so the verdict is on disk for the stamp chokepoint.
    if not run_failed:
        _run_finalize_validation_panel(ctx, report, project_dir)
```

(The `if not run_failed:` preserves `_finalize`'s existing behavior — the panel was already inside the normal-completion flow; keep it gated the same way the regrade above it is.)

- [ ] **Step 5: Run tests + lint**

Run: `.venv/bin/python -m pytest tests/rlm/test_finalize_validation_panel.py -v`
Expected: PASS (5 tests).

Run: `uvx ruff@0.15.16 check backend/agents/rlm/run.py`
Expected: clean.

---

### Task 2: Call the panel from the fatal-abort and hard-stop paths

**Files:**
- Modify: `backend/agents/rlm/run.py` (`_finalize_fatal_primitive_abort`, `_hard_stop_with_report`)

**Interfaces:**
- Consumes: `_run_finalize_validation_panel` (Task 1).
- Produces: no new symbol — the two paths now run the panel (gated; byte-identical when `OPENRESEARCH_EXTERNAL_VALIDATOR` is off).

- [ ] **Step 1: Wire the fatal-abort path**

In `_finalize_fatal_primitive_abort`, find the regrade block immediately followed by the report write:

```python
    try:
        from backend.agents.rlm import finalize_regrade as _fr
        _fr.regrade_and_emit(ctx, report, emit)
    except Exception:  # noqa: BLE001
        logger.warning("_finalize_fatal_primitive_abort: regrade failed (non-fatal)", exc_info=True)
    json_path, _md_path = write_final_report_rlm(
```

Insert the panel call between the regrade `except` and the `json_path` write:

```python
    try:
        from backend.agents.rlm import finalize_regrade as _fr
        _fr.regrade_and_emit(ctx, report, emit)
    except Exception:  # noqa: BLE001
        logger.warning("_finalize_fatal_primitive_abort: regrade failed (non-fatal)", exc_info=True)
    # Run the adversarial validator on the abort path too (gated; byte-identical when off) —
    # a fabrication-guard abort must not skip the critic. Before write so the stamp sees it.
    _run_finalize_validation_panel(ctx, report, project_dir)
    json_path, _md_path = write_final_report_rlm(
```

- [ ] **Step 2: Wire the hard-stop path**

In `_hard_stop_with_report`, find the salvage call immediately followed by the write `try`:

```python
    salvaged_score = _salvage_partial_report(
        report, project_dir, stop_kind=stop_kind, stop_detail=status_error,
    )
    try:
```

Insert a guarded panel call between the salvage and the write `try` (the hard-stop path already does an LLM regrade above, so a sync panel here is consistent; `ctx` may be None):

```python
    salvaged_score = _salvage_partial_report(
        report, project_dir, stop_kind=stop_kind, stop_detail=status_error,
    )
    # Run the adversarial validator on the wall-clock/SIGTERM hard-stop path too (gated;
    # byte-identical when off). ctx may be None here (the watchdog can fire pre-bind).
    if ctx is not None:
        _run_finalize_validation_panel(ctx, report, project_dir)
    try:
```

- [ ] **Step 3: Lint + full-suite regression**

Run: `uvx ruff@0.15.16 check backend/agents/rlm/run.py`
Expected: clean.

Run: `.venv/bin/python -m pytest tests/rlm/ -q`
Expected: PASS — same counts as before plus the 5 new tests; no regression (the panel is gated off by default, so `_finalize`/abort/hard-stop behavior is unchanged under the default config).

**Note on coverage:** the three call sites are one-liners verified by inspection + the helper's 5 unit tests + the full-suite regression. A full finalize-path integration test (real RLM run) is out of scope; the final Codex review is the backstop.

---

### Task 3: Milestone commit

- [ ] **Step 1: Controller reviews the diff** (`git --no-pager diff backend/agents/rlm/run.py` + the new test file).

- [ ] **Step 2: Commit (controller, standing authorization — no push)**

```bash
git add backend/agents/rlm/run.py tests/rlm/test_finalize_validation_panel.py \
        docs/superpowers/plans/2026-06-20-validator-on-every-finalize-path.md
git commit -m "Run the adversarial validator on every finalize path, not just the happy one

Extracts the _finalize validation-panel block into a shared run.py helper
_run_finalize_validation_panel and calls it from _finalize, _finalize_fatal_primitive_abort,
and _hard_stop_with_report. Closes the confirmed evasion where a fabrication-guard abort or
a wall-clock/SIGTERM hard-stop shipped an unvalidated report (the 6h-timeout case). Gated by
OPENRESEARCH_EXTERNAL_VALIDATOR + ctx.validator_client (default off -> byte-identical); the
hard-stop path already does an LLM regrade, so a sync panel there is consistent. 5 unit tests."
```

Do NOT push.

---

## Self-Review

**1. Spec coverage:** Implements spec §5.1 "make the grok validator fire on all terminal paths via a unified finalize step." The full `FinalizeContext`/`finalize_pipeline` extraction (regrade + champion + floor + validator under one object) is deferred; this plan does the highest-value slice — the validator on every path — with minimal blast radius. Champion-restore on the abort/hard-stop paths is a follow-on (tracked in the ledger).

**2. Placeholder scan:** None. The helper body is the verbatim current block (reproduced in full); both wirings give exact before/after anchors; all 5 tests are complete.

**3. Type consistency:** `_run_finalize_validation_panel(ctx, report, project_dir)` signature matches all three call sites. `run_validation_panel(validator_client=, panel_models=, metrics=, project_dir=, leaf_records=, separation=, report_claims=)` and `persist_verdict(project_dir, verdict)` and `load_verdict(project_dir, expect_fingerprint=)` match the extracted block + the test mocks + the verified `external_validator` API. `ValidatorVerdict` fields used (`status`, `veto_set`, `separation`) match the test's `fake_verdict`.
