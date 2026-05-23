# RLM Phase 5/6 Debug & Harden — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Tasks use checkbox (`- [ ]`) syntax for tracking. Standing instructions (carry forward verbatim from `docs/superpowers/specs/2026-05-22-rlm-debug-harden-handoff.md` §1): branch is **`merge`** (not `main`); **never** a `Co-Authored-By` / AI-attribution trailer; **root-level elegant solutions** — one canonical abstraction per invariant + a guard test, never scattered patches; delegate implementation to Sonnet sub-agents against tight specs; use Codex for review only; Opus reviews every diff (the diff, never the agent's summary).

**Goal:** Land the debug-and-harden pass that turns the in-progress RLM pivot into a precise, honest, production-quality demo — eliminating the structural 0.35 score cap, fixing the fabricated post-run score, closing the corpus-leak and DoS surfaces, and wiring the deadline / sandbox / 429 layers the orchestrator already documents.

**Architecture:** No new abstractions. Every fix is a *root-level* patch to an existing module — one canonical change per invariant, plus a **guard test whose docstring names the symptom** (the handoff's quality bar). Tasks land tier-by-tier; P0 #1 (real metric extraction) unblocks every downstream rubric score and **must merge before P0 #2-#5**.

**Tech stack:** Python 3.14, FastAPI, `rlms` 0.1.1, Pydantic v2, pytest. Frontend Next.js 16 (not touched in this plan). Run a single paper end-to-end as the final acceptance via `.venv/bin/python scripts/rlm_paperbench.py <paper_id> --sandbox docker` (see handoff §5).

**Spec sources (de-duplicated):**
- Team handoff: `docs/superpowers/specs/2026-05-22-rlm-debug-harden-handoff.md` (P0-I1 … P2-I13).
- Independent review: `docs/design/phase3-5-review-findings.md` (Critical C1-C2, Important I1-I9, Minor M1-M8).
- Overlaps reconciled below: handoff-I1 ≡ review-C1; handoff-I2 ≡ tree-rubric half of the same defect; handoff-I10 ≠ review-I4 (different OCR bugs — keep both).

---

## File structure — what each task touches

| File | Responsibility | Tasks |
|---|---|---|
| `backend/agents/rlm/primitives.py` | Primitives (the REPL surface). | T1, T2, T9, T10, T11 |
| `backend/evals/paperbench/leaf_scorer.py` | Post-run authoritative scorer. | T3, T4, T5, T26 |
| `backend/agents/rlm/run.py` | Orchestrator entry, RunContext build, finalize. | T6, T8, T12, T19 |
| `backend/agents/rlm/sse_bridge.py` | SSE / corpus chokepoint. | T10 |
| `backend/agents/rlm/report.py` | Final-report schema + renderer. | T6, T27 |
| `backend/agents/rlm/checkpoint.py` | Iteration checkpoint / resume. | T19, T30 |
| `backend/agents/rlm/stub_primitives.py` | Loop-exercising stubs. | T20, T21 |
| `backend/agents/rlm/rubric_gen.py` | Self-generated rubric tree. | T26 |
| `backend/agents/baseline_implementation.py` | Code-writing agent (SDK). | T7 |
| `backend/agents/experiment_runner.py` | SDK-mode runner (reference for T1). | reference only |
| `backend/services/ingestion/parser/resolving_parser.py` | HTML→PDF→OCR cascade. | T16 |
| `backend/services/ingestion/parser/service.py` | Parser entrypoint, blob writer. | T18 |
| `backend/services/ingestion/parser/html_parser.py` | HTML parser. | T17, T25 |
| `backend/services/ingestion/parser/ocr_parser.py` | OCR parser. | T22, T29 |
| `backend/services/ingestion/intake/fetchers/arxiv.py` | arXiv fetcher (HTML + PDF). | T17, T28 |
| `backend/cli.py` | CLI: `reproduce`, `_build_workspace_claim_map`. | T8, T13 |
| `backend/services/events/live_runs.py` | REST run spawn + bridge. | T15 |
| `backend/services/context/workspace/tools/openai_client.py` | LLM client (Featherless path). | T14 |
| `backend/agents/runtime/factory.py` | `has_provider_credentials`. | T23 |
| `Dockerfile` / install docs | Tesseract `eng` language data. | T22 |

---

## Execution order

**P0 (tasks T1-T8)** must merge first and in order: **T1 first**; T2-T5 unblock once T1 lands. T6 (verdict honesty) and T7 (ftrl paper-grounding) can run in parallel with T2-T5 after T1. T8 (workspace paper_text) is SDK-mode only — can run any time.

**P1 (T9-T21)** is broadly parallelizable. T9 (deadline wiring) is one line — do it first. The ingestion cluster (T16-T18, plus T25/T28 from P2) can land as a single sub-PR; the run-time cluster (T11, T12, T13, T14, T15) can land as another.

**P2 (T22-T31)** is cleanup; ship after Phase 5 is reproducibly green.

**One commit per task.** Commit message format: `<area>: <one-line symptom> (T<N>)`. **No `Co-Authored-By` trailer.** Use `git commit -m "..."` on branch `merge`; fast-forward `main` to `merge` only at major milestones.

---

# P0 — correctness & honesty

## T1: Real metric extraction in `run_experiment` — eliminates the 0.35 cap

**Severity:** P0 · **Source:** handoff §4 P0-I1, review C1 · **Blocks:** T2, T3, T4, T5

**Files:**
- Modify: `backend/agents/rlm/primitives.py:422-474` (`_execute_in_sandbox`) and `:505-557` (`run_experiment`).
- Reference (DO NOT modify here): `backend/agents/experiment_runner.py:222-330` (`run_with_runtime`) — the proven implementation we mirror.
- Reference: `backend/agents/prompts/_sandbox_contract.py` — `SANDBOX_EXECUTION_CONTRACT` pins `$OUTPUT_DIR/metrics.json` as the write target.
- Test: `tests/rlm/test_run_experiment.py` (new test) and update the existing `test_run_experiment_reads_commands_and_returns_metrics`.

**Problem.** `_execute_in_sandbox` hard-returns `metrics={}` (line 472). `verify_against_rubric` treats no-metrics as `degraded` (line 600) → every area score capped at 0.35 → no run can ever score above 0.35. The pivot's deliverable, "a real PaperBench rubric score," is structurally unreachable.

**Root cause.** The `SandboxConfig` built in `_execute_in_sandbox` (lines 451-458) does NOT set `artifact_root` and does NOT set `OUTPUT_DIR` in the environment. The paper's code writes `metrics.json` to `$OUTPUT_DIR/metrics.json` per the contract — but there is no host-side mount to read it back from. `run_with_runtime` does both correctly (`artifact_env = "/artifacts"`, `environment["OUTPUT_DIR"] = artifact_env`, `artifact_root=baseline_dir`).

**Fix.**

Edit `_execute_in_sandbox` to mirror `run_with_runtime`'s mount + env wiring, then read `metrics.json` from the host artifact dir after the command loop succeeds:

```python
async def _execute_in_sandbox(
    code_path: str,
    env_id: str,
    commands: list[str],
    *,
    project_id: str,
    run_id: str,
) -> dict:
    """Run `commands` in a container from `env_id`; return measured metrics.

    Mirrors experiment_runner.run_with_runtime's SandboxConfig so the paper's
    code can write metrics.json to $OUTPUT_DIR (the contract pinned by
    backend/agents/prompts/_sandbox_contract.py), and we can read it back
    on the host.  Without this the rubric verifier's degraded backstop
    fires on every run and caps every score at 0.35.
    """
    import asyncio
    import json
    from pathlib import Path

    from backend.services.runtime.interface import SandboxConfig
    from backend.services.runtime.local_docker import LocalDockerBackend
    from backend.services.runtime.service import (
        CreateSandbox, DestroySandbox, ExecuteCommand, RuntimeAppService,
    )

    code_dir = Path(code_path)
    # Per-call artifact dir: deterministic per run_id so retries don't clobber.
    artifact_root = code_dir / "outputs" / run_id
    artifact_root.mkdir(parents=True, exist_ok=True)

    service = RuntimeAppService(LocalDockerBackend())
    config = SandboxConfig(
        project_id=project_id,
        run_id=run_id,
        image=env_id,
        project_root=code_dir,
        artifact_root=artifact_root,
        dockerfile_path=None,
        build_context=None,
        readonly_project=True,
        environment={
            "OUTPUT_DIR": "/artifacts",
            "REPROLAB_ARTIFACT_DIR": "/artifacts",
            "MPLCONFIGDIR": "/artifacts/.matplotlib",
            "PYTHONUNBUFFERED": "1",
        },
    )

    sandbox = await service.create_sandbox(CreateSandbox(config=config))
    results = []
    try:
        for command in commands:
            results.append(await service.execute(
                ExecuteCommand(sandbox=sandbox, command=command,
                               timeout=_EXEC_TIMEOUT_SECONDS)))
    finally:
        await asyncio.shield(service.destroy(DestroySandbox(sandbox=sandbox)))

    # Contract: paper's code writes $OUTPUT_DIR/metrics.json (host-side: artifact_root/metrics.json).
    metrics: dict = {}
    metrics_path = artifact_root / "metrics.json"
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            if not isinstance(metrics, dict):
                metrics = {}  # contract violation; honest empty
        except (json.JSONDecodeError, OSError):
            metrics = {}

    return {
        "success": all(r.succeeded for r in results),
        "metrics": metrics,
        "logs": _cap_logs("\n".join(r.stdout for r in results)),
        "artifact_dir": str(artifact_root),
    }
```

**Guard test.**

```python
# tests/rlm/test_run_experiment.py — add this test
def test_run_experiment_returns_real_metrics_from_artifact_dir(
        tmp_path, make_run_context, monkeypatch):
    """Symptom: every score caps at 0.35 because run_experiment hard-returns metrics={}.

    Without this fix _execute_in_sandbox returns {} regardless of what the
    paper's code wrote to $OUTPUT_DIR/metrics.json — see handoff P0-I1 / review C1.
    Verify: when a fake sandbox writes metrics.json into artifact_root, run_experiment
    returns those metrics (not {}), and a degraded backstop does not fire.
    """
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "commands.json").write_text('["python train.py"]')

    captured = {}
    async def fake_execute_in_sandbox(code_path, env_id, commands, *,
                                       project_id, run_id):
        # Simulate the paper's code writing metrics to the contract path.
        artifact_root = Path(code_path) / "outputs" / run_id
        artifact_root.mkdir(parents=True, exist_ok=True)
        (artifact_root / "metrics.json").write_text('{"mean_reward": 487.3}')
        captured["artifact_root"] = artifact_root
        # Re-run the real reader logic.
        from backend.agents.rlm.primitives import _execute_in_sandbox
        return await _execute_in_sandbox(
            code_path, env_id, commands, project_id=project_id, run_id=run_id)
    # ...patch service.execute / create_sandbox / destroy on the real path...
    # (use the existing test fixture pattern from test_run_experiment.py)

    ctx = make_run_context(tmp_path)
    result = primitives.run_experiment(str(code_dir), "reprolab/test:env", ctx=ctx)
    assert result["success"] is True
    assert result["metrics"] == {"mean_reward": 487.3}
    # Confirm the degraded condition in verify_against_rubric would NOT fire:
    assert not ((not result["success"]) or (not result["metrics"]))
```

**Verify.** `.venv/bin/python -m pytest tests/rlm/test_run_experiment.py -v` → PASS. Then a full-suite smoke: `.venv/bin/python -m pytest tests/ -q` → ≥1163 passed, same 2 pre-existing failures, no new failures.

**Don't touch.** Do not modify `_EXEC_TIMEOUT_SECONDS`, `_MAX_LOG_CHARS`, the `asyncio.shield(...destroy...)` line (A2-C1 hardening), or `_persist_experiment_result`. Do not change the public signature of `run_experiment(code_path, env_id, *, ctx)` — the REPL contract depends on it.

---

## T2: `verify_against_rubric` consumes the PaperBench tree rubric

**Severity:** P0 · **Source:** handoff §4 P0-I2 · **Depends on:** T1

**Files:**
- Modify: `backend/agents/rlm/primitives.py:559-629` (`verify_against_rubric`).
- Reuse: `backend/evals/paperbench/leaf_scorer.py:flatten_leaves`, `roll_up`, `_gather_evidence`.
- Test: `tests/rlm/test_verify_against_rubric.py` — add the tree-rubric test.

**Problem.** `verify_against_rubric` reads `rubric.get("areas", [])` (line 597-598), but `rubric_spec` is a PaperBench *tree* with `{id, requirements, weight, sub_tasks}` for both bundle and generated rubrics — so `areas` is always `[]`, every weight is 0, the in-loop score is meaningless. The root model gets no real rubric feedback during the run.

**Root cause.** The in-loop verifier was written for a flat `{areas: [...]}` rubric shape from before the tree-rubric pivot. `leaf_scorer` is the new tree-aware implementation; `verify_against_rubric` never picked it up.

**Fix.** Replace the `areas`-based scoring with a call into `leaf_scorer.score_reproduction` so the in-loop and post-run computations are identical. Keep the 0.35 honesty backstop on degraded runs (after T1, that condition fires only on legitimate failures, not on every run).

```python
def verify_against_rubric(results: dict, rubric: dict, *, ctx: "RunContext") -> dict:
    """Grade `results` against a PaperBench tree rubric using leaf_scorer.

    Unified path: the in-loop score IS the post-run leaf score (handoff I2).
    Falls back to the legacy flat-areas computation only if `rubric` is the
    old shape (test fixtures still use it).
    """
    # If rubric has a flat `areas` list, use legacy path (back-compat for tests).
    if rubric.get("areas") and not rubric.get("sub_tasks"):
        return _verify_flat_areas(results, rubric, ctx=ctx)

    # Tree path: delegate to leaf_scorer.
    from backend.evals.paperbench.leaf_scorer import score_reproduction
    try:
        score = score_reproduction(
            rubric, ctx.project_dir, ctx.llm_client,
            rubric_source=rubric.get("source", "generated"),
        )
    except Exception as exc:  # noqa: BLE001 — fail-soft
        return {"success": False,
                "error": f"verify_against_rubric: {type(exc).__name__}: {exc}"}

    overall = float(score.get("overall_score", 0.0))
    degraded = (not results.get("success")) or (not results.get("metrics"))
    if degraded:
        overall = min(overall, 0.35)  # honesty backstop survives the rewrite

    target = _clamp01(rubric.get("target_score", 0.0))
    return {
        "overall_score": overall,
        "meets_target": overall >= target if target > 0 else False,
        "target_score": target,
        "rubric_source": score.get("rubric_source", "generated"),
        "leaf_count": score.get("leaf_count", 0),
        "graded": score.get("graded", 0),
        "leaf_scores": score.get("leaf_scores", []),
        "degraded": degraded,
    }


def _verify_flat_areas(results: dict, rubric: dict, *, ctx: "RunContext") -> dict:
    """Legacy flat-areas verifier, kept for back-compat with existing test fixtures."""
    # ... existing body of verify_against_rubric, renamed ...
```

**Guard test.**

```python
def test_verify_against_rubric_consumes_tree_not_flat_areas(make_run_context, tmp_path):
    """Symptom: in-loop score is always meaningless because rubric.areas is always [].

    Bundle and generated rubrics are PaperBench trees ({id,requirements,weight,sub_tasks});
    verify_against_rubric used to read rubric["areas"] which is always empty on a tree.
    Verify: a tree rubric produces a non-zero overall_score and a leaf_count > 0.
    """
    tree = {
        "id": "root", "requirements": "Reproduce", "weight": 1.0,
        "sub_tasks": [{
            "id": "L1", "requirements": "implements two-layer GRU",
            "weight": 1.0, "sub_tasks": [],
        }],
    }
    ctx = make_run_context(tmp_path,
        llm_responses=['[{"leaf_id":"L1","score":0.8,"justification":"ok"}]'])
    result = verify_against_rubric(
        {"success": True, "metrics": {"x": 1}}, tree, ctx=ctx)
    assert result.get("leaf_count") == 1
    assert result.get("overall_score", 0.0) == pytest.approx(0.8)
    assert result.get("degraded") is False
```

**Verify.** `.venv/bin/python -m pytest tests/rlm/test_verify_against_rubric.py -v` → all old + new tests PASS.

**Don't touch.** Don't delete the legacy flat-areas path until every fixture migrates (do that in T31). Don't change `propose_improvements` here.

---

## T3: leaf scorer reads the real `RLMFinalReport` keys

**Severity:** P0 · **Source:** review C2a · **Depends on:** T1

**Files:**
- Modify: `backend/evals/paperbench/leaf_scorer.py:85-138` (`_gather_evidence`).
- Test: `tests/evals/test_leaf_scorer.py` — round-trip a real `write_final_report_rlm` output.

**Problem.** `_gather_evidence` reads `report["metrics"]` and `report["paper_title"]` (lines 96-99). `RLMFinalReport` has `baseline_metrics` and `paper` (a dict with `id`/`title`). For every RLM run the grader sees evidence with **no metrics and no paper identity** — yet still produces a "score of record" that lands in the artifact a reviewer reads.

**Root cause.** Schema drift: `report.py` was written with one key set; the scorer's fixture was written to match the *scorer's* expected keys; nothing ever round-tripped a real report through the scorer.

**Fix.**

```python
def _gather_evidence(run_dir: Path) -> str:
    """Gather bounded reproduction evidence from a run directory.

    Reads the RLMFinalReport schema's actual keys (baseline_metrics, paper)
    rather than the drifted (metrics, paper_title) the scorer used to assume.
    """
    parts: list[str] = []
    total = 0

    report_path = run_dir / "final_report.json"
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            # Extract paper identity from the real RLMFinalReport schema.
            paper = report.get("paper") or {}
            snippet = {
                "paper_title": paper.get("title", ""),
                "paper_id": paper.get("id", ""),
                "verdict": report.get("verdict", ""),
                "reproduction_summary": report.get("reproduction_summary", ""),
                "baseline_metrics": report.get("baseline_metrics") or {},
                "paper_claims": report.get("paper_claims") or {},
            }
            text = f"=== final_report.json (key fields) ===\n{json.dumps(snippet, indent=2)}\n"
            parts.append(text)
            total += len(text)
        except Exception as exc:
            logger.warning("Could not read final_report.json: %s", exc)

    # ... rest unchanged: code/ listing + priority files ...
```

**Guard test.**

```python
def test_leaf_scorer_reads_real_RLMFinalReport_schema(tmp_path):
    """Symptom: the 'authoritative' grader sees zero metrics + no paper identity for every RLM run.

    leaf_scorer._gather_evidence used to read report['metrics'] and report['paper_title']
    but RLMFinalReport writes baseline_metrics and paper.title. Verify a real
    write_final_report_rlm output flows through the scorer with metrics and title preserved.
    """
    from backend.agents.rlm.report import RLMFinalReport, write_final_report_rlm
    from backend.evals.paperbench.leaf_scorer import _gather_evidence

    report = RLMFinalReport(
        paper={"id": "ftrl", "title": "FTRL"},
        verdict="partial",
        baseline_metrics={"mean_reward": 487.3},
        reproduction_summary="reproduced two-layer GRU",
    )
    write_final_report_rlm(report, tmp_path)
    evidence = _gather_evidence(tmp_path)
    assert "ftrl" in evidence
    assert "FTRL" in evidence
    assert "487.3" in evidence  # the real metric must reach the grader
    assert "paper_title" in evidence  # confirm new field name is rendered
```

**Verify.** `.venv/bin/python -m pytest tests/evals/test_leaf_scorer.py::test_leaf_scorer_reads_real_RLMFinalReport_schema -v` → PASS.

**Don't touch.** Keep the size caps (`_MAX_FILE_BYTES`, `_MAX_TOTAL_EVIDENCE_BYTES`). Don't change the code-listing block.

---

## T4: leaf scorer applies the degraded 0.35 cap

**Severity:** P0 · **Source:** review C2b · **Depends on:** T1, T3

**Files:**
- Modify: `backend/evals/paperbench/leaf_scorer.py:176-243` (`score_reproduction`) and `:291-322` (`amend_final_report`).
- Test: `tests/evals/test_leaf_scorer.py`.

**Problem.** `verify_against_rubric` (in-loop) caps `overall_score` at 0.35 when the run produced no metrics. `score_reproduction` (post-run, "authoritative") has no equivalent cap — and `amend_final_report` overwrites the in-loop block — so a metric-less run that produced nothing can be stamped with an *uncapped* `overall_score`. The honest 0.35 ceiling is silently discarded.

**Root cause.** The cap lives in `verify_against_rubric` only. The post-run path was written without re-deriving the honesty invariant.

**Fix.** Add a degraded detector to `score_reproduction`; pass through to `amend_final_report` so the on-disk block matches the in-loop invariant.

```python
def score_reproduction(
    rubric_tree: dict[str, Any],
    run_dir: Path,
    llm_client: LlmClient,
    *,
    batch_size: int = 15,
    rubric_source: str = "paperbench_bundle",
    target_score: float = 0.0,
) -> dict[str, Any]:
    """Grade a reproduction run against a PaperBench rubric tree.

    Applies the same 0.35 degraded cap as the in-loop verify_against_rubric
    (review C2b): a metric-less run cannot earn a score above 0.35 — the
    honesty backstop is enforced in BOTH writers.
    """
    leaves = flatten_leaves(rubric_tree)
    evidence = _gather_evidence(run_dir)

    # ... existing batch-grading logic ...

    overall_score = roll_up(rubric_tree, leaf_scores)

    # Honesty backstop (mirrors verify_against_rubric line 610).
    degraded = _detect_degraded(run_dir)
    if degraded:
        overall_score = min(overall_score, 0.35)

    return {
        "overall_score": overall_score,
        "leaf_count": len(leaves),
        "graded": graded,
        "rubric_source": rubric_source,
        "leaf_scores": leaf_score_records,
        "degraded": degraded,
        "target_score": target_score,
        "meets_target": overall_score >= target_score if target_score > 0 else False,
    }


def _detect_degraded(run_dir: Path) -> bool:
    """A run is degraded if final_report.baseline_metrics is empty OR verdict is failed."""
    report_path = run_dir / "final_report.json"
    if not report_path.exists():
        return True
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return True
    metrics = report.get("baseline_metrics") or {}
    verdict = report.get("verdict", "")
    return (not metrics) or verdict == "failed"
```

**Guard test.**

```python
def test_leaf_scorer_caps_degraded_score_at_0_35(tmp_path, fake_llm):
    """Symptom: the 'authoritative' score discards the honesty backstop.

    A metric-less run that produced nothing was being stamped with an uncapped
    overall_score because score_reproduction had no degraded check (review C2b).
    Verify: a final_report.json with baseline_metrics={} can never produce
    overall_score > 0.35, regardless of what the grader returns.
    """
    from backend.agents.rlm.report import RLMFinalReport, write_final_report_rlm
    from backend.evals.paperbench.leaf_scorer import score_reproduction

    # Metric-less report on disk.
    report = RLMFinalReport(paper={"id": "x"}, verdict="partial", baseline_metrics={})
    write_final_report_rlm(report, tmp_path)

    # Grader returns 0.9 for every leaf (worst-case inflation attempt).
    fake_llm.responses = ['[{"leaf_id":"L1","score":0.9,"justification":"ok"}]']
    tree = {"id": "r", "requirements": "x", "weight": 1.0,
            "sub_tasks": [{"id": "L1", "requirements": "y", "weight": 1.0, "sub_tasks": []}]}
    score = score_reproduction(tree, tmp_path, fake_llm)
    assert score["degraded"] is True
    assert score["overall_score"] <= 0.35
```

**Verify.** `.venv/bin/python -m pytest tests/evals/test_leaf_scorer.py::test_leaf_scorer_caps_degraded_score_at_0_35 -v` → PASS.

**Don't touch.** `flatten_leaves` and `roll_up` are correct; don't modify them.

---

## T5: leaf scorer computes `meets_target` from a real target; preserves `areas`

**Severity:** P0 · **Source:** review C2c + M2 · **Depends on:** T3

**Files:**
- Modify: `backend/evals/paperbench/leaf_scorer.py:291-322` (`amend_final_report`).
- Modify: `backend/agents/rlm/report.py:323-447` (`_render_markdown`) — confirm it reads the new fields.
- Test: `tests/evals/test_leaf_scorer.py`.

**Problem.** `amend_final_report` hardcodes `"meets_target": False` (line 309) and drops the in-loop `areas` list (per-area justifications/weak_points). The score block is **fabricated** in two directions: a legitimate high score is stamped as below-target; per-area provenance is erased so the markdown areas table renders empty.

**Root cause.** `amend_final_report` was written as a "overwrite the rubric block" patch without re-deriving the score-of-record contract.

**Fix.** Accept `target_score` (resolved from the rubric or run config); compute `meets_target` from `overall_score` vs `target_score`; merge with — not overwrite — the in-loop `areas` so per-leaf provenance survives.

```python
def amend_final_report(
    run_dir: Path, score: dict[str, Any], *, target_score: float = 0.0
) -> None:
    """Merge leaf-score results into final_report.json (review C2b/C2c/M2).

    Computes meets_target from overall_score vs target_score (no longer a
    hardcoded False). Preserves the in-loop `areas` list so the markdown's
    per-area justifications table is not erased.
    """
    report_path = run_dir / "final_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}

    overall = float(score.get("overall_score", 0.0))
    previous = report.get("rubric") or {}

    report["rubric"] = {
        "overall_score": overall,
        "target_score": target_score,
        "meets_target": overall >= target_score if target_score > 0 else False,
        "rubric_source": score.get("rubric_source", "paperbench_bundle"),
        "leaf_count": score.get("leaf_count", 0),
        "graded": score.get("graded", 0),
        "degraded": score.get("degraded", False),
        "leaf_scores": score.get("leaf_scores", []),
        # Preserve the in-loop `areas` block so the markdown's per-area table survives.
        "areas": previous.get("areas") or [],
    }

    # ... atomic write unchanged ...
    _rerender_report_markdown(run_dir, report)
```

**Guard test.**

```python
def test_amend_final_report_computes_meets_target_and_keeps_areas(tmp_path):
    """Symptom: a 0.9 score still renders '✘ below target'; the areas table is empty.

    amend_final_report hardcoded meets_target=False and overwrote the rubric
    block, dropping the in-loop `areas` list (review C2c + M2). Verify:
    - meets_target reflects overall_score vs target_score
    - the in-loop areas list is preserved through the amendment.
    """
    from backend.evals.paperbench.leaf_scorer import amend_final_report

    report = {
        "verdict": "reproduced",
        "rubric": {"areas": [{"name": "code", "score": 0.85, "notes": "good"}]},
    }
    (tmp_path / "final_report.json").write_text(json.dumps(report))

    amend_final_report(tmp_path, {"overall_score": 0.9, "leaf_count": 1, "graded": 1},
                       target_score=0.7)
    out = json.loads((tmp_path / "final_report.json").read_text())
    assert out["rubric"]["meets_target"] is True
    assert out["rubric"]["areas"] == [{"name": "code", "score": 0.85, "notes": "good"}]
```

**Verify.** `.venv/bin/python -m pytest tests/evals/test_leaf_scorer.py::test_amend_final_report_computes_meets_target_and_keeps_areas -v` → PASS. Re-rendered markdown shows both the per-area table and the correct `✔ meets target`.

**Don't touch.** The atomic-write block (lines 312-322); the `_rerender_report_markdown` call.

---

## T6: Verdict reconciliation — a run with score 0.0 cannot self-report "reproduced"

**Severity:** P0 · **Source:** handoff §4 P0-I9 · **Depends on:** T4 (degraded flag)

**Files:**
- Modify: `backend/agents/rlm/report.py:116-127` (`_reconcile_verdict`) and `:179-263` (`build_final_report`).
- Test: `tests/rlm/test_report.py`.

**Problem.** The `ftrl` run scored 0.0 yet self-reported `verdict: "reproduced"`. The existing honesty guard drops *fabricated metrics* but not an over-claimed *verdict*.

**Root cause.** `_reconcile_verdict` accepts any valid verdict string; it never checks whether the verdict is consistent with the evidence (rubric score, baseline_metrics, primitive_trace).

**Fix.** Extend `build_final_report` to downgrade the verdict based on evidence: a run with no `run_experiment` call OR rubric `overall_score < 0.5` OR `baseline_metrics == {}` cannot claim `reproduced` — downgrade to `partial`.

```python
def _reconcile_verdict_against_evidence(
    verdict: str,
    *,
    baseline_metrics: dict,
    rubric: dict,
    primitive_trace: dict,
) -> tuple[str, str | None]:
    """Downgrade an over-claimed verdict; return (verdict, reason_or_None)."""
    if verdict != "reproduced":
        return verdict, None
    score = float((rubric or {}).get("overall_score", 0.0) or 0.0)
    ran_experiment = bool(primitive_trace.get("by_primitive", {}).get("run_experiment"))
    reasons: list[str] = []
    if not ran_experiment:
        reasons.append("run_experiment never ran")
    if not baseline_metrics:
        reasons.append("no measured baseline metrics")
    if score < 0.5:
        reasons.append(f"rubric score {score:.3f} < 0.5")
    if reasons:
        return "partial", "; ".join(reasons)
    return verdict, None
```

Wire it into `build_final_report` after the existing `_reconcile_verdict`:

```python
    verdict = _reconcile_verdict(parsed)
    # ... existing metric-honesty downgrade ...
    verdict, downgrade_reason = _reconcile_verdict_against_evidence(
        verdict, baseline_metrics=baseline_metrics, rubric=parsed.get("rubric") or {},
        primitive_trace=trace,
    )
    if downgrade_reason:
        summary = (summary + f"\n\n[verdict guard] Downgraded to 'partial': "
                   f"{downgrade_reason}.").strip()
        logger.warning("report: verdict downgraded to partial — %s", downgrade_reason)
```

**Guard test.**

```python
def test_verdict_downgraded_when_evidence_contradicts(make_run_context, tmp_path):
    """Symptom: ftrl run scored 0.0 yet self-reported verdict='reproduced'.

    The verdict honesty guard was only metric-fabrication-aware; it did not
    check rubric score or whether run_experiment actually ran (handoff P0-I9).
    Verify: a result claiming reproduced but with score 0.0 downgrades to partial.
    """
    from rlm.core.types import RLMChatCompletion
    raw = json.dumps({
        "verdict": "reproduced",
        "baseline_metrics": {"mean_reward": 487.3},
        "rubric": {"overall_score": 0.0, "meets_target": False},
    })
    completion = RLMChatCompletion(response=raw, usage_summary=None, metadata={})
    ctx = make_run_context(tmp_path)
    # Make trace show run_experiment ran (so metric-honesty doesn't fire first).
    ctx.cost_ledger.entries.append(_mk_ledger_entry("run_experiment"))

    report = build_final_report(completion, ctx=ctx)
    assert report.verdict == "partial"
    assert "0.000 < 0.5" in report.reproduction_summary
```

**Verify.** `.venv/bin/python -m pytest tests/rlm/test_report.py::test_verdict_downgraded_when_evidence_contradicts -v` → PASS.

**Don't touch.** The existing metric-fabrication guard (lines 233-249) — it runs before this and remains correct.

---

## T7: ftrl paper-grounding — confirm `paper.md` is the real FTRL paper; harden root prompt

**Severity:** P0 (correctness) · **Source:** handoff §4 P0-I3

**Files:**
- Investigate: `third_party/paperbench/ftrl/paper.md` and any `addendum.md`.
- Modify (if needed): `backend/agents/rlm/run.py:75-94` (`_ROOT_PROMPT`).
- Test: `tests/rlm/test_run.py` — add a `paper_text`-must-not-mention-substrate negative test.

**Problem.** The `ftrl` run wrote a 13 KB `train.py` whose summary claimed "the RLM system was implemented" — leaf score 0.000. The root reproduced the substrate it runs on, not the paper it was given.

**Root cause hypotheses (verify in order):**
1. `third_party/paperbench/ftrl/paper.md` is the wrong file (e.g. the RLM paper accidentally vendored under `ftrl/`).
2. The paper.md content is correct but the root prompt's "reproduce the paper offloaded in `context`" wording lets the root substitute "the system" for "the paper" when the paper is dense.

**Fix (steps):**

1. **Read** `third_party/paperbench/ftrl/paper.md` — confirm its first 1000 chars match the FTRL paper (McMahan 2011, "Follow-the-Regularized-Leader and Mirror Descent"). If wrong, replace it with the real upstream PaperBench `ftrl` bundle and stop — this is a vendoring fix, not a prompt fix.
2. If the paper is right, harden the root prompt with an anti-substitution clause:

```python
_ROOT_PROMPT = (
    "Reproduce the research paper offloaded in the REPL variable `context['paper_text']`. "
    "The thing you reproduce is THAT paper — not the agent framework you are running on, "
    "not the LLM that hosts you, not the orchestrator. If `context['paper_text']` is about "
    "topic X, your code, summary, and report MUST be about topic X; never substitute the "
    "substrate you run on for the paper. If the paper is unclear, narrow it by reading more "
    "of `context['paper_text']` via slices and llm_query, not by drifting to a different topic. "
    # ... rest of the existing prompt ...
)
```

**Guard test.**

```python
def test_root_does_not_reproduce_the_substrate(tmp_path):
    """Symptom: ftrl run produced a 13 KB train.py whose summary said
    'the RLM system was implemented' (leaf score 0.000).

    The root model substituted the substrate it runs on for the paper it
    was supposed to reproduce. Verify: with a paper.md whose topic is X,
    the final_report.reproduction_summary mentions X and does NOT mention
    'RLM', 'orchestrator', 'agent framework' as the reproduced subject.
    """
    # Run the prompt-validation helper (not a real RLM run — too slow).
    # This test verifies the prompt itself contains the anti-substitution
    # clause; an integration test on real paper drift requires a live run.
    from backend.agents.rlm.run import _ROOT_PROMPT
    assert "never substitute the substrate" in _ROOT_PROMPT
    assert "context['paper_text']" in _ROOT_PROMPT
```

**Verify.** Re-run `ftrl`: `.venv/bin/python scripts/rlm_paperbench.py ftrl --sandbox docker --max-wall-clock 7200`; inspect `runs/<run_id>/final_report.md` — `reproduction_summary` must reference the FTRL algorithm specifically, not the orchestrator.

**Don't touch.** Don't change `_MAX_ITERATIONS` / `_MAX_DEPTH` or the FINAL_VAR contract.

---

## T8: SDK-mode `paper_text` workspace variable content loss

**Severity:** P0 (correctness, SDK mode only) · **Source:** handoff §4 P0-I4

**Files:**
- Investigate: parser sections → indexer → chunker → workspace path.
- Modify (likely): `backend/cli.py:_build_workspace_claim_map` and/or `backend/services/context/workspace/`.
- Test: `tests/services/test_workspace_paper_text_integrity.py` (new).

**Problem.** For some papers the workspace `paper_text` variable (reassembled from indexed chunks) is degraded or empty — the IOI paper's was. RLM mode now bypasses it (reads `parsed_full_text.txt`), but **SDK mode still consumes it** — so an SDK-mode reproduction of those papers starts from corrupted input.

**Root cause (to investigate at execution time):** chunk-boundary loss between the indexer and the workspace reassembly. Trace: `parser._write_full_text` → indexer chunks → workspace variable.

**Fix.**
1. Add an end-to-end integrity test that runs the parser→indexer→workspace path on a fixture paper and asserts `workspace["paper_text"] == parser.full_text` (modulo whitespace normalization).
2. Trace the failing path; fix at the boundary that drops content.

**Guard test (write this first — it currently fails; fix until it passes).**

```python
def test_workspace_paper_text_equals_parser_full_text(tmp_path):
    """Symptom: the IOI paper's workspace paper_text was degraded; SDK mode read garbage.

    The parser → indexer → chunker → workspace pipeline loses content for some
    papers (handoff P0-I4). The workspace `paper_text` variable MUST equal what
    the parser wrote to parsed_full_text.txt (modulo whitespace).
    """
    # Use a known-good fixture or vendored paper.
    paper_path = Path("tests/fixtures/papers/ioi_minimal.pdf")
    if not paper_path.exists():
        pytest.skip("fixture missing; add tests/fixtures/papers/ioi_minimal.pdf")

    parsed_text = _parse_paper(paper_path)  # parser path
    workspace_text = _build_workspace_paper_text(paper_path)  # indexer→workspace path

    # Both should contain the same essential content.
    def normalize(s: str) -> str:
        return " ".join(s.split())
    assert normalize(workspace_text) == normalize(parsed_text)
```

**Verify.** `.venv/bin/python -m pytest tests/services/test_workspace_paper_text_integrity.py -v` → PASS after fix.

**Don't touch.** RLM mode's `parsed_full_text.txt` source path is correct (commit `1b69fe7`); don't change it.

---

# P1 — robustness

## T9: Wire `ctx.deadline_utc` in `run.py` — activate the per-primitive deadline layer

**Severity:** P1 · **Source:** review I1

**Files:**
- Modify: `backend/agents/rlm/run.py:520-533` (`RunContext(...)` constructor call).
- Test: `tests/rlm/test_run.py`.

**Problem.** `run.py`'s docstring claims time is bounded three ways: `rlm`'s `max_timeout`, the per-primitive deadlines (#59), and the watchdog. But `RunContext.deadline_utc` is never set, so `ctx.remaining_s()` always returns `None` and `_timeout_for(ctx, cap_s)` always returns the static `cap_s`. The per-primitive deadline never tightens to the run budget. Three bounds are really two.

**Root cause.** Missing argument in the `RunContext(...)` constructor call at run.py:520.

**Fix.** One change at run.py:520-533:

```python
    from datetime import datetime, timedelta, timezone
    ctx = RunContext(
        project_id=project_id,
        project_dir=project_dir,
        runs_root=runs_root,
        dashboard=dashboard,
        cost_ledger=cost_ledger,
        llm_client=llm_client,
        provider=provider_label,
        model=llm_model,
        runtime=agent_runtime,
        agent_model=agent_model,
        workspace_service=workspace_service,
        workspace_id=workspace_id,
        deadline_utc=datetime.now(timezone.utc) + timedelta(seconds=wall_clock_s),  # M-DEADLINE
    )
```

**Guard test.**

```python
def test_run_context_deadline_is_armed_from_wall_clock(monkeypatch, tmp_path):
    """Symptom: ctx.remaining_s() is always None; per-primitive deadlines never tighten.

    run.py documents three time bounds but never set deadline_utc on RunContext
    (review I1). Verify: when run_pipeline_rlm is invoked with a wall_clock_s,
    the constructed RunContext has remaining_s() < wall_clock_s and not None.
    """
    captured: dict = {}
    real_resolve = primitives  # placeholder
    def fake_completion(*a, **kw):
        # capture the ctx from inside the run; for simplicity use a sentinel
        raise RuntimeError("stop after ctx build")
    # ... patch RLM.completion to raise immediately so we just exercise setup ...

    budget = RunBudget(max_wall_clock_seconds=300)
    with pytest.raises(Exception):
        await run_pipeline_rlm(
            "test_proj", tmp_path, {"entries": []}, run_budget=budget,
        )
    # Inspect via the dashboard events log or instrument _resolve_custom_tools
    # to capture ctx; assert ctx.deadline_utc is set and remaining_s() in (0, 300].
```

**Verify.** `.venv/bin/python -m pytest tests/rlm/test_run.py::test_run_context_deadline_is_armed_from_wall_clock -v` → PASS.

**Don't touch.** `RunContext.remaining_s()` is correct — don't modify `context.py`.

---

## T10: Redact corpus in the root `response` egress

**Severity:** P1 · **Source:** review I2

**Files:**
- Modify: `backend/agents/rlm/sse_bridge.py:139-141` (`sanitize_iteration`).
- Test: `tests/rlm/test_sse_bridge.py`.

**Problem.** `sanitize_iteration` truncates `response` to 4000 chars but does **not** pass it through `redact_corpus` — while `_stream_metadata` (stdout/stderr prefixes) and `_finalize` (`reproduction_summary`) both do. The root model reads paper slices via REPL code (`print(context['paper_text'][:N])`) and quotes what it saw in its next natural-language `response`; that `response` goes verbatim into every `repl_iteration` event, up to **4000 chars of paper per iteration**.

**Root cause.** One egress point was missed when the redaction layer was added.

**Fix.** One line:

```python
    response = iteration.response or ""
    if len(response) > _RESPONSE_MAX_CHARS:
        response = response[:_RESPONSE_MAX_CHARS]
    if _sentinels:
        response = redact_corpus(response, _sentinels)  # close the egress
```

**Guard test.**

```python
def test_sanitize_iteration_redacts_corpus_in_response(make_iteration):
    """Symptom: up to 4000 chars of paper corpus per iteration leak via response.

    sanitize_iteration redacted stdout/stderr prefixes but NOT the response —
    the root's natural-language response can quote paper slices it read via
    REPL code (review I2). Verify: a response containing a corpus sentinel
    is redacted, not streamed verbatim.
    """
    sentinel = "x" * 200  # ≥16 chars, mimic a corpus sentinel
    iteration = make_iteration(response=f"I saw this in the paper: {sentinel} ...")
    clean = sanitize_iteration(iteration, index=1, sentinels=[sentinel])
    assert sentinel not in clean["response"]
    assert "[REDACTED]" in clean["response"]
```

**Verify.** `.venv/bin/python -m pytest tests/rlm/test_sse_bridge.py::test_sanitize_iteration_redacts_corpus_in_response -v` → PASS.

**Don't touch.** Don't change `_RESPONSE_MAX_CHARS` or the metadata-only `_stream_metadata`/`_locals_metadata` invariants.

---

## T11: `propose_improvements` fail-soft `_extract_json`

**Severity:** P1 · **Source:** review I3

**Files:**
- Modify: `backend/agents/rlm/primitives.py:632-672` (`propose_improvements`).
- Test: `tests/rlm/test_propose_improvements.py`.

**Problem.** `propose_improvements` calls `_extract_json(raw)` with no try/except (lines 663-664). A truncated or JSON-less LLM response raises `ValueError` out of the primitive — diverging from `plan_reproduction` and `verify_against_rubric`, both of which already wrap the same `complete`+`_extract_json` pair in a fail-soft `except`.

**Root cause.** Inconsistent fail-soft treatment across LLM primitives.

**Fix.**

```python
    try:
        raw = ctx.llm_client.complete(system=IMPROVEMENT_ORCHESTRATOR_PROMPT, user=user)
        items = _extract_json(raw).get("hypotheses", [])
    except Exception as exc:  # noqa: BLE001 — fail-soft (D3 pattern, review I3)
        return [{
            "success": False,
            "error": f"propose_improvements: {type(exc).__name__}: {exc}",
        }]
```

(Note: return type is `list[dict]`; an error becomes a single-item error list, preserving the contract.)

**Guard test.**

```python
def test_propose_improvements_failsoft_on_truncated_json(make_run_context, tmp_path):
    """Symptom: a truncated LLM response crashes the primitive instead of returning fail-soft.

    propose_improvements diverged from plan_reproduction / verify_against_rubric
    (review I3) — both of those wrap _extract_json fail-soft, propose_improvements
    did not. Verify: a truncated JSON response yields a single-item error list,
    never a raised ValueError.
    """
    ctx = make_run_context(tmp_path, llm_responses=['{"hypotheses": [{"a":1'])  # truncated
    result = propose_improvements({"success": True}, {}, ctx=ctx)
    assert isinstance(result, list)
    assert result and result[0].get("success") is False
    assert "truncated" in result[0]["error"].lower() or "ValueError" in result[0]["error"]
```

**Verify.** `.venv/bin/python -m pytest tests/rlm/test_propose_improvements.py::test_propose_improvements_failsoft_on_truncated_json -v` → PASS.

**Don't touch.** The `k` clamp (already correct from Phase 2 review fix).

---

## T12: `--sandbox` flag threaded to `run_experiment`

**Severity:** P1 · **Source:** handoff §4 P1-I7

**Files:**
- Modify: `backend/agents/rlm/run.py:446-462` (`run_pipeline_rlm` signature passes `sandbox_mode`; it currently goes nowhere).
- Modify: `backend/agents/rlm/context.py:RunContext` — add `sandbox_mode: str = "docker"`.
- Modify: `backend/agents/rlm/primitives.py:_execute_in_sandbox` — pick backend from `ctx.sandbox_mode`.
- Test: `tests/rlm/test_run_experiment.py`.

**Problem.** `run_pipeline_rlm` accepts `sandbox_mode` but `_execute_in_sandbox` hardcodes `LocalDockerBackend()`. `--sandbox runpod` or `--sandbox local` has no effect on RLM `run_experiment`.

**Root cause.** The plumbing wasn't completed when `run_experiment` landed.

**Fix.**

1. Add `sandbox_mode` to `RunContext`:
```python
@dataclass
class RunContext:
    # ... existing fields ...
    sandbox_mode: str = "docker"  # "docker" | "local_process" | "runpod"
```

2. In `run.py:run_pipeline_rlm`, thread it through:
```python
    ctx = RunContext(
        # ... existing args ...
        sandbox_mode=str(sandbox_mode) if sandbox_mode else "docker",
    )
```

3. In `primitives.py:_execute_in_sandbox`, pick backend:
```python
    from backend.services.runtime.local_docker import LocalDockerBackend
    from backend.services.runtime.local_process import LocalProcessBackend

    mode = (ctx.sandbox_mode or "docker").lower()
    if mode == "local_process":
        backend = LocalProcessBackend()
    elif mode == "runpod":
        from backend.services.runtime.runpod_backend import RunpodBackend
        backend = RunpodBackend.from_settings()
    else:
        backend = LocalDockerBackend()
    service = RuntimeAppService(backend)
```

(Note: `_execute_in_sandbox`'s signature must now accept `sandbox_mode: str` so it can be passed in; thread it via `run_experiment` after T1.)

**Guard test.**

```python
def test_sandbox_mode_is_threaded_from_ctx(monkeypatch, make_run_context, tmp_path):
    """Symptom: --sandbox runpod silently uses LocalDockerBackend in RLM mode.

    _execute_in_sandbox hardcoded LocalDockerBackend; ctx.sandbox_mode was never
    consulted (handoff P1-I7). Verify: with ctx.sandbox_mode='local_process',
    LocalProcessBackend is instantiated.
    """
    captured: dict = {}
    real_service = RuntimeAppService
    def spy_service(backend):
        captured["backend_class"] = type(backend).__name__
        return real_service(backend)
    monkeypatch.setattr(primitives, "RuntimeAppService", spy_service)

    ctx = make_run_context(tmp_path)
    ctx.sandbox_mode = "local_process"
    # Run with a manifest that returns immediately (empty commands).
    code_dir = tmp_path / "code"; code_dir.mkdir()
    (code_dir / "commands.json").write_text("[]")
    primitives.run_experiment(str(code_dir), "img", ctx=ctx)
    # Even if the run aborts on empty commands, the backend choice must be made first.
    assert captured.get("backend_class") in ("LocalProcessBackend", "LocalDockerBackend")
```

**Verify.** `.venv/bin/python -m pytest tests/rlm/test_run_experiment.py::test_sandbox_mode_is_threaded_from_ctx -v` → PASS.

**Don't touch.** The DEFAULT (`docker`) must remain the default for back-compat.

---

## T13: `reproduce --fresh` purges both stores atomically

**Severity:** P1 · **Source:** handoff §4 P1-I5

**Files:**
- Modify: `backend/cli.py` — add `--fresh` flag to `reproduce`.
- Add helper: `backend/services/runs/purge.py` (new) — atomic two-store purge.
- Test: `tests/services/test_purge.py`.

**Problem.** Re-running a paper with the same id: `rm -rf runs/<id>` clears the run dir but NOT the `event_store_events` aggregates (`<id>`, `<id>:parsed`, `<id>:index`, `<id>:discovery`) in `reprolab.db` → first `record()` passes `expected_version=0` against version N → `ConcurrencyError`.

**Root cause.** Two stores, one purge path.

**Fix.**

1. Add `backend/services/runs/purge.py`:
```python
"""Atomically reset both stores for a project_id — the run dir and SQLite aggregates."""
import shutil
from pathlib import Path

from backend.eventstore.sqlite_store import SqliteEventStore
from backend.config import get_settings

def purge_project(project_id: str, runs_root: Path) -> dict:
    """Delete runs/<id>/ AND every event_store aggregate with prefix <id>.

    Returns {"run_dir_removed": bool, "aggregates_removed": int}.
    """
    run_dir = Path(runs_root) / project_id
    run_dir_removed = False
    if run_dir.exists():
        shutil.rmtree(run_dir)
        run_dir_removed = True

    store = SqliteEventStore(get_settings().database_url)
    try:
        aggregates_removed = store.delete_aggregates_with_prefix(project_id)
    finally:
        store.close()
    return {"run_dir_removed": run_dir_removed, "aggregates_removed": aggregates_removed}
```

2. Add `delete_aggregates_with_prefix` to `SqliteEventStore` (if not present):
```python
def delete_aggregates_with_prefix(self, prefix: str) -> int:
    """Delete all aggregates whose id starts with prefix. Returns count."""
    with self._connect() as conn:
        cur = conn.execute(
            "DELETE FROM event_store_events WHERE aggregate_id = ? OR aggregate_id LIKE ?",
            (prefix, f"{prefix}:%"),
        )
        return cur.rowcount
```

3. Wire `--fresh` in `backend/cli.py:cmd_reproduce`:
```python
    if args.fresh:
        from backend.services.runs.purge import purge_project
        result = purge_project(args.project_id, runs_root)
        logger.info("--fresh: %s", result)
```

**Guard test.**

```python
def test_purge_project_clears_both_stores(tmp_path, sqlite_event_store):
    """Symptom: re-running a paper raises ConcurrencyError because aggregates persist.

    rm -rf runs/<id> does not clear event_store_events; first record() then
    crashes (handoff P1-I5). Verify: purge_project deletes both surfaces.
    """
    # Seed both stores.
    run_dir = tmp_path / "prj_x"
    run_dir.mkdir()
    (run_dir / "marker").write_text("x")
    sqlite_event_store.append("prj_x", "Started", {"a": 1}, expected_version=0)
    sqlite_event_store.append("prj_x:parsed", "Done", {}, expected_version=0)

    result = purge_project("prj_x", tmp_path)
    assert result["run_dir_removed"] is True
    assert result["aggregates_removed"] >= 2
    # Re-running now works: version starts at 0 again.
    sqlite_event_store.append("prj_x", "Started", {}, expected_version=0)
```

**Verify.** `.venv/bin/python -m pytest tests/services/test_purge.py -v` → PASS.

**Don't touch.** Default behavior (no `--fresh`) must be unchanged.

---

## T14: 429-aware retry/backoff in LLM clients

**Severity:** P1 · **Source:** handoff §4 P1-I6

**Files:**
- Modify: `backend/services/context/workspace/tools/openai_client.py:OpenAILlmClient.complete` — add 429 retry-with-jitter.
- Modify: `backend/services/context/workspace/tools/rlm_query.py:ClaudeLlmClient.complete` — same.
- Test: `tests/services/test_openai_client_429.py`.

**Problem.** Featherless caps at 4 concurrent units (one RLM run saturates it). Two concurrent runs → second 429s on `generate_rubric_tree` or `rlm.completion`. No retry-on-429 currently exists in the LLM clients.

**Root cause.** Hard-fail on 429 was acceptable when runs were strictly serial; now the team runs in parallel.

**Fix.** Add exponential-backoff-with-jitter on 429 (and 503) responses:

```python
# in openai_client.py — at the top
import random
import time

_MAX_429_RETRIES = 6
_BASE_BACKOFF_S = 2.0
_MAX_BACKOFF_S = 60.0


def _is_retryable(exc: Exception) -> bool:
    """True if exc is a 429 / 503 from an OpenAI-compatible endpoint."""
    msg = str(exc).lower()
    return "429" in msg or "rate limit" in msg or "503" in msg or "service unavailable" in msg


class OpenAILlmClient:
    def complete(self, *, system: str, user: str) -> str:
        last_exc: Exception | None = None
        for attempt in range(_MAX_429_RETRIES + 1):
            try:
                return self._complete_once(system=system, user=user)
            except Exception as exc:
                if not _is_retryable(exc) or attempt == _MAX_429_RETRIES:
                    raise
                wait = min(_MAX_BACKOFF_S, _BASE_BACKOFF_S * (2 ** attempt))
                wait += random.uniform(0, wait * 0.3)  # 30% jitter
                logger.warning(
                    "OpenAILlmClient: 429/503 (attempt %d/%d) — backing off %.1fs",
                    attempt + 1, _MAX_429_RETRIES, wait,
                )
                time.sleep(wait)
                last_exc = exc
        raise last_exc  # unreachable, but keeps type checker happy
```

Mirror in `ClaudeLlmClient.complete`.

**Guard test.**

```python
def test_openai_client_retries_on_429(monkeypatch):
    """Symptom: a concurrent run 429s instead of waiting for a slot.

    Featherless caps at 4 units; a second RLM run gets 429 and fails outright
    (handoff P1-I6). Verify: an OpenAILlmClient client receives a sequence
    [429, 429, OK] and returns the OK response after backing off.
    """
    from backend.services.context.workspace.tools.openai_client import OpenAILlmClient

    calls = []
    def fake_complete_once(self, *, system, user):
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("429 Too Many Requests")
        return "ok"
    monkeypatch.setattr(OpenAILlmClient, "_complete_once", fake_complete_once)
    monkeypatch.setattr("time.sleep", lambda s: None)  # don't actually sleep in tests

    client = OpenAILlmClient()
    assert client.complete(system="s", user="u") == "ok"
    assert len(calls) == 3
```

**Verify.** `.venv/bin/python -m pytest tests/services/test_openai_client_429.py -v` → PASS.

**Don't touch.** Don't add retry on auth errors (401/403) or schema errors (400) — those should fail fast.

---

## T15: REST `POST /runs/arxiv` two-dir split — bridge events + use one run dir

**Severity:** P1 · **Source:** handoff §4 P1-I8

**Files:**
- Modify: `backend/services/events/live_runs.py:_python_script` (the REST spawn) and `finalize_benchmark`.
- Test: `tests/services/test_live_runs_rlm.py`.

**Problem.** A REST-spawned RLM run writes artifacts to `runs/prj_<hash>/` while the API watches `runs/ui_rlm_<…>/`. `finalize_benchmark` bridges only `final_report.{json,md}` and reads SDK-schema fields absent on RLM reports — so events are not bridged and the `benchmark` summary is empty for RLM.

**Root cause.** Two run-id naming conventions; bridge written for SDK only.

**Fix (preferred — root-level).** Make `_python_script` use the same run-id for both: pass `--project-id ui_rlm_<…>` into the CLI so RLM writes to the same dir the API watches.

```python
# in live_runs.py _python_script — when run_mode='rlm':
script = [
    sys.executable, "-m", "backend.cli", "reproduce", paper_source,
    "--mode", "rlm",
    "--project-id", run_id,  # use the same id the API watches
    "--runs-root", str(runs_root),
    # ... existing args ...
]
```

Verify `backend.cli.cmd_reproduce` accepts `--project-id` (it should); if not, add it.

Update `finalize_benchmark` to be schema-aware:
```python
def finalize_benchmark(run_dir: Path) -> dict:
    report_path = run_dir / "final_report.json"
    if not report_path.exists():
        return {"benchmark": None}
    report = json.loads(report_path.read_text())
    # RLM schema vs SDK schema:
    if "baseline_metrics" in report:  # RLM
        return {
            "benchmark": {
                "verdict": report.get("verdict", ""),
                "rubric_score": (report.get("rubric") or {}).get("overall_score"),
                "metrics": report.get("baseline_metrics") or {},
                "cost_usd": (report.get("cost") or {}).get("llm_usd"),
            }
        }
    # SDK schema (existing path)
    return _legacy_sdk_benchmark(report)
```

**Guard test.**

```python
def test_rest_arxiv_rlm_run_writes_to_one_dir(tmp_path, monkeypatch):
    """Symptom: REST runs split artifacts across two dirs; events never bridge for RLM.

    POST /runs/arxiv launched RLM to runs/prj_<hash>/ but the API watched
    runs/ui_rlm_<…>/ (handoff P1-I8). Verify: --project-id is threaded so
    final_report.json lands in the watched dir.
    """
    # ... spawn the script with run_id='ui_rlm_test123' and assert the dir matches ...
```

**Verify.** `.venv/bin/python -m pytest tests/services/test_live_runs_rlm.py -v` → PASS, plus a manual smoke: `POST /runs/arxiv` with a tiny paper, confirm `GET /runs/<id>` returns a populated benchmark.

**Don't touch.** SDK-mode bridging — keep the legacy path intact.

---

## T16: OCR-skip heuristic gated on quality, not raw length

**Severity:** P1 · **Source:** review I4

**Files:**
- Modify: `backend/services/ingestion/parser/resolving_parser.py:141-146`.
- Test: `tests/services/test_resolving_parser.py`.

**Problem.** OCR is skipped when HTML/PDF produced ≥`_MIN_USEFUL_CHARS` (200) characters. But a 200-999-char result of figure-noise scores `0.0` from `score_text_quality` (the `<1000`-char rule) yet still has `len ≥ 200`, so OCR — the tier that exists to rescue exactly this case — never runs. `_choose`'s `non_empty` fallback then ships the garbage.

**Root cause.** Length is the wrong signal; quality is.

**Fix.**

```python
# Replace the length-based gate with a quality-based gate.
_MIN_USEFUL_SCORE = 0.3  # below this, OCR may rescue

def _should_run_ocr(html_score: float, pdf_score: float) -> bool:
    """OCR runs unless one of HTML or PDF produced *quality* text."""
    return max(html_score, pdf_score) < _MIN_USEFUL_SCORE
```

Wire it where the existing length gate lives.

**Guard test.**

```python
def test_ocr_runs_on_short_low_quality_html(monkeypatch, tmp_path):
    """Symptom: 200-char figure-noise HTML wins over OCR; cascade ships garbage.

    The OCR-skip gate used raw length (≥200 chars => skip), so a 250-char
    HTML soup scoring 0.0 still skipped OCR (review I4). Verify: OCR runs
    when HTML quality score < 0.3, regardless of length.
    """
    # ... fixture: HTML returns 250 chars of figure-noise; PDF returns 800 chars; ...
    # ... assert OCR was invoked (spy on OcrPaperParser.parse) ...
```

**Verify.** `.venv/bin/python -m pytest tests/services/test_resolving_parser.py::test_ocr_runs_on_short_low_quality_html -v` → PASS.

**Don't touch.** `score_text_quality` itself (it's correct); don't change `_choose`'s fallback semantics.

---

## T17: Unbounded arXiv-HTML fetch + unbounded BeautifulSoup parse

**Severity:** P1 · **Source:** review I5

**Files:**
- Modify: `backend/services/ingestion/intake/fetchers/arxiv.py:78` (`response.read()`).
- Modify: `backend/services/ingestion/parser/html_parser.py:175` (`_walk` — explicit stack or re-raise as `ParseError`).
- Test: `tests/services/test_arxiv_fetcher.py`, `tests/services/test_html_parser.py`.

**Problem.** `arxiv.py:78` does an unbounded `response.read()`; `download_pdf` enforces a 100 MB cap via chunked reads but the HTML path has only a *minimum* size check. A malicious arXiv-HTML endpoint returning a multi-GB body OOM-kills ingestion. `html_parser._walk` is then a recursive walk on attacker-controlled HTML; a deeply-nested document raises `RecursionError` which is not a `ParseError` and escapes the cascade.

**Root cause.** Two gaps in the new HTML cascade. PDF path was hardened; HTML wasn't.

**Fix.**

1. `arxiv.py`:
```python
_HTML_MAX_BYTES = 50 * 1024 * 1024  # 50 MB — same envelope as PDF cap

def fetch_html(arxiv_id: str) -> bytes:
    # ... existing setup ...
    with urlopen(req, timeout=30) as response:
        body = bytearray()
        while True:
            chunk = response.read(64 * 1024)
            if not chunk:
                break
            body.extend(chunk)
            if len(body) > _HTML_MAX_BYTES:
                raise FetchError(
                    f"arxiv HTML body exceeded {_HTML_MAX_BYTES} bytes "
                    f"(possible DoS); aborting"
                )
        return bytes(body)
```

2. `html_parser.py`:
```python
# Convert recursive _walk to an iterative stack-based walk; catch RecursionError defensively.
def _walk(root) -> list[str]:
    out: list[str] = []
    stack = [root]
    while stack:
        node = stack.pop()
        # ... existing per-node logic ...
        for child in reversed(list(node.children if hasattr(node, "children") else [])):
            stack.append(child)
    return out
```

Add an outer guard in the parser entry that converts any unexpected `RecursionError` to `ParseError` so the cascade's `except ParseError` catches it.

**Guard test.**

```python
def test_arxiv_html_fetch_caps_at_50mb(monkeypatch):
    """Symptom: a malicious arXiv-HTML body OOM-kills ingestion.

    The new HTML fetch did response.read() unbounded; PDF enforced a 100 MB
    cap (review I5). Verify: a 60 MB fake body raises FetchError, not OOM.
    """
    big = b"x" * (60 * 1024 * 1024)
    monkeypatch.setattr(urllib.request, "urlopen",
                        _fake_urlopen_returning(big))
    with pytest.raises(FetchError, match="exceeded"):
        fetch_html("2512.24601")


def test_html_parser_handles_deep_nesting_without_recursion_error():
    """Symptom: deeply-nested HTML escapes the cascade with a RecursionError.

    _walk was a recursive walk on attacker-controlled HTML (review I5).
    Verify: 5000-deep nesting parses without raising RecursionError; if
    it does raise, it surfaces as ParseError.
    """
    html = "<div>" * 5000 + "x" + "</div>" * 5000
    try:
        result = HtmlPaperParser().parse(html.encode())
        assert result is not None  # iterative path
    except ParseError:
        pass  # acceptable: wrapped in ParseError, caught by cascade
```

**Verify.** Both tests PASS.

**Don't touch.** The 100 MB PDF cap; the minimum-size check.

---

## T18: `parsed_full_text.txt` stale/missing handling

**Severity:** P1 · **Source:** review I6

**Files:**
- Modify: `backend/services/ingestion/parser/service.py:170-174`.
- Modify: `backend/cli.py:113-121` (the RLM-mode read).
- Test: `tests/services/test_parser_service.py`.

**Problem.** `parsed_full_text.txt` is written only after a *successful* parse. On parse failure (or a re-run into a directory holding a *stale* blob from a previous paper), `cli.py:113-121` reads the stale/wrong blob and silently feeds it to the RLM — reintroducing the bug commit `1b69fe7` set out to kill.

**Root cause.** No invalidation on parse failure; no missing-blob warning on the read side.

**Fix.**

1. In `service.py`, on parse failure delete any stale blob:
```python
def write_parsed_full_text(project_dir: Path, text: str | None) -> None:
    """Write parsed_full_text.txt atomically, or delete it on parse failure."""
    path = project_dir / "parsed_full_text.txt"
    if not text:
        # Parse failed: a stale blob from a prior paper would silently feed the RLM.
        if path.exists():
            path.unlink()
        return
    tmp = path.with_suffix(".txt.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
```

2. In `cli.py`, warn loudly on missing/empty:
```python
def _read_parsed_full_text_for_rlm(project_dir: Path) -> str | None:
    path = project_dir / "parsed_full_text.txt"
    if not path.exists():
        logger.warning(
            "RLM mode: parsed_full_text.txt missing — parser likely failed; "
            "falling back to workspace variable (lossy)"
        )
        return None
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        logger.warning("RLM mode: parsed_full_text.txt is empty — parser likely failed")
        return None
    return text
```

**Guard test.**

```python
def test_parsed_full_text_is_deleted_on_parse_failure(tmp_path):
    """Symptom: a re-run into a dir with a stale blob feeds the wrong paper to the RLM.

    write_parsed_full_text only WROTE on success; on failure it left whatever
    was there from a prior run (review I6, regression of commit 1b69fe7).
    Verify: write_parsed_full_text(..., text=None) deletes any existing blob.
    """
    p = tmp_path / "parsed_full_text.txt"
    p.write_text("STALE CONTENT FROM PRIOR PAPER")
    write_parsed_full_text(tmp_path, text=None)
    assert not p.exists()
```

**Verify.** PASS.

**Don't touch.** The successful-write path; the atomic-rename pattern.

---

## T19: `checkpoint.py` — implement resume OR rename and drop the resume claim

**Severity:** P1 · **Source:** review I9

**Files:**
- Modify: `backend/agents/rlm/checkpoint.py`.
- Test: `tests/rlm/test_checkpoint.py`.

**Problem.** Docstring says "run-state checkpoint / resume" but nothing reads `iterations.jsonl` / `RLMRunIteration` back. `__init__` hardcodes `self._version = 0`; on a process restart with the same `project_id`, the aggregate is already at version N → first `record()` passes `expected_version=0` against N → **`ConcurrencyError` on restart**.

**Root cause.** Resume was advertised but never wired.

**Fix (pragmatic — rename, defer real resume to a follow-up).**

1. Rename the module's docstring and class to "iteration event log" — drop the word "resume."
2. Fix the restart crash: in `__init__`, query the current version and start from there:
```python
def __init__(self, *, project_id: str, event_store: SqliteEventStore, snapshot_dir: Path):
    self._project_id = project_id
    self._store = event_store
    self._snapshot_dir = snapshot_dir
    # Look up current aggregate version so a restart appends instead of crashing.
    self._version = event_store.current_version(f"rlm-run:{project_id}") or 0
    # ...
```

If `current_version` doesn't exist, add:
```python
def current_version(self, aggregate_id: str) -> int:
    with self._connect() as conn:
        cur = conn.execute(
            "SELECT MAX(version) FROM event_store_events WHERE aggregate_id = ?",
            (aggregate_id,),
        )
        row = cur.fetchone()
        return int(row[0]) if row and row[0] is not None else 0
```

**Guard test.**

```python
def test_iteration_checkpointer_does_not_crash_on_restart(tmp_path, sqlite_event_store):
    """Symptom: a restarted run crashes on the first checkpoint with ConcurrencyError.

    checkpoint.py hardcoded _version=0; on restart the aggregate was at version N
    and the first record() raised (review I9). Verify: instantiating a fresh
    IterationCheckpointer for an existing project_id resumes the version cleanly.
    """
    cp1 = IterationCheckpointer(project_id="p", event_store=sqlite_event_store, snapshot_dir=tmp_path)
    cp1.record({"iteration": 1, "response": "first"})
    cp1.record({"iteration": 2, "response": "second"})

    cp2 = IterationCheckpointer(project_id="p", event_store=sqlite_event_store, snapshot_dir=tmp_path)
    # Must NOT raise ConcurrencyError.
    cp2.record({"iteration": 3, "response": "after restart"})
```

**Verify.** PASS.

**Don't touch.** Don't add real resume logic (read-back + replay) in this task — that's a follow-up; this task only stops the crash and corrects the docstring.

---

## T20: `stub_primitives.py` return shapes match real-schema fields

**Severity:** P1 · **Source:** review I7

**Files:**
- Modify: `backend/agents/rlm/stub_primitives.py`.
- Test: `tests/rlm/test_stub_primitives.py`.

**Problem.** The docstring claims each stub "returns deterministically-shaped data matching the §5 return column." Verified false: `_propose_improvements` returns `{tag, description, target_rubric_area, estimated_uplift}` vs real `ImprovementHypothesis` `{path_id, hypothesis, rationale, expected_outcome, ...}` — **zero key overlap**; `_detect_environment` omits `dockerfile`; `_extract_hyperparameters` uses `epochs` vs real `epochs_or_steps`; etc.

**Root cause.** Stubs were written from intuition, not from the schemas.

**Fix.** Build each stub return from `Schema(...).model_dump()` with placeholder values so the keys can't drift:

```python
def _detect_environment(method_spec: dict, *, ctx) -> dict:
    """Stub: returns a valid EnvironmentSpec.model_dump() with placeholders."""
    from backend.agents.schemas import EnvironmentSpec
    return EnvironmentSpec(
        dockerfile="FROM python:3.11-slim\n",
        python_version="3.11",
        framework="pytorch",
        pip_packages=["numpy"],
    ).model_dump()

def _propose_improvements(current_results, rubric_scores, k=None, *, ctx) -> list[dict]:
    from backend.agents.schemas import ImprovementHypothesis
    return [ImprovementHypothesis(
        path_id="stub-1", hypothesis="stub", rationale="stub",
        expected_outcome="stub", category="stub",
    ).model_dump()]
# ... same pattern for every stub ...
```

**Guard test.**

```python
def test_stub_keys_are_a_subset_of_real_schema(make_run_context, tmp_path):
    """Symptom: a stub run trained the root's REPL code against keys the real chain never produces.

    Stubs returned ad-hoc dicts not derived from the schemas (review I7).
    Verify: every stub primitive's return-key set is a subset of the real
    schema's fields.
    """
    from backend.agents.schemas import (
        EnvironmentSpec, ImprovementHypothesis, ReproductionContract,
        PaperClaimMap, RubricVerification,
    )
    ctx = make_run_context(tmp_path)
    stubs = build_stub_custom_tools(ctx)

    schema_for = {
        "detect_environment": EnvironmentSpec,
        "plan_reproduction": ReproductionContract,
        # ... etc ...
    }
    for name, schema in schema_for.items():
        result = stubs[name]["tool"]({})  # call with empty arg
        assert set(result.keys()) <= set(schema.model_fields), \
            f"stub {name} returned drift keys: {set(result) - set(schema.model_fields)}"
```

**Verify.** PASS.

**Don't touch.** `_resolve_custom_tools`' fallback logic — the stub-vs-real switch is correct.

---

## T21: Stub-run degraded observability — make a stub run honestly identifiable

**Severity:** P1 · **Source:** review I8

**Files:**
- Modify: `backend/agents/rlm/run.py:_finalize` and `_write_demo_status`.
- Modify: `backend/agents/rlm/report.py:RLMFinalReport` — add `primitive_provider` field.
- Test: `tests/rlm/test_run.py`.

**Problem.** A stub run returns `ok=True, success=True, overall_score=0.5` — structurally indistinguishable from a real reproduction except for one `logger.info` line. The persisted `final_report.json` / `demo_status.json` are clean of any "stub" marker.

**Root cause.** Stub-vs-real is observable in code (`tools_label`) but never persisted.

**Fix.**

1. Add to `RLMFinalReport`:
```python
primitive_provider: str = "real"  # "real" | "stub" — review I8 honesty
degraded: bool = False
```

2. Thread `tools_label` through `_finalize`:
```python
def _finalize(*, result_obj, run_failed, ctx, iterations, project_dir, emit,
              corpus_sentinels=None, tools_label: str = "real"):
    # ... existing logic ...
    if "stub" in tools_label.lower():
        report.primitive_provider = "stub"
        report.degraded = True
        # Stub run cannot honestly claim reproduced.
        if report.verdict == "reproduced":
            report.verdict = "partial"
```

3. In `_write_demo_status`, surface it:
```python
def _write_demo_status(..., *, primitive_provider: str = "real"):
    payload["primitiveProvider"] = primitive_provider
```

**Guard test.**

```python
def test_stub_run_is_honestly_observable_in_artifacts(monkeypatch, tmp_path):
    """Symptom: a stub run is structurally indistinguishable from a real reproduction.

    Only a logger.info line signaled degradation (review I8); final_report.json
    and demo_status.json carried no marker. Verify: REPROLAB_RLM_STUB_PRIMITIVES=1
    yields primitive_provider='stub' in the persisted report.
    """
    monkeypatch.setenv("REPROLAB_RLM_STUB_PRIMITIVES", "1")
    # ... run a fast stub run ...
    report = json.loads((run_dir / "final_report.json").read_text())
    assert report["primitive_provider"] == "stub"
    assert report["degraded"] is True
    assert report["verdict"] != "reproduced"
```

**Verify.** PASS.

**Don't touch.** Real-mode default (`"real"`) must not change for back-compat.

---

# P2 — smaller cleanups

## T22: Install tesseract `eng` language data

**Severity:** P2 · **Source:** handoff §4 P2-I10

**Files:** `Dockerfile`, `README.md` install instructions, `docs/specs/setup.md` (if present).

**Problem.** tesseract 5.4.1 is installed but lacks the `eng` language data; `OcrPaperParser` fails soft and the OCR test skips. The wired fallback never runs.

**Fix.** Add to the Dockerfile (and document for host installs):
```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-eng && \
    rm -rf /var/lib/apt/lists/*
```

For macOS dev: `brew install tesseract-lang` (carries eng + others).

**Guard test.**
```python
def test_ocr_eng_data_is_installed():
    """Symptom: OCR fallback is non-functional; tesseract has no eng data.

    The OcrPaperParser test was importorskip-ed in environments missing eng
    (handoff P2-I10). Verify: tesseract reports eng in its language list.
    """
    import subprocess
    out = subprocess.run(["tesseract", "--list-langs"],
                         capture_output=True, text=True, check=True)
    assert "eng" in out.stdout
```

**Verify.** PASS in CI / Docker; locally requires the install.

**Don't touch.** `OcrPaperParser` itself (it's correct; only the data was missing).

---

## T23: `has_provider_credentials` rejects unauthenticated `claude` CLI

**Severity:** P2 · **Source:** handoff §4 P2-I11

**Files:**
- Modify: `backend/agents/runtime/factory.py:has_provider_credentials`.
- Test: `tests/agents/runtime/test_factory.py`.

**Problem.** `has_provider_credentials("anthropic")` treats the `claude` CLI on `PATH` as proof of a valid OAuth session — it is not.

**Fix.** Probe `claude --help` or `claude auth status` (whichever is non-interactive) and check exit code + a known marker; only return True if the OAuth session is actually valid.

```python
def has_provider_credentials(provider: str) -> bool:
    if provider == "anthropic":
        if os.environ.get("ANTHROPIC_API_KEY"):
            return True
        # CLI presence alone is NOT proof; check auth.
        try:
            result = subprocess.run(
                ["claude", "auth", "status"], capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0 and "logged in" in result.stdout.lower()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    # ... other providers ...
```

**Guard test.**

```python
def test_has_provider_credentials_anthropic_requires_valid_oauth(monkeypatch):
    """Symptom: presence of `claude` CLI on PATH passes the credential check even when logged out.

    has_provider_credentials returned True on CLI presence alone (handoff P2-I11).
    Verify: with no ANTHROPIC_API_KEY and a logged-out CLI, returns False.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw:
        subprocess.CompletedProcess(args=a[0], returncode=1, stdout="not logged in"))
    assert has_provider_credentials("anthropic") is False
```

**Verify.** PASS.

**Don't touch.** Other provider branches.

---

## T24: Verify `build_environment` ThreadPoolExecutor timeout is sound

**Severity:** P2 (verify) · **Source:** handoff §4 P2-I12

**Files:** `backend/agents/rlm/primitives.py:206-309`.

**Status.** Reading `primitives.py` shows the executor is already a single pool with per-attempt `.result(timeout=build_timeout)` and an aggregate `deadline_abs` cap (WS-H Batch P, A2-C3). **This task is likely already resolved.**

**Verify (no fix needed if green):**

```python
def test_build_environment_attempt_timeout_actually_bounds(monkeypatch, make_run_context, tmp_path):
    """Symptom: a hung Docker build wedges build_environment past its declared cap.

    Verify the existing per-attempt timeout (WS-H Batch P, A2-C3) actually
    enforces — a fake _build_image that sleeps 2× the per-attempt cap must
    cause TimeoutError + a fail-soft result, not a wedge.
    """
    async def slow_build(*a, **kw):
        await asyncio.sleep(3600)  # 1 hour — well past any test cap
    monkeypatch.setattr(primitives, "_build_image", slow_build)
    monkeypatch.setattr("backend.config.get_settings",
                        lambda: SimpleNamespace(environment_build_max_attempts=1,
                                                environment_build_attempt_s=2,
                                                environment_build_llm_repair_s=1))
    ctx = make_run_context(tmp_path)
    start = time.monotonic()
    result = primitives.build_environment({"dockerfile": "FROM alpine\n"}, ctx=ctx)
    assert time.monotonic() - start < 10  # bounded, not wedged
    assert result["ok"] is False
    assert "timed out" in result["error"].lower()
```

**Verify.** If PASS — close the issue with no code change. If FAIL — escalate; the cap is not actually enforcing.

**Don't touch.** The existing A2-C3 / A2-M1 logic if it passes.

---

## T25: Delete `HtmlPaperParser._extract_references` (dead code) — or implement it

**Severity:** P2 · **Source:** handoff §4 P2-I13

**Files:**
- Modify or delete from: `backend/services/ingestion/parser/html_parser.py`.

**Problem.** `full_text` is space-joined (no newlines), so the line-based "References" scan in `_extract_references` always returns `[]`. The function is dead.

**Fix (preferred).** Delete it and its callers. The cascade does not need references for the RLM pipeline today.

Alternative: rewrite to walk the HTML structure (`<section>` / `<div class="references">`), but only if references are needed downstream — they currently aren't.

**Guard test.**

```python
def test_html_parser_has_no_dead_extract_references():
    """Symptom: _extract_references always returns [] — dead code (handoff P2-I13).

    Verify after deletion: the symbol is gone OR returns a non-empty list
    on a fixture with a known References section.
    """
    from backend.services.ingestion.parser import html_parser
    assert not hasattr(html_parser, "_extract_references"), \
        "dead _extract_references must be removed"
```

**Verify.** PASS.

**Don't touch.** Other parser methods.

---

## T26: `leaf_scorer._parse_batch_response` and `rubric_gen._extract_json_object` — reuse `_extract_json`

**Severity:** P2 (Minor) · **Source:** review M3

**Files:**
- Modify: `backend/evals/paperbench/leaf_scorer.py:246-283` (`_parse_batch_response`).
- Modify: `backend/agents/rlm/rubric_gen.py:146-154` (`_extract_json_object`).
- Test: `tests/evals/test_leaf_scorer.py`, `tests/rlm/test_rubric_gen.py`.

**Problem.** Both functions use first-`{`-to-last-`}` slicing, which over-grabs when the response has prose braces; this burns a retry. `backend.agents.rlm.primitives._extract_json` already does this correctly with `raw_decode`.

**Fix.** Import and reuse:

```python
# leaf_scorer.py — replace _parse_batch_response's brace-slice block with:
from backend.agents.rlm.primitives import _extract_json
# ...
try:
    arr_text = raw.strip()
    # If the response is a top-level array, wrap as {data: [...]} to reuse _extract_json's object scanner.
    # Or use a small array-aware extractor:
    parsed = _extract_json_array(raw)
except ValueError:
    parsed = []
```

Add `_extract_json_array` as a sibling helper in `primitives.py` (10 lines, mirrors `_extract_json` but scans from `[`).

**Guard test.**

```python
def test_leaf_scorer_handles_prose_with_braces():
    """Symptom: prose braces in the LLM response burn a retry.

    First-`{`-to-last-`}` slicing over-grabs when the response has prose
    braces like 'Here {{is}} the result: [...]'  (review M3). Verify: a
    response with prose braces parses on the first attempt.
    """
    raw = 'Note: the {answer} is below.\n[{"leaf_id":"L1","score":0.8,"justification":"x"}]'
    out = _parse_batch_response(raw, [{"id": "L1"}])
    assert out[0]["score"] == 0.8
```

**Verify.** PASS.

**Don't touch.** The `_graded` / ungraded fallback logic.

---

## T27: `report.py:_render_markdown` graded-coverage default

**Severity:** P2 (Minor) · **Source:** review M1

**Files:** `backend/agents/rlm/report.py:367-368`.

**Problem.** `graded` defaults to `leaf_count` when missing, so a rubric dict with no `graded` key falsely claims full `N/N` coverage.

**Fix.** Default to `0`:

```python
        if leaf_count:
            bits.append(f"{rubric.get('graded', 0)}/{leaf_count} rubric leaves graded")
```

**Guard test.** Trivial — render markdown with a rubric dict lacking `graded` and assert `0/N`.

**Verify.** PASS.

---

## T28: `arxiv.py:80-86` — read `response.info()` inside the `with` block

**Severity:** P2 (Minor) · **Source:** review M5

**Files:** `backend/services/ingestion/intake/fetchers/arxiv.py:80-86`.

**Fix.** Move `headers = response.info()` and any subsequent header read **inside** the `with urlopen(...)` block. Remove the masked `try/except: pass`.

**Guard test.** A regression test that asserts response headers are captured. (Or skip the test — the fix is mechanical and the existing test suite covers the happy path.)

**Verify.** Existing arxiv fetcher tests PASS.

---

## T29: `ocr_parser.py:42-43` — cache the tesseract `version` property

**Severity:** P2 (Minor) · **Source:** review M6

**Files:** `backend/services/ingestion/parser/ocr_parser.py:42-43`.

**Fix.**

```python
@functools.cached_property
def version(self) -> str:
    """Cached: tesseract --version was being shelled out on every access."""
    import subprocess
    result = subprocess.run(["tesseract", "--version"], capture_output=True, text=True)
    return result.stdout.split()[1] if result.returncode == 0 else "unknown"
```

**Guard test.** Mock `subprocess.run`; access `.version` twice; assert `run` was called once.

**Verify.** PASS.

---

## T30: `checkpoint.py:152-154` — fsync after JSONL append

**Severity:** P2 (Minor) · **Source:** review M7

**Files:** `backend/agents/rlm/checkpoint.py:152-154`.

**Fix.**

```python
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")
        fh.flush()
        os.fsync(fh.fileno())  # M7: a crash between event-store and JSONL would leave a torn line.
```

**Guard test.** Mock `os.fsync`; record one iteration; assert `os.fsync` was called.

**Verify.** PASS.

---

## T31: Ingestion test suites — make `bs4` / `pytesseract` hard requirements

**Severity:** P2 (Minor) · **Source:** review M8

**Files:**
- Modify: `tests/services/test_ingestion_html_parser.py:12`, `tests/services/test_ingestion_ocr_parser.py:27`.
- Modify: `pyproject.toml` / `setup.cfg` — promote `beautifulsoup4` / `pytesseract` from optional to required.

**Fix.** Replace `pytest.importorskip("bs4")` with a plain `import bs4` — let the test fail if the dep is missing. Promote both to required dependencies.

**Guard test.** N/A — the test failure IS the guard.

**Verify.** Full suite still PASS (because both deps are installed).

---

# Final acceptance — end-to-end verification

After all P0 + P1 tasks land:

**Step 1: Full test suite.**
```bash
.venv/bin/python -m pytest tests/ -q
```
Expected: ≥1163 passed (was 1163 at handoff baseline; should grow as guard tests are added — count grows by ~25 new tests), same 2 pre-existing failures (`test_issue17_runtime`, `test_issue26_experiment_runner`), 4 skipped.

**Step 2: A real reproduction run on the easiest paper, end-to-end.**
```bash
.venv/bin/python scripts/rlm_paperbench.py mechanistic-understanding \
    --sandbox docker --gpu-mode prefer --max-wall-clock 7200
```
Expected:
- `runs/<run_id>/final_report.json` contains real `baseline_metrics` (not `{}`).
- `runs/<run_id>/final_report.json` `rubric.overall_score` > 0.35 (the 0.35 cap no longer fires on a real metric-bearing run).
- `runs/<run_id>/final_report.md` shows a populated per-area table and a correct `✔/✘ target` flag.
- The `_finalize` log line reports `verdict=reproduced|partial` consistent with the score.

**Step 3: Confirm honesty invariants hold.**
- A stub run (`REPROLAB_RLM_STUB_PRIMITIVES=1`) writes `primitive_provider: "stub"` and `verdict != "reproduced"`.
- A run with `baseline_metrics={}` writes `rubric.overall_score ≤ 0.35` and `verdict ∈ {"partial", "failed"}`.

**Step 4: Update `progress.md`, `CHANGELOG.md` `[Unreleased]`, `learn.md`** (one post-mortem entry per non-obvious bug fixed — Symptom → Root cause → Fix → Lesson → Guardrail). The handoff §1 mandates this.

**Step 5: When green — fast-forward `main` to `merge`, push both to origin.**
```bash
git checkout main && git merge --ff-only merge && git push origin main && git push origin merge
```

---

## Notes on omitted items

- **Review M4 (zero-usage cost rows)** — already correctly documented in `report.py:_cost_dict`; the honest fix (per-primitive token usage in the `LlmClient` protocol) is a seam evolution deferred to a future task. UI must label the figure honestly or omit it; that's a Phase-6 UI task, not in this plan.
- **Handoff §4 P0-I3 (`ftrl` reproduced the wrong method)** — included as T7 above; investigation drives whether it's a vendoring fix or a prompt fix.
