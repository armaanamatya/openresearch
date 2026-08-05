# Tier-3 Sub-project A — Clean run on the GCP single-VM path Implementation Plan

> **STATUS 2026-08-01 — EXECUTED (2026-07-22/23).** The checkboxes below are the as-authored
> roadmap and were not ticked during execution. Ground truth for what actually landed:
> `docs/progress/2026-07-22-tier3-adam-progress.md` and the status banner in
> `docs/superpowers/specs/2026-07-22-tier3-scheduler-adam-ab-design.md`. Tasks 1–4 and 6
> executed. Note two disclosed deviations: (1) the 5-field checkpoint/resume work
> (spec A-item-2) moved to sub-project B; (2) Task 5 (autonomous-profile routing) was
> explicitly deferred as an operator decision — not implemented (see the "BLOCKED on an
> operator decision" marker at Task 5 below).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the GCP single-VM path land one clean paper reproduction — the in-VM child's `final_report.json` (`evidence_gate_passed: true` + a `success=True` experiment row) is collected back cleanly, on a graceful `FINALIZE` path, without billing idle GPU to the max-run-duration ceiling — and harden the "GKE is not used" posture in code + docs.

**Architecture:** The GCP single-VM path (`VmComputeProvider`) SSHes `reproduce --sandbox local` *inside* one GPU VM. The child writes `final_report.json` + `experiment_runs.jsonl` and evaluates its own evidence gate in-process (where the `run_experiment` calls happened). The orchestrator's job is lifecycle only: detect the child's terminal sentinel, tear down (stop billing), and collect the artifacts — including the evidence artifacts needed to audit the gate. This sub-project adds terminal-detection to `watch()`, widens `collect()`'s artifact allow-list, rewords the (already-existing, already-fail-closed) GKE guard + all authoritative docs to "not used," and handles the autonomous-profile routing break the guard now surfaces.

**Tech Stack:** Python 3.11+, pytest (hermetic — fake `runner` callables, no live `gcloud`/`ssh`/`scp`), the existing golden-argv test pattern in `tests/services/runtime/test_vm_compute_provider.py`.

---

## Scope note — read before starting

**What A delivers (exit criterion):** one clean paper reproduction on one GCP VM whose collected `final_report.json` shows `evidence_gate_passed: true` and whose `experiment_runs.jsonl` has a `success=True` row — reached on the graceful `FINALIZE` state, not `RECOVERED`.

**Deviations from the design spec (`docs/superpowers/specs/2026-07-22-tier3-scheduler-adam-ab-design.md`), decided during planning with advisor review — surface these to the operator at handoff:**

1. **The 5-field checkpoint/resume work (spec's A-item-2) moves to sub-project B.** It is the freeze/revive *substrate*, not a prerequisite for a *first clean run*: a run that finishes inside its budget never needs to resume. The spec's "A is a hard prerequisite for B" becomes **"B builds its own checkpoint/resume substrate."** B's plan MUST open by absorbing it as a stated prerequisite — do not let it fall in the gap.
   - **Tripwire:** give A's real run generous `--max-gpu-hours` / `max_run_duration_s` so the minutes-scale experiments finish in one shot. If A's run repeatedly gets STOP'd mid-run (budget ceiling or A100-stockout preemption), that is the signal to pull checkpoint/resume forward from B into A. Note it in the run journal (`docs/progress/2026-07-22-tier3-adam-progress.md`) if it happens.

2. **A's exit bar is the per-run evidence gate, NOT terminal `REPRODUCED`.** The default-OFF `OPENRESEARCH_EXTERNAL_VALIDATOR` soft-quarantines every attempt, so campaign-terminal `REPRODUCED` is unreachable without wiring a validator backend — out of A's scope. A reads `report.py`'s per-run `evidence_gate_passed` + the success row (distinct from campaign `run_level_clean`). The external validator and terminal `REPRODUCED` are sub-project C concerns.

3. **The "status stuck `running`" bug that killed `cutout_val1` was GKE-specific** (`k8s_job_cell_runner`). The local cell path (`gpu_cell_runner`, what runs under `--sandbox local` inside the VM) has no such Job-status flip, so moving to the single-VM path inherently avoids it. No cell-status-flip code is in A.

**RED LINE — do NOT cross (evidence-not-grade):** never edit `attempt_assessment.py:673-679` or `campaign_policy.py:908-922` to force cleanliness. A clean verdict must come from real upstream artifacts (the child's evidence gate + the collected bundle), never from an adjusted gate. If you find yourself editing an assessment/policy predicate to make a run "pass," stop — that is the fabrication vector the whole harness is built to prevent.

**GKE resolution (settles the recon's "self-contradiction" flag — just implement it, do not re-litigate):** docs say **"GKE is not used"**; the `OPENRESEARCH_ALLOW_GKE` revive branch **stays** in the code as an inert operator-only escape hatch (the approved "keep inert + guard + doc-hardened" choice). The guard already fail-closes on both the direct and `OPENRESEARCH_CLOUD_FAILOVER` paths — this sub-project rewords, adds one failover-path regression test, and does NOT remove the revive branch.

**Doc-reword boundary:** reword only the *current/authoritative* docs (Task 4). **Leave the dated runbooks, `docs/periods/*`, and `CHANGELOG.md` alone** — they are incident narrative per repo norm, and one records the opposite "keep parked" decision. Rewording them fights the norm.

---

## File Structure

| File | Responsibility | Tasks |
|---|---|---|
| `backend/services/runtime/vm_compute_provider.py` | VM lifecycle: `watch()` terminal-detection + `collect()` evidence allow-list | 1, 2 |
| `backend/agents/rlm/reproduction_run.py` | State machine: handle the new `completed` terminal state on the graceful path | 1 |
| `backend/agents/rlm/primitives.py` | `_backend_for_sandbox_mode` GKE guard message | 3 |
| `backend/services/events/live_runs.py` | Autonomous-profile routing (guard now fail-closes `sandbox='gcp'`) | 5 |
| `README.md`, `docs/architecture.md`, `docs/operations.md`, `docs/engineering-guide.md`, `backend/services/runtime/CLAUDE.md`, root `CLAUDE.md`, `docs/reference/flags.md` | "GKE not used" doc posture | 4 |
| `tests/services/runtime/test_vm_compute_provider.py` | watch()/collect() hermetic tests | 1, 2 |
| `tests/services/runtime/test_gke_alias.py`, `tests/services/runtime/test_backend_factory_gcp.py` | GKE guard message + failover fail-closed regression | 3 |
| `tests/services/events/test_live_runs*.py` | autonomous-profile routing regression | 5 |
| `configs/` GCP run-spec | arm `OPENRESEARCH_CELL_ERROR_SALVAGE` where it applies | 6 |

---

## Task 1: `watch()` terminal-detection via the in-VM report sentinel

**Why:** Today `watch()` classifies only VM *status* (RUNNING vs not). When the in-VM child finishes and writes `final_report.json`, the VM stays RUNNING — so `watch()` keeps polling until the max-run-duration ceiling STOPs the VM, which routes through `stopped_uncollected → _emergency_shutdown → recover → RECOVERED`. Result: **every clean run currently lands as `RECOVERED` and bills idle GPU from child-finish to the ceiling.** Detecting the sentinel yields a graceful `completed` terminal state, stops billing immediately, and takes the clean `FINALIZE` path.

**Files:**
- Modify: `backend/services/runtime/vm_compute_provider.py` (the `watch()` method ~line 655; add a `_run_completed_on_vm` helper)
- Modify: `backend/agents/rlm/reproduction_run.py:299-303` (add a `completed` branch in the watch-consumer loop)
- Test: `tests/services/runtime/test_vm_compute_provider.py`

- [ ] **Step 1: Write the failing tests for the sentinel probe**

Append to `tests/services/runtime/test_vm_compute_provider.py`. Note the imports at the top of the file already bring in `VmComputeProvider`, `VmExecResult`, `ComputeLease`; add `RunHandle` to the existing `from backend.services.runtime.compute_provider import ...` line (grep `class RunHandle` to confirm the module — it is the same one `recover()` imports from). The existing tests construct the provider as `VmComputeProvider(profile, runner=<callable>)`; this test also injects `sleep` — if the constructor kwarg differs, Step 2's run will show the `TypeError` and you adjust.

```python
def test_watch_returns_completed_when_report_present():
    """A RUNNING VM whose in-VM child already wrote final_report.json yields a
    single terminal 'completed' status and stops polling -- no idle-GPU billing
    to the max-run-duration ceiling, and the graceful FINALIZE path downstream."""
    def fake_runner(argv):
        joined = " ".join(argv)
        if "final_report.json" in joined:      # the ssh sentinel probe
            return VmExecResult(returncode=0, stdout="DONE\n")
        return VmExecResult(returncode=0, stdout="RUNNING\n")  # instances describe
    prov = VmComputeProvider(_gcp_profile(), runner=fake_runner, sleep=lambda s: None)
    handle = RunHandle(id="prj_test", lease=_lease())
    states = [s.state for s in prov.watch(handle)]
    assert states == ["completed"]


def test_watch_keeps_polling_until_report_present():
    """While the child is still running (no final_report.json), watch() yields
    'running' and keeps polling -- until the sentinel flips to DONE."""
    describe_calls = {"n": 0}
    def fake_runner(argv):
        joined = " ".join(argv)
        if "final_report.json" in joined:
            # WAIT on the first poll, DONE once two describes have happened.
            return VmExecResult(returncode=0, stdout=("DONE\n" if describe_calls["n"] >= 2 else "WAIT\n"))
        describe_calls["n"] += 1
        return VmExecResult(returncode=0, stdout="RUNNING\n")
    prov = VmComputeProvider(_gcp_profile(), runner=fake_runner, sleep=lambda s: None)
    handle = RunHandle(id="prj_test", lease=_lease())
    states = [s.state for s in prov.watch(handle)]
    assert states == ["running", "completed"]


def test_watch_probe_error_does_not_false_complete():
    """A failed/empty sentinel probe must NEVER be read as DONE -- watch() must
    fail-open to 'running' and keep polling (never a false terminal)."""
    def fake_runner(argv):
        joined = " ".join(argv)
        if "final_report.json" in joined:
            return VmExecResult(returncode=255, stdout="", stderr="ssh: connect timeout")
        return VmExecResult(returncode=0, stdout="RUNNING\n")
    prov = VmComputeProvider(_gcp_profile(), runner=fake_runner, sleep=lambda s: None)
    handle = RunHandle(id="prj_test", lease=_lease())
    # Take just the first two yields (it would otherwise poll forever on RUNNING).
    import itertools
    states = [s.state for s in itertools.islice(prov.watch(handle), 2)]
    assert states == ["running", "running"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/services/runtime/test_vm_compute_provider.py::test_watch_returns_completed_when_report_present tests/services/runtime/test_vm_compute_provider.py::test_watch_keeps_polling_until_report_present tests/services/runtime/test_vm_compute_provider.py::test_watch_probe_error_does_not_false_complete -v`
Expected: FAIL — the first yields `running` forever (no `completed` state exists yet). If instead you get a `TypeError` on the `sleep=` kwarg or an `ImportError` on `RunHandle`, fix the import/kwarg to match the real signature and re-run to reach the real assertion failure.

- [ ] **Step 3: Add the sentinel helper + wire it into `watch()`**

In `backend/services/runtime/vm_compute_provider.py`, replace the `watch()` body's RUNNING branch to probe the sentinel, and add the helper method just below `watch()`:

```python
    def watch(self, handle: RunHandle) -> Iterator[RunStatus]:
        """Poll ``describe`` status until the VM leaves RUNNING **or the in-VM
        child writes its terminal ``final_report.json``**.

        A non-RUNNING VM (stopped by the max-run-duration ceiling, a
        preemption, or an external actor) is classified
        ``stopped_uncollected`` -- the state machine reacts by calling
        ``recover()``. A RUNNING VM whose child has already written
        ``final_report.json`` is classified ``completed`` -- a graceful
        terminal that routes the state machine straight to COLLECT ->
        RELEASE_GPU -> FINALIZE, so a finished run does not keep billing idle
        GPU until the ceiling. ``synced`` is always ``False`` here: this method
        never pulls artifacts (that is ``collect``'s job).
        """
        while True:
            argv = self._gcloud(
                "compute", "instances", "describe", self._instance,
                "--zone", self._zone, "--format=value(status)",
            )
            result = self._run(argv)
            status = (result.stdout or "").strip()
            if status == "RUNNING":
                if self._run_completed_on_vm(handle):
                    yield RunStatus(state="completed", detail="final_report.json present", synced=False)
                    return
                run_status = RunStatus(state="running", detail=status, synced=False)
            elif status:
                run_status = RunStatus(state="stopped_uncollected", detail=status, synced=False)
            else:
                run_status = RunStatus(state="stalled", detail="empty status", synced=False)
            yield run_status
            if run_status.state != "running":
                return
            self._sleep(self._poll_interval_s)

    def _run_completed_on_vm(self, handle: RunHandle) -> bool:
        """True once the in-VM ``reproduce`` child has written its terminal
        ``final_report.json`` -- the DONE sentinel the bash watcher polls
        (``scripts/sdar_gcp_watch.sh``). Any ssh error / non-DONE output is
        treated as 'not done yet' (fail-OPEN to keep polling -- never a false
        DONE that would abandon a still-running GPU job)."""
        remote_run_dir = f"{self._remote_dir}/runs/{handle.id}"
        probe = f"test -f {remote_run_dir}/final_report.json && echo DONE || echo WAIT"
        result = self._run(self._ssh_argv(probe))
        return result.returncode == 0 and "DONE" in (result.stdout or "")
```

- [ ] **Step 4: Wire the `completed` state into the reproduction state machine**

In `backend/agents/rlm/reproduction_run.py`, inside the `for status in self._provider.watch(handle):` loop, add a `completed` branch right after the `stopped_uncollected` check (currently lines 299-303). The generator already `return`s after yielding `completed`, so the loop would fall through to the graceful COLLECT path anyway — this `break` is explicit + skips the budget check on the final poll:

```python
                if status.state == "stopped_uncollected":
                    # Invariant 4: compute vanished before COLLECT could run.
                    return self._emergency_shutdown(
                        lease, reason="stopped_uncollected", decision=decision.decision
                    )

                if status.state == "completed":
                    # The in-VM child wrote final_report.json: the run finished
                    # cleanly. Break to the graceful COLLECT -> RELEASE_GPU ->
                    # FINALIZE path (invariant 3) instead of billing idle GPU to
                    # the max-run-duration ceiling.
                    break
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/services/runtime/test_vm_compute_provider.py -v -k watch`
Expected: PASS (all three new tests, plus any pre-existing watch tests still green).

- [ ] **Step 6: Run the reproduction-run state-machine tests to confirm no regression**

Run: `.venv/bin/python -m pytest tests/rlm/ -v -k "reproduction_run" 2>/dev/null || .venv/bin/python -m pytest -v -k reproduction_run`
Expected: PASS — existing `stopped_uncollected → RECOVERED` and `FINALIZE` invariant tests unchanged. If a test constructs a fake provider whose `watch()` yields a bespoke sequence, the new `completed` branch is additive and should not affect it.

- [ ] **Step 7: Commit**

```bash
git add backend/services/runtime/vm_compute_provider.py backend/agents/rlm/reproduction_run.py tests/services/runtime/test_vm_compute_provider.py
git commit -m "VM watch(): detect in-VM final_report sentinel -> graceful FINALIZE, no idle-GPU billing to ceiling"
```

---

## Task 2: `collect()` — bring back the evidence artifacts the gate keyed on

**Why:** `collect()` tars a fixed allow-list; anything unlisted is silently dropped. Today it copies `final_report.{json,md}`, `demo_status.json`, `cost_ledger.jsonl`, `experiment_runs.jsonl`, `dashboard_events.jsonl`, `code/metrics.json` — but NOT the artifacts that let an operator audit *why* the child's evidence gate passed/failed: `generated_rubric.json`, `rlm_state/evidence_bundle.json`, `rlm_state/validation_verdict.json`, `rubric_tree.json`. A clean run should return an auditable bundle, not just the verdict.

**Files:**
- Modify: `backend/services/runtime/vm_compute_provider.py` (the `collect()` method ~lines 692-717, the `tar_cmd` file list)
- Test: `tests/services/runtime/test_vm_compute_provider.py`

- [ ] **Step 1: Write the failing test**

```python
def test_collect_tars_evidence_artifacts():
    """collect() must include the evidence-audit artifacts (rubric, evidence
    bundle, validation verdict, rubric tree) in the remote tar allow-list, not
    just the report + ledger -- so a collected clean run is auditable."""
    captured = {"tar_cmd": None}
    def fake_runner(argv):
        joined = " ".join(argv)
        if "tar czf" in joined:
            captured["tar_cmd"] = joined
        return VmExecResult(returncode=0, stdout="")
    prov = VmComputeProvider(_gcp_profile(), runner=fake_runner)
    handle = RunHandle(id="prj_test", lease=_lease())
    bundle = prov.collect(handle)
    assert bundle.ok is True
    tar = captured["tar_cmd"]
    # existing artifacts still present
    for existing in ("final_report.json", "experiment_runs.jsonl", "code/metrics.json"):
        assert existing in tar, f"regression: {existing} dropped from collect tar"
    # new evidence-audit artifacts
    for evidence in (
        "generated_rubric.json",
        "rlm_state/evidence_bundle.json",
        "rlm_state/validation_verdict.json",
        "rubric_tree.json",
    ):
        assert evidence in tar, f"{evidence} missing from collect tar allow-list"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/services/runtime/test_vm_compute_provider.py::test_collect_tars_evidence_artifacts -v`
Expected: FAIL — the four evidence artifacts are not in the current `tar_cmd`.

- [ ] **Step 3: Widen the allow-list**

In `collect()`, extend the `tar_cmd` file list. `tar czf ... 2>/dev/null` already tolerates missing files (the `2>/dev/null` swallows "No such file"), so listing an artifact a given run didn't produce is safe:

```python
        tar_cmd = (
            f"cd {remote_run_dir} 2>/dev/null && tar czf {tgz_path} "
            f"final_report.json final_report.md demo_status.json cost_ledger.jsonl "
            f"experiment_runs.jsonl dashboard_events.jsonl code/metrics.json "
            f"generated_rubric.json rubric_tree.json "
            f"rlm_state/evidence_bundle.json rlm_state/validation_verdict.json "
            f"2>/dev/null; echo tarred"
        )
```

- [ ] **Step 4: Run it to verify it passes**

Run: `.venv/bin/python -m pytest tests/services/runtime/test_vm_compute_provider.py -v -k collect`
Expected: PASS (new test + any pre-existing collect tests).

- [ ] **Step 5: Commit**

```bash
git add backend/services/runtime/vm_compute_provider.py tests/services/runtime/test_vm_compute_provider.py
git commit -m "VM collect(): tar evidence-audit artifacts (rubric, evidence bundle, validation verdict) so a clean run is auditable"
```

---

## Task 3: Reword the GKE guard message to "not used" (keep the revive branch) + failover-path regression

**Why:** The fail-closed GKE guard already exists (`_backend_for_sandbox_mode`, gcp branch) and already covers the `OPENRESEARCH_CLOUD_FAILOVER` path (`cloud_failover._real_backend_factory` re-enters the same guarded function). Two gaps: (1) the guard's `RuntimeError` message still says "PARKED … set OPENRESEARCH_ALLOW_GKE=1 to revive," which reads as "parked/temporary" rather than the operator's "not used" posture; (2) no test pins that the *failover* path is not a bypass hole. We reword the message (keeping the inert `ALLOW_GKE` escape hatch), add the failover regression, and update the two existing tests that match on `"PARKED"`.

**Files:**
- Modify: `backend/agents/rlm/primitives.py` (the `if mode is SandboxMode.gcp:` branch, the `RuntimeError(...)` message)
- Modify: `tests/services/runtime/test_gke_alias.py` (the `match="PARKED"` assertions)
- Modify: `tests/services/runtime/test_backend_factory_gcp.py` (the `match="PARKED"` assertion)
- Test: `tests/services/runtime/test_gke_alias.py` (new failover-path test)

- [ ] **Step 0: VERIFY the failover behavior before writing its test (read, no edit)**

The failover test below asserts `_resolve_run_backend(gcp)` with `OPENRESEARCH_CLOUD_FAILOVER=gcp` **raises**. That depends on whether `cloud_failover.select_backend_with_failover` treats a backend-construction `RuntimeError` as terminal or catches it and falls through to the next candidate. The recon *inferred* it fails closed (failover preference `(gcp, azure)`) but did not prove it. **Read `backend/services/runtime/cloud_failover.py::select_backend_with_failover` first.** If a construction `RuntimeError` propagates → the test as written is correct. If it is caught and the next candidate (Azure) is constructed instead → change the test to assert it returns an `AksJobBackend` (NOT a GKE bypass — Azure ≠ GKE, so the no-GKE directive still holds — but the "no bypass hole" claim must be written accurately). **Do NOT weaken the guard to make a raising-test pass.**

- [ ] **Step 1: Update the existing guard tests to the new "not used" wording + add the failover regression**

In `tests/services/runtime/test_gke_alias.py`, change `test_gke_token_parked_raises_without_flag` to match the new message, and add a failover-path regression + keep the Azure-untouched assertion (already present as `test_aks_path_still_resolves_to_aks_backend`):

```python
def test_gke_token_not_used_raises_without_flag(monkeypatch):
    """GKE is NOT USED: the gke alias raises a fail-closed RuntimeError unless
    the inert operator-only OPENRESEARCH_ALLOW_GKE escape hatch is set."""
    monkeypatch.delenv("OPENRESEARCH_ALLOW_GKE", raising=False)
    from backend.agents.rlm.primitives import _backend_for_sandbox_mode

    with pytest.raises(RuntimeError, match="not used"):
        _backend_for_sandbox_mode(SandboxMode("gke"), run_budget=None)


def test_gcp_fails_closed_through_failover_path(monkeypatch):
    """The OPENRESEARCH_CLOUD_FAILOVER path must NOT be a bypass hole: routing
    gcp through _resolve_run_backend with failover set still fail-closes,
    because _real_backend_factory re-enters the same guarded function."""
    monkeypatch.delenv("OPENRESEARCH_ALLOW_GKE", raising=False)
    monkeypatch.setenv("OPENRESEARCH_CLOUD_FAILOVER", "gcp")
    from backend.agents.rlm.primitives import _resolve_run_backend

    with pytest.raises(RuntimeError, match="not used"):
        _resolve_run_backend(SandboxMode.gcp, run_budget=None)
```

Delete the now-renamed old `test_gke_token_parked_raises_without_flag`. Leave `test_gke_token_constructs_gke_backend` and `test_force_sandbox_gke_threads_run_budget` untouched — the revive branch stays, so `ALLOW_GKE=1` still constructs `GkeJobBackend`.

In `tests/services/runtime/test_backend_factory_gcp.py`, change the one `match="PARKED"` assertion (line ~28) to `match="not used"`. Leave the `ALLOW_GKE=1 → GkeJobBackend` cases untouched.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/services/runtime/test_gke_alias.py tests/services/runtime/test_backend_factory_gcp.py -v -k "not_used or failover or parked"`
Expected: FAIL — the guard message still says "PARKED", and `_resolve_run_backend` may need importing (if the failover test errors on import, confirm the symbol name via `grep "def _resolve_run_backend" backend/agents/rlm/primitives.py`).

- [ ] **Step 3: Reword the guard message (keep the revive branch + `ensure_gcp_available` unchanged)**

In `backend/agents/rlm/primitives.py`, the `if mode is SandboxMode.gcp:` branch — reword only the `RuntimeError` string. Keep the `ALLOW_GKE` gate and the `GkeJobBackend` construction exactly as-is:

```python
    if mode is SandboxMode.gcp:
        if os.environ.get("OPENRESEARCH_ALLOW_GKE", "").strip().lower() not in (
            "1",
            "true",
            "yes",
        ):
            raise RuntimeError(
                "sandbox=gcp/gke routes to GKE, which is NOT USED. The supported "
                "GCP GPU path is the campaign single-VM route: `campaign "
                "--campaign-driver unified --sandbox local --billing-sandbox gcp`. "
                "(OPENRESEARCH_ALLOW_GKE is an inert operator-only escape hatch, "
                "not a supported path.)"
            )
        import backend.services.runtime as _runtime
        from backend.services.runtime.gke_job_backend import GkeJobBackend

        _runtime.ensure_gcp_available()
        return GkeJobBackend(run_budget=run_budget, gpu_plan=gpu_plan)
```

Also update the function's own docstring line "``SandboxMode.gcp`` (and its ``gke`` alias) is PARKED:" → "is NOT USED (fail-closed):" so the source comment matches.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/services/runtime/test_gke_alias.py tests/services/runtime/test_backend_factory_gcp.py -v`
Expected: PASS (whole files — including the untouched `ALLOW_GKE=1 → GkeJobBackend`, the Azure/AWS-untouched assertions, and the HTTP 422 contract test).

- [ ] **Step 5: Commit**

```bash
git add backend/agents/rlm/primitives.py tests/services/runtime/test_gke_alias.py tests/services/runtime/test_backend_factory_gcp.py
git commit -m "GKE guard: reword to 'not used' + pin failover path fail-closed (revive branch stays inert)"
```

---

## Task 4: Reword all authoritative docs to "GKE is not used"

**Why:** The operator directive (memory `no-gke-directive`) is that every cloud-posture doc must state GKE is *not used*, not "parked." Reword only the current/authoritative docs; leave dated runbooks / `docs/periods/*` / `CHANGELOG.md` alone (incident narrative).

**Files (exact edits):**
- Modify: `README.md:149`, `README.md:199`
- Modify: `docs/architecture.md:80`
- Modify: `docs/operations.md:40`
- Modify: `docs/engineering-guide.md:49`, `:58`
- Modify: `backend/services/runtime/CLAUDE.md:20`, `:22-23`
- Modify: root `CLAUDE.md:49`, and the "GKE runs go through the cell-matrix" rule (~`:102-104`)
- Modify: `docs/reference/flags.md` (any `OPENRESEARCH_ALLOW_GKE` / GKE-parked row) — regenerate, do not hand-edit (see Step 6)

- [ ] **Step 1: README.md**

Line 149 — change:
```
GKE is **parked** behind a fail-loud guard (`OPENRESEARCH_ALLOW_GKE` to revive)
and is not the go-forward GCP route.
```
to:
```
GKE is **not used** — a fail-closed guard rejects `--sandbox gcp/gke` on the
reproduction path (`OPENRESEARCH_ALLOW_GKE` is an inert operator-only escape
hatch, not a supported path); the single-VM path above is the GCP GPU route.
```
Line 199 — change the table cell `GCP config (single-VM GPU path is the supported route; GKE parked)` to `GCP config (single-VM GPU path is the supported route; GKE not used)`.

- [ ] **Step 2: docs/architecture.md:80** — change the `gcp` / `gke` row:
```
| `gcp` / `gke` | Not used | GKE Kubernetes-job backend is disabled by a fail-closed guard (`OPENRESEARCH_ALLOW_GKE` is an inert escape hatch, not a supported path); the GCP GPU route is the single-VM `local` path above |
```

- [ ] **Step 3: docs/operations.md:40** — change:
```
GKE is parked behind a fail-loud guard (`OPENRESEARCH_ALLOW_GKE` to revive).
```
to:
```
GKE is not used — a fail-closed guard rejects it (`OPENRESEARCH_ALLOW_GKE` is an
inert operator-only escape hatch, not a supported path).
```

- [ ] **Step 4: docs/engineering-guide.md** — line 49 change the matrix row `parked (OPENRESEARCH_ALLOW_GKE to revive)` → `not used (fail-closed guard; OPENRESEARCH_ALLOW_GKE is an inert escape hatch)`; line 58 change `NOT GKE (parked)` → `NOT GKE (not used)`.

- [ ] **Step 5: backend/services/runtime/CLAUDE.md** — line 20 change `**`gcp`/`gke` is PARKED** (see below)` → `**`gcp`/`gke` is NOT USED** (fail-closed; see below)`. Rewrite the line-22 heading `### GKE / GCP backend (`--sandbox gke` = `--sandbox gcp`) — PARKED` → `### GKE / GCP backend (`--sandbox gke` = `--sandbox gcp`) — NOT USED`, and in the line-23 body change `**PARKED: `_backend_for_sandbox_mode` RAISES a clear `RuntimeError` on the `gcp` branch unless `OPENRESEARCH_ALLOW_GKE` ∈ {`1`,`true`,`yes`}**` → `**NOT USED: `_backend_for_sandbox_mode` RAISES a fail-closed `RuntimeError` on the `gcp` branch; `OPENRESEARCH_ALLOW_GKE` ∈ {`1`,`true`,`yes`} is an inert operator-only escape hatch, not a supported path**`. Leave the SKU catalog + machine-type override sentences (they document the inert code, still accurate).

- [ ] **Step 6: root CLAUDE.md** — line 49 change `gcp`/`gke` is PARKED and raises unless `OPENRESEARCH_ALLOW_GKE=1`` → `gcp`/`gke` is NOT USED and fail-closes unless the inert `OPENRESEARCH_ALLOW_GKE=1` escape hatch is set`. In the "GKE runs go through the cell-matrix" rule (~line 102), prefix with the posture: `**GKE is not used** (fail-closed guard). The historical rule, for the inert code path: the monolithic ...`.

- [ ] **Step 7: Regenerate the flag registry (do NOT hand-edit `docs/reference/flags.md`)**

The `OPENRESEARCH_ALLOW_GKE` row is generated. Regenerate and verify no drift:

Run: `.venv/bin/python scripts/gen_flag_registry.py && git diff --stat docs/reference/flags.md`
Expected: either no change (the registry pulls the docstring/description from code) or a clean regenerated diff. If the row's description text is sourced from a code comment, update that comment to say "not used / inert escape hatch" and regenerate.

- [ ] **Step 8: Run the doc-fidelity + docs-check gates**

Run: `.venv/bin/python -m pytest tests/test_claude_md_fidelity.py -v`
Expected: PASS — the fidelity test reads root + nested CLAUDE.md; the primitive count (19) and default sandbox (local) anchors are unchanged, and `OPENRESEARCH_ALLOW_GKE` is still git-grep-able under `backend/` (the flag stays in code), so `_DOCUMENTED_ENV_VARS` still resolves.
Run (if present): `.venv/bin/python scripts/docs_check.py 2>/dev/null || true`
Expected: green / no forbidden entries.

- [ ] **Step 9: Commit**

```bash
git add README.md docs/architecture.md docs/operations.md docs/engineering-guide.md backend/services/runtime/CLAUDE.md CLAUDE.md docs/reference/flags.md
git commit -m "Docs: GKE is not used (authoritative docs only; dated runbooks/periods untouched)"
```

---

## Task 5: Autonomous-profile routing — SURFACE as an operator decision, do NOT implement in A

**Status: BLOCKED on an operator decision — not on A's critical path. Do not implement until the operator picks a route.**

**The finding (must not stay silent):** `apply_autonomous_profile_override` (`live_runs.py:518-544`) routes the opt-in autonomous profile to `sandbox="gcp"` unless the request is Azure. With the GKE guard fail-closing `sandbox="gcp"`, the autonomous profile — the product's unattended "bare arXiv ID → full pipeline" path — now hard-fails at backend construction. This is a **pre-existing break** (the guard predates this plan) and is **off A's critical path** (A uses the campaign `--sandbox local --billing-sandbox gcp` route, which never calls `apply_autonomous_profile_override`). But the doc-hardening (Task 4) makes the contradiction prominent, so it must be surfaced.

**Why this is NOT a "pick the obvious default" call:** the naive fix (`cloud = "local"`) may be *actively worse* than the current break — `local` = host subprocess execution, and the deepinvent.ai backend host likely has **no GPU**, so unattended runs would run CPU-only or fail differently. "What should unattended GPU runs use" cannot be resolved from the code; it is a product/infra decision. Options to put to the operator:
- **`local`** — matches the shipped default + no-GKE posture, but only correct if the autonomous host actually has a GPU (or always bills through a campaign VM).
- **Azure-or-fail** — select `azure` when configured, else raise a clear "no GPU route configured for autonomous mode" error instead of silently routing to a fail-closed `gcp`.
- **Campaign-VM** — reshape the autonomous path to drive the single-VM campaign route (larger change; the honest GPU answer, but not a one-line profile edit).

**Action for the executor:** do nothing here until the operator answers. When they do, this becomes a small TDD task on `apply_autonomous_profile_override` (test the chosen route + preserve explicit Azure + OFF-is-identity + update the line-181 `gcp/gke parked` comment to `gcp/gke not used`). Until then, Task 4's docs correctly say "GKE not used"; this dangling product path is tracked here and in the run journal.

---

## Task 6: Arm `OPENRESEARCH_CELL_ERROR_SALVAGE` in the GCP run-spec (verify it stays verdict-only)

**Why:** `OPENRESEARCH_CELL_ERROR_SALVAGE` (default-OFF) caps a `cell_execution_error` run whose cells executed-then-errored with real graded metrics from `reproduced → partial` — turning a lost-but-real cloud run into an honest partial instead of a bare failure. It is a report-time verdict cap gated on a `partial_cell_error` ledger stamp + a `cell_manifest.json` error receipt (a REPL-forged row still fails closed), NOT a resume mechanism. Arming it on the GCP run-spec makes A's expensive cloud run degrade gracefully rather than losing signal.

**Files:**
- Modify / create: the GCP campaign run-spec JSON (grep `grep -rl "billing-sandbox\|OPENRESEARCH_MAX_RUN_GPU_USD" configs/` — likely `configs/campaign_run_spec.json` or a `configs/*gcp*` spec). Add the flag.
- Test: reference the existing `tests/rlm/test_cell_error_salvage.py` (no new code — confirm it still asserts verdict-only, independent of resume).

- [ ] **Step 1: Confirm the flag is verdict-only (read, no edit)**

Run: `.venv/bin/python -m pytest tests/rlm/test_cell_error_salvage.py -v`
Expected: PASS — this is the regression that `CELL_ERROR_SALVAGE` caps `reproduced→partial` on a real cell-error receipt and fails closed on a forged row. It confirms arming the flag cannot fabricate a verdict.

- [ ] **Step 2: Add the flag to the GCP run-spec**

Locate the run-spec used by the GCP campaign path (the one loaded via `--run-spec`). Add:
```json
{
  "OPENRESEARCH_CELL_ERROR_SALVAGE": "1"
}
```
merged into the existing spec object (do NOT add the driver-owned per-attempt keys `OPENRESEARCH_SEED_BEST_ATTEMPT`/`OPENRESEARCH_TARGET_BEST_FLOOR`/`OPENRESEARCH_BASELINE_EXTRA_GUIDANCE`/`OPENRESEARCH_MAX_RUN_GPU_USD` — the campaign INIT fail-closes on overlap with driver env). Every key must pass `run_spec_contract.run_spec_key_applies`.

- [ ] **Step 3: Verify the run-spec round-trips through the campaign INIT contract**

Run: `.venv/bin/python -m pytest -v -k "run_spec_contract or run_spec_key_applies"`
Expected: PASS — `OPENRESEARCH_CELL_ERROR_SALVAGE` is an accepted key (it is read at report time in-process, applies to the child).

- [ ] **Step 4: Commit**

```bash
git add configs/
git commit -m "GCP run-spec: arm OPENRESEARCH_CELL_ERROR_SALVAGE (verdict-only salvage for lost-but-real cloud cells)"
```

---

## Task 7: Full-suite green + the A dry-run gate

**Why:** Before spending GPU$ on the real cloud run, the whole hermetic suite must be green and the argv-parity of the VM path must hold.

- [ ] **Step 1: Run the runtime + rlm suites**

Run: `.venv/bin/python -m pytest tests/services/runtime/ tests/rlm/ tests/services/events/ -n auto -q`
Expected: PASS (0 failures). Investigate any failure before proceeding — a red suite here means a regression in the VM lifecycle / guard / routing changes.

- [ ] **Step 2: Lint**

Run: `uvx ruff@0.15.16 check backend/services/runtime/vm_compute_provider.py backend/agents/rlm/reproduction_run.py backend/agents/rlm/primitives.py backend/services/events/live_runs.py`
Expected: no errors.

- [ ] **Step 3: Full suite (socket-hermetic) sanity**

Run: `.venv/bin/python -m pytest tests/ -n auto -q`
Expected: PASS — same green baseline as session start (10,203 tests, 0 collection errors). A new failure is a regression introduced by this plan.

- [ ] **Step 4: Commit any incidental fixes, then hand off to the cloud run**

The **real** cloud run is operator-money work, gated behind the preflight cost estimate + operator checkpoint (design spec §Cost & safety) and the mandatory GCP-run journal (`docs/progress/2026-07-22-tier3-adam-progress.md`, per the `backend/services/runtime/CLAUDE.md` journaling rule). Command shape (operator-run, generous budget so ADAM finishes in one shot — the tripwire in the Scope note):

```bash
python -m backend.cli campaign 1412.6980 \
  --campaign-driver unified --sandbox local --billing-sandbox gcp \
  --max-llm-usd <X> --max-gpu-usd <Y> --max-gpu-hours <Z generous> \
  --run-spec <gcp-run-spec.json>
```

**A is done when** the collected `runs/<id>/final_report.json` shows `evidence_gate_passed: true` and `runs/<id>/experiment_runs.jsonl` has a `success=True` row, reached on `ReproductionOutcome.state == "FINALIZE"` (not `RECOVERED`).

---

## Self-Review (completed inline)

- **Spec coverage:** A-item-1 (clean run) → Tasks 1+2+7; A-item-2 (checkpoint/resume) → **explicitly deferred to B** (Scope note, with tripwire); A-item-3 (arm salvage flags) → Task 6; A-item-4 (GKE guard + doc rewrite) → Tasks 3+4. The autonomous-profile break (surfaced by the doc-hardening, flagged in recon + advisor) → Task 5, **surfaced as an operator decision, NOT implemented in A** (off A's critical path; the naive `local` default may be actively wrong). Exit-bar pinning + red-line guard → Scope note.
- **Placeholder scan:** every code step shows complete before/after code; every command has an expected result. The one intentional operator branch (Task 5 Option A/B) shows both and implements the recommended default.
- **Type/name consistency:** the new `RunStatus(state="completed", ...)` string is consumed by the matching `status.state == "completed"` branch in `reproduction_run.py`; `_run_completed_on_vm` is defined in Task 1 and called only there; the guard message substring `"not used"` matches the `pytest.raises(match="not used")` assertions in Task 3.
