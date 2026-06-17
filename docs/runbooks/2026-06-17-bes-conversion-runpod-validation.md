# BES conversion-correctness — RunPod validation runbook (2026-06-17)

Validate the BES conversion + archival-correctness work
(branch `feat/bes-conversion-correctness`, spec
`docs/superpowers/specs/2026-06-17-bes-evidence-first-and-conversion-remediation-design.md`)
on **real GPU evidence** via `--sandbox runpod`, and run the **A1 kill-experiment**
that decides whether static LLM SELECT is signal or noise.

## What this validates (and what it does NOT)

The conversion/champion/archive/guard logic all runs in the **local orchestrator
process**; RunPod only supplies real `run_experiment` evidence (a populated
`code/metrics.json`) so the conversion path has something real to be coherent
about. So these tiers exercise the new code end-to-end with genuine GPU artifacts.

**Not in scope here:** A2 (short-slice predictiveness) and C3 (the GPU cascade) —
deferred to a follow-on plan, gated on A1's verdict. This runbook does **not** flip
any default; every new behavior stays flag-gated.

## Prerequisites

```bash
# On the orchestrator host (the box that launches the run; experiments run on the pod):
git checkout feat/bes-conversion-correctness
uv venv --python 3.12 .venv 2>/dev/null || python3.12 -m venv .venv
.venv/bin/pip install -r backend/requirements.txt

# RunPod + auth (LLM $0 via OAuth; you pay only pod GPU-hours)
export OPENRESEARCH_RUNPOD_API_KEY=...            # your RunPod key
export OPENRESEARCH_RUNPOD_SSH_KEY_PATH=~/.ssh/id_ed25519
export REPROLAB_RUNPOD_CLOUD_TYPE=COMMUNITY        # ~$0.34/hr RTX 4090 (cheapest)
export OPENRESEARCH_RUNPOD_SKIP_BUILD=1            # no wasted local docker build under runpod
unset ANTHROPIC_API_KEY                            # a no-credit key shadows OAuth -> 400
claude login                                       # subscription OAuth for sub-agents
```

Cost: each tier below uses a **cheap CNN paper** (~10–30 min on one 4090 ≈ $0.10–0.20),
not SDAR. Budget ~$1–3 total. **Stop/delete pods when idle.**

---

## Tier 0 — CPU sanity (no GPU, do first)

The 60 conversion-correctness tests must be green on the host before spending GPU:

```bash
.venv/bin/python -m pytest \
  tests/rlm/test_champion_artifact.py tests/rlm/test_binding_champion_record.py \
  tests/rlm/test_champion_coherence.py tests/rlm/test_conversion_guard.py \
  tests/rlm/test_report_provenance_repair.py tests/rlm/test_archive_completeness.py \
  tests/rlm/test_select_stability.py tests/rlm/test_staged_search_scope_gaps.py \
  tests/scripts/test_ab_compare_archive_gate.py tests/scripts/test_bes_a1_capture.py -q
```
**PASS criterion:** 60 passed.

---

## Tier 1 — Parity smoke (all new flags OFF) on real GPU

Confirm the new code does not regress a normal run. Pick a cheap paper.

```bash
PAPER=1412.6806   # All-CNN (fast CNN; or 1412.6980 Adam)
env -u ANTHROPIC_API_KEY .venv/bin/python -m backend.cli reproduce $PAPER \
  --mode rlm --sandbox runpod --model claude-oauth \
  --max-wall-clock 5400 --max-usd 5 --project-id bes_t1_parity
```
**PASS criteria** (inspect `runs/bes_t1_parity/final_report.json`):
1. Run completes with a real `rubric.overall_score` (not `null`, not a crash).
2. `baseline_metrics` is **populated** (a real `run_experiment` produced metrics) —
   i.e. the conversion guard had nothing to repair on a healthy run.
3. The advisory `rubric.evidence_cites_metrics` key is present (the only additive
   change on a non-BES run; it must not alter the score).

Coherence check helper:
```bash
.venv/bin/python - <<'PY'
import json
r = json.load(open("runs/bes_t1_parity/final_report.json"))
ru = r.get("rubric", {})
print("overall_score:", ru.get("overall_score"))
print("baseline_metrics empty?:", not r.get("baseline_metrics"))
print("evidence_cites_metrics:", ru.get("evidence_cites_metrics"))
PY
```

---

## Tier 2 — Champion coherence + conversion repair (flags ON)

```bash
env -u ANTHROPIC_API_KEY \
  OPENRESEARCH_CHAMPION_ARTIFACT=1 \
  .venv/bin/python -m backend.cli reproduce 1412.6806 \
  --mode rlm --sandbox runpod --model claude-oauth \
  --max-wall-clock 5400 --max-usd 5 --project-id bes_t2_champion
```
**PASS criteria** (`runs/bes_t2_champion/final_report.json`):
1. If `rubric.champion_restored == true`: the shipped `rubric.overall_score`
   **reconciles with `rubric.leaf_scores`** (the top-line is no longer detached
   from its leaves — the core fix). Verify:
   ```bash
   .venv/bin/python - <<'PY'
   import json
   ru = json.load(open("runs/bes_t2_champion/final_report.json"))["rubric"]
   ls = ru.get("leaf_scores") or []
   print("champion_restored:", ru.get("champion_restored"),
         "sample_count:", ru.get("champion_sample_count"))
   # leaf_scores is a list of records; spot-check the score field exists & is non-stale
   print("n_leaves:", len(ls), "overall:", ru.get("overall_score"))
   PY
   ```
2. `rlm_state/champions.json` entries carry `sample_count` (1 unless
   `OPENRESEARCH_GRADER_SAMPLES>=3`), and `rlm_state/champions/<key>/rubric_block.json`
   exists (the snapshotted graded block).
3. **Provenance repair fires only when needed:** if any run produced a populated
   `code/metrics.json` but the report would have shipped empty `baseline_metrics`,
   the report now carries `provenance_repaired: true` with `baseline_metrics`
   populated. On a healthy run this key is absent (no-op).

---

## Tier 3 — A1 kill-experiment (the decisive one; zero GPU for the regrade, bounded LLM)

Decide whether static LLM SELECT is signal or noise on a **fresh** (first-attempt)
paper. The candidate *capture* runs on the pod (cheap, code-only static grade — no
training); the *re-grade* loop is CPU/LLM only.

```bash
# 1) Capture N>=3 competing candidates on a fresh paper (forces a first-attempt lineage
#    via a project-id suffix so adaptive gating engages the pool).
env -u ANTHROPIC_API_KEY \
  OPENRESEARCH_BES_ENABLED=1 OPENRESEARCH_BES_CANDIDATES_PER_CLUSTER=3 \
  OPENRESEARCH_BES_ADAPTIVE=0 \
  .venv/bin/python -m backend.cli reproduce 1412.6980 \
  --mode rlm --sandbox runpod --model claude-oauth \
  --max-wall-clock 5400 --project-id bes_a1_capture --project-id-suffix a1

# 2) The candidate snapshots land in runs/bes_a1_capture*/candidates/rlm_impl_*/ ;
#    re-grade each K=10 times at temperature=0 (no GPU) and assemble `regrades`
#    (list of {candidate_id: score}) via scripts/calibrate_grader.py, then:
.venv/bin/python - <<'PY'
from scripts.bes_a1_capture import summarize_regrades
# regrades: fill from the K re-grade outputs, e.g.
# regrades = [{"rlm_impl#0": 0.55, "rlm_impl#1": 0.56, "rlm_impl#2": 0.54}, ... x10]
regrades = [...]   # <-- assemble from calibrate_grader output
print(summarize_regrades(regrades, repeatability_sigma=0.02))
PY
```
**Verdict interpretation:**
- `verdict == "select_is_noise"` (top-1 flips across re-grades at margins ≤ σ) →
  **static LLM SELECT is noise; the selection-based BES line is falsified.** Keep
  only the binary runnable/not smoke gate. Do NOT proceed to A2/C3.
- `verdict == "select_stable"` → SELECT has signal; A2 (budget-matched short-slice
  predictiveness) becomes worth designing.

Wire the K-regrade loop per the `NOTE` in `scripts/bes_a1_capture.py::main` (it is an
operator-gated stub by design). Archive the capture (candidate snapshots +
`bes_candidates.json` + `dashboard_events.jsonl` + `rubric_evaluation.json` +
`final_report.json` + `metrics.json` + `generated_rubric.json`) — **no complete
archive, no efficacy claim** (the gate in Tier 4 enforces this).

---

## Tier 4 — Stamped A/B + archival gate (optional)

Only if Tier 3 says `select_stable`. Run a stamped BES arm and a control arm on the
same paper, then confirm the archival gate accepts complete archives and refuses
incomplete ones:

```bash
# BES arm
env -u ANTHROPIC_API_KEY OPENRESEARCH_BES_ENABLED=1 OPENRESEARCH_BES_CANDIDATES_PER_CLUSTER=3 \
  OPENRESEARCH_AB_ARM=bes OPENRESEARCH_AB_PAIR_ID=allcnn-conv-val \
  .venv/bin/python -m backend.cli reproduce 1412.6806 --sandbox runpod --model claude-oauth \
  --project-id bes_t4_bes --project-id-suffix bes
# control arm
env -u ANTHROPIC_API_KEY OPENRESEARCH_AB_ARM=control OPENRESEARCH_AB_PAIR_ID=allcnn-conv-val \
  .venv/bin/python -m backend.cli reproduce 1412.6806 --sandbox runpod --model claude-oauth \
  --project-id bes_t4_control --project-id-suffix control
# gated compare (refuses on an incomplete arm archive)
OPENRESEARCH_REQUIRE_STAMPED_AB=1 .venv/bin/python scripts/ab_compare.py --pair-id allcnn-conv-val
```
**PASS criteria:** with complete archives the compare emits `runs/_ab/<key>/ab_report.{md,json}`;
delete one required artifact from an arm and re-run → it refuses with the missing-artifact list.

---

## Cleanup

```bash
# pods auto-terminate on run end, but confirm none are left running:
.venv/bin/python -m backend.cli  # (or the RunPod console) — stop/delete idle pods
```

## Success summary

| Tier | Proves |
|---|---|
| 0 | conversion-correctness unit suite green |
| 1 | no regression on a normal GPU run; provenance coherent on healthy runs |
| 2 | champion-restored reports ship coherent leaves; `sample_count` recorded; repair fires only when needed |
| 3 | **the decisive A1 verdict** — is static SELECT noise? |
| 4 | archival gate accepts complete / refuses incomplete A/B archives |

If Tier 3 returns `select_is_noise`, the evidence-first thesis is confirmed and the
BES selection line stops here (smoke-gate only). If `select_stable`, open the A2/C3
follow-on plan.
