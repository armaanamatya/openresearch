# Durable Controller + CPU Cloud Lane — Session Handoff (2026-07-12)

- **Branch:** `feat/gke-gpu-path-reproduction-reliability` (pushed to **deepinvent**).
- **What this session did:** made the durable GKE controller *real* (was a `NotImplementedError` stub),
  added a *CPU cloud lane* so CPU-class papers run on cloud with no laptop, added the WS1-H1 demo_status
  guard, and fixed a lease-fence correctness bug. All flag-gated **default-OFF, byte-identical off**.
- **Predecessors:** design `docs/superpowers/specs/2026-07-12-cloud-native-durable-and-cpu-lane-design.md`
  (Codex-reviewed, Revision 2); plan `docs/superpowers/plans/2026-07-12-cloud-native-durable-and-cpu-lane.md`;
  operator drill `docs/runbooks/2026-07-12-cpu-lane-and-durable-drill-operator-checklist.md`; prior wave
  `docs/runbooks/2026-07-12-phase3-ws3-ws2-implementation-handoff.md` (whose deferred items §5/§3a this finished).

## 1. Commits (this session, on top of the prior WS3 wave)
| Commit | What |
|---|---|
| `2c5f5a4e` | design spec (Codex-reviewed) |
| `c4a2fb6a` | **fence_epoch** — renew-invariant fence + `reap_stale_fence_epochs` (fixes the self-reap bug) |
| `2eac72f4` | controller Job builder + fence env-threading + durable default-for-gcp scaffolding |
| `f5b09f94` | **real durable submit** — takeover-safe ordering + heartbeat + sweeper + `ControllerHandle` |
| `c12462b9` | **CPU cloud lane** — classifier + CPU Job manifest branch + local fallback |
| `04406fd0` | **WS1-H1** — demo_status stale-republish guard |
| (docs) | operator checklist + CHANGELOG + this handoff |

## 2. State — DONE + verified
- Full suite: **19 failed / 9563 passed** — the 19 are the known pre-existing env failures
  (oauth/keychain/OCR/`TestResolveAuto`/demo-gate/repo-hygiene), **zero new**. Registry still **19** primitives;
  `test_single_verdict_authority_guard` green; `gen_flag_registry --check` clean.
- Every new path fail-soft to today's local behavior; GPU cell manifest byte-identical (golden test);
  off-state byte-identical for every flag.

## 3. How to resume (Linux server)
```bash
git clone <deepinvent> && cd <repo> && git checkout feat/gke-gpu-path-reproduction-reliability
uv venv --python 3.12 .venv && .venv/bin/pip install -r backend/requirements.txt -r backend/requirements-dev.txt
.venv/bin/python -m pytest tests/ -n auto -q            # expect the 19 known env failures only
uvx ruff@0.15.16 check .
```
"On for gcp" is a DEPLOYMENT setting (Option Y — keeps the byte-identical-OFF invariant). To activate,
set in the gcp deployment env: `OPENRESEARCH_DURABLE_CONTROLLER=1`, `OPENRESEARCH_CPU_CLOUD_CELLS=1`,
`OPENRESEARCH_CPU_POOL_LABEL=reprolab/pool=cpu`, `OPENRESEARCH_GCP_GCS_BUCKET=<bucket>` — then run the
operator checklist (CPU node pool + KSA RBAC + Pod-kill drill).

## 4. What remains
### 4a. WS-Ext — `result_fidelity` per-claim `kind` (NOT done, deliberately — verdict red line)
Every real claim has `kind == ""` → `result_fidelity._evaluate_claim` returns `missing_kind` → always
`unmeasured` → the deterministic verdict can never be `reproduced`/`contradicted`. Making claims measurable
**changes verdicts**, so it needs its own flag-gated pass + frozen-Adam A/B (the "worst error" is a false
`contradicted`). **Corrected anchors** (the prior handoff's `_normalize_claim_from_llm` does not exist):
- `backend/agents/rlm/repro_spec_extractor.py::_EXTRACTOR_SYSTEM` (line ~634) emits `proposed_value`/
  `baseline_value`/`estimate_kind` but NO test `kind`; `build_repro_spec` (line ~377) builds the `comparison`
  dict at line ~428 with NO `kind`/`baseline_value`/`proposed_value`.
- `backend/agents/rlm/result_fidelity.py::_evaluate_claim` (line ~236) already READS `claim.get("kind")` +
  `claim.get("baseline_value")` off the raw dict (via `normalize_repro_spec_claims` → `bind_claims`), so it
  measures the moment those fields are populated — **no `ComparisonSpec` change needed** (that dataclass
  ignores extras). Confirm `normalize_repro_spec_claims` preserves the new fields when lifting nested→flat.
- Relax A6a `_reconcile_with_blinded._cmp`: a one-sided `None` should not count as disagreement.
- **Design:** add `kind ∈ {numeric,relative,trend,qualitative}` to the extractor prompt; read it in
  `build_repro_spec` ONLY if it is one of the four literals (never inferred from `estimate_kind`/value
  presence — that's the false-`contradicted` trap); thread `baseline_value`/`proposed_value` into the
  comparison dict. **Flag-gate** (e.g. `OPENRESEARCH_RESULT_FIDELITY_KIND`, default-OFF → no `kind` →
  byte-identical `unmeasured`). **Acceptance:** a real non-ambiguous numeric claim measures; the frozen
  `runs/prj_adam_local_1` primary (genuinely ambiguous) STAYS `inconclusive`; NO false `contradicted`;
  the existing `test_normalize_lifts_..._real_adam` (frozen artifact) stays green.

### 4b. Operator/drill-gated (cannot be done by an agent — real GKE + money)
CPU node-pool provisioning, controller KSA RBAC, and the Pod-kill durability drill — see the operator
checklist. Also the Helm `reproduce`→`campaign` command rewrite + controller budget-meter values (prior
handoff §4).

### 4c. Design non-goals (not deferrals)
Azure/AKS lease parity (WS3 GCS-only by design); the Owner-6 GCS-mirror resume-promotion (drill-gated,
resume-skip correctness = lost work — safe only with the drill).
