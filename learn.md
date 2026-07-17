# learn.md — cross-cutting reliability rules (active)

> **Active, append-only.** This log holds cross-cutting reliability RULES in
> **Rule / How / Why** shape (per the `/iterate` discipline) — not per-bug
> postmortems. Append only: improve formatting or wording in place, never
> delete a rule. One rule per section. **Newest rule on top.**
>
> Incident *narratives* (symptom → root cause → fix, with the specific
> `prj_*` run and commit) live in per-bug memory files and
> `docs/superpowers/specs/` — this file cites them but doesn't retell them.
> The pre-2026-06 postmortem log (`Symptom → Root cause → Fix → Lesson →
> Guardrail` shape) is frozen and archived at
> [`docs/archive/learn.md`](docs/archive/learn.md).

---

## 2026-07-16 — When adding an integrity check, prove hash byte-agreement with the existing canonical artifact before shipping

**Rule.** An integrity guard that computes a digest of a shared artifact (e.g. `rubric_tree.json`) must use the **exact same serialization** as the code that already stamps the canonical hash — every JSON option (`sort_keys`, `separators`, `ensure_ascii`, `default`) must match byte-for-byte. Even an innocent `ensure_ascii=False` produces a different SHA-256 for any rubric containing a non-ASCII character (β, σ, λ…), so the guard always fires "tampered" even when nothing has changed.

**How.** W1-M1 rubric pinning (branch `feat/evidence-integrity-w1`, 2026-07-16): first draft of `rubric_fingerprint` used `ensure_ascii=False, default=str` while the campaign-level canonical hash `attempt_assessment.rubric_sha256` uses bare `json.dumps(tree, sort_keys=True, separators=(",", ":"))` (Python default `ensure_ascii=True`, no `default`). A consistency test `test_rubric_fingerprint_matches_canonical_campaign_hash` caught the divergence in RED. Fix: drop both non-default kwargs. The general pattern: find the canonical producer, extract its exact `json.dumps` call, copy it verbatim, and add a paired test that hands the same rubric object to both producers and asserts the digests are identical. Do this BEFORE wiring the guard; catch it RED, fix it, wire it GREEN.

**Why.** A false-positive integrity block (`evidence_tampered`) on an unmodified rubric halts the grading call and returns a failure before any LLM spend — it's completely silent under flag-OFF but fires on every single run once the flag is on. Because the comparison is opaque (two hex strings), the mismatch is not surfaced in logs until someone digs into `rlm_state/rubric_pin.json`. The serialization mismatch is exactly the kind of subtle correctness bug that a unit test catches in 5 minutes but a production incident catches after wasted GPU hours.

---

## 2026-07-16 — An autouse fixture that only `delenv`s named vars does NOT protect against production code that writes raw `os.environ`

**Rule.** When a test fixture isolates environment state by calling `monkeypatch.delenv(var)` for a hard-coded list of vars, it fails silently if the code under test writes to `os.environ` **directly** (not via monkeypatch). The leaked var persists into later tests in full-suite order, causing order-dependent failures that pass in isolation — the hardest class of test pollution to diagnose.

**How.** `test_campaign_composition.py` exercised a run-spec profile loader that calls `os.environ[key] = value` for every key in the profile (real production behavior). The autouse fixture only `monkeypatch.delenv`'d 3 known vars, so `OPENRESEARCH_EXTERNAL_VALIDATOR=1` leaked into the test session; the 3 tests that assert `OPENRESEARCH_EXTERNAL_VALIDATOR` is off-by-default failed only when run after `test_campaign_composition.py` in full-suite order. Fix: snapshot `os.environ` at fixture entry, `yield`, then `os.environ.clear(); os.environ.update(snapshot)` — this restores the exact pre-test state regardless of what the production code wrote. The snapshot/restore costs ~1 µs and catches every raw write without maintaining a var-list.

**Why.** The 3 failures (`test_external_validator_disabled_by_default`, `test_flag_off_panel_unavailable_with_none_client`, `test_mismatched_fingerprint_leaves_validation_empty`) passed in isolation, so CI never saw them — only a full-suite `pytest tests/` run in filesystem order exposed them. Because `monkeypatch` tracks its own mutations, not raw `os.environ` writes, any test that calls real production code doing `os.environ[...] = ...` is invisible to the cleanup machinery. Use snapshot/restore for any fixture exercising run-spec/config loaders.

---

## 2026-07-08 — A low rubric score can be an artifact of IMPOSSIBLE experiments, not a bad reproduction

**Rule.** Before reading a partial/low rubric score as "the reproduction failed," check how much of the rubric is **physically unreproducible** — leaves that require proprietary or too-large datasets (JFT-300M, Google-internal speech, ImageNet-1k, COCO) that no bounded sandbox can run. Those leaves score 0 and sit in the denominator, so a faithful reproduction of the *feasible* core reads far lower than it is. Grade the reproducible core, not the full paper, when the paper's headline experiments are out of reach.

**How.** Overnight CV run (measured 2026-07-08): KD's MNIST distillation was near-perfectly reproduced (distilled 111 < baseline 137, the core claim) yet the full rubric read **0.458** — because **13 of 24 leaves** were impossible speech/JFT experiments scored 0. Re-rolling KD's *already-graded* leaves without the impossible ones → **0.933**. ResNet 0.468→0.574, WRN 0.491→0.516 (ImageNet leaves). The graders found ~zero implementation bugs; the architectures were correct. Root cause of the disconnect: the agent KNEW the experiments were infeasible (its grader justifications say "speech_data_efficiency infeasible") but `code/metrics.json::scope.gaps` was **empty**, so the existing `_detect_data_unavailable_leaves` exclusion never fired. Fix landed: **`OPENRESEARCH_FEASIBILITY_SCOPE`** (default-OFF) — `leaf_scorer._detect_infeasible_dataset_leaves` identifies leaves that ONLY require a HARNESS-catalogued infeasible dataset (`imagenet`/`jft`/`coco`, extensible via `OPENRESEARCH_INFEASIBLE_DATASETS`) the run produced no evidence for. **FOUR integrity properties keep this HONEST measurement, not score-gaming — a first cut that lacked #2 wrongly dropped 2 ResNet ImageNet-*architecture* leaves that had scored 1.0, and an audit caught it:** (1) HARNESS-owned catalog, never agent prose — the graded party can't launder its own failures out (the 0.188 SDAR lesson); (2) **`_feasibility_roll_up_exclusions` drops a candidate ONLY if it scored 0** — impossible-experiment FAILURES leave the denominator, earned credit NEVER does (a correctly-implemented ImageNet architecture stays counted even with no ImageNet data); (3) candidates are still GRADED (not skipped), and the exclusion list + the un-excluded full-paper score are both recorded (`feasibility_excluded` / `overall_score_full`) so every drop is independently auditable and reversible; (4) mixed "ImageNet vs CIFAR" leaves stay gradeable, and a run that actually trained ImageNet keeps its leaves. The reproduction's measured evidence and per-leaf grades are untouched — only which zero-scored-impossible leaves count toward the aggregate changes. For proprietary EXPERIMENT leaves with no dataset-name token (KD speech), the complementary paths are `--scope-spec`/`--paper-hint default_scope` (operator-stated) and making the agent populate `scope.gaps` (so the existing exclusion fires). A default-ON flip needs the standard ≥3 paired-A/B + grader-σ gate. Narrative: [`docs/runbooks/2026-07-08-multipaper-gcp-overnight.md`](docs/runbooks/2026-07-08-multipaper-gcp-overnight.md).

**Why.** The evidence-not-grade red line cuts both ways: just as a green LLM grade doesn't prove reproduction, a low rubric score doesn't prove *failure* when the denominator is padded with experiments the universe won't let you run. Scoping to the feasible core is what makes the fitness signal honest — and it must stay HARNESS-controlled so it isn't a gaming vector.

---

## 2026-07-08 — The per-run GPU-$ cap is checked only at `run_experiment` return, so a long/stuck cell runs uncapped

**Rule.** `OPENRESEARCH_MAX_RUN_GPU_USD` (`RunBudget.check_run_gpu_usd`) does NOT bound a cell that never returns. It fires only when `run_experiment` completes, so a cell stuck downloading (slow dataset mirror) or doing a multi-hour multi-config train blows past the nominal cap silently — only `--max-wall-clock` actually stops it. Treat the `$10`/run GPU cap as best-effort, not a hard ceiling; watch A100 node-hours (`kubectl get nodes`) directly and kill stuck cells by hand.

**How.** Overnight 4-paper GCP run: NiN held 2 A100s ~2h purely downloading CIFAR-10/100 at ~30 KB/s (ledger `$0`, cap never checked → uncapped leak, ~$16); WRN/ResNet each ran a single cell ~4h (multi-config train / re-download) to ~$16, all past the `$10` cap. Mitigations that worked: SIGTERM the run process (writes `killed`/`failed`, preserves `runs/<id>/`) + `kubectl delete job` its cells for a pure stuck-download leak with no banked evidence; and `kubectl logs <pod> > runs_logs/…` BEFORE any kill, because a long cell writes no `experiment_runs.jsonl` success row until it returns (results live only in pod stdout until then). The SAME knob also has the OPPOSITE failure at the other end: `--max-run-gpu-usd` is ALSO a **pre-flight estimate gate** — the cell-matrix refuses to launch a cell (`failure_class=budget_exhausted`, "per-run GPU budget would be exceeded") when the cell's ESTIMATED cost exceeds it, BEFORE any GPU runs. Set it too tight and the run produces **zero experiment evidence → evidence-gate downgrade → 0.0**, which reads as a reproduction failure but is a pure config problem (2026-07-08 KD demo: `--max-run-gpu-usd 12` refused an MNIST cell whose estimate — inflated by the agent passing an invalid `compute_scope="gpu_full"` string — exceeded 12; the identical run with a comfortable cap runs fine). So the cap is a real gate at launch but NOT during a running cell. Rule of thumb: size `--max-run-gpu-usd` to `~1.25 × (gpu_count × rate × est_hours)`, never below the single-cell estimate; MNIST/CIFAR cells are cheap in absolute terms (a 1× A100 cell is $3.93/hr), so a too-low cap costs a whole run for no saving. Follow-up fix (unimplemented): enforce the GPU-$ cap on a heartbeat inside the cell runner (poll node-hours×rate mid-cell) and/or a per-cell max-runtime so a stuck download self-terminates. Companion cost sink: `torchvision` datasets download from `cave.cs.toronto.edu` at ~30 KB/s inside the pod (~$6/paper of idle A100) — pre-cache CIFAR/MNIST/SVHN into `gke-cell-base` or a GCS mirror. Full narrative: [`docs/runbooks/2026-07-08-multipaper-gcp-overnight.md`](docs/runbooks/2026-07-08-multipaper-gcp-overnight.md).

**Why.** A `$0` ledger + a cap that only checks at cell-return means a runaway cell reads as "within budget" while burning A100-hours. Same trust-the-ledger trap as the sibling `$0 ≠ $0 spent` rule below, but the mechanism is the cap's evaluation point, not Foundry-blindness.

---

## 2026-07-07 — A hyperparameter guard must key on the variable's ROLE, not an ambiguous name

**Rule.** A preflight/sanity guard that range-checks a hyperparameter (learning rate, dropout, …) must scope to names that unambiguously denote that role — never a Greek-letter/coefficient name (`alpha`, `beta`, `lambda`, `tau`, `eta`) shared with RL/regularization coefficients, where values outside the "sane LR range" (`0.0` ablation, `>1.0` weight) are legitimate.

**How.** `_check_absurd_learning_rate` listed `alpha` in `LR_NAMES`, so UCPO's faithful `alpha=0.0` sharpening ablation tripped the "outside sane LR range [1e-7,1.0]" hard block. Fix: drop `alpha`; keep the unambiguous LR names (`lr`/`learning_rate`/`base_lr`/…) and both bounds. More generally: when you widen a guard's scope (e.g. `OPENRESEARCH_PREFLIGHT_UNION_SCOPE`), re-audit its name/value heuristics against RL configs — a supervised-training assumption (loss>0, lr∈[1e-7,1], variance>0) false-fails legitimate RL / degenerate-slice values.

**Why.** `prj_618` (UCPO) blocked for a full run + repair loop on `alpha=0.0` — the executor even documented the false-positive in-code. Same family as the RL-smoke false-fail (`[[project_sdar_gcp_rl_smoke_fix]]`) and the SDAR file-scoping false-block.

---

## 2026-07-07 — `$0` in `cost_ledger.jsonl` / `demo_status.json` does NOT mean $0 spent

**Rule.** Never trust the cost ledger alone to conclude a run spent nothing.

**How.** Real LLM spend lives in `tokens_total.json`; Foundry-routed models
(`claude-opus-4-8`, `claude-sonnet-5`) are unpriced in the ledger, and GPU
node-idle time is invisible to it entirely. Verify actual spend via
`kubectl get nodes` (catches stray/idle A100s) **plus** `tokens_total.json`
— never the ledger in isolation.

**Why.** All 6 runs in the 2026-07-07 triage
(`docs/runbooks/2026-07-07-all-runs-triage-and-hardening-handoff.md`) showed
`$0` in the ledger while `tokens_total.json` showed real spend (e.g. 137K
output tokens).

---

## 2026-07-07 — A faithful implementation can fail preflight because the guard reads code MID-WRITE (harvest race)

**Rule.** Never conclude "surrogate/toy" from a preflight block — inspect the
FINAL code snapshot and file mtimes before ruling on faithfulness.

**How.** `implement_baseline` may report `ok` ~140s in while the writer keeps
landing files 12–14 minutes later; the guard then validates a stale
snapshot. Guardrail: `OPENRESEARCH_IMPL_ABANDON_GUARD` returns
`implement_timeout_abandoned` and refuses to cache `ok` on an aclose-stall
give-up.

**Why.** `prj_23f04429cd3beaf7` false-blocked a genuinely faithful SDAR
implementation this way — real `from_pretrained` via the model registry,
real GRPO, `grep nn.Linear|nn.Embedding` = 0.

---

## 2026-07-07 — The monolithic exec path (`k8s_job_backend.exec`) is NOT GKE-ready

**Rule.** On GKE, training must route through the cell-matrix
(`code/cells.json` + `train_cell.py`), or the `OPENRESEARCH_GKE_SYNTH_CELL`
synthesis that generates them — never the monolithic exec path.

**How.** The monolithic exec path runs `commands.json` via `sh -c`,
overrides the base-image GCS-download ENTRYPOINT, and never restages the
uploaded code into the pod; its smoke bootstrap also emits a
host-absolute `cd <laptop path>`.

**Why.** `prj_618445173e9ae4f2` and `prj_c912f5df415f410c` both died with
`sh: cd: can't cd to .../code`.

---

## 2026-07-07 — Reliability fixes ship default-OFF and protect NOTHING until enabled in the run-spec

**Rule.** Landing a fix behind a default-OFF flag is not the same as
shipping the fix — turning it ON in the run-spec is a required deployment
step, not a nicety.

**How.** Repo flag discipline requires new flags default-OFF and
byte-identical-off. A landed fix (`OPENRESEARCH_PREFLIGHT_UNION_SCOPE`,
`OPENRESEARCH_IMPL_ABANDON_GUARD`, `OPENRESEARCH_HARDEXIT_CLEANUP`) does
nothing until it's turned on in `configs/*run_spec.json`.

**Why.** Every run in the 2026-07-07 triage failed with the relevant fix
present in the codebase but OFF in the run-spec.

---

## 2026-07-07 — Cosmetic template defaults masquerade as agent failures; verify against artifacts, not the manifest

**Rule.** Judge "toy vs faithful" from the real `plan_reproduction` output
and the code itself — never from `demo_status.json` / `reprolab_manifest.json`.

**How.** The `ppo-cartpole-v1` / `mean_reward=475` "benchmark" block is a
hardcoded demo stub (`backend/services/events/live_runs.py`), stamped on
EVERY run and never overwritten when a run fails pre-scoring. It looks
exactly like a toy-task substitution but isn't.

**Why.** This stub produced a phantom "critical (6) UCPO→CartPole"
substitution finding and briefly misled triage; both the SDAR and UCPO
implementations were in fact faithful.

---

## 2026-07-07 — A killed driver is a durability failure, not a science failure

**Rule.** Launch long GPU runs from a durable host (or `nohup`, relying on
the primitive cache) — don't re-derive the science after a driver dies.

**How.** Run drivers die from host-suspend, OOM, or operator SIGTERM (this
WSL2 laptop host is not durable). A run left `orphaned`/`interrupted` with
`$0` is resumable via `rlm_state/primitive_cache.jsonl`.

**Why.** `prj_c912f5df415f410c` was orphaned by host-suspend/OOM;
`prj_618445173e9ae4f2` and `prj_13f7eef55bd0b55c` by SIGTERM.

---

## 2026-07-07 — Evidence-not-grade when auditing an implementation for "surrogate/toy"

**Rule.** Before ruling an implementation "fake," check the FINAL code
snapshot against deterministic markers — never a grade or a first
impression.

**How.** Check `grep nn.Linear|nn.Embedding` = 0 and a real
`AutoModelForCausalLM.from_pretrained(...)` (literal, or a grounded
variable-arg call resolved through the model registry) in the final
snapshot. Preflight scope must cover the training-file union, not just
`train.py` (`OPENRESEARCH_PREFLIGHT_UNION_SCOPE`).

**Why.** Two separate SDAR "surrogate" verdicts were overturned as faithful
once the real files were read.
