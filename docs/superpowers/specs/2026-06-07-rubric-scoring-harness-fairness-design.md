# Rubric scoring + harness fairness redesign — design spec

**Status:** PARTIALLY IMPLEMENTED (2026-06-07). Core scoring path landed + tested on branch `5.30.26_sdar` (uncommitted); validating end-to-end on the Adam paper. See §12.

## 12. Implementation status (2026-06-07)

Landed + tested (~700 tests green across touched areas, 0 regressions):
- **D1** anchored grader scale — `leaf_scorer.py::_SYSTEM_PROMPT` rewritten (anchored 1.0/0.7/0.4/0.0, default flips to "met→1.0, deduct only named gaps"), **default-on** (rail green = the spec's flip criterion; git is rollback) keeping the adversarial anti-narrative + evidence-citation floor (`tests/test_p3_grader_stance.py` green). `deductions[]` field added + parsed.
- **D5** result-match = trend/direction within tolerance, never exact magnitude — routed on the new `task_category` payload field.
- **D2a** `backend/agents/rlm/provenance.py` (emit_provenance / emit_figure_sidecar / assert_provenance, stdlib-only, series-summarized) + registered in `_HARNESS_CODE_HELPERS` + `_PROVENANCE_BLOCK` prompt. `tests/agents/rlm/test_provenance.py` (12).
- **D2c** `_gather_evidence` consumes `provenance.json` + `fig_*.json` sidecars; caps 6 KB/40 KB → 32 KB/200 KB. `tests/evals/paperbench/test_evidence_manifest.py` (5).
- **D4 (plumbing)** top-level `final_report.json::overall_score`/`meets_target` now mirrored from the rubric (the `None`-while-0.831 bug).
- **D7a/b** `tests/evals/paperbench/test_discrimination_rail.py` — 4 golden fixtures + LLM-free dual-run kill criterion (fake ≤0.4 **and** fake doesn't rise **and** spread widens); observed faithful 0.60→0.97, fake 0.00→0.00.
- **D6a (foundation only)** `backend/agents/rlm/env_pin.py` (CORE_DENYLIST + COMPAT_MATRIX + harden_requirements + pin_install_specs) + `tests/agents/rlm/test_env_pin.py` (7). **NOT yet wired** into the live requirements path (wiring threads `sandbox_mode` through several call sites — focused pass; will be gated `OPENRESEARCH_HERMETIC_ENV` default-OFF).

Deferred (focused passes after the Adam core validates): D6a wiring + D6b unified preflight gate + D6c new failure classes; D3 plan/code critics; D7c refute pass; D4 eval-model/Opus knob; D2b cell-path promotion; D2 vision path; the `grader_version` field (§9).

---

**Original design status:** DESIGN (grilled, grounded). Codex-reviewed 2026-06-07 (two "blockers" were hallucinations — verified void; 4 conceptual findings incorporated). Awaiting human approval for the deferred work + a commit.
**Date:** 2026-06-07
**Branch target:** 5.30.26_sdar (sibling to `2026-06-07-bes-integration/`)
**Author provenance:** Grilled one-decision-at-a-time with the operator (9 locked decisions), then grounded by 5 parallel read-only agents against the working tree. Every `file:line` below was verified by a grounding agent; Codex review must re-confirm the load-bearing ones.

---

## 0. Why

A *faithful* reproduction of the Adam paper (arXiv 1412.6980) scored **0.848** on the PaperBench-style rubric this session. The operator's question — *"why isn't it ~100, and how do we consistently score higher without it being near-impossible?"* — decomposed, under grilling, into a precise diagnosis. The 15 missing points are **three different problems**, and only the third is "the reproduction was incomplete":

| Bucket | What it is | Share of weak leaves | Fault |
|---|---|---|---|
| **1. Evidence-visibility** | The grader is shown a truncated, figure-blind, text-only view of a run that *did* the work | ~9 of 13 | Harness **+** agent |
| **2. Result-match strictness** | Docked for exact-magnitude / exact-ordering on near-tied, physically-unreproducible numbers | ~1–2 | Grader/rubric |
| **3. Real method gaps** | Agent genuinely skipped work (true ZCA whitening, the VAE early-instability measurement, momentum grid) | ~3 | Agent |

Two structural facts confirmed in code make Bucket 1 the dominant lever:

- **The grader is text-only and truncates code.** `LlmClient.complete(*, system, user) -> str` (`backend/evals/paperbench/leaf_scorer.py:47-49`) cannot accept images; evidence is capped at **6 KB/file, 40 KB total** (`leaf_scorer.py:190-191`). The run produced **9 PNGs** the judge never saw, and `models.py`/`optimizers.py` were truncated — hence *"log-scale axis not verifiable," "full code is truncated."*
- **The agent's figures + provenance never reach the level the grader reads.** Per-cell figures, `training_curves.json`, `config_used.json` are written into per-cell subdirs `outputs/<run_id>/<cell>/`; the **aggregated** `outputs/<run_id>/` holds only `metrics.json`. Both `_gather_evidence` and `rubric_contract._check_required_artifacts` read the aggregated level and find nothing — hence *"45-epoch not confirmed," "batch=128 only an assumption."*

And the grader's posture guarantees it withholds the top even with perfect evidence: `_SYSTEM_PROMPT` says *"strict… Be conservative: score 0.0 when there is no evidence either way"* (`leaf_scorer.py:739-756`) — a **default-to-zero** judge with no anchored partial-credit scale.

**Goal:** a complete, faithful reproduction should reach **~0.95**, while a surrogate/fake/degraded run still scores **low** — i.e. *raise the ceiling (calibration + legibility) and the floor (robustness), and prove the spread between faithful and fake widens, not narrows.*

### Design principles (the "robust elegant" bar)
1. **Every new seam is fail-soft.** A missing manifest, a critic exception, a preflight hiccup → degrade to today's behavior, never a new way to die. No new hard-fail on light papers.
2. **Reuse existing patterns, don't invent machinery.** `rubric_guard.RubricGuardFailure` for enforcement, `forced_iteration` for loop-bounding + wall-clock bypass, `preflight_ast._check_env_interface_contract`'s self-scoping for "only fire when the paper needs it," the `contract_violations` channel for repair feedback.
3. **All new behavior flag-gated, default-OFF** (mirrors the BES integration convention). Flag-off ⇒ byte-identical to today.
4. **The deterministic discrimination floor is untouchable.** The hard/soft invariant gates (`INVARIANT_HARD_CAP=0.0`, `INVARIANT_SOFT_CAP=0.5`) and `DEGRADED_LEAF_CEILING=0.35` run *after* grading and are never bypassed by any calibration change.

---

## 1. Locked decisions (the grill output)

| # | Decision | One-line |
|---|---|---|
| **D1** | Anchored grader scale + flip default | Replace free-form "default 0, be conservative" with anchors (1.0/0.7/0.4/0.0); faithful repro **starts at 1.0**, loses points only for **named, cited** deductions. Keep the invariant gates. |
| **D2** | Evidence legibility (manifest + vision) | Agent emits `provenance.json` + per-figure JSON sidecars; harness **promotes** them to the aggregated level + un-truncates code; grader gains an optional **vision** path for figure leaves. |
| **D3** | Pre-GPU adversarial critic (plan + code) | A plan-stage critic (conceptual coverage) and a code-stage critic (emission coverage) block the GPU run until rubric-leaf + evidence gaps close. |
| **D4** | Eval-provider/model knob | One knob `oauth-sonnet` (default) \| `anthropic-opus` \| `openai-gpt-5` for grader+critics+refute; cross-provider independence option; **fix the silent `model=None` grader bug**. |
| **D5** | Result-match = claim-direction + tolerance | Grade whether the repro supports the paper's **trend/direction within tolerance**; a **contradiction** still fails. Exact magnitude is **never** required. |
| **D6** | Robustness floor | Hermetic version-pinned env + forbid agent breaking core packages; ONE ordered pre-GPU preflight gate; strengthened reactive repair (defense-in-depth). |
| **D7** | Discrimination rail | Golden-set fixtures + dual-run regression with a **kill criterion** (fakes ≤0.4, faithful rises, spread widens) + an adversarial refute pass. |
| **D8** | Rollout | Flag-gated, staged; raise `OPENRESEARCH_TARGET_SCORE` so the agent climbs toward ~0.9 not 0.6; gains require a re-run (no retroactive inflation). |

---

## 2. D1 — Anchored grader scale (calibration)

**Seam:** `_SYSTEM_PROMPT` (`leaf_scorer.py:739-756`), `_USER_TEMPLATE` (`758-768`), per-leaf payload assembly (`1196-1200`), parse/clamp (`_parse_batch_response`, `1316-1351`).

**Change (prompt-only; no signature change):**
- Rewrite the `_SYSTEM_PROMPT` tail to an explicit anchor ladder:
  - **1.0** — requirement met and the evidence confirms it.
  - **0.7** — met, but **one named** non-critical detail differs or is unconfirmed.
  - **0.4** — attempted but a core part is missing or wrong.
  - **0.0** — absent, or the evidence contradicts the requirement.
- Flip the default posture: *"Assume a faithful reproduction. START at 1.0 and deduct only for SPECIFIC, NAMED gaps you can cite from the evidence."*
- Add a required per-leaf JSON field `"deductions": [str]` — a non-1.0 score **must** cite at least one named deduction. Keep the existing `max(0.0, min(1.0, …))` clamp (`1336`).
- Add `task_category` (and `finegrained_task_category` when present) to the per-leaf payload at `1196-1200` so the grader can route D5 (result-match) and the D2 vision path. *(Today only `{leaf_id, requirements}` is sent — the grader can't tell a result-match leaf from a figure leaf.)*

**Fail-soft / floor:** the degraded branch (`1126-1168`) zeroes leaves in **code** before roll-up — D1 is a *prompt* instruction only and cannot lift a degraded run. `_apply_invariant_gate` (`1289`) runs after grading — D1 cannot lift a surrogate. Both invariants preserved.

**Risk:** D1 raises leaf scores on a **fresh grade**, so realizing gains requires a re-run; A/B across the prompt change is apples-to-oranges (mitigated by D7's recorded-grade dual-run). `finalize_rescore` re-rolls *persisted* leaves (no re-grade), so **existing scored runs are not retroactively inflated** — confirmed at `leaf_scorer.py:934-1037`; the BLOCKER-2 bail at `1023` still holds.

---

## 3. D2 — Evidence legibility (manifest + vision)

This is the dominant lever and has **three** parts: agent-emit, harness-promote, grader-consume. **The promotion step is the actual bug fix** — without it the manifest is invisible regardless of what the agent emits.

### 3a. Agent-emit — new copyable helper `backend/agents/rlm/provenance.py`
Stdlib-only, zero-dep; added to `_HARNESS_CODE_HELPERS` (`baseline_implementation.py:59`), auto-copied by `_copy_harness_helpers_to_code_root` (called `baseline_implementation.py:2357`). Mirrors the `gpu_cell_runner.py`/`rubric_guard.py` copyable-helper pattern.

```
emit_provenance(output_dir, *, experiments: dict) -> Path        # writes provenance.json
emit_figure_sidecar(png_path, *, shows, axis, series) -> Path     # writes <png>.json next to each PNG
assert_provenance(output_dir, *, require_series=False) -> None    # raises rubric_guard.RubricGuardFailure
```

`provenance.json` schema: `{schema_version, run_id, generated_at, experiments:{<exp_id>:{model_key, env, baseline, seed, epochs, steps, batch_size, per_optimizer:{<name>:{lr, betas?, weight_decay?}}, hardware, framework_versions, convergence:{iteration:[…], <metric>:[…]}}}, figures:[{path, shows, axis:{x:{label,scale}, y:{label,scale}}, series_ref}]}`. Figure sidecar: `{shows, axis, series}`. The sidecar's `axis.scale:"log"` is exactly what answers *"log-scale axis not verifiable."*

**Instruction surface (two, mirroring `rubric_guard`):** a `_PROVENANCE_BLOCK` prompt string appended near `_ARTIFACT_COMPLETENESS_BLOCK` (`baseline_implementation.py:~2157`), plus an optional end-of-`train_cell.py` `assert_provenance(...)` call. **`require_series` defaults False** and is flipped on only by a paper-level signal (the paper YAML already declares curve requirements for SDAR — `docs/papers/2605.15155.yaml:97`). Self-scoping like `preflight_ast.py:895-897`: no series requirement on a light paper ⇒ best-effort emit, never hard-fail. Reuse `rubric_guard.RubricGuardFailure` so a missing manifest rides the **same** repair_context channel — do not invent a new exception.

### 3b. Harness-promote — the bug fix
At aggregate time, promote per-cell sidecars/figures to the aggregated `outputs/<run_id>/` level **and** synthesize a top-level `provenance.json` from the per-cell leaves. Seam: `cell_matrix.aggregate_cell_metrics` (`cell_matrix.py:445-570`, it already folds per-cell leaves at `511-543`) writing an extra `provenance` block, and/or `_persist_metrics` (`primitives.py:3761`) copying `<cell>/fig_*.png`+sidecars up namespaced. **Cell-path only**; the legacy monolithic path already writes a flat `outputs/<run_id>/` so `emit_provenance(output_dir=outputs/<run_id>)` lands where the grader reads with no promotion.

### 3c. Grader-consume — `_gather_evidence` (`leaf_scorer.py:238-312`)
- Raise caps: `_MAX_FILE_BYTES` 6 KB→**32 KB**, `_MAX_TOTAL_EVIDENCE_BYTES` 40 KB→**~200 KB** (`190-191`). Safe on Sonnet/Opus; keep a provider-aware ceiling for `openai-gpt-5` eval (tighter window).
- Add two labelled sections under their **own byte budget** (so the code-file loop can't starve them): `=== provenance.json ===` and `=== figure <name> (sidecar) ===`. New helpers `_gather_provenance(run_dir)`, `_gather_figure_sidecars(run_dir)`. **The manifest must carry a series *summary* (first/last/min/max/len), not the full array**, or a long convergence series blows the cap. Fail-soft: missing manifest ⇒ skip, never error.
- **Vision path (D2c):** extend the `LlmClient` protocol with an **optional** `complete_with_images(*, system, user, images: list[tuple[str, bytes]]) -> str` (default → text-only `complete`). Implement on `ClaudeLlmClient` reusing the base64 image block at `backend/services/ingestion/parser/vision.py:71-95`. `_grade_batch` (`1196-1207`) checks `hasattr(client, "complete_with_images")` **and** whether the batch contains figure-`task_category` leaves; otherwise the string path is unchanged. Non-vision providers (`OpenAILlmClient`) unaffected. Skip leaves already in `unavailable_ids`.

**Cross-component:** the code-stage critic (D3) checks for the **same** filenames D2 emits — co-version the contract. Manifest schema drift must fail-soft to the legacy path (`primitives.py:4099-4101`), never crash.

---

## 4. D3 — Pre-GPU adversarial critic (plan + code)

**Key structural finding:** in **RLM mode the root model drives ordering via REPL Python — there is no controller boundary to wrap.** The gate must be enforced **inside the primitives**. (In RDR mode there is a real linear controller.) **No critic exists today** — D3 is greenfield.

**New module `backend/agents/rlm/pre_gpu_critic.py`** (stdlib + `ctx.llm_client`):
```
critique_plan(plan, rubric_leaves, paper_text, *, ctx) -> CritiqueResult   # conceptual gaps
critique_code(code_dir, rubric_leaves, *, ctx) -> CritiqueResult            # emission gaps
```
Gap schema: `{"gaps":[{"leaf_id, kind:"conceptual"|"emission", detail, fix}], "blocking": bool}` — shaped to drop into the existing `contract_violations` list.

**RLM seams (enforce in-primitive, return `repairable`, never hard-stop):**
- **Plan critic** at the top of `implement_baseline` (`primitives.py:1345`), after `repair_context = plan.get("repair_context")` (`~1420`), before SDK dispatch. Blocking ⇒ `_baseline_error_envelope(..., repairable=True)` (`150`/`1596`).
- **Code critic** at the top of `run_experiment` (`primitives.py:3875`), after the `cells.json`/`commands.json` presence check (`3951-3959`), before `_execute_in_sandbox` — mirroring the existing `contract_guard` precedent (`3927-3949`) which already returns `repairable` without touching the GPU. Blocking ⇒ `_with_outcome({…, "contract_violations": gaps, "failure_class":"code_critic"}, PrimitiveOutcome.repairable)`.
- Both emit a `run_warning` SSE (`build_run_warning_event`, `sse_bridge.py:692`; pattern at `primitives.py:1117/1403/1538`).

**Rubric source (critical):** `rubric_tree.json` is written only *after* the first `verify_against_rubric` (`primitives.py:4775`) — **absent on the first pre-GPU pass.** Source leaves from `generated_rubric.json` (written pre-loop, `run.py:1522`) or `bundle.rubric()`, flattened via the shared `flatten_leaves` (`leaf_scorer.py:57`). **Add a `ctx.rubric_tree` field** populated in `run.py` (~`1521`) — serves both modes and avoids a re-walk.

**Loop-bounding (mandatory — mirror `forced_iteration`):** a per-run `_critic_iter_count` capped by `OPENRESEARCH_MAX_CRITIC_ITERATIONS` (mirror `_MAX_REFUSALS_PER_RUN=16`), and a **wall-clock bypass** when `ctx.remaining_s() <= 60` (`forced_iteration.py:57,189`). After the cap → log + proceed (fail-soft). **Extend the `run.py:706` repair-floor wrapper** to also call `policy.record_repair_attempt` on a `repairable` `implement_baseline` (today it only watches `run_experiment`) — else nothing stops the root skipping straight to `FINAL_VAR` after a never-improved plan.

**RDR seams:** plan critic after `decompose(rubric, …)` (`controller.py:939`); code critic **extends the existing pre-GPU gate** `_run_rdr_preflight_gate` (`controller.py:727-786`, invoked `1126-1133`) — add the LLM emission-gap pass beside the AST scan, reuse the `rdr_preflight_blocked` event + cluster-regen loop (`max_regens` is the existing bound).

**Flags:** `OPENRESEARCH_PRE_GPU_CRITIC` (master, default-OFF) + `OPENRESEARCH_PLAN_CRITIC`/`OPENRESEARCH_CODE_CRITIC` sub-flags.

**Cost:** 2 extra text calls per pass, bounded by the iteration cap, cheap vs a GPU run (and they *prevent* GPU re-runs). Note: `LlmClient` returns no usage, so these are invisible to `cost_ledger.jsonl` (acceptable; note it).

**Anti-gaming (Codex review 2026-06-07) — load-bearing.** D3 makes the rubric leaves visible to the agent *before* the run. That is the *point* (this is a declared score-maximizer), but it opens a gaming surface: the root could write code that *emits a `metrics.json` echoing the rubric's expected keys with plausible numbers* without running real training. D3 alone does not detect this. Mitigation is **mandatory and already partly present**: (1) the deterministic surrogate/invariant gate (`paper_hints.py` `real_qwen_weights_not_surrogate`, hard-cap → 0.0) checks for real `from_pretrained('Qwen/…')` + the algorithm invariants — a metric-echoing script trips it; (2) the **D2 provenance manifest must carry execution evidence** (real weights/data fingerprints, convergence series, hardware) that a faked `metrics.json` cannot fabricate cheaply; (3) the **D7c refute pass** is the backstop on any high leaf whose metrics look hardcoded. **Therefore D3 MUST NOT ship before D2's provenance + the invariant gate are in force** — D3-on with D2-off is a strictly worse anti-gaming posture. Encode this as a flag precondition: enabling `OPENRESEARCH_PRE_GPU_CRITIC` warns (or refuses) if `OPENRESEARCH_EVIDENCE_MANIFEST` is off. A dedicated provenance-scored leaf (grades "execution happened" independently of metric *values*) is the stronger long-term fix — deferred, tracked in §10.

**Wall-clock escape (document):** near timeout, the forced-iteration policy bypasses (`ctx.remaining_s() <= 60`), so a slow run can `FINAL_VAR` past an open D3 gap. This is the intended wall-clock-wins precedence — D3 enforcement is **best-effort near timeout**, not a hard guarantee. Acceptable because a timed-out run ships a *partial* report, which the rubric already scores conservatively.

---

## 5. D4 — Eval-provider/model knob (+ a real bug fix)

**Bug found:** the grader is `ClaudeLlmClient()` with **`model=None`** (`run.py:223-225` → `rlm_query.py:610-612`) — it silently rides whatever the local `claude` CLI defaults to, **not** the `claude-sonnet-4-6` declared in `models.py:196`. Fix while here.

**Change:**
- CLI `--eval-model {oauth-sonnet|anthropic-opus|openai-gpt-5}` default `oauth-sonnet` (repurpose the **dead** `--verification-provider`, `cli.py:1633`/`637`). Plumb as `eval_model` on `run_pipeline_rlm` (mirror `provider` at `cli.py:936/1462`).
- In `run.py` after `_build_llm_client` (`~1311`): `eval_client = llm_client` by default; if `eval_model` set, `eval_client, _ = _build_llm_client(provider, resolve_root_model(eval_model))`. Carry as a new `RunContext.eval_llm_client` (default None). `verify_against_rubric` uses `getattr(ctx, "eval_llm_client", None) or ctx.llm_client` (`primitives.py:4750`).
- **Pin the grader model:** pass `ClaudeLlmClient(model="claude-sonnet-4-6")` (or the opus slug) so it stops inheriting the CLI default. The accelerator-grader-offload (`run.py:172-182/1340-1342`) is the existing precedent for swapping the grader model — mirror, then supersede.
- **Cross-provider independence:** when the root is Claude, allow the eval to be `openai-gpt-5` (independence — no shared blind spots) or `anthropic-opus` (max calibration). The knob carries both; operator picks per goal.

The same `eval_llm_client` feeds the D3 critics and the D7 refute pass (refute is a yes/no check — well-suited to a cheaper model).

---

## 6. D5 — Result-match = claim-direction + tolerance

**Operator directive:** *"we don't need the exact magnitudes."* Grade the **trend/direction within a tolerance band**, never exact equality.

**Change (prompt + routing; no roll-up change):**
- Route on `task_category` (now in the payload per D1). For leaves whose category starts `"Result match"` (5 in the SDAR rubric), the system prompt instructs: *full credit if the reproduction supports the paper's claimed direction/ordering within tolerance; a **contradiction** (claimed direction inverts — e.g. SDAR < GRPO when the paper claims >) scores 0.4/0.0 (real fail).* Exact magnitude is irrelevant.
- Surface `metrics.json::comparison`/`per_model` deltas into the evidence so the grader can judge "supports trend?" with data. The signal half-exists: `MetricDelta.direction` (`schemas.py:682`), `MetricFloor.direction`/`relative_error_vs_paper` (`schemas.py:276,685`).

**Risk / scope note:** leaves carry only free-text `task_category` + `requirements` — **no structured claim-direction/metric-key/tolerance field** (`finegrained_task_category` is `None` on every leaf). So direction-grading is **inferred by the grader from prose + the comparison deltas in evidence**, not a structured contract. A stronger version (optional per-leaf `claim_direction`/`metric_key`/`tolerance` in the rubric schema + generator) is **deferred** — it touches the rubric producer (`rubric_gen.py`) and is a bigger change. The prose+category path is grading-only and ships first.

---

## 7. D6 — Robustness floor (the consistency lever)

A degraded run caps every leaf at **0.35**; a dead run scores **0**. This session's first Adam experiment **failed on `torch_redundancy`** (agent reinstalled torch). Defense-in-depth: **prevent at a gate (D6a+b)** and **survive what slips through (D6c)**.

### D6a — Hermetic pinned env + forbid core-package reinstall
**Today's gaps:** the torch-redundancy strip/guard is **runpod-only** (`requirements_derive._strip_preinstalled:253-270` gated on `_is_runpod_pytorch_base`; `pre_flight_validator._check_requirements_torch_redundancy:1186-1238` gated on `base_image.startswith("runpod/pytorch")`). Local relies on a single best-effort torch-index pin (`primitives.py:2818-2825`) the agent can still clobber. **No compatibility matrix exists.**

**New `backend/agents/rlm/env_pin.py`** (data + one function; refresh-quarterly like `gpu_catalog.py`):
- `CORE_DENYLIST = {"torch","torchvision","torchaudio","numpy","triton", …}`; `COMPAT_MATRIX: {base_tag -> PinSet}` of proven-compatible exact pins.
- `harden_requirements(req_path, *, base_tag, allow_override) -> list[str]` — neutralizes core-package lines that **conflict** with the active PinSet (not a blanket strip), returns dropped specs for the repair surface. Run it right before each `pip install -r` (`primitives.py:2762`/`2827`); install the full PinSet (torch+vision+numpy) before agent deps. Generalize the strip + torch-redundancy checks off `env_pin` for **local + docker + runpod**.

**Escape hatch (mandatory):** `OPENRESEARCH_ALLOW_CORE_PIN_OVERRIDE` + a per-paper YAML key (same dispatcher as `_check_paper_invariants`, `pre_flight_validator.py:1333`) flips `harden_requirements(allow_override=True)` to warn-and-honor. Default deny, opt-in allow — some papers need a specific torch/CUDA (flash-attn ABI, custom ops).

### D6b — ONE ordered pre-GPU preflight gate
**Today the checks are split across THREE phases and the import smoke runs *after* the pod is up (`primitives.py:3036`), i.e. not actually pre-GPU on cloud.** New `backend/agents/rlm/preflight_gate.py::run_preflight_gate(ctx, code_path, *, base_image, caps) -> PreflightResult`, called once in `run_experiment` after the Dockerfile rebuild and **before** any cell-route/escalation branch (replace `primitives.py:4009-4036`, hoist the import smoke + capacity/dataset checks up). Ordered cheapest/most-decisive first:
1. **shell-vs-.env precedence warn** — lift `_warn_on_shell_env_override` (`cli.py:78-105`) into a shared util so **batch + UI paths** get it too (today CLI-only; warn-only, never blocks). *(BUG-LR-014 is built for CLI but absent from `batch_reproduce.py` + the `/runs` spawn path.)*
2. **version-compat** (new, D6a) — assert requirements vs PinSet; auto-harden + record drops.
3. **env-contract + correctness AST** — `validate_code_pre_flight` (already wraps `scan_code_dir`/`_check_env_interface_contract` + torch_redundancy); hard → block.
4. **import smoke** — run **here on CPU** (`CUDA_VISIBLE_DEVICES=""`), genuinely pre-GPU. *(Flip `OPENRESEARCH_PREFLIGHT_SMOKE` to default-ON inside the gate.)*
5. **capacity_gate + dataset_url_preflight** — call for **all** GPU backends (today cell-route only, inside `_execute_cell_matrix`, `primitives.py:3787-3789`); the cell route then consumes the already-gated set rather than re-gating.

### D6c — Strengthen reactive repair
New `FAILURE_CLASSES` (`failure_classifier.py:31-59`): `version_conflict` ("ResolutionImpossible"), `cuda_driver_mismatch` ("no kernel image"/"driver too old"), `core_pkg_reinstalled` (from `commands.json` + logs), `env_construct_error` (an `*Env` ABC construction escapee). Each gets a detector branch (mirror `239-280`) + a `_suggest` entry. **Harness-side auto-repair:** for `version_conflict`/`core_pkg_reinstalled`, strip the offending pin and re-run via `harden_requirements` **before** consuming an agent iteration; extend patch-mode (`primitives.py:1437`) to trigger on these classes.

**Risks:** the gate must **return a `success:False`/`pre_flight_blocked` repairable result, never raise** (preserve the die-without-a-report guard); version-compat (step 2) must precede import smoke (step 4); moving `describe_capacity` (`primitives.py:4089`) up must keep its fail-soft "unknown capacity ⇒ don't block." Pinning's escape hatch covers the genuine-different-torch paper.

---

## 8. D7 — Discrimination rail (the kill criterion)

D1 flips the default to 1.0 — inflation-prone. D7 is the mandatory counterweight. **No refute/adversarial code exists today.** The recorded-grade replay mechanism already exists (`finalize_rescore`, `leaf_scorer.py:934`, reads `rubric_evaluation.json::leaf_scores` and re-rolls via `roll_up` with **zero LLM calls**) — that is the template for an LLM-free regression.

### D7a — Golden-set fixtures
Under a new `tests/evals/paperbench/__fixtures__/golden/<class>/` (mirrors `tests/agents/rlm/__fixtures__/`). Each class = a minimal run-dir (`code/train.py` + `final_report.json` + `code/metrics.json`) so `score_reproduction` runs with a **stub** LlmClient — no live grader. Seeds:

| Class | Built from | Expected band |
|---|---|---|
| (a) **surrogate** | `code/train.py` with `class TinyLM` + `# surrogate model` | ≤0.4 → **0.0** (hard gate) |
| (b) **degraded** | faithful code, `baseline_metrics={}` | ≤0.35 |
| (c) **faithful** | `from_pretrained('Qwen/…')` + `sigmoid(beta*Δ).detach()` + `λ=0.1, β=10`, both losses (copy invariant-gate test `L606-624`); seed from `best_runs/adam/` (0.7413) | rises to **~0.95** under D1 |
| (d) **sabotaged-faithful** | faithful but `.detach()`/`from_pretrained` removed | drops (soft gate → 0.5) |

*(Caveat: `best_runs/adam/rubric_evaluation.json` on disk is a degraded snapshot — author the faithful per-leaf grades, don't copy verbatim. `prj_6d41d2f09c026403` is NOT a usable faithful fixture: `overall_score=None`, `baseline_metrics={}`.)*

### D7b — Dual-run regression (kill criterion)
A committed pytest, **LLM-free for CI**: two stub clients `_OldAnchorLlm` (floor-anchored grades) and `_NewAnchorLlm` (anchored-to-1.0 grades) driven through `score_reproduction` over the 4 fixtures. **Ship only if ALL hold:**
```
fake_new   ≤ 0.4                                    # absolute fake ceiling
fake_new   ≤ fake_old + ε         (ε = 0.05)        # fakes must NOT rise (anti-gaming)
faithful_new > faithful_old                         # faithful rises
(faithful_new − fake_new) > (faithful_old − fake_old)   # spread WIDENS
```
The **`fake_new ≤ fake_old + ε`** clause is load-bearing (Codex review 2026-06-07): "spread widens" *alone* is gameable — both scores can rise proportionally (fake 0.1→0.45, faithful 0.3→0.7 widens the spread yet the fake rose 0.35). Requiring the fake to stay flat closes that hole. Sound because the prompt change only alters per-leaf LLM scores; gate/ceiling/roll-up are deterministic Python. (Note: this is a **prompt** A/B on the *same* grader model — not a model swap, which is D4 — so old-vs-new is a legitimate comparison.)

**Fixture independence (required):** the `faithful`/`fake` fixtures must NOT be the same runs the rubric was authored against. `best_runs/` is a convenient *seed* for realistic leaf grades, but the committed fixtures are hand-authored synthetic run-dirs (D7a) precisely so the kill criterion is not measuring rubric-overfit. Keep at least one faithful + one fake fixture from a paper family *outside* the SDAR/Adam tuning set.

An optional `@pytest.mark.live` variant runs the real two prompts (non-deterministic; opt-in, not the CI gate).

### D7c — Adversarial refute pass
```
refute_high_leaves(leaf_records, evidence, code_dir, *, high_threshold=0.8,
                   llm_client=None, cap=0.4) -> (records, refutation_log)
```
Slots in `score_reproduction` **between `L1284` (grading complete) and `L1289` (`_apply_invariant_gate`)** — **before** the invariant gate, so the deterministic floor remains the un-overridable last word. For records ≥ `high_threshold`, **one batched** meta-grader call (mirror `_grade_batch`) asks "hardcoded / surrogate / not actually executed, given the evidence?"; caps to 0.4 **only on an evidence-citing affirmation**, never on "uncertain." Default-OFF (`OPENRESEARCH_REFUTE_HIGH_LEAVES`). Takes its own `llm_client` (default = grader) so D4 can point it at a cheaper model. Surface a `refutations` key for `amend_final_report` (`1386-1412`). Does **not** run on the degraded branch (it returns before `1289`).

**Over-skepticism guard:** the faithful fixture (c) must stay ~0.95 *through* the refute pass — that test catches a refuter that re-introduces the 0.85 cap.

---

## 9. D8 — Rollout & sequencing

**Flags (all default-OFF; flag-off ⇒ byte-identical to today):** `OPENRESEARCH_ANCHORED_GRADER` (D1), `OPENRESEARCH_EVIDENCE_MANIFEST` (D2), `OPENRESEARCH_PRE_GPU_CRITIC` + sub-flags (D3), `--eval-model` (D4, default `oauth-sonnet`), `OPENRESEARCH_RESULT_MATCH_TREND` (D5), `OPENRESEARCH_HERMETIC_ENV` + `OPENRESEARCH_PREFLIGHT_GATE` (D6), `OPENRESEARCH_REFUTE_HIGH_LEAVES` (D7). Plus raise `OPENRESEARCH_TARGET_SCORE` (forced-iteration target) from 0.6 toward ~0.9 so the agent keeps climbing — **but only once D7's rail is green** (raising the target on an un-calibrated grader just burns GPU).

**Build order (each step independently shippable, gated, reversible):**
1. **D7a + D7b first** — fixtures + the dual-run harness with *today's* prompt as `old`. This is the instrument; nothing else ships without it green.
2. **D2 (3a→3b→3c)** — the manifest + the **promotion bug fix** + grader-consume. Biggest leverage, no calibration risk. Re-score a fixture to confirm legibility lifts the faithful score.
3. **D1** — anchored scale, validated immediately against D7b (faithful rises, fakes don't, spread widens).
4. **D4** — provider knob + the `model=None` pin fix (small, unblocks Opus/GPT graders + cheap refute).
5. **D5** — result-match trend grading (prose+category path).
6. **D3** — pre-GPU critics (depends on D2's manifest contract for the code critic).
7. **D6** — robustness floor (independent track; can land in parallel anytime).
8. **D7c** — refute pass last (it's the belt over D1's suspenders).

**Rollback & grader-version hygiene (Codex review 2026-06-07):** because D1/D5 change *what a score means*, a run scored under the new grader and a run scored under the old grader are **not comparable** — silently mixing them on `/leaderboard` is a correctness bug. Stamp a **`grader_version`** field on `GradeResult` → `rubric_evaluation.json` → `final_report.json` (bump on any D1/D5/D7c change). The leaderboard groups/filters by `grader_version`; a flag flipped off after a run was scored does not retroactively rewrite stored scores (they carry their version). This is also the clean rollback story: revert the flag, new runs score under the old version, old runs keep their stamped scores.

**Score-plumbing bug observed live (fix as part of D2/D8):** the Adam run that motivated this spec finished with `final_report.json::overall_score = None` while `final_report.json::rubric.overall_score = 0.831` — the top-level field is **not populated from the rubric sub-dict**. The watcher (and any leaderboard reader keying on top-level `overall_score`) sees `None` for a fully-scored 0.83 run. Populate top-level `overall_score`/`meets_target` from `rubric.*` at report-write time; add a regression asserting they agree. (This is exactly the legibility/plumbing fragility class D2 targets.)

---

## 10. Risks & open questions (for Codex review)
- **Inflation vs discrimination:** is the D7b kill criterion (spread-widens) strict enough, or do we also need an absolute fake ceiling per class? Are the 4 fixtures representative enough (only SDAR + Adam families)?
- **D5 without structured metadata:** grading direction from prose + deltas is softer than a `claim_direction` contract. Is the deferred structured-leaf version actually required for fairness, or is prose+category sufficient?
- **D2 vision cost/length:** sending PNGs + 200 KB evidence on every figure leaf — batch? cache? provider-aware byte budget for `openai-gpt-5` eval.
- **D3 in RLM:** can a primitive cleanly force a re-plan given the root drives ordering, or will the root ignore the `repairable` refusal and `FINAL_VAR`? The `run.py:706` extension + forced-iteration floor is the backstop — is it sufficient?
- **D6a pinning:** is a static `COMPAT_MATRIX` maintainable, or should we resolve pins dynamically from the installed driver? Escape-hatch coverage for exotic-dep papers.
- **Manifest ↔ gate co-versioning:** D2's schema and D6b's capacity_gate both read `cells.json`/manifest fields — a single schema version + a fail-soft-on-drift test.

---

## 11. Grounding provenance (verify these in review)
- Grader is text-only, `model=None`: `leaf_scorer.py:47-49`; `run.py:223-225` → `rlm_query.py:610-612`; `models.py:196`.
- Evidence caps + `_gather_evidence`: `leaf_scorer.py:190-191, 238-312`. Per-leaf payload drops `task_category`: `1196-1200`.
- The promotion bug: per-cell sidecars in `outputs/<run_id>/<cell>/`, aggregated level has only `metrics.json`; `cell_matrix.aggregate_cell_metrics:445-570`, `_persist_metrics primitives.py:3761`; `rubric_contract._check_required_artifacts` flat-globs `outputs/<run_id>`.
- Helper-copy: `baseline_implementation.py:59-83, 2357`. Enforcement: `rubric_guard.py:165, 283`.
- RLM has no controller boundary; critic seams `implement_baseline primitives.py:1345`, `run_experiment 3875` (contract_guard precedent `3927-3949`); rubric absent on first pass `4775`; `generated_rubric.json` `run.py:1522`; repair-floor `run.py:686-729`; bound via `forced_iteration.py:57,64,189,388`.
- RDR: `controller.py:939` (decompose), `727-786`/`1126-1133` (preflight gate).
- Robustness: venv `batch_reproduce.py:382-387, 413-425`; req blocks `primitives.py:2718-2769, 2789-2833`; torch strip/guard runpod-only `requirements_derive.py:253-270`, `pre_flight_validator.py:1186-1238`; classifier `failure_classifier.py:31-59, 168-296`; shell-warn `cli.py:78-105, 1113`; scattered preflights inside `_execute_cell_matrix primitives.py:3787-3789`.
- Discrimination: gates `leaf_scorer.py:37,40,150,776,909-931`; refute slot `1284→1289`; surrogate invariant `paper_hints.py:118-133`; LLM-free replay `finalize_rescore` **`leaf_scorer.py:944`** (confirmed present); stub-LLM tests `tests/rlm/test_leaf_scorer_invariant_gate.py:60-67`; fixtures `tests/agents/rlm/__fixtures__/sample_leaf_metrics.json`, `best_runs/{adam,vae,sdar}`.

**Verification note (2026-06-07):** a Codex adversarial pass disputed several anchors as "wrong-line" or "non-existent." Each contested anchor was re-checked directly against the working tree; the spec's anchors held in every case: `finalize_rescore` **exists** (`leaf_scorer.py:944`); the grader payload is **only** `{leaf_id, requirements}` — `task_category` **is** dropped (`leaf_scorer.py:1207-1208`) and D1 must add it; evidence caps are at `190-191`; the `model=None` grader is built at `run.py:225`. Codex's two "blockers" were both based on incorrect grounding (its run made a single tool call) and are **void**. Its four *conceptual* findings — kill-criterion absolute-fake-ceiling (§8), fixture independence (§8), D3 anti-gaming precondition (§4), grader-version/rollback (§9) — were valid and are incorporated above. Line numbers throughout are agent-verified but approximate; re-confirm at implementation time.
