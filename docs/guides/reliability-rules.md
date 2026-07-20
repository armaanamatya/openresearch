<!-- doc-meta: status=current; last-verified=2026-07-20 -->
# Reliability rules

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
> [`docs/archive/learn.md`](../archive/learn.md).

---

## 2026-07-17 — A cloud handoff must preserve both intent and authentication

**Rule.** Treat every process/Job boundary as a typed protocol: explicitly
forward the selected model and execution policy, then prove that the matching
credential source exists without embedding its value in the manifest.

**How.** Persist root model, execution mode, GPU mode, and compute-minimization
in campaign directives; project provider keys through CSI; pass only non-secret
endpoint/model coordinates as Job environment; test both the positive routing
and the absence of the key value.

**Why.** The first durable path successfully launched an autonomous Pod but
dropped its requested root model and never mounted the Foundry key required by
the forced `opus-foundry` profile. Infrastructure readiness hid a guaranteed
model-initialization failure.

---

## 2026-07-17 — A submitted Job is not yet a durable controller

**Rule.** Call a run durable only after the in-cluster process owns and renews
the cloud CAS lease, its pod is actually Running, and its resumable state is on
persistent storage.

**How.** Give each launch a unique owner that survives only that Job's pod
retries; require Running/Succeeded pod readiness; mount the RWX run directory;
fail closed on ambiguous cluster/storage errors; and keep intentional money
halts out of Kubernetes crash retries.

**Why.** `Job.status.active` includes Pending pods, a project-id owner lets a
second laptop impersonate a restart, and local fallback after a timed-out create
can produce split-brain. All three failure modes existed in the first durable
controller merge.

---

## 2026-07-13 — A default-OFF flag with no forcing function is a fix that never ships; make "undecided" fail CI

**Rule.** Every integrity/reliability flag must carry an EXPLICIT per-profile decision in
`configs/flag_decisions.json`, enforced by `tests/config/test_flag_decision_manifest.py`. `off`
is a valid decision *with a stated reason*; `off` by accident is the bug. A new
`*_GUARD/_GATE/_KILL/_VERDICT/_VALIDATOR/_CHECK/_DETECTION` flag cannot merge without a decision,
and the gate asserts manifest⇄profile consistency both ways so a "should-be-on" lever can never
silently be dark.

**How.** The repo's flag policy (new flags default-OFF + byte-identical; default-flip needs ≥3
paired A/B) is individually sound but produced 471 flags with no promotion pipeline, so fixes
ship dark. It is NOT fixable by care: on the day this rule was written, the lead diagnosed the
disease, quoted it, built the run-spec profiles to route around it — and still shipped four new
integrity flags dark, incl. a verdict-ceiling whose whole job was blocking a false "reproduced".
The manifest gate immediately found a fifth: both profiles set `EVIDENCE_GATE=1` believing they
had enabled the per-leaf veto `LEAF_EVIDENCE_GATE` (a DIFFERENT var) — the corpus's
"highest-value correctness lever" was dark in both tiers.

**Why.** `learn.md` had already recorded "fixes ship default-OFF and protect NOTHING until
enabled in the run-spec." A postmortem is not a control; a failing test is. See
[[flag-decision-discipline]] and the two-var `EVIDENCE_GATE` vs `LEAF_EVIDENCE_GATE` untangling.

---

## 2026-07-13 — Closing os.environ does not close /proc; only sandboxing closes a REPL that execs on the host

**Rule.** Do not treat credential-scrubbing mitigations as closing the leak class when the root
model `exec`s in the orchestrator process. Scrub `os.environ` and hand credentials over an
inherited pipe (not `env=`, which freezes into `/proc/self/environ`) as defense-in-depth — but
the class is only closed by running the reproduction in a disposable sandbox.

**How.** The credential vault takes the child's `os.environ` and `/proc/self/environ` from 3/3
harvestable sentinels to 0 (proven with the real exploit). But `/proc/<ppid>/environ` still holds
the uvicorn parent's keys, and — verified on this host at `kernel.yama.ptrace_scope=1`, the
*protective* value — a same-UID child CAN read it (`ptrace_scope` gates `ptrace()` memory
attach, not `/proc/<pid>/environ` reads). `__import__("os").system(...)` RCE is wholly
unaddressed. The paper is attacker-influenceable (arbitrary uploaded PDF), so a prompt injection
reaches all of this.

**Why.** Once the attacker has `exec` in your process the game is lost from inside it; each
mitigation only raises cost. Operational control until the sandbox lands: do NOT expose public
PDF upload — trusted arXiv IDs / operator papers only. Full analysis: the capability spec §5;
[[openresearch-audit-2026-07-13]].

---

## 2026-07-13 — An equality-based veto is structurally blind to NaN; a filter that DROPS non-finite readings turns "diverged" into "no data"

**Rule.** Any guard that detects degenerate values must test `math.isfinite` FIRST, before any `==` comparison — and must never silently drop non-finite readings. A non-finite result-claiming metric is an automatic veto: strictly worse than zero, never exempt (there is no honest reading of a NaN result, so not even the provenance exemption applies).

**How.** `zero_metrics_detection`'s predicates were `v == 0.0` and `v == values[0]`. Under IEEE-754 self-inequality both are `False` for NaN, so a run that diverged to NaN sailed past the veto built to catch degenerate results. `no_learning_signal` / `dead_training_guard` compounded it by *filtering out* non-finite points — a curve that goes NaN and stays NaN read as "no data" rather than "diverged." `dead_training_guard`'s number regex could not even *match* `nan`/`inf`, so its `isfinite` filter sat downstream of a parser that was already blind. Fix: non-finite checked first and vetoed unconditionally; the trend guards key divergence on the curve's **tail**, so a transient fp16 spike that recovers is deliberately NOT flagged (see the 2026-07-07 false-block history).

**Why.** The fabrication guards are the fitness signal. A veto that cannot see the single most common way training dies is not a backstop.

---

## 2026-07-13 — A trust predicate must exist EXACTLY ONCE; every other site delegates

**Rule.** A guard/veto predicate gets one canonical implementation. Any second copy is a second place for it to be wrong — and it will be, silently, in the tier that matters most.

**How.** `external_validator.check_not_all_constant` re-derived the degenerate-metrics test inline (`all(v == 0.0 ...)` + `len(set(values)) == 1`) instead of calling `looks_like_zero_metrics`. When the canonical predicate was made NaN-aware, the copy stayed blind — so the Tier-2 adversarial panel, the one gate whose entire job is precision, reported a NaN-diverged run as *healthy*. It now delegates.

**Why.** Defence-in-depth only works if the depth is real. Duplicated logic converts N gates into 1 gate plus N-1 places to rot.

---

## 2026-07-13 — The test suite must be ENV-hermetic, not just socket-hermetic

**Rule.** Tests must never read the developer's real `.env`. A test asserting a Settings-backed DEFAULT must inject what it depends on explicitly; ambient environment is not a fixture.

**How.** `backend/config.py`'s `SettingsConfigDict(env_file=".env")` makes pydantic-settings read `.env` **from disk on every `Settings()` construction**, regardless of `os.environ` — deliberate in production, wrong under test. `delenv` alone cannot fix it (the disk read bypasses `os.environ`, and `factory.py` separately copies `.env` secrets *into* `os.environ`). `tests/conftest.py` now scrubs the leaking namespaces and sets `Settings.model_config["env_file"] = None`; a guard test plants a fake `.env` and fails loudly if hermeticity ever regresses.

**Why.** 18 suite failures were this ONE root cause — four separate agents each independently filed them as "pre-existing, unrelated." A live `AZURE_FOUNDRY_API_KEY` was printed in full into pytest assertion output (rotate on exposure). Worst of all: every "default-OFF / byte-identical-when-off" assertion was silently unprovable, because the suite was asserting against a developer's `.env`, not the code default.

---

## 2026-07-13 — Stage-after-archive: capture the archive's RETURN path, never the pre-archive path

**Rule.** When a step relocates a directory, any pointer staged afterwards must be built from the mover's returned destination — never from the path captured at plan time.

**How.** `attempt_driver.launch` called `force_archive_incomplete` (which `shutil.move`s `code/` into `attempts/<ts>/`), discarded its return value, and *then* staged a seed marker whose `source_code_dir` was the plan-time `run_dir/code` — the directory the archive had just emptied. `seed_reference_code` fails closed on a missing source, so seeding silently never happened. `_stage_seed_marker` now takes `source_code_dir` as a REQUIRED keyword, so no future caller can reintroduce the ordering.

**Why.** Cross-attempt learning had never once fired. Receipt: campaign `prj_09047604e591d969` ($24.23 LLM, 4.5h, EXHAUSTED) staged `seed_staging.json → runs/.../code`; `_BEST_ATTEMPT_README.txt` exists nowhere on disk, repo-wide. All three attempts cold-started, and the terminal recorded `champion_attempt_n: 1` while attempt 2 fell back to `runner_up` — the champion arm skipped for want of a pointer. The campaign was a retry loop wearing a learning loop's clothes.

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
