<!-- doc-meta: status=current; last-verified=2026-08-01 -->
# configs/ — run-spec profiles

`--run-spec <path.json>` (`backend/cli.py`) loads a flat JSON object of
`OPENRESEARCH_*`/`REPROLAB_*` keys (+ the `models`/`baseline_extra_guidance`
special keys) into the process env before flag resolution. Every key is checked
against `backend/agents/rlm/run_spec_contract.py::run_spec_key_applies` at
campaign INIT, before any money moves (F15): a renamed/typo'd key fails the
round-trip at $0 instead of silently no-opping for the rest of the campaign.
Full flag semantics live in `backend/agents/rlm/CLAUDE.md`'s feature-flag catalog.

> **Every integrity flag's per-profile decision is recorded and enforced.**
> `flag_decisions.json` is the authoritative manifest: every integrity-shaped
> `OPENRESEARCH_*` flag (`*_GUARD`/`*_GATE`/`*_KILL`/`*_VERDICT`/`*_VALIDATOR`/
> `*_CHECK`) carries an EXPLICIT `on`/`off`/value decision per tier, with a
> rationale. `tests/config/test_flag_decision_manifest.py` fails the suite if a
> guard exists in `backend/` without a decision, or if a decision drifts from
> the profile JSON below — so a reliability fix can no longer ship "dark"
> (present in code, absent from every profile, protecting nothing —
> `learn.md` 2026-07-07). Add a new guard flag ⇒ decide it in the manifest.

> **A key that passes the contract can still be inert.** The contract checks the
> key *name prefix* only. See "Two kinds of env key" below before adding a key —
> a shipped profile already has this bug.

## The two-tier triage funnel

`triage_screen_run_spec.json` (Tier 1) and `verify_deep_run_spec.json` (Tier 2)
implement a cheap-then-expensive funnel:

| | Tier 1 — SCREEN | Tier 2 — VERIFY |
|---|---|---|
| Purpose | Cheap, broad, **recall**-oriented pass over many candidate papers; ranks confidence + feasibility | Expensive, **precision**-first, fail-closed confirmation — shortlist only |
| Root model | `gpt-5` (reliable + ~$1/run) | `opus-foundry` (Claude Opus 4.8, best-known config) |
| Sub-roles | default (Sonnet executor) | `executor`/`grader`/`verifier` = `sonnet-foundry` |
| Grader | decoupled + cheap: `openai`/`gpt-4o-mini` | `sonnet-foundry`, `GRADER_SAMPLES=3` |
| Scope / seeds | reduced scope, single seed (via `--scope-spec`/`--scope-ladder`) | full scope, multi-seed |
| Trust/memory machinery | off | on (external validator, champion-artifact, leaf-actuate, recipe/lesson/experience memory, two-axis verdict) |
| Can it mint `REPRODUCED`? | **No** (see below) | Yes — the only tier allowed to |

**Both tiers are execute-mode-first** (`USE_AUTHOR_REPO=1`,
`REPRODUCTION_MODE=auto`, `LIFECYCLE_PRIMARY=1`): *prefer* running the authors'
published code behind a verified metrics shim over having the LLM re-implement the
paper. Evidence: from-scratch SDAR scored **0.0**; the authors' trainer scored
**0.456**. `auto` — not a pinned `execute` — is what makes this a *preference*
rather than a precondition: it resolves to `execute` when the paper has usable code
and falls back to a disclosed from-scratch attempt when it does not, so a paper
that published no code is still screened instead of crashing the run (see
"`REPRODUCTION_MODE=auto`" below — this is a recall red line).
`LIFECYCLE_PRIMARY` supplies the deterministic understand→implement→run→verify FSM
— without it the run depends on the root LLM self-sequencing itself, which is the
documented degenerate-loop failure mode.

**The pure-stdlib, conservative, fail-soft fabrication/evidence guards are
enabled in BOTH tiers** — disabling them would not make Tier 1 cheaper, only
make it *lie* about what it found, which breaks the red-line invariant: the
deterministic evidence layer, never the LLM grade, is the fitness signal. This
includes `OPENRESEARCH_LEAF_EVIDENCE_GATE` (the A7 per-leaf veto — the corpus's
single highest-value correctness lever; note it is a **distinct** var from the
verdict-level `OPENRESEARCH_EVIDENCE_GATE`, so setting the latter did **not**
enable it), now on in both tiers.

A few integrity flags are deliberately **split** by the recall-vs-precision
economics below rather than on in both — e.g. `OPENRESEARCH_METRIC_SEMANTICS_GUARD`
is verify-only (its rate-range branch false-blocks a paper that stores a rate as
a percentage, a false negative the recall tier can't afford; triage keeps NaN
protection via `ZERO_METRICS_GUARD`), and the cost-bearing / certification-only
gates (`EXTERNAL_VALIDATOR`, `CODE_REVIEW_GATE`, `SPEC_VALIDATOR`,
`REPORT_CLAIM_GATE`, `EVIDENCE_AUDIT`, two-axis verdict) are verify-only. The
authoritative per-flag decision + rationale is `flag_decisions.json`.

## Cost levers — what Tier 1 economizes on, and what it must never touch

Tier 1 is the **recall-critical** stage of a needle-in-a-haystack funnel. Its
expensive error is a **false negative**: screening out a paper that would in fact
have reproduced. So Tier 1 economizes only on levers that cost *compute*, never
on levers that cost *recall*:

**Cheapen freely** — reduced scope rung; single seed; fewer training steps;
smaller training-cell models; a cheap **decoupled grader** (`gpt-4o-mini` — the
grader only reads text, and the σ-gate measured grader noise at σ=0.0067).

**Never cheapen the root model.** A weak root (qwen / kimi / grok) is exactly the
one CLAUDE.md documents as degenerating, needing `OPENRESEARCH_ARG_CONTRACTS`,
and not being paper-validated. A root that fails to *drive the harness* produces
a spurious "couldn't reproduce" — i.e. a false negative, precisely the error the
screen tier must not make. Root LLM spend is also a small fraction of GPU spend,
so it is a poor place to economize. Tier 1 therefore pins `gpt-5`, which CLAUDE.md
names as the recommended reliable root *and* costs ~$1/run.

If a future edit ever does pin a weak root, `OPENRESEARCH_ARG_CONTRACTS=1` must
go on alongside it — `tests/config/test_triage_and_verify_run_specs.py` enforces
this.

## Why Tier 1 cannot mint "reproduced" — and how to keep it that way

The campaign's REPRODUCED gate (`backend/agents/rlm/campaign_policy.py::decide`,
rule 1) requires, among other predicates:

```python
scope_rung_by_attempt.get(a.attempt_n) == full_rung   # campaign_policy.py:788
```

where `full_rung = config.ladder_len - 1` and `ladder_len = len(opts.scope_ladder) or 1`
(`campaign_composition.py::_decide_impl`), fed straight from the CLI's
`--scope-ladder`. A reduced rung can equal the full rung only when the ladder has
exactly one rung — so the guarantee holds only when Tier 1 is invoked with:

- an explicit **multi-rung** `--scope-ladder` (e.g. `reduced,full`) so
  `full_rung >= 1`, **and**
- `--max-attempts 1`. Attempt 1 always starts at rung 0
  (`scope_rung_by_attempt = {1: 0}`); with no second attempt the rung can never
  climb to `full_rung`.

**Do not omit `--scope-ladder` for Tier 1.** Unset, it defaults to a *single*-rung
ladder (`("full",)`), making `full_rung == 0` — immediately satisfied by attempt 1.
Tier 2 wants the opposite and omits it, defaulting to a single full rung.

## How to run each

```bash
# Tier 1 -- cheap screen: single attempt, reduced scope + single seed, tight budget
python -m backend.cli campaign <paper> \
  --run-spec configs/triage_screen_run_spec.json \
  --scope-ladder reduced,full \
  --scope-spec '{"seeds":[0]}' \
  --max-attempts 1 \
  --max-llm-usd 3 --max-gpu-usd 5 --max-gpu-hours 1 \
  --sandbox local

# Tier 2 -- shortlist only: full scope, multi-seed, all guards + external validator
python -m backend.cli campaign <paper> \
  --run-spec configs/verify_deep_run_spec.json \
  --scope-spec '{"seeds":[0,1,2]}' \
  --max-attempts 6 \
  --max-llm-usd 50 --max-gpu-usd 300 --max-gpu-hours 12 \
  --sandbox gcp
```

Credentials: Tier 1 needs `OPENAI_API_KEY` (root + grader) and the usual Anthropic
executor auth. Tier 2 needs `AZURE_FOUNDRY_*` (root + all Claude sub-roles) **and**
`OPENAI_API_KEY` (the validator). Tier 2's Claude sub-roles are pinned to Foundry
deliberately: an `opus-foundry` root co-resident with a `claude-oauth` sub-role
trips `run.py::assert_no_foundry_oauth_coresidency`. The validator sits on OpenAI
so it is **cross-family** from the Claude executor — a genuinely independent judge
(`role_models.separation_strength` → `independent`), not a same-model echo. This is
not optional once `OPENRESEARCH_EXTERNAL_VALIDATOR=1`:
`grader_transport.build_validator_client` is **fail-closed** and raises rather than
silently judging with the executor's own lineage.

## `REPRODUCTION_MODE=auto` — why neither tier pins `execute`

Both tiers set `OPENRESEARCH_REPRODUCTION_MODE=auto`, **not** `execute`. `auto`
resolves the mode **per paper**, at repo-resolution time in
`run.py::_build_context()`:

| Paper | Resolves to | Evidence quality |
|---|---|---|
| Usable author repo, cloned OK | `execute` — run the authors' own pipeline behind a value-preserving metrics shim | **High** — the published code actually ran |
| No repo found / clone failed / repo unusable | `scratch` — a real from-scratch attempt, loudly disclosed | **Lower** — the model reimplemented the paper |

A hard-pinned `execute` **hard-fails every paper that published no code**:
`run.py::assert_execute_mode_stamped` raises `RuntimeError` when `execute` was
requested but the resolver/clone found nothing and silently downgraded. For a
needle-in-a-haystack triage funnel that is the *worst available error* — it turns
"this paper shipped no code" into "this run crashed", which reads downstream as a
**false negative**: a paper that might well have reproduced from scratch gets
discarded. The screen tier's expensive error is exactly that, so a no-code paper
must still get a real attempt.

**The backstop is not weakened.** `assert_execute_mode_stamped` still raises for an
*explicit* `execute` request that silently became `adapt`/`scratch` — a silent
downgrade is a lie about what ran, and that is the bug it exists to catch. `auto`
is exempt because its fallback is **disclosed, not silent**.

### How a fallback is disclosed (read this before trusting a screened score)

An execute-mode result and a from-scratch result are **not** the same evidence —
from-scratch SDAR scored **0.0** where the authors' trainer scored **0.456**. A
triage consumer must never conflate them, so a fallback surfaces in four places:

- a loud `execute_mode_no_repo` `run_warning` on the SSE/event stream;
- `rlm_state/repo_spec.json` → `mode` (the **resolved** mode: ground truth for what
  ran, and what every downstream consumer reads) plus `requested_mode` /
  `fallback_from_execute`;
- `final_report.json` → `reproduction.mode` (`execute` vs `scratch`) with an
  explicit `reproduction.fallback{from,to,reason,evidence_note}` sub-dict;
- `final_report.json` → `degradations_taken[]`, which now carries
  `execute_mode_no_repo` — a fallback *is* a degradation, of evidence quality.

`final_report.md` also renders an **Evidence provenance** line under the
Reproduction Summary on `auto` runs, so a human reader cannot miss it either.

### Don't pin `OPENRESEARCH_EXECUTE_OWNS_DEPS` alongside `auto`

Neither tier sets it, deliberately. `=1` is a **hard** opt-in — `_execute_owns_deps`
returns `True` on the literal flag *without* consulting `repo_spec.json`. On a
from-scratch fallback there is no authors' conda env to own the dependencies, so a
pinned `1` would suppress the harness's own pip/torch bootstrap and the training
cell would die on `import torch`. Left **unset** it auto-derives from the resolved
`repo_spec` mode (`execute` ⇒ `True`, `scratch` ⇒ `False`) — correct on *both*
branches of `auto`. `tests/config/test_triage_and_verify_run_specs.py` pins this.

(`configs/sdar_execute_run_spec.json` still pins both `execute` and
`EXECUTE_OWNS_DEPS=1` — that is fine and intended: it targets one specific paper
that definitely *has* a repo, so there is no no-code branch to fall back to.)

## Two kinds of env key (why a contract-valid key can still be inert)

- **`os.environ`-read flags** (`os.environ.get("OPENRESEARCH_X")` at run time) —
  a run-spec entry works, and *overrides* a shell export.
- **pydantic-`Settings`-backed keys** (`backend/config.py`, resolved via
  `env_prefix="OPENRESEARCH_"`) — a run-spec entry is **silently inert**. The
  settings cache is already warm by the time `_load_run_spec` writes to
  `os.environ`, and nothing force-reloads it. These keys only take effect as a
  **shell export set before the process starts**.

Both pass `run_spec_key_applies`, so the F15 contract check cannot tell them
apart. `tests/config/test_triage_and_verify_run_specs.py` closes the gap by
asserting every profile key is actually read back out of `os.environ` somewhere
in `backend/`.

Known-affected (verified empirically): `OPENRESEARCH_REPO_CLONE_TIMEOUT_S`,
`OPENRESEARCH_REPO_CLONE_MAX_MB`, `OPENRESEARCH_REPO_CLONE_LFS`,
`OPENRESEARCH_REPO_LOCAL_PATH`, `OPENRESEARCH_REPO_COMMIT`. The last two are
currently set in `configs/sdar_execute_run_spec.json`, where they are inert —
export them in the shell instead. (That file's bare `HF_HOME` key is inert for a
different reason: it fails `run_spec_key_applies` outright and would *raise* at
campaign INIT — that spec is `reproduce`-only, not campaign-safe.)
