<!-- doc-meta: status=current; last-verified=2026-07-06 -->
# Design — Relevance-gated, agent-selected skill activation

> **Date:** 2026-07-06 · **Status:** Current (design, pre-implementation) ·
> **Author surface:** `backend/agents/rlm/` skill subsystem.
> **Prereq context:** the OpenScience skill-library port (Release-1, commit `8f4944bf`, now
> merged to `deepinvent/main` via `62ae4f73`). This spec adds a *selection* layer on top of the
> already-merged `consult_skill` primitive — it does not re-port the library.

## 0. TL;DR

The subject-matter skill library is merged but **dormant** (no run-spec enables it) and, even when
enabled, invocation is **not gated to the paper**: the root sees the whole 200-file catalog and
must self-select, and the **verifier cannot consult skills at all**. This design adds a
**relevance-gated, agent-selected activation** step that runs in the **understand-paper phase**:
a deterministic-but-thorough matcher proposes candidate skills from the paper's already-extracted
subject matter, a bounded agent/LLM pick prunes them to the relevant set, and that **active skill
set** is surfaced to both the root (implement) and the verifier (grade). All behind flags,
default-OFF and byte-identical when off; the deterministic evidence layer remains the sole fitness
signal.

## 1. Problem — what's true today (grounded in code)

`consult_skill` (19th primitive, `backend/agents/rlm/primitives.py:8946`, gated by
`OPENRESEARCH_SKILLS`) is agent-driven, on-demand, progressive-disclosure: no args → category
index; `category=` → browse; `name=` → full playbook body + copies that skill's `scripts/` into
`code/skill_scripts/<name>/`. When `OPENRESEARCH_SKILLS` is on, `system_prompt.py::_skill_catalog_section`
appends a compact catalog overview (categories + 3 example names each) to the **root** prompt.

Two gaps versus the goal ("skills invoked only when that subject matter is required; the
paper-understanding agent *and* the verifier choose"):

- **Gap 1 — relevance is not tied to the paper.** The root self-selects from the *entire* catalog;
  there is no "given THIS paper's datasets/methods/frameworks, here are the relevant skills" step.
  Targeting quality depends entirely on the root recognizing relevance from a generic list.
- **Gap 2 — the verifier has no skill access.** `consult_skill` and the catalog section are
  **root-only**. `verify_against_rubric` (`primitives.py:8041`) delegates to
  `leaf_scorer.score_reproduction` → `_grade_batch`, a grader LLM call that never calls primitives.
  "The verifier chooses what skills to invoke" is impossible today.

Already satisfied (no work): **entry-point-agnostic** — `consult_skill` lives in the RLM loop, which
runs identically for PDF upload (`POST /api/demo`) and CLI/GCP; and **on-demand agent choice** — the
root already pulls playbooks itself.

## 2. Goals / non-goals

**Goals**
- Select skills **in the understand-paper phase**, producing a per-run **active skill set**.
- Selection = **deterministic thorough recall** (paper subject matter → candidate skills) **then a
  bounded agent/LLM precision pick** — the choice stays with the agent.
- Make the active set available to **both** the root (focused implement guidance) and the
  **verifier** (grade-time skill consultation) — closing Gap 2.
- Flag-gated, default-OFF, byte-identical when off; opt-in per-run via run-spec for the SDAR e2e.
- Deterministic candidate provenance on disk; the LLM pick is advisory and logged.

**Non-goals**
- Not re-porting the skill library (already merged).
- Not making skills a fitness signal — they sharpen the LLM verifier but the **deterministic
  evidence layer stays authoritative** (zero-metrics / eval-provenance / env-liveness /
  no-learning / evidence-gate + external validator).
- Not flipping any global default (a default-ON flip needs the ≥3-paired-A/B + grader-σ gate).
- Not a new retrieval/embedding index — the catalog is small and already tag-annotated.

## 3. Architecture overview

```
understand_section ─► PaperClaimMap (datasets, metrics, training recipe, frameworks, keywords)
detect_environment ─► EnvSpec      (frameworks, libraries)
        │
        ▼   (understand-paper phase; after detect, before plan)
  skill_selection.select_active_skills(subject_matter, catalog)
        │
        ├─ (1) match_candidate_skills()   deterministic, thorough recall  ── no LLM
        │        paper tokens × skill{tags,category,name,description}  (reuses _fuzzy_score)
        │
        └─ (2) llm_prune_candidates()     bounded agent/LLM precision pick ── fail-soft
                 candidates + subject matter → relevant subset (+ 1-line reason each)
        │
        ▼
  rlm_state/active_skills.json  {selected, candidates, subject_matter_keys, selector, reasons}
  SSE: skills_selected
        │
        ├─► ROOT prompt: "Skills relevant to THIS paper: …"  (focused, replaces full-catalog overview)
        │      root still calls consult_skill(name=…) on demand
        │
        └─► VERIFIER (_grade_batch): active-skill context injected
               grader may consult the selected playbooks when judging fidelity  (advisory)
```

## 4. Components

### 4.1 Deterministic candidate matcher — `skill_selection.py::match_candidate_skills`
Pure Python, zero LLM, deterministic, reproducible. Reuses `skill_catalog` primitives.

- **Inputs:** a normalized `SubjectMatter` extracted from the understand-phase outputs —
  `datasets`, `metrics`, `methods`/`training_recipe` terms (from the PaperClaimMap), `frameworks` +
  `libraries` (from EnvSpec), and `keywords` (title/abstract tokens). Tokenized + lowercased.
- **Scoring:** for each `SkillMeta`, relevance = best match of any subject token against the skill's
  high-signal fields — `tags` (curated, weighted highest), then `category`, then `name`, then
  `description`. A token matches a field via a direct substring hit **or** `skill_catalog._fuzzy_score`
  (bigram-Dice) ≥ a generous `SKILL_MATCH_THRESHOLD` (default lenient, e.g. 0.5 for fuzzy; substring
  always wins). Recall-biased on purpose: a candidate is admitted on ANY field hit.
- **Output:** candidates ranked by best score, capped at `SKILL_CANDIDATES_MAX` (default ~15) with a
  per-category floor so no relevant domain is starved. Deterministic tiebreak (score desc, name asc).
- **Fail-soft:** empty catalog or empty subject matter → `[]` (never raises).

### 4.2 Bounded agent/LLM precision pick — `skill_selection.py::llm_prune_candidates`
- **Call:** one bounded completion via the paper-understanding transport (`ctx.llm_client`; the same
  model that understood the paper). Prompt = a compact subject-matter summary + the candidate
  shortlist (`name`, `category`, `description` only — never full bodies) → returns the subset of
  candidate **names** genuinely needed to reproduce THIS paper, each with a one-line reason.
- **Grounding:** the LLM only **prunes** the deterministic candidate set (it cannot invent names or
  reach past the shortlist), keeping the call cheap and the output bounded/verifiable.
- **Fail-soft:** any error / timeout / unparseable output → fall back to the deterministic candidate
  set (selection never blocks a run). `selector` stamped `"deterministic+llm"` or `"deterministic"`.
- **Escape hatch:** `OPENRESEARCH_SKILL_SELECT_DETERMINISTIC=1` skips the LLM entirely (selected =
  top-K candidates) — the zero-extra-call variant.

### 4.3 Understand-phase hook — `lifecycle_driver.py` + root-loop nudge
- **Lifecycle-primary path (SDAR uses this):** in `_run_lifecycle_chain`, after
  `understand_section` + `detect_environment` and **before** `plan_reproduction`, call
  `select_active_skills(...)` (guarded by the flag). Emits a `lifecycle_drive_step` / `skills_selected`
  event; writes `active_skills.json`. Fail-soft: a selection error logs + continues (plan/implement
  proceed with no active set, i.e. the full-catalog fallback).
- **Normal root-loop path:** selection is triggered during understanding by one of two mechanisms
  (decided at plan time, §10) — either a lightweight `select_skills()` **primitive** the root calls
  (thin wrapper over `select_active_skills`; this moves the primitive count 19→20), or **lazy
  auto-run** on first skill access (the catalog section builder / first `consult_skill` call runs
  selection from the latest understand outputs / intra-run context map when no `active_skills.json`
  exists yet). Either way writes the same artifact + event, keeping both drivers aligned. The
  lifecycle-primary path above never needs the primitive (it uses the internal hook).

### 4.4 Root/implement consumption — `system_prompt.py`
- When `active_skills.json` exists (SKILL_SELECT on), `_skill_catalog_section` renders an **ACTIVE
  SKILLS** block (selected names + descriptions + "consult these first") instead of the full-catalog
  overview. The root can still `consult_skill(name=…)` on non-active skills, but its attention is
  focused on the subject matter.
- SKILL_SELECT off (SKILLS on) → today's full-catalog overview, unchanged.

### 4.5 Verifier consumption — `leaf_scorer.py::score_reproduction` / `_grade_batch` (closes Gap 2)
- When SKILL_SELECT is on and `active_skills.json` exists, `verify_against_rubric` reads the active
  set and passes it into `score_reproduction`, which injects a bounded **"skill playbooks relevant to
  this paper"** context into the grader prompt — the selected skills' `name`+`description` (+ the
  bodies of the top-`SKILL_VERIFIER_BODIES` most-relevant, size-capped) — so the grader judges
  fidelity against the domain playbook, not just the paper text.
- **Reach (sub-decision, default = active set):** the verifier sees the *active* set. A wider variant
  (verifier browses the full catalog via its own `consult_skill`) is deliberately deferred — the
  active set is the focused, size-bounded default.
- **Advisory only:** this improves the LLM grade's quality; it is NOT a new fitness signal. The
  evidence gate + fabrication guards remain the authoritative red line.

### 4.6 Persistence + provenance
`runs/<id>/rlm_state/active_skills.json`:
```json
{
  "selected": ["serving-llms-vllm", "grpo-rl-training"],
  "candidates": [{"name": "serving-llms-vllm", "category": "ml-inference", "score": 0.83}, "..."],
  "subject_matter_keys": {"datasets": ["Search-QA"], "frameworks": ["verl", "vllm"], "methods": ["GRPO", "OPSD"]},
  "selector": "deterministic+llm",
  "reasons": {"serving-llms-vllm": "paper serves Qwen via vLLM for rollout generation"}
}
```
SSE event `skills_selected` (`{count, selected}`), fail-soft via `_safe_emit`.

## 5. Flags (default-OFF; byte-identical when off)

| Flag | Default | Effect |
|---|---|---|
| `OPENRESEARCH_SKILLS` | off | Master (existing). Off → no catalog section, `consult_skill` returns `{"status":"disabled"}`, no selection. |
| `OPENRESEARCH_SKILL_SELECT` | off | New. On (requires SKILLS on) → understand-phase selection → `active_skills.json` → focused root prompt + verifier access. Off → today's full-catalog behavior. |
| `OPENRESEARCH_SKILL_SELECT_DETERMINISTIC` | off | Skip the LLM pick; selected = top-K deterministic candidates. |
| `OPENRESEARCH_SKILL_CANDIDATES_MAX` | 15 | Candidate cap (recall knob). |
| `OPENRESEARCH_SKILL_VERIFIER_BODIES` | small (e.g. 2) | How many top playbook bodies to inline into the grader prompt (size guard). |

State table: `SKILLS=off` → byte-identical to pre-R1. `SKILLS=on, SKILL_SELECT=off` → today's R1
behavior. `SKILLS=on, SKILL_SELECT=on` → the new relevance-gated selection. This makes a clean
three-arm A/B possible.

## 6. Invariants preserved
- **Evidence-not-grade red line** — skills never become fitness; the deterministic evidence layer is
  authoritative. Verifier skill context is advisory.
- **Default-OFF / byte-identical** — every new path is flag-gated; unset ⇒ no new files, no prompt
  changes, no extra calls.
- **Fail-soft everywhere** — a matcher/LLM/IO error degrades to the full-catalog fallback; selection
  never aborts a run (mirrors `consult_skill`'s and the context-map's fail-soft contracts).
- **Entry-point-agnostic** — selection runs in the understand phase of the shared RLM loop → identical
  for upload and CLI/GCP.
- **No prompt-injection regression** — reuses `skill_catalog`'s existing injection sanitizer; selected
  bodies passed to the verifier go through `get_skill_body` (already strips injection lines).

## 7. Testing
- `tests/rlm/test_skill_selection.py`:
  - **Recall:** SDAR subject matter (`{Search-QA, verl, vllm, GRPO, OPSD}`) → candidates include the
    `ml-training` (GRPO/RL) + `ml-inference` (vLLM) + `research` (agentic env) skills.
  - **Determinism:** identical subject matter → identical candidate list + order.
  - **Fail-soft:** empty catalog / empty subject matter / LLM error → `selected=[]` or deterministic
    fallback, never raises.
  - **Off-state byte-identical:** `SKILL_SELECT` off → no `active_skills.json`, root prompt = today's
    full-catalog section, verifier prompt unchanged.
  - **Verifier injection:** with an active set, the grader prompt carries the selected playbook
    context; without, it does not.
- Extend `tests/rlm/test_registry.py` only if a `select_skills` primitive is added (keeps the
  primitive-count fidelity test honest — would move 19 → 20; update root + nested CLAUDE.md + the
  fidelity anchor together).
- Off-state regression sweep + `uvx ruff` clean.

## 8. Integration + SDAR end-to-end (downstream of this design)
1. **Commit prerequisites** (separate, off this design): branch off `main`; land the two validated
   SDAR-execute fixes — `run.py` (Foundry-executor beta-header disable) + `provisioner.py`
   (symlink-preserve) + the `sdar_execute_run_spec.json` commit-pin. Keep the external-runs workstream
   OUT (per the 2026-07-05 merge handoff §3).
2. **Enable skills in the SDAR run-spec:** add `OPENRESEARCH_SKILLS=1` + `OPENRESEARCH_SKILL_SELECT=1`
   (opt-in per-run; global default stays OFF).
3. **A/B (evidence-not-grade):** OFF arm = the existing skills-OFF Phase-1 (`val/success_rate=0.456`,
   adjudicate from GCS — **$0**); ON arm = one skills-ON Phase-1 Search-3B (~$30, autostop,
   evidence-gated + external-validator). Compare on measured evidence + PASS gate.
4. **On healthy ON arm → grid** (~$400, staged, checkpoint before full spend).

## 9. Risks / mitigations
- **Grader context bloat** → inject descriptions + only top-`K` bodies, size-capped.
- **LLM-pick latency/cost** → bounded single call, fail-soft to deterministic; `_DETERMINISTIC` hatch.
- **Recall miss** (a relevant skill unmatched) → thorough recall-biased matcher + per-category floor +
  the root can still consult non-active skills.
- **LLM-pick non-determinism** → the candidate set is deterministic + logged; the pick is advisory,
  logged with reasons, and never touches the evidence layer.
- **Primitive-count drift** → if `select_skills` is added, update the 19→20 fidelity anchors in the
  same change.

## 10. Open questions (resolved defaults, flag to change)
- Selection call: **bounded LLM pick** (default) vs pure-deterministic (`_DETERMINISTIC`). — default set.
- Verifier reach: **active set** (default) vs full-catalog browse. — default set.
- New `select_skills` primitive vs internal-only hook for the root-loop path: prefer **internal hook +
  lifecycle call**; add the primitive only if the normal root loop needs an explicit callable
  (decided at plan time; affects the primitive count).

## 11. Implementation status (2026-07-06 — shipped, default-OFF)

Implemented; three deviations from the pre-implementation design above, each a *simplification* found
during recon (they lower blast radius while meeting every goal + invariant):

1. **One trigger = `detect_environment`, not a `lifecycle_driver` hook.** Recon found the deterministic
   matcher (`skill_matcher.match_skills`) was **already computed + persisted** (`rlm_state/skill_match.json`)
   inside `detect_environment` when `OPENRESEARCH_SKILLS` is on — a primitive that runs in **both** the
   lifecycle-primary and normal root-loop paths, for every entry point. So the selection layer extends
   that existing hook (§4.3's two-mechanism question is moot): no `lifecycle_driver` edit, **no new
   primitive** (resolving Q10 / the count stays **19**), entry-point-agnostic by construction. Idempotent
   (skipped when `active_skills.json` already exists) so the bounded LLM call runs ≤1×/run.
2. **Root focus via `consult_skill()`'s index, not `system_prompt._skill_catalog_section` (§4.4).** The
   root system prompt is built once at t=0, before understand/detect run, so `active_skills.json` never
   exists at prompt-build time (and `build_system_prompt` has no `project_dir` to read it). The correct
   dynamic seam is `consult_skill()` — which the root already calls on-demand and which *does* have
   `ctx.project_dir`: its index now surfaces the active set (`"active"`/`"active_note"`) once selection has
   run. `_skill_catalog_section` is left byte-identical.
3. **Recall widened for dependency-only signals.** `skill_matcher` only tokenizes `env["framework"]`, so a
   library the paper depends on but never names in that field (e.g. SDAR's vLLM, which lives in
   `pip_packages`) was missed. `skill_selection._augment_env_for_recall` folds pip/system packages into the
   recall input **in the selection layer only** — the shared matcher (and its implementer-shortlist
   consumer) is untouched. Verified: SDAR subject matter now recalls `grpo-rl-training` + `verl-rl-training`
   + `serving-llms-vllm`.

Files: new `backend/agents/rlm/skill_selection.py`; wiring in `primitives.py` (`detect_environment`,
`consult_skill`, `verify_against_rubric`) + `leaf_scorer.score_reproduction` (`skill_context` param);
tests `tests/rlm/test_skill_selection.py` (26 cases). Flags documented in `backend/agents/rlm/CLAUDE.md`.
Off-state proven byte-identical; 26 new + 177 touched-surface tests green; ruff clean; count 19.
