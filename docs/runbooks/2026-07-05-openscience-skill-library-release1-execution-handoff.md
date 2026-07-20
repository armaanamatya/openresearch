# Session Handoff — OpenScience Skill Library, **Release-1 EXECUTION** (in progress)

> **Date:** 2026-07-05 · **Author:** Opus (orchestrator/reviewer) · **Status:** Release-1
> implementation ~80% done — 4 of 5 subagent tasks accepted, T3 (wiring) running, T4
> (per-provider prompt tails) not started, final integration pending.
> **Branch:** `reconcile/grounded-self-improvement-on-main` (do NOT switch/commit/push without
> following §8). This doc is **self-contained**: a fresh session reads THIS + the design spec and
> finishes Release 1 without re-doing recon.
>
> **Source-of-truth design spec (read second):**
> `docs/history/specs/2026-07-05-openscience-skill-library-and-harness-enhancements-design.md`
> **Original design handoff (background):**
> `docs/runbooks/2026-07-05-openscience-skill-library-handoff.md`
> **OpenScience reference clone (read-only, Apache-2.0):** `/home/abheekp/openscience-ref`

---

## 0. TL;DR — pick up here

Release 1 (spec §7 / design-handoff §6) is being executed as **6 tasks** (T1..T6) delegated to
Sonnet subagents; Opus (this role) reviews every diff. State:

| Task | What | Status |
|---|---|---|
| **T1** | Vendor 40 seed `SKILL.md` playbooks → `backend/agents/rlm/skills/` + NOTICE | ✅ accepted |
| **T2** | Pure `skill_catalog.py` (loader) + `skill_matcher.py` (matcher) + tests | ✅ accepted |
| **T5** | Literature connectors + advisory rubric-input claim gate | ✅ accepted (+ Opus wired the emitter) |
| **T6** | "Provenance & Evidence" markdown section in `report.py` | ✅ accepted |
| **T3** | `consult_skill` primitive + registry + `detect_environment` hook + system-prompt catalog section + shortlist block | ✅ accepted + integrated (§3) |
| **T4** | WS① per-provider prompt tails (`OPENRESEARCH_PROVIDER_PROMPTS`) | ⛔ **NOT STARTED** — full spec + verbatim content in §4 |
| **INT** | Final integration: full `pytest` + `ruff`, OFF-byte-identical checks, commit | ⛔ pending — §5/§6 |

**Immediate next steps for the fresh session:**
1. **Dispatch T4** (Sonnet) using the §4 spec — the prompt tail **content is pre-authored verbatim
   in §4.3; copy it, do not re-write it**. T4 edits `system_prompt.py` (which T3 already edited and
   is settled — just read the current file).
2. Review T4's diff (brace invariant + OFF byte-identical per §3 item 5 / §4.5), then run **final
   integration** (§5): the Release-1 sweep + T4's test file + a broad `tests/rlm/ tests/services/`
   regression + `ruff@0.15.16`.
3. Commit per §6 (spec + both handoffs + all Release-1 code together; no co-author trailer; push
   only to `deepinvent` **and only if asked**).

**Current green state (as of this handoff):** T1/T2/T3/T5/T6 all accepted + integrated. A **137-test
Release-1 sweep passes** and **`ruff@0.15.16` is clean** across every touched file (exact commands in
§5). Only **T4 + its test pass + the commit** remain.

---

## 1. Standing rules (from operator memory — do not violate)

- **Opus designs + reviews every diff; Sonnet executes.** Delegate implementation to Sonnet
  subagents (`Agent` tool, `subagent_type: general-purpose`, `model: sonnet`, background). Opus
  reviews the actual **diff**, not the summary. Use the **`/implement` skill** for implementation
  work; **never** `implement_codex` / `codex:*`.
- **Default-OFF, byte-identical-when-off, TDD, hermetic OFF+ON tests, `ruff@0.15.16` clean.** This
  program flips **no** defaults (a default-flip needs ≥3 paired A/B runs + the grader-σ gate — out
  of scope here).
- **Env-flag gate convention for ALL new flags:** `os.environ.get("FLAG","").strip().lower() in
  ("1","true","yes")` — **NOT** `"on"`. (Some *existing* primitives like `inspect_repository`
  include `"on"`; the new skill/literature/report flags deliberately use the 3-tuple per the design.)
- **North star:** *copy faithfully · adapt surgically · ground natively · reuse existing analogs ·
  default-OFF.* The 292 upstream playbooks are real + Apache-2.0 → **copy verbatim**, never
  regenerate. Native code is written against real, cited seams (never invented APIs).
- **Git:** branch `reconcile/grounded-self-improvement-on-main`; push **only** to `deepinvent`
  (`Deepinvent/scientific_article_generator`), never `origin`/`openresearch`, never on `main`
  without branching. **No `Co-Authored-By`/AI-attribution trailer.** Author = local config
  (lolout1 / appradhann@gmail.com). No Conventional-Commit prefixes; descriptive present-tense
  headline. Commit **infrequently** (milestones). Commit/push only when the user asks.
- **Concurrency hygiene:** subagents editing disjoint files may run in parallel; two agents must
  never edit the same file concurrently. Each subagent runs ONLY its own new test files (not the
  full suite). Opus runs the full sweep at integration.

---

## 2. DONE work (accepted) — files, flags, decisions, verdicts

### T1 — Skill library vendored ✅
- **Added:** `backend/agents/rlm/skills/<category>/<name>/SKILL.md` (+ `references/`, `scripts/`)
  — **40 skills** across ml-inference(5), ml-training(27), cloud-compute(1: skypilot),
  research(2), writing(2), visualization(2), scholar-evaluation(1). Plus
  `backend/agents/rlm/skills/NOTICE` (Apache-2.0 attribution to Synthetic Sciences/OpenScience).
- **Not a Python package** — no `__init__.py` under `skills/` (it is DATA the loader globs).
- **Decisions/fixes (Opus, during review):** the upstream corpus had a global scraper splice
  (`Orchestra` → `Synthetic Sciences`). Fixed 3 body/tag/reference instances (`verl`, `ray-train`,
  `skypilot`, `openrlhf/references`) to the intended word. **Two files needed corpus repair so all
  40 index (see T2):**
  - `ml-training/ray-train/SKILL.md` frontmatter had invalid YAML `dependencies: [ray[train], …]`
    (unescaped `[`); quoted to `["ray[train]", …]` (documentation-only field, content preserved).
  - `scholar-evaluation/SKILL.md` shipped with **no frontmatter fence** upstream; Opus prepended a
    faithful minimal frontmatter (`name: scholar-evaluation`, `category: scholar-evaluation`,
    `description` lifted verbatim from the file's own Overview, content-derived `tags`).
- **Verify:** `find backend/agents/rlm/skills -name SKILL.md | wc -l` → 40; `grep -rn "Sciencest"
  backend/agents/rlm/skills` → clean.

### T2 — Pure catalog loader + matcher ✅
- **Added:** `backend/agents/rlm/skill_catalog.py`, `backend/agents/rlm/skill_matcher.py`,
  `tests/rlm/test_skill_catalog.py`, `tests/rlm/test_skill_matcher.py`.
- **`skill_catalog.py` public API (import; do not reimplement):**
  - `SkillMeta(name, description, category, tags: tuple, path: Path)` (frozen). Skill dir =
    `meta.path.parent` (`references/`, `scripts/` live under it). Catalog is keyed by frontmatter
    `name` (NOT dir — e.g. `ml-inference/vllm/` → `serving-llms-vllm`).
  - `load_catalog(skills_dir: Path | None = None) -> dict[str, SkillMeta]` (default = vendored dir,
    memoized per resolved dir + lock, fail-soft: malformed/injection/missing-name skipped+warned).
  - `get_skill_body(name, skills_dir=None) -> str | None` (re-reads body, strips injection lines).
  - `fuzzy_did_you_mean(query, names, *, limit=5) -> list[str]` (ported `fuzzyScore`).
  - `group_by_category(catalog) -> dict[str, list[SkillMeta]]`; `clear_cache()` (test-only).
- **`skill_matcher.py` public API:** `SkillMatch(domain: str, skill_names: tuple, reasons: tuple)`
  (frozen); `match_skills(claim_map: Mapping, environment_spec: Mapping, catalog, *, top_k=8) ->
  SkillMatch`. Takes plain dicts (`.model_dump()` output). Deterministic domain table
  (`ml-rl`/`ml-inference`/`ml-vision`/`ml-nlp`/`ml-interp`, non-ML only on strict dominance, else
  `ml-other`). Fail-soft → empty match.
- **Verdict:** clean, faithful to `openscience-ref` `skill.ts`/`tool/skill.ts`. All 40 index after
  the T1 corpus repair; the test asserts `len(catalog) == 40` and both repaired skills present.
- **Test:** `.venv/bin/python -m pytest tests/rlm/test_skill_catalog.py
  tests/rlm/test_skill_matcher.py -q` → 39 passed.

### T5 — Literature connectors + advisory claim gate ✅ (+ Opus wired the emitter)
- **Added:** `backend/services/knowledge/__init__.py`, `backend/services/knowledge/connectors/`
  (`__init__.py`, `base.py`, `arxiv.py`, `openalex.py`, `semantic_scholar.py`),
  `backend/agents/rlm/literature_claim_gate.py`, `tests/services/knowledge/` (12 tests),
  `tests/rlm/test_literature_claim_gate.py` (11 tests).
- **Edited:** `backend/agents/rlm/rubric_gen.py` — `generate_rubric_tree(...)` gained append-only
  `project_dir: Path | None = None`, `emit_warning: Any | None = None` kwargs + a
  `_apply_literature_claim_gate(...)` call right before `return tree`. Byte-identical when off
  (callee returns `[]` before any import/network when the flag is unset).
- **Flags:** `OPENRESEARCH_LITERATURE_GROUNDING` (gates connectors' network use),
  `OPENRESEARCH_LITERATURE_CLAIM_GATE` (gates the gate). Advisory only — never mutates the rubric;
  emits `run_warning code="literature_claim_ungrounded"` per uncorroborated claim.
- **Connectors:** ported 1:1 from `openscience-ref/backend/cli/src/science/connectors/literature/`
  (`arxiv`, `openalex`, `semantic-scholar`); key-free (S2 optional `SEMANTIC_SCHOLAR_API_KEY`);
  injectable transport (`fetch_json`/`fetch_text`, httpx default, fail-soft → hermetic tests, no
  live network; confirmed under pytest-socket). ~600-char snippets. Sibling disk cache reusing
  `primitive_cache.make_key` (did NOT edit the closed `primitive_cache.CACHEABLE_PRIMITIVES`).
- **Opus wiring (the last mile):** `backend/agents/rlm/run.py` at the single
  `generate_rubric_tree(...)` call site (was ~line 3550, in `run_pipeline_rlm`) now passes
  `project_dir=project_dir` and a real emitter adapter:
  ```python
  def _lit_emit(code: str, message: str) -> None:
      emit(build_run_warning_event(level="warn", code=code, message=message))
  # ... emit_warning=_lit_emit if callable(emit) else None
  ```
  Ruff-safe (nested `def`, not a lambda), byte-identical when the gate flag is off (gate returns
  before ever calling `emit_warning`). `emit`/`build_run_warning_event` were already in scope.
- **Test:** `.venv/bin/python -m pytest tests/rlm/test_literature_claim_gate.py
  tests/services/knowledge/ -q` → 23 passed (incl. a blocked-socket fail-soft assertion);
  `tests/rlm/test_rubric_gen.py` → 8 passed (no regression).

### T6 — "Provenance & Evidence" report section ✅
- **Edited:** `backend/agents/rlm/report.py` (+148 lines, purely additive) — added
  `_evidence_report_section_enabled()` and `_render_evidence_section(report)`, plus one gated call
  site before the footer in `_render_markdown` (~line 2622).
- **Added test:** `tests/rlm/test_evidence_report_section.py` (21 tests).
- **Flag:** `OPENRESEARCH_EVIDENCE_REPORT_SECTION`. Renders `report.evidence_bundle` (sha256
  receipt — real keys `attempt_id`/`metrics_sha256`/`code_tree_digest`/`ledger_sequence`/
  `coordinates`, verified against `evidence_bundle.py` + `report.py:1143`) + `report.validation`
  (status/veto_set/predicate table). No `claim_grounding` field exists on `RLMFinalReport` (it's
  stamped on the serialized JSON dict only) — correctly NOT rendered.
- **Copy discipline (enforced by test):** a clean panel reads "no suspicion raised", never
  "verified correct".
- **Verdict:** verified against real schemas; byte-identical off (test proves a fully-populated
  report renders identically with the flag unset); fail-soft (whole section in try/except).

---

## 3. T3 — wiring (RUNNING) — REVIEW CHECKLIST

**T3 scope (files):** `backend/agents/rlm/primitives.py`, `backend/agents/rlm/system_prompt.py`,
`backend/agents/baseline_implementation.py`, `tests/rlm/test_registry.py`, `CLAUDE.md`, new
`tests/rlm/test_consult_skill.py`. Flag: **`OPENRESEARCH_SKILLS`** (master). Disjoint from T4 only
by sequencing (both touch `system_prompt.py`).

**What T3 was told to build (verify each in the diff — do NOT trust the summary):**

1. **`consult_skill(name="", category="", *, ctx) -> dict` primitive** in `primitives.py`, modeled
   exactly on `inspect_repository` (`primitives.py` ~8855) / `read_context_map` (~8835):
   flag-gated `{"status":"disabled"}` when `OPENRESEARCH_SKILLS` off; fail-soft (returns
   `{"status":"error",...}`, never raises); lazy imports of the skill modules. On-state:
   `name` found → `{status:ok, name, category, tags, body, references, scripts}`; not found →
   `{status:not_found, did_you_mean:[...]}`; `category` → browse; neither → category index.
   **Scripts→code/:** when a found skill has `scripts/` AND `ctx.project_dir/"code"` exists, copies
   into `code/skill_scripts/<name>/` (fail-soft), surfaces dest paths.
   - ☐ **Registered** in `PRIMITIVE_REGISTRY` (~8908) AND `PRIMITIVE_DESCRIPTIONS` (~8929).
2. **Registry test:** ☐ `tests/rlm/test_registry.py::EXPECTED` gains `"consult_skill"` → **19
   entries** (was 18). The existing asserts (REGISTRY == DESCRIPTIONS == build_custom_tools ==
   EXPECTED) then enforce coverage. Run: `.venv/bin/python -m pytest tests/rlm/test_registry.py -q`.
3. **CLAUDE.md counts:** ☐ the primitive-count claims ("bound `custom_tools` set is **18**",
   "PRIMITIVE_REGISTRY actually holds **18**", the aux-primitive enumeration listing
   `heartbeat, recommend_next_tool, resolve_gpu_requirements, codex_repair, read_context_map,
   inspect_repository`) bumped to **19** with `consult_skill` added. **Verify ONLY registry-count
   "18"s changed** — not dates/other counts. (Review this edit carefully; it was flagged as
   delicate.)
4. **`detect_environment` matcher hook** (`primitives.py` ~1108-1214): after `spec_dict` is fully
   built (after the runtime-capacity annotation, before `result = _with_outcome(...)`), flag-gated
   + fail-soft: `match_skills(method_spec, spec_dict, load_catalog())` → stash on
   `spec_dict["extra"]["skill_match"]` + persist `rlm_state/skill_match.json`. ☐ **Cache-payload
   correctness:** `_payload["skills"] = True` added **only when the flag is on** (so the OFF cache
   key is byte-identical; a matcher error must never block detection).
5. **System-prompt catalog section** (`system_prompt.py` ~578 `build_system_prompt`): a
   `_skill_catalog_section()` (categories by count desc, ≤3 example names each, + `consult_skill`
   contract) appended to `parts` **only when `OPENRESEARCH_SKILLS` on**, mirroring the
   `_CONTEXT_MAP_SECTION` append (~line 629), wrapped in try/except (omit on error).
   - ☐ **CRITICAL brace invariant** (`system_prompt.py` ~658-676): the whole body is brace-escaped
     (`{`→`{{`) then the single `[[OPENRESEARCH_CUSTOM_TOOLS_SECTION]]` marker → `{custom_tools_section}`,
     guarded by `assert count == 1`. The new section text MUST NOT contain that marker or a raw
     `{custom_tools_section}`. **Verify the ON-flag prompt still has exactly one
     `{custom_tools_section}`** (the T3 test must assert this).
6. **Matched-shortlist guidance block** (`baseline_implementation.py` `_compute_constraint_guidance`
   ~2300): a `_skill_shortlist_block(project_dir)` reading `rlm_state/skill_match.json`, house
   `_SCREAMING_SNAKE_BLOCK` voice (like `_RL_SCAFFOLD_BLOCK` ~1422), appended near the
   `_load_paper_override` point (~1944), gated `OPENRESEARCH_SKILLS`, fail-soft. ☐ Off ⇒
   `_compute_constraint_guidance` output byte-identical.
7. **Tests** `tests/rlm/test_consult_skill.py`: ☐ OFF (`consult_skill` returns disabled;
   `build_system_prompt` has NO catalog section + still `.count("{custom_tools_section}")==1`;
   guidance no block) + ON (body returned; did-you-mean on miss; catalog section present + brace
   assert holds; shortlist renders from a fixture `skill_match.json`).

**Review commands after T3 lands:**
```bash
cd /home/abheekp/openresearch
git diff --stat backend/agents/rlm/primitives.py backend/agents/rlm/system_prompt.py \
  backend/agents/baseline_implementation.py tests/rlm/test_registry.py CLAUDE.md
git diff backend/agents/rlm/system_prompt.py          # eyeball the brace-escape region
grep -n "18\|19" tests/rlm/test_registry.py            # EXPECTED must be 19
.venv/bin/python -m pytest tests/rlm/test_registry.py tests/rlm/test_consult_skill.py -q
uvx ruff@0.15.16 check backend/agents/rlm/primitives.py backend/agents/rlm/system_prompt.py \
  backend/agents/baseline_implementation.py tests/rlm/test_consult_skill.py
```
**T3 verdict:** ✅ **ACCEPTED + integrated.** Every checklist item verified against the diff:
`consult_skill` (`primitives.py` ~8946 — faithful disabled-sentinel + progressive disclosure +
`scripts/`→`code/skill_scripts/<name>/` copy, fail-soft per file); registry = **19** (`EXPECTED` +
**4 collateral count-tests** bumped: `tests/test_claude_md_fidelity.py`,
`tests/rlm/test_integration_custom_tools.py`, `tests/rlm/test_run.py`,
`tests/rlm/test_inspect_repository.py`); CLAUDE.md (only the 2 registry-count sentences changed);
`detect_environment` hook + `_payload["skills"]=True` **only-when-on** cache gating (a dedicated
regression test proves the cache key separates flag states); system-prompt catalog section (contains
**zero `{`/`}`** → brace invariant safe; ON and OFF tests both assert exactly one
`{custom_tools_section}`); shortlist block in `_compute_constraint_guidance` (which already carried a
`project_dir` param). **Opus integration fix:** `tests/rlm/test_run.py::_fake_generate` mock
signature widened to accept the `project_dir`/`emit_warning` kwargs the `run.py` literature-gate
wiring (§2 T5) now passes — the single failure T3 correctly flagged as caused by the concurrent T5
signature change, not a T3 defect. **Post-fix:** 137-test Release-1 sweep green, `ruff` clean.

---

## 4. T4 — WS① per-provider prompt tails (NOT STARTED) — full spec

**Goal:** give each root-model family a small system-prompt tail addressing **openresearch's own
documented failure modes** (NOT openscience's mapping). Flag **`OPENRESEARCH_PROVIDER_PROMPTS`**,
default-OFF, byte-identical when off. Additive on top of the existing `root_model.prompt_addendum`
(do not modify existing addenda). **Runs AFTER T3** (shared file `system_prompt.py`).

### 4.1 Seams (verified)
- `backend/agents/rlm/models.py`: `RootModel` (~line 51) has `.key` (e.g. `"claude-oauth"`,
  `"azure-foundry"`, `"gpt-5"`), `.prompt_addendum` (already appended verbatim in
  `build_system_prompt` ~line 643), `.paper_validated`. `resolve_root_model(name)` (~684) →
  `RootModel`. Registry keys present: `gpt-5`, `qwen3-coder`, `kimi-k2.5`, `claude`, `claude-oauth`,
  `opus-foundry`, `sonnet-foundry`, `qwen3-coder-featherless`, `azure-gpt-4o`, `azure-foundry`.
- `build_system_prompt(*, context_metadata, root_model, include_hints=True)` in `system_prompt.py`
  (~578) composes `parts: list[str]`, appends `root_model.prompt_addendum` (~643), then brace-escapes
  the whole body and restores the single `{custom_tools_section}` placeholder (~666, asserted ==1).
  **The per-provider tail appends to `parts` the same way as the `_CONTEXT_MAP_SECTION` gate
  (~629), keyed off `root_model.key`, gated on `OPENRESEARCH_PROVIDER_PROMPTS`.**

### 4.2 Family classifier (by `RootModel.key`)
- **Sonnet-family (anti-premature-stop tail):** `claude-oauth`, `claude`, `sonnet-foundry`.
  (These are the documented degenerate-`FINAL_VAR`-without-implementing roots — see
  `project_lifecycle_driver`, `project_reasoning_chat_root_guardrails`, the OAuth degenerate-loop
  detector.)
- **Reasoning-chat-family (anti-churn tail):** `azure-foundry` (grok/foundry). (Documented churn /
  stub-metrics / over-continuation — see `project_reasoning_chat_root_guardrails`,
  `project_foundry_gptchat_root_not_executor`.)
- **No tail (empty):** `gpt-5` (paper-validated, reliable — do not perturb), `opus-foundry` (Opus,
  the reliable-root fix), `azure-gpt-4o`, `qwen3-coder`, `qwen3-coder-featherless`, `kimi-k2.5`
  (qwen/kimi already carry `_QWEN_PROMPT_ADDENDUM`; leave it).

### 4.3 Tail CONTENT — **COPY VERBATIM into T4 (do not re-author)**

**Sonnet-family tail:**
```
═══════════════════════════════════════════════════════════════
  MODEL-SPECIFIC RELIABILITY (Sonnet/Claude root)
═══════════════════════════════════════════════════════════════

Your known failure mode on this harness is STOPPING TOO EARLY — calling FINAL_VAR
(or repeatedly refusing to continue) BEFORE any real experiment has run. Guard against it:

1. You MUST call implement_baseline AND run_experiment and obtain a real, non-empty
   metrics result BEFORE any FINAL_VAR. A report without a successful run_experiment
   scores 0 — a hypothetical, described, or summarized result is worthless here.
2. Do not narrate a plan and stop. Each iteration must ADVANCE a concrete stage
   (understand_section → detect_environment → implement_baseline → run_experiment →
   verify_against_rubric). If you are about to summarize instead of act, call the next
   primitive instead.
3. FINAL_VAR is only legitimate AFTER verify_against_rubric has scored a real run (or the
   wall-clock is nearly exhausted). Until then, keep working — repair and re-run.
```

**Reasoning-chat-family tail:**
```
═══════════════════════════════════════════════════════════════
  MODEL-SPECIFIC RELIABILITY (reasoning-chat root, e.g. grok/foundry)
═══════════════════════════════════════════════════════════════

Your known failure mode on this harness is CHURN — re-planning, re-reading, or
re-summarizing the same state across iterations without producing runnable code, and
emitting placeholder/stub metrics instead of a real measurement. Guard against it:

1. Write REAL code that trains/evaluates on the ACTUAL data and produces real metric keys
   — never placeholder metrics (e.g. total_length / chunk_count) and never a stub that
   reports success without running.
2. Every iteration must call a STATE-CHANGING primitive (implement_baseline / run_experiment
   / verify_against_rubric). Do not spend an iteration only re-reading the paper or
   re-emitting a plan you already have.
3. When a stage is done, MOVE ON. Reproduce the FULL paper (every required model / dataset /
   baseline in scope), then call FINAL_VAR once — do not loop.
```

### 4.4 Suggested implementation shape
In `system_prompt.py`: a module-level `_SONNET_RELIABILITY_TAIL` / `_REASONING_CHAT_RELIABILITY_TAIL`
(the two blocks above) + `_provider_prompt_tail(root_model) -> str` mapping `.key` → tail (else "").
In `build_system_prompt`, after the `prompt_addendum` append, add:
```python
if _os.environ.get("OPENRESEARCH_PROVIDER_PROMPTS", "").strip().lower() in ("1","true","yes"):
    tail = _provider_prompt_tail(root_model)
    if tail:
        parts.append(tail)
```
(Brace-safe: the blocks contain no `{`/`}`.)

### 4.5 Tests (`tests/rlm/test_provider_prompts.py`)
- OFF: `build_system_prompt` for a `claude-oauth` root has NO reliability tail (assert a
  distinctive phrase absent) and still `.count("{custom_tools_section}") == 1` — byte-identical.
- ON (`monkeypatch.setenv("OPENRESEARCH_PROVIDER_PROMPTS","1")`): `claude-oauth`/`sonnet-foundry`
  → Sonnet tail present; `azure-foundry` → churn tail present; `gpt-5` → neither tail; the brace
  assert still holds for each. Build `RootModel` via `resolve_root_model("<key>")`.

### 4.6 T4 dispatch prompt skeleton (Sonnet)
Same house rules as §1. "Add `OPENRESEARCH_PROVIDER_PROMPTS` per-provider reliability tails to
`build_system_prompt` in `backend/agents/rlm/system_prompt.py`, keyed off `RootModel.key` per the
family map in §4.2, using the VERBATIM content in §4.3 (do not re-author). Preserve the brace-escape
invariant (§3 item 5). Add `tests/rlm/test_provider_prompts.py` (OFF byte-identical + ON per §4.5).
Run only that test file + `tests/rlm/` prompt tests; `ruff@0.15.16` clean. Do not touch `models.py`
addenda. Report the diff + the ON-flag placeholder count."

---

## 5. Final integration (INT) — after T3 + T4 accepted

1. **Full targeted sweep** (the suite is socket-hermetic; `-n auto` needs requirements-dev):
   ```bash
   cd /home/abheekp/openresearch
   .venv/bin/python -m pytest tests/rlm/test_skill_catalog.py tests/rlm/test_skill_matcher.py \
     tests/rlm/test_consult_skill.py tests/rlm/test_registry.py tests/rlm/test_provider_prompts.py \
     tests/rlm/test_literature_claim_gate.py tests/services/knowledge/ \
     tests/rlm/test_evidence_report_section.py tests/rlm/test_rubric_gen.py \
     tests/rlm/test_render_markdown_telemetry.py -q
   # then a broad regression pass on the touched modules' neighbors:
   .venv/bin/python -m pytest tests/rlm/ tests/services/ -q
   ```
2. **Ruff** across everything new/edited:
   ```bash
   uvx ruff@0.15.16 check backend/agents/rlm/skill_catalog.py backend/agents/rlm/skill_matcher.py \
     backend/agents/rlm/literature_claim_gate.py backend/services/knowledge/ \
     backend/agents/rlm/primitives.py backend/agents/rlm/system_prompt.py \
     backend/agents/baseline_implementation.py backend/agents/rlm/report.py \
     backend/agents/rlm/rubric_gen.py backend/agents/rlm/run.py tests/rlm/ tests/services/knowledge/
   ```
3. **OFF byte-identical invariants** (all flags unset ⇒ current harness unchanged):
   - `PRIMITIVE_REGISTRY` == 19 (consult_skill always registered; returns `{"status":"disabled"}`
     off — the ONLY permanent structural change, mirroring how `inspect_repository` made it 18).
   - `build_system_prompt(...)` with `OPENRESEARCH_SKILLS` + `OPENRESEARCH_PROVIDER_PROMPTS` unset:
     no catalog section, no reliability tail, exactly one `{custom_tools_section}`.
   - `_compute_constraint_guidance` output unchanged with `OPENRESEARCH_SKILLS` unset.
   - `_render_markdown` unchanged with `OPENRESEARCH_EVIDENCE_REPORT_SECTION` unset.
   - `generate_rubric_tree` produces the same tree with the literature flags unset.
4. **Skill scripts copy sanity:** if T3 ships `scripts/` into `code/`, confirm it's fail-soft when
   `code/` is absent.

---

## 6. Commit guidance (only when the user asks)

- The **spec + BOTH handoffs are currently UNCOMMITTED** — commit them WITH the Release-1 code in
  one milestone commit (or when asked). Files: `docs/history/specs/2026-07-05-openscience-*.md`,
  `docs/runbooks/2026-07-05-openscience-skill-library-handoff.md`, THIS doc, all `skills/`,
  `skill_catalog.py`, `skill_matcher.py`, `literature_claim_gate.py`, `backend/services/knowledge/`,
  and the edits to `primitives.py`/`system_prompt.py`/`baseline_implementation.py`/`report.py`/
  `rubric_gen.py`/`run.py`/`CLAUDE.md`/`test_registry.py` + new tests.
- **`git status` at session start already had unrelated untracked/modified files** (`backend/app.py`,
  `backend/config.py`, `frontend/.../lab-sidebar.tsx`, `backend/routes/external_runs.py`,
  `backend/services/external_monitor/`, many `runs/…`, `image.webp`, `error.log`, etc.) — **do NOT
  commit those**; stage only the Release-1 paths above.
- **CLAUDE.md maintenance:** add a Release-1 skill-library bullet to the feature-flags section
  (mirroring the existing default-OFF flag entries) documenting `OPENRESEARCH_SKILLS`,
  `OPENRESEARCH_PROVIDER_PROMPTS`, `OPENRESEARCH_LITERATURE_GROUNDING`,
  `OPENRESEARCH_LITERATURE_CLAIM_GATE`, `OPENRESEARCH_EVIDENCE_REPORT_SECTION`, and bump the
  primitive count to 19 (T3 does the count; add the flag prose at integration/commit).
- No co-author trailer; descriptive present-tense headline (e.g. "Add the OpenScience skill library
  + Release-1 harness enhancements (consult_skill, literature grounding, evidence report, per-provider
  prompts) — all default-OFF").
- Push **only to `deepinvent`**, only if asked.
- **Memory:** update `project_openscience_skill_port` (in `MEMORY.md`) from
  "UNCOMMITTED/unimplemented" → "Release-1 implemented (T1–T6 + integration), default-OFF, <commit
  sha>; Releases 2–5 pending".

---

## 7. Flags introduced by Release 1 (all default-OFF, `("1","true","yes")`)

| Flag | Gates | Task |
|---|---|---|
| `OPENRESEARCH_SKILLS` | skill library load, matcher hook in `detect_environment`, `consult_skill` advertisement + catalog section, shortlist block | T3 |
| `OPENRESEARCH_PROVIDER_PROMPTS` | per-provider reliability prompt tails | T4 |
| `OPENRESEARCH_LITERATURE_GROUNDING` | literature connectors' network use | T5 |
| `OPENRESEARCH_LITERATURE_CLAIM_GATE` | advisory claim-grounding gate on the rubric input | T5 |
| `OPENRESEARCH_EVIDENCE_REPORT_SECTION` | "Provenance & Evidence" markdown section | T6 |

`consult_skill` is ALWAYS registered (returns `{"status":"disabled"}` when `OPENRESEARCH_SKILLS`
off) — registry count is permanently 19, exactly the `inspect_repository` precedent. Reused
existing flags: `OPENRESEARCH_EXTERNAL_VALIDATOR`, `OPENRESEARCH_VALIDATOR_PANEL_N`,
`OPENRESEARCH_REPORT_CLAIM_GATE`.

---

## 8. Later releases (spec §7 — NOT this session)

- **R2 — Native cloud skills** (spec §5.E): author `SKILL.md`s against the real cell contract
  (`gpu_cell_runner.run_matrix`, `cells.json` axes, `services`/`command`/`metrics_source`, the
  three multi-GPU paths). Seams in the design-handoff §3 "native-cloud".
- **R3 — Specialist + reliability depth:** domain-workflow prompt (`OPENRESEARCH_SKILL_DOMAIN_WORKFLOW`)
  + critique-gate REUSING `external_validator.run_validation_panel`; progress journal; bounded-iter
  exit menu; angle-scoped panelists.
- **R4 — Self-improvement + honesty:** usage→outcome recipe/lesson weighting; structured executor
  reports; operator standing-notes; findings ledger.
- **R5 — Full-catalog backfill + new capability:** copy remaining non-ML categories as inert
  library content; blind reviewer (`OPENRESEARCH_BLIND_REVIEWER`); grounded figure pipeline.
- Each release: its own `writing-plans` plan; default-OFF; Opus reviews every diff.

---

## 9. Grounded seams appendix (verified by direct read this session)

- `detect_environment` primitive — `primitives.py` ~1108-1214 (cached via `primitive_cache`;
  builds `spec_dict = spec.model_dump()`, appends a runtime-capacity `ENV-RT1` assumption, then
  `_with_outcome` + `_cache.put`). `_payload` already carries `repo_first`/`repo_commit` (precedent
  for adding `skills`). `EnvironmentSpec.extra` overflow slot = `schemas.py` ~371; `PaperClaimMap` =
  `schemas.py` ~88.
- `PRIMITIVE_REGISTRY` = `primitives.py` ~8908; `PRIMITIVE_DESCRIPTIONS` ~8929;
  `read_context_map` ~8835 / `inspect_repository` ~8855 = the disabled-sentinel model for
  `consult_skill`. `tests/rlm/test_registry.py::EXPECTED` currently 18 entries.
- `build_system_prompt` = `system_prompt.py` ~578; flag-gated section append precedent =
  `_CONTEXT_MAP_SECTION` (~629) / `_REPO_AWARE_SECTION` (~656); `prompt_addendum` append ~643;
  brace-escape + single `{custom_tools_section}` placeholder + `assert count==1` ~666-675.
- `_HARNESS_CODE_HELPERS` (~65) / `_copy_harness_helpers_to_code_root` (~82) /
  `refresh_harness_helpers` (~98) in `baseline_implementation.py` = the code-helper copy pattern.
  `_compute_constraint_guidance` ~2300; `_load_paper_override` ~1944 (shortlist-injection precedent);
  house guidance-block style `_RL_SCAFFOLD_BLOCK` ~1422 / `_SDAR_BASELINES_BLOCK` ~1499.
- `report.py`: `_render_markdown` ~2422; `RLMFinalReport.evidence_bundle` field ~61 (populated
  ~1133-1152); `report.validation` field ~199 (stamped ~1979-2006). `evidence_bundle.py` receipt
  keys = `attempt_id`/`ledger_sequence`/`metrics_sha256`/`code_tree_digest`/`artifact_dir`/
  `coordinates` (CLAUDE.md's `{attempt,…}` is loose shorthand — the real key is `attempt_id`).
- `rubric_gen.generate_rubric_tree` = `rubric_gen.py` ~128 (single caller `run.py` ~3550, in
  `run_pipeline_rlm`, where `emit`/`build_run_warning_event`/`project_dir` are in scope).
- `external_validator.run_validation_panel` ~451 / `PredicateVerdict` ~36 / `ValidatorVerdict` ~46
  (the REUSE target for R3's critique gate — do not build a new panel).
- `models.py`: `RootModel` ~51 (`.key`/`.prompt_addendum`), registry ~182, `resolve_root_model`
  ~684.
- OpenScience sources ported: `openscience-ref/backend/cli/src/skill/skill.ts` (loader),
  `.../src/tool/skill.ts` (fuzzy + browse + injection strip),
  `.../src/science/connectors/literature/{arxiv,openalex,semantic-scholar,shared}.ts`,
  `.../src/session/prompt/{anthropic,beast,qwen}.txt` (per-provider tail PATTERN, R-content
  re-authored to openresearch failure modes).

---

## 10. Delegation pattern used (reproduce for T4)

`Agent` tool, `subagent_type: "general-purpose"`, `model: "sonnet"`, `run_in_background: true`.
Each prompt: repo+branch+house-rules block (§1) → confirmed dependency APIs → exact seams (symbol +
approx line, tell them to grep) → deliverables → scope discipline ("touch ONLY these files; another
agent owns X concurrently") → tests (OFF byte-identical + ON) → "report back = raw data for the
reviewer: files+lines, flags, exact test cmd + pass/fail, ruff result, deviations + reasons." Opus
then reviews the DIFF (`git diff <file>`), verifies against real schemas (not the summary), fixes
corpus/integration gaps, and only then accepts.
