# OpenScience-Inspired Skill Library + Harness Enhancements — DESIGN

> **Status:** DRAFT (brainstorming → spec). Author: Opus (design + review). Date: 2026-07-05.
> Reference repo analyzed: `synthetic-sciences/openscience` (Apache-2.0), cloned read-only at
> `/home/abheekp/openscience-ref`. Every seam cited below was verified by direct file read
> against `/home/abheekp/openresearch` — no invented APIs.
> Memory: [[project_rlm_pivot]], [[project_lifecycle_driver]],
> [[project_reasoning_chat_root_guardrails]], [[project_sdar_gcp_harness_refactor]],
> [[feedback_solution_quality]].

## 1. Summary

Port the genuinely useful, openresearch-fitting parts of **OpenScience** — an open-source
(Apache-2.0) AI research workbench with 292 expert `SKILL.md` playbooks, domain-specialist
agents, per-provider prompt routing, and a claim→evidence review culture — into openresearch's
RLM paper-reproduction harness.

The centerpiece is a **reusable, framework/technique-keyed skill library** so the agent gets a
tested expert playbook (vLLM, verl, GRPO, DeepSpeed, FSDP, PEFT, TRL, eval harnesses, cloud
orchestration, and every scientific domain) surfaced *automatically by the paper's subject
matter*, plus a `consult_skill` primitive for on-demand deep-dive. Around it, four grounded
workstreams the analysis surfaced as high-value:

- **① Root-reliability** — per-provider system-prompt tails, a narrative progress journal,
  a bounded-iteration exit menu, and angle-scoped validator panelists. Targets openresearch's
  single most-documented failure: weak/keyless roots degenerating.
- **② Knowledge-grounding** — key-free literature connectors (arXiv / OpenAlex / Semantic
  Scholar) + a claim-grounding gate on the **rubric input**, extending the anti-hallucination
  line onto the paper's own claimed numbers and baselines.
- **③ Self-improvement** — a usage→outcome feedback loop that fixes recency-only recipe/lesson
  selection, structured executor reports, and durable operator standing-notes.
- **④ Deliverable & honesty** — surface the evidence openresearch already computes into the
  human report, an itemized validator findings ledger, an agentic **blind reviewer** sub-agent
  (the one genuinely new capability), and a grounded figure pipeline.

**Guiding principle (the north star, per the operator's explicit "optimize for our repo, do not
hallucinate" mandate):** *copy faithfully · adapt surgically · ground natively · reuse existing
analogs · default-OFF.* The 292 playbooks are real and Apache-2.0 — we **copy them verbatim**
(with attribution), never regenerate them by hand; that is the anti-hallucination guarantee for
ported content. Native content (openresearch's own cell/cloud wiring, the report renderer) is
written against real, cited seams. Every module reuses an existing openresearch analog where one
exists (`external_validator`, `claim_grounding`, `context_map`, `_HARNESS_CODE_HELPERS`,
`_compute_constraint_guidance`) rather than duplicating it.

**Discipline (standing repo convention):** all changes ship **default-OFF, byte-identical when
the flag/field is absent**, TDD, hermetic ON+OFF tests, `ruff` clean. Unset ⇒ the current harness
is unchanged to the byte.

## 2. Problem statement

openresearch reproduces ML papers autonomously (no human-in-loop per run), but three gaps recur:

1. **No shared technique knowledge.** Framework-specific expertise (how to run vLLM with weight
   sync, wire GRPO reward variance, shard with FSDP, avoid the DeepSpeed ZeRO footguns) is
   re-derived per paper or hand-written per-paper in `paper_hints.py` and the opt-in
   `_RL_SCAFFOLD_BLOCK` / `_SDAR_BASELINES_BLOCK` (`baseline_implementation.py:1422-1566`). These
   are effectively bespoke one-paper "skills" with no reuse across the many papers using the same
   technique. There is no library, no catalog, no on-demand lookup.
2. **Weak-root degeneration.** Documented repeatedly ([[project_lifecycle_driver]],
   [[project_reasoning_chat_root_guardrails]]): `claude-oauth` loops `FINAL_VAR` without
   implementing; `gpt-chat`/grok churn. openresearch drives **7 root families**
   (`gpt-5`, `qwen3-coder`, `kimi-k2.5`, `claude`, `claude-oauth`, `azure-gpt-4o`,
   `azure-foundry`/`grok`) through **one** `system_prompt.py`.
3. **Trust stops at the run boundary.** openresearch has strong *output-side* evidence machinery
   (`evidence_gate`, `eval_provenance`, `evidence_bundle`, `external_validator`, `claim_grounding`)
   but (a) no grounding of the *paper's own* claimed numbers/baselines that become hard rubric
   targets in `rubric_gen.py:59`, and (b) none of the computed evidence reaches the human — the
   `final_report.md` renderer (`report.py:_render_markdown`, ~line 2422) never references
   `report.validation` or `report.evidence_bundle`, both of which exist in the JSON.

OpenScience solves (1) with a skill library + `skill` tool, (2) with per-provider prompt tails,
and has partial patterns for (3) (a blind `reviewer` sub-agent, a claim→evidence discipline). We
port the mechanisms, keyed to openresearch's own architecture and failure modes.

## 3. What OpenScience does (verified mechanism, for grounding)

- **Skills** = `SKILL.md` files (YAML frontmatter `name`/`description`/`category`/`tags` + a
  Markdown body of workflows/checklists/pitfalls + optional `references/*.md` and `scripts/*.py`).
  A disk loader builds a **catalog** (names+descriptions only). The catalog is exposed as a single
  function-calling `skill` tool; calling `skill({name})` returns the **full body** on demand
  (Tier-1 progressive disclosure); `references/`/`scripts/` are Tier-2 (read by the agent's own
  tools). Injection-phrase descriptions ("always run this skill") are blocked at catalog-build.
  292 playbooks across 17 categories (biology, chemistry, cloud-compute, coding, data-engineering,
  databases, document-parsing, llm-tools, ml-inference, ml-training, physics, quantum, research,
  scholar-evaluation, visualization, writing, other).
- **Specialist agents** `biology`/`physics`/`ml` + read-only subagents (`critique`, `reviewer`,
  `literature-review`, `explore`, `write`). Domain routing is **explicit user selection** (a
  dropdown) — no classifier. The specialist's staged workflow prompt (e.g. `ml.txt`: SCOPE →
  LITERATURE → DATA → DESIGN → TRAIN → EVALUATE → CRITIQUE-GATE → REPORT, with a keyword→skill
  routing table) is injected every turn.
- **Critique/reviewer gate** — blind by design (verifiers do **not** see the generator's
  chain-of-thought; cites Feng et al. 2026, arXiv:2602.10177). `reviewer` is a genuinely agentic
  read-only sub-agent that recomputes numbers from artifacts.
- **Per-provider prompts** — `session/system.ts:19-26` routes to a different prompt tail per model
  family (Claude/GPT-5/GPT-4/Gemini/Qwen), each encoding that family's known agentic quirks
  (`beast.txt` = anti-premature-stop reinforcement for non-reasoning GPT chat models).

**Licensing:** repo is Apache-2.0 (`LICENSE`/`NOTICE`). Skill `license:` frontmatter describes the
*documented tool's* license, not a grant over the prose. We vendor the prose under Apache-2.0 with
a `NOTICE` attribution (§9).

## 4. Architecture

```
                          ┌─────────────────────────────────────────────┐
  paper text  ─┐          │  SKILL LIBRARY  backend/agents/rlm/skills/   │
  detect_env ──┼──▶ B. matcher (pure) ──▶ shortlist ──┐  <cat>/<name>/SKILL.md (+refs/scripts) │
  paper_hint ──┘   framework/domain → skill names      │  A. loader → catalog {name,desc,tags}  │
                                                        │                                        │
                          ┌─────────────────────────────┴──────────────┐  C. injection            │
                          ▼                             ▼               ▼                          │
                 system_prompt.py            _compute_constraint_        consult_skill(name)        │
                 catalog section (flag)      guidance() shortlist block  19th primitive (Tier-1)    │
                          │                             │               (refs/scripts = Tier-2 via   │
                          ▼                             ▼                _HARNESS_CODE_HELPERS copy)  │
                 D. domain-workflow prompt   E. native cloud skills                                 │
                 (by detected domain)        (vLLM/FSDP on GKE/AKS/RunPod cells)                    │
                          │                                                                          │
                 critique gate → REUSE external_validator.run_validation_panel                      │
                          └─────────────────────────────────────────────────────────────────────────┘

  WS① root-reliability   : per-provider prompt tails · progress journal · bounded-iter menu · angled panel
  WS② knowledge-grounding: literature connectors (arXiv/OpenAlex/S2) · claim gate on rubric INPUT · contradiction
  WS③ self-improvement   : usage→outcome loop · structured executor reports · operator standing-notes
  WS④ deliverable/honesty: render evidence section · findings ledger · blind reviewer · grounded figures
```

Master flag `OPENRESEARCH_SKILLS` gates the whole skill program; each workstream item has its own
sub-flag (§8). All default-OFF.

## 5. Detailed design — the skill program

### 5.A Skill library + loader
- **Location:** `backend/agents/rlm/skills/<category>/<name>/SKILL.md` (+ `references/`,
  `scripts/`). Verbatim copies of the upstream playbooks. Category set matches upstream.
- **Adaptation pass (surgical, per skill):** (i) strip upstream-only tooling that openresearch
  can't honor (Modal/Tinker/Atlas-specific invocations) or reframe to openresearch's backends;
  (ii) where openresearch already owns a bespoke version (RL scaffold, SDAR baselines), the skill
  *points at* the harness helper rather than duplicating a divergent copy; (iii) the "Cloud GPUs
  only / cost-approval" mandates are reconciled with openresearch's own budget meters.
- **Loader:** `backend/agents/rlm/skill_catalog.py` — pure, stdlib + `PyYAML` (already used — `import yaml` in `baseline_implementation.py`),
  disk-only. `load_catalog() -> dict[name, SkillMeta]` globs `skills/**/SKILL.md`, parses
  frontmatter (`name`, `description`, `category`, `tags`), skips bodies. Mirrors openscience's
  `skill.ts:compute()` minus the 6 network/Atlas sources. Injection-phrase sanitization ported
  (reject a description containing "always run", "must always run"). Duplicate-name = last-wins +
  warn. Memoized per process. Fail-soft: a malformed file is skipped with a warning, never raises.

### 5.B Detection + matching (autonomous — replaces the human dropdown)
- **Module:** `backend/agents/rlm/skill_matcher.py` — pure/deterministic, zero LLM calls.
  `match_skills(claim_map, environment_spec, catalog) -> SkillMatch{domain, skill_names, reasons}`.
- **Signal sources (all already in scope at `detect_environment`):** `PaperClaimMap`
  (`core_contribution`, `claims`, `model_architecture`, `training_recipe.optimizer`, `datasets`,
  `metrics`, `evaluation_protocol`) + `EnvironmentSpec.framework`. Match against the skills' own
  `tags`/`name` tokens (grounded — the tags are real).
- **`detected_domain`:** a coarse label (ml-rl / ml-vision / ml-inference / ml-interp / ml-nlp /
  physics / chemistry / biology / …) used by 5.D and WS① to pick the workflow prompt. **New
  field** — `detect_environment` does *not* classify domain today (`environment_detective.py`
  `_infer_framework` @ 133-149 is a crude pytorch/tf/jax 3-way). Stored in `EnvironmentSpec.extra`
  (the existing overflow slot) + persisted to `rlm_state/skill_match.json`.
- **Hook:** extend the `detect_environment` primitive (`primitives.py:1108-1214`) to call
  `match_skills` after `run_offline` and stash the result on the returned dict + on disk.
  Fail-soft; unset flag ⇒ not called.

### 5.C Injection (progressive disclosure)
- **`consult_skill(name)` — 19th primitive.** Returns the full `SKILL.md` body (Tier-1). Fuzzy
  "did-you-mean" on miss (ported from `tool/skill.ts:14-30`). Registered in `PRIMITIVE_REGISTRY`
  (`primitives.py:8908`); **`tests/rlm/test_registry.py::EXPECTED` updated 18 → 19** and the
  primitive count references in `CLAUDE.md` updated. Advertised in the system prompt only when
  `OPENRESEARCH_SKILLS` is on.
- **Catalog section in `system_prompt.py`.** A new flag-gated section (mirrors the existing
  `_CONTEXT_MAP_SECTION` / `_REPO_AWARE_SECTION` pattern in `build_system_prompt`, line 578+):
  lists categories + a few skill names each + the `consult_skill` contract. Off ⇒ omitted,
  prompt byte-identical.
- **Matched-shortlist block in the implementer guidance.** A `_skill_shortlist_block(...)` folded
  into `_compute_constraint_guidance` (`baseline_implementation.py:2300-2726`) near the
  `_load_paper_override` step (line ~1944, the existing precedent for "look up something keyed to
  this paper, format as markdown, append"). Voice matches the house `_SCREAMING_SNAKE_BLOCK` style;
  gated `OPENRESEARCH_SKILLS in ("1","true","yes")`.
- **Skill `scripts/` → `code/`.** Shipping an executable helper reuses the existing
  `_HARNESS_CODE_HELPERS` auto-copy (`baseline_implementation.py:65-95`) — add the skill script
  filenames (or a directory-copy variant) so a consulted skill's `scripts/*.py` land in the run's
  `code/` verbatim, exactly like `provenance.py`/`eval_provenance.py` do today. Tier-2
  `references/*.md` are read by the agent's own tools on demand.

### 5.D Domain-workflow + critique gate (the specialist layer)
- **Domain-workflow prompts** seeded from `ml.txt`, split by `detected_domain` (5.B): an
  `ml-rl` workflow (reward wiring, on-policy vLLM sync, GRPO/PPO), `ml-vision`, `ml-inference`
  (compression/serving), `ml-interp`, etc. Injected into the root prompt by detected domain — the
  same append pattern as the catalog section. Content is imperative + grounded in the matched
  skills (routes to `consult_skill`), never fabricated numbers.
- **Critique gate = REUSE `external_validator.run_validation_panel`.** Do **not** build a new
  panel. The existing entry (`external_validator.py:451-561`) already accepts an arbitrary
  `validator_client`, `panel_models`, `metrics`, `leaf_records`, `report_claims` and returns a
  `ValidatorVerdict{status, veto_set, predicates[], separation, evidence_fingerprint}` with
  machine-checked predicates and min-aggregation. The domain workflow's "critique gate" step simply
  triggers this existing panel at the workflow's report stage. `build_validator_client`
  (fail-closed, `grader_transport.py:358`) already enforces cross-lineage separation.
  Flags reused: `OPENRESEARCH_EXTERNAL_VALIDATOR`, `OPENRESEARCH_VALIDATOR_PANEL_N`.

### 5.E Native cloud skills (the "vLLM for cluster work on GCP/Azure" ask)
Authored `SKILL.md`s written against openresearch's **real** cell contract (verified):
- **Cell contract** (`gpu_cell_runner.py:run_matrix` @ 1273; `_run_cell_subprocess`): a
  `train_cell.py` reads `OPENRESEARCH_CELL_PARAMS` (JSON of the cell), `OPENRESEARCH_CELL_OUTPUT_DIR`,
  argv `--cell-id`/`--output-dir`, honors `OPENRESEARCH_CELL_BATCH_SCALE`/`_GRAD_CHECKPOINT`, and
  writes a **flat** `metrics.json`. `cells.json` axes: `model_key`/`env`/`baseline` (+ synonyms),
  `est_vram_gb`, `gpus`, `cell_env`, `services`, `command`, `metrics_source`.
- **"Serve vLLM alongside a training cell"** → grounded in the real `services` field (a co-located
  GPU service reserving a disjoint GPU slice via `_partition_cell_gpus`) — **not invented**.
- **"Run the authors' launcher verbatim"** → the execute-mode `command` seam (`bash -lc`, cwd =
  cell dir, `OUTPUT_DIR`/`OPENRESEARCH_CELL_ID` exported) + the `metrics_source` verl adapter.
- **Multi-GPU discipline — three distinct paths, stated exactly so a skill picks the right one:**
  (a) legacy monolithic `commands.json` → `accelerate launch`+FSDP2, **marker-gated**
  (`primitives.py:_resolve_distributed_launch` @ 3712, `_DISTRIBUTED_MARKERS`); (b) GKE in-pod →
  plain `torchrun` (`gke_cell_entrypoint.py:build_cell_launch_argv` @ 519, via
  `OPENRESEARCH_CELL_GPU_COUNT`); (c) **the mainstream cells route does neither** — a `>1`-GPU
  cell is **model-parallel only** (`device_map="auto"`), never launcher-rewritten. A native
  "multi-GPU cell" skill must say this or it will mislead the agent.
- Backends: RunPod (`runpod_backend.py`), GCP/GKE (`GkeJobBackend`), Azure AKS
  (`k8s_job_cell_runner.py` — sets `OPENRESEARCH_CELL_GPU_COUNT`, `OPENRESEARCH_GCP_GCS_BUCKET` /
  `OPENRESEARCH_AZURE_STORAGE_ACCOUNT`+`_BLOB_CONTAINER`).

## 6. Detailed design — the four workstreams

### 6.① Root-reliability
- **Per-provider prompt tails** (`OPENRESEARCH_PROVIDER_PROMPTS`). Split `system_prompt.py` into
  the shared core (RLM operating model, primitives, FINAL_VAR contract, iteration mechanics) + a
  small **per-family tail** keyed off the resolved root token (`role_models.py:resolve_root_model`
  / `models.RootModel`). **Content keyed to openresearch's own documented failure modes, not
  openscience's mapping:** a `claude-oauth`/Sonnet tail hammering anti-premature-stop ("you MUST
  call `implement_baseline` **and** `run_experiment` before any `FINAL_VAR`; do not summarize a
  hypothetical result") — borrowing the `beast.txt` *pattern*, applied to the model that actually
  degenerates here; a `gpt-chat`/grok tail addressing churn/over-continuation. Natural extension of
  the existing `root_model.prompt_addendum` hook (`system_prompt.py:643`). Off ⇒ current single
  prompt, byte-identical.
- **Narrative progress journal** (`OPENRESEARCH_PROGRESS_JOURNAL`). A self-authored
  `rlm_state/progress.md` ("current stage / what worked / what didn't / open questions"), re-read
  at the top of each iteration alongside `check_user_messages()`, compaction-immune. Distinct from
  `context_map.json` (structured tool-output cache) and `NEGATIVE_LESSONS` (post-hoc cross-run
  mining). Gives `lifecycle_driver` a richer in-run signal than raw event logs.
- **Bounded-iteration exit menu** (`OPENRESEARCH_ITER_EXIT_MENU`). Generalize the existing
  `repair_exhausted` terminal (`OPENRESEARCH_REPAIR_MAX_ITERATIONS`) to the general
  score-improvement loop: triage-before-counting (a pure bugfix retry doesn't consume the budget)
  + on exhaustion ship a report that names which of continue / pivot / abandon applies. Small
  `forced_iteration.py` tweak.
- **Angle-scoped validator panelists** (`OPENRESEARCH_VALIDATOR_ANGLES`). When
  `OPENRESEARCH_VALIDATOR_PANEL_N > 1`, scope each panelist to a **distinct predicate subset**
  instead of N identical prompts — better coverage per dollar, no architecture change
  (`external_validator.py`).

### 6.② Knowledge-grounding
- **Literature connectors** (`OPENRESEARCH_LITERATURE_GROUNDING`). New
  `backend/services/knowledge/connectors/` — a thin `Connector` protocol (`search`/`fetch`),
  ported 1:1 from openscience's `science/connectors/literature/{arxiv,openalex,semantic-scholar}`
  (all key-free; S2 optional key at a higher limit). Result summaries capped (~600 chars, mirrors
  upstream `snippet()`) and cached via the existing `primitive_cache`. **Wired as a deterministic
  helper inside `rubric_gen.py`/`understand_section` — NOT a new root primitive** (keeps the tested
  primitive surface; per the knowledge-sweep's explicit recommendation).
- **Claim-grounding gate on the rubric input** (`OPENRESEARCH_LITERATURE_CLAIM_GATE`). A
  `literature_claim_gate.py` mirroring the existing *output-side* `report_claim_gate.py`: resolve
  the paper's extracted baselines / headline numbers against connector records; emit an advisory
  `run_warning` (never a hard veto — fuzzy title match) when a claimed number/baseline can't be
  corroborated. Same "typed check over external state, not LLM trust" philosophy as
  `evidence_gate`/`claim_grounding`, applied to rubric *inputs* (`rubric_gen.py:59`).
- **Paper-vs-code contradiction detection** (`OPENRESEARCH_CONTRADICTION_CHECK`). Surface the
  SDAR-λ/β discrepancy class (paper text vs. authors' released scripts, cf.
  `paper_hints.py:44-47`) as an advisory note in `rlm_state/`, using connectors + the cloned repo
  (`OPENRESEARCH_USE_AUTHOR_REPO`). Never silently rewrites a rubric target.

### 6.③ Self-improvement
- **Usage→outcome feedback loop** (`OPENRESEARCH_RECIPE_OUTCOME_WEIGHTING`). Both stores select by
  recency (`recipe_library.py:565` `active[-INJECT_TOP_K:]`) and the lesson `occurrences` counter
  is a *perverse* proxy (climbs when a fix isn't working). Stamp the recipe/lesson IDs *shown* to
  an attempt at the assembly chokepoint (`experience_memory.py:254-262`), correlate against that
  attempt's **deterministic** `meets_target` / failure-class-recurrence at the `admit_recipe` call
  sites (`run.py:2136,2282,4517`), and weight future selection by measured efficacy. **Advisory /
  evidence-not-grade** — the outcome signal is the same deterministic signal already gating
  admission, never an LLM grade (the red line).
- **Structured executor reports** (`OPENRESEARCH_EXECUTOR_REPORTS`). Capture a normalized
  `{status, findings, failures, assumptions, parameters, artifact_refs, suggestions}` block from a
  **successful** `implement_baseline` (nothing captures assumptions/suggestions today —
  `failure_capsule.py` is failure-path-only, `context_map.py` is 3 primitives). Port openscience's
  `state.ts` schema **and build the prompt-teaching half openscience skipped** (instruct the
  sub-agent to emit the block — otherwise it's the dead plumbing the self-improvement sweep flagged).
  Pairs with `provenance.py` (parameters) + `leaf_triage.py` (suggestions).
- **Operator standing-notes** (`OPENRESEARCH_OPERATOR_NOTES`). A durable `runs/_memory/
  operator_notes.json` (global) folded into the same guidance chokepoint — cross-run policy config
  ("this cluster caps at 24 GB", "avoid bitsandbytes here") that today requires re-specifying a
  flag every invocation. Pre-run configuration, not live chat (openresearch runs unattended).

### 6.④ Deliverable & honesty
- **"Provenance & Evidence" report section** (`OPENRESEARCH_EVIDENCE_REPORT_SECTION`). Render the
  already-computed `report.evidence_bundle` (sha256 receipt, `report.py:61`/1133-1152) +
  `report.validation` (`report.py:199`/1979-2006) + `claim_grounding` results into
  `_render_markdown` (`report.py:2422`). **Pure disclosure, zero new trust surface** — an
  absent/`bundle_unverified` receipt now shows in the deliverable instead of being buried in JSON.
  Highest-ROI, lowest-risk item.
- **Itemized validator findings ledger** (`OPENRESEARCH_FINDINGS_LEDGER`). Promote the flat
  `veto_set` to a rendered per-item table (claim · predicate · severity · on-disk evidence) — the
  `PredicateVerdict{predicate, metric_ref, violated, detail}` dataclass already carries the shape
  (`external_validator.py:35-42`). UI copy: a clean panel reads "no suspicion raised," never
  "verified correct."
- **Agentic blind reviewer** (`OPENRESEARCH_BLIND_REVIEWER`). The one genuinely new capability. A
  read-only sub-agent (Read/Grep/Glob + recompute-only Bash; Write/Edit **denied**) that roams
  `final_report.{json,md}` + `code/` + `experiment_runs.jsonl`, recomputes means/percentages, and
  catches cross-section inconsistencies the fixed 5-predicate panel structurally can't (abstract ≠
  table ≠ caption, N mismatches). Reuses the existing sub-agent tool-scoping
  (`claude_runtime.py:81,224-286` — `allowed_tools`/`permission_mode`/`setting_sources=[]`).
  **Invariants (non-negotiable):** blind to the root's REPL transcript/`dashboard_events.jsonl`
  (preserve blind-review, Feng et al. 2026); **advisory/veto-only, never verdict-upgrading**
  (mirror `held_out_gate` — evidence never let LLM judgment lift a score). Biggest single
  investment → sequenced **last**.
- **Grounded figure pipeline** (`OPENRESEARCH_FIGURE_PIPELINE`). Render the already-grounded
  `fig_auto_*.json` sidecars (`leaf_actuator.py:emit_figure_sidecars` @ 428-480 — measured
  `per_model` only, guarded) into honest PNGs (real error bars, no truncated axes, colorblind
  palette — openscience's `scientific-visualization` checklist) embedded in `final_report.md`.
  **Fed only by evidence-gate-passed numbers**, never an agent-produced unaudited PNG (that would
  reintroduce the "polished but false" risk).

## 7. Release phasing (highest-ROI-first, operator-selected)

Each release is independently shippable and default-OFF.

- **Release 1 — Foundation + near-free wins.** 5.A loader + library (ML/cluster + `research`/
  `scholar-evaluation` seed first), 5.B matcher, 5.C `consult_skill` + catalog + shortlist;
  **①** per-provider prompt tails; **②** claim-grounding gate + connectors; **④** evidence-report
  section. → delivers the core "invoke skills per subject matter" + "vLLM skills" ask *and* the
  three highest-value grounded wins.
- **Release 2 — Native cloud.** 5.E native cloud skills on the real cell seams.
- **Release 3 — Specialist + reliability depth.** 5.D domain-workflow + critique-gate reuse; **①**
  progress journal, bounded-iteration menu, angled panelists; **②** contradiction detection.
- **Release 4 — Self-improvement + honesty depth.** **③** usage→outcome loop, structured executor
  reports, operator notes; **④** findings ledger.
- **Release 5 — Full-catalog backfill + the new capability.** Copy the remaining non-ML categories
  (physics/chem/bio/quantum/…) as inert library content the matcher surfaces; **④** blind reviewer
  + grounded figure pipeline (the largest investments) last.

Because the whole program exceeds one implementation plan, `writing-plans` will produce a
**Release-1 plan first**; later releases get their own plan when Release 1 lands. Implementation is
delegated to Sonnet against the plan; Opus reviews every diff.

## 8. Flags (all default-OFF; unset ⇒ byte-identical)

| Flag | Gates |
|---|---|
| `OPENRESEARCH_SKILLS` | master: library load, matcher, `consult_skill`, catalog section, shortlist block |
| `OPENRESEARCH_SKILL_DOMAIN_WORKFLOW` | 5.D domain-workflow prompt injection |
| `OPENRESEARCH_PROVIDER_PROMPTS` | ① per-provider prompt tails |
| `OPENRESEARCH_PROGRESS_JOURNAL` | ① narrative progress journal |
| `OPENRESEARCH_ITER_EXIT_MENU` | ① bounded-iteration exit menu |
| `OPENRESEARCH_VALIDATOR_ANGLES` | ① angle-scoped panelists |
| `OPENRESEARCH_LITERATURE_GROUNDING` | ② connectors |
| `OPENRESEARCH_LITERATURE_CLAIM_GATE` | ② claim-grounding gate on rubric input |
| `OPENRESEARCH_CONTRADICTION_CHECK` | ② paper-vs-code contradiction |
| `OPENRESEARCH_RECIPE_OUTCOME_WEIGHTING` | ③ usage→outcome loop |
| `OPENRESEARCH_EXECUTOR_REPORTS` | ③ structured executor reports |
| `OPENRESEARCH_OPERATOR_NOTES` | ③ operator standing-notes |
| `OPENRESEARCH_EVIDENCE_REPORT_SECTION` | ④ evidence report section |
| `OPENRESEARCH_FINDINGS_LEDGER` | ④ itemized findings ledger |
| `OPENRESEARCH_BLIND_REVIEWER` | ④ blind reviewer sub-agent |
| `OPENRESEARCH_FIGURE_PIPELINE` | ④ grounded figures |

Reused (existing) flags: `OPENRESEARCH_EXTERNAL_VALIDATOR`, `OPENRESEARCH_VALIDATOR_PANEL_N`,
`OPENRESEARCH_USE_AUTHOR_REPO`, `OPENRESEARCH_REPORT_CLAIM_GATE`.

## 9. Licensing / attribution

OpenScience is Apache-2.0. We vendor copied `SKILL.md` prose under Apache-2.0 with a
`backend/agents/rlm/skills/NOTICE` crediting "Synthetic Sciences / OpenScience
(github.com/synthetic-sciences/openscience), Apache-2.0" and preserving the two upstream-flagged
third-party drops (`markitdown` MIT; `hugging-face-trackio` Apache-2.0) if copied. No skill body
carries a per-file copyright header upstream; attribution lives at the library root. Connector code
(§6②) is a clean-room Python reimplementation of the small `Connector` protocol, credited the same.

## 10. Testing & rollout discipline

- **Every item ships hermetic OFF + ON tests.** OFF test asserts byte-identical behavior
  (prompt unchanged, primitive count 18, `_compute_constraint_guidance` output unchanged,
  `_render_markdown` unchanged). ON test asserts the new behavior with a fixture.
- `tests/rlm/test_registry.py::EXPECTED` updated 18 → 19 (only when `consult_skill` lands).
- Loader/matcher are pure → unit-tested against real on-disk fixtures (no mocks, house rule).
- Blind reviewer: hermetic test with a canned artifact tree; assert it can veto but the verdict
  never *raises* a score.
- Default-flip of any flag follows the standing bar: **≥3 paired A/B runs + the grader-σ gate**
  before a default changes. This spec flips **no** defaults.
- `uvx ruff@0.15.16 check .` clean; full `pytest` green.

## 11. Rejected candidates (analysis was adversarial, not accretive)

- **Provenance DAG, `_script_manifest.jsonl`, RLMArtifacts, compaction protect-list, `max-steps`,
  stage-gates** — redundant with openresearch's stricter existing infra (`evidence_bundle`,
  `arg_contracts`, wall-clock watchdog, `context_map`). Porting would be a downgrade.
- **Perplexity/OpenRouter search, Atlas research-graph** — paid/closed + LLM-trust; against the
  anti-hallucination philosophy (Atlas has no portable code — closed SaaS).
- **`scientific-schematics` (AI-drawn diagrams scored on aesthetics)** — **actively dangerous**:
  manufactures the exact "polished but false" failure mode the evidence machinery exists to
  prevent. Not ported. Any illustrative method diagram (if ever wanted) stays outside the
  verdict/scoring surface, watermarked "illustrative, not evidence."
- **Dual-loop `<rlm_result>` XML compression** — dead plumbing upstream (never taught to any
  model); we take the *schema idea* (§6③ structured executor reports) and build the teaching half.
- **Generic plotting / writing-skill docs** — low fit (openresearch emits a reproduction verdict,
  not a citable paper); revisit only if a citation surface ever exists.

## 12. Risks & open questions

- **Prompt bloat.** The catalog section + domain workflow + per-provider tail add tokens to the
  root prompt. Mitigation: catalog is names+descriptions only (Tier-0); everything else is
  on-demand (`consult_skill`) or a small shortlist. Measure input-token delta in the Release-1 A/B.
- **Matcher precision.** A wrong domain label surfaces the wrong shortlist. Mitigation: matcher is
  advisory (the root/implementer still chooses); shortlist is additive, never blocking; fail-soft.
- **Skill staleness.** Vendored playbooks pin a moment in time (some upstream skills are raw
  doc-scrapes — e.g. the DeepSpeed one leaks scraper artifacts). Mitigation: seed with the
  hand-curated ML/cluster skills first; treat the backfill (Release 5) as lower-trust inert content.
- **Connector rate limits / offline runs.** arXiv/OpenAlex/S2 are network calls. Mitigation:
  key-free + cached + fail-soft; a blocked network degrades to "no grounding," never an error
  (matches the `OPENRESEARCH_USE_AUTHOR_REPO` blocked-clone posture).
- **Open question:** should the domain-workflow prompt (5.D) replace or *augment* the generic root
  prompt for a detected domain? Recommendation: augment (append), never replace — the RLM operating
  model + FINAL_VAR contract must always be present. To confirm at Release-3 planning.
```
