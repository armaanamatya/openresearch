# Session Handoff — OpenScience Skill Library + Harness Enhancements

> **Date:** 2026-07-05 · **Author:** Opus · **Status:** design approved, scope locked, **nothing
> implemented yet**. Written for a fresh (context-cleared) session to pick up and execute.
> **Read this first, then the spec.**

## 0. TL;DR — what this is and what to do next

We analyzed **OpenScience** (`synthetic-sciences/openscience`, Apache-2.0, cloned read-only at
`/home/abheekp/openscience-ref`) and are porting its useful, openresearch-fitting parts into the
RLM paper-reproduction harness. The operator approved the full scope. **Design doc (source of
truth):**

```
docs/superpowers/specs/2026-07-05-openscience-skill-library-and-harness-enhancements-design.md
```

**Immediate next step:** run the `writing-plans` skill to produce the **Release-1** implementation
plan from §7 of the spec (Release 1 breakdown is also pre-drafted in §6 of *this* handoff), then
delegate implementation to Sonnet against that plan (Opus reviews every diff). Everything ships
**default-OFF, byte-identical when off**, TDD, hermetic ON+OFF tests, `ruff` clean.

**North star (operator's explicit mandate):** *copy faithfully · adapt surgically · ground
natively · reuse existing analogs · default-OFF.* Do **not** hallucinate skill content — the 292
upstream `SKILL.md` playbooks are real and Apache-2.0, so **copy them verbatim** (with attribution)
rather than regenerating; native content is written against the cited real seams below.

## 1. The centerpiece + the four workstreams (all approved, all in the spec)

- **Skill program** — a reusable, framework/technique-keyed **skill library** surfaced
  automatically by the paper's subject matter, + a `consult_skill(name)` primitive (progressive
  disclosure), + a domain-workflow/critique layer, + native cloud skills, + full-catalog backfill.
- **① Root-reliability** — per-provider prompt tails · narrative progress journal ·
  bounded-iteration exit menu · angle-scoped validator panelists.
- **② Knowledge-grounding** — key-free literature connectors (arXiv/OpenAlex/Semantic-Scholar) ·
  claim-grounding gate on the *rubric input* · paper-vs-code contradiction detection.
- **③ Self-improvement** — usage→outcome feedback loop · structured executor reports · operator
  standing-notes.
- **④ Deliverable & honesty** — render already-computed evidence into the report · itemized
  validator findings ledger · agentic **blind reviewer** (new capability) · grounded figure pipeline.

## 2. Locked decisions (do not re-litigate)

1. **Scope = everything**, but optimized for openresearch and grounded (operator: "do everything
   but ensure they are optimized for our repo … do not hallucinate be careful").
2. **All four workstreams** folded into the design.
3. **Sequencing = highest-ROI-first**: Release 1 = skills foundation + per-provider prompts +
   claim-grounding gate + evidence-report section. Release 2 = native cloud. Release 5 (last) =
   full-catalog backfill + blind reviewer + figure pipeline. (Full phasing in spec §7.)
4. Critique gate **reuses** `external_validator.run_validation_panel` — do not build a new panel.
5. **Rejected** (do not port): provenance DAG, `_script_manifest.jsonl`, RLMArtifacts, compaction
   protect-list, `max-steps`, stage-gates (redundant with stricter existing infra); Perplexity/
   OpenRouter search + Atlas (paid/closed/LLM-trust); **`scientific-schematics` AI-diagrams
   (dangerous — aesthetic-scored, manufactures "polished but false" — never port)**. Reasons in
   spec §11.

## 3. Grounded seams appendix — THE payload (verified by direct file read; don't re-analyze)

All paths under `/home/abheekp/openresearch`. These were each confirmed by reading the file; two
early assumptions were **wrong and corrected here**: `environment_detective.py` is under
`backend/agents/` (not `rlm/`), and the guidance assembler is `_compute_constraint_guidance`
(not `_assemble_*`).

### Skill program seams
- **`detect_environment`** primitive — `backend/agents/rlm/primitives.py:1108-1214`. Returns
  `EnvironmentSpec.model_dump()` + an `"outcome"` key. It calls
  `environment_detective.run_offline(...)` (`backend/agents/environment_detective.py:50`).
  Framework detection is a crude 3-way (`_infer_framework` @ 133-149: pytorch/tf/jax). **No
  domain/task_type field exists** — the matcher adds it, stored in `EnvironmentSpec.extra` (the
  overflow slot, `backend/agents/schemas.py:357-371`). Input signal = `PaperClaimMap`
  (`schemas.py:88-101`). This primitive is cached via `primitive_cache` and writes
  `environment_spec.json` — the right hook to also run `match_skills` + persist `skill_match.json`.
- **Harness-helper auto-copy** — `backend/agents/baseline_implementation.py`: `_HARNESS_CODE_HELPERS`
  tuple @ 65-79, `_copy_harness_helpers_to_code_root` @ 82-95 (`shutil.copy2` per file from
  `backend/agents/rlm/`), `refresh_harness_helpers` @ 98-121, called in `run_with_sdk` @ ~2771.
  **Model for shipping skill `scripts/` into `code/`.**
- **Implementer guidance assembly** — `_compute_constraint_guidance` @
  `baseline_implementation.py:2300-2726` (~30 append points). `OPENRESEARCH_BASELINE_EXTRA_GUIDANCE`
  read @ 2616. `_load_paper_override` @ 1944-1979 reads `docs/papers/<arxiv_id>.yaml`, dumps as
  markdown — **the precedent for the matched-shortlist injection** (slot the shortlist block near
  it). Called from `run_with_sdk` @ 2853-2862, interpolated as `{sandbox_guidance}`.
- **House style for a guidance block** — `_RL_SCAFFOLD_BLOCK` @ 1422-1482
  (`OPENRESEARCH_RL_SCAFFOLD`), `_SDAR_BASELINES_BLOCK` @ 1499-1566 (`OPENRESEARCH_SDAR_BASELINES`):
  a big triple-quoted `_SCREAMING_SNAKE_BLOCK` constant, appended verbatim behind a lowercase env
  gate `in ("1","true","yes")` (note: **not** `"on"`), imperative numbered STEPs, literal
  identifiers/constants inline, "why = which failure this prevents." Match this voice.
- **System prompt** — `build_system_prompt` @ `backend/agents/rlm/system_prompt.py:578`; composes
  ordered sections; flag-gated appends already exist (`_CONTEXT_MAP_SECTION`, `_REPO_AWARE_SECTION`);
  `root_model.prompt_addendum` appended @ 643. Add the catalog section + (WS①) the per-family tail
  here. NB: the whole prompt is brace-escaped then `.format(custom_tools_section=...)` — exactly one
  `{custom_tools_section}` placeholder must survive (asserted @ 671).
- **Primitive registry** — `PRIMITIVE_REGISTRY` @ `primitives.py:8908`; test `EXPECTED` @
  `tests/rlm/test_registry.py` (currently **18** → **19** with `consult_skill`). Update the count
  references in `CLAUDE.md` too.

### Critique-gate / validator seams (REUSE, don't duplicate)
- `backend/agents/rlm/external_validator.py`: `run_validation_panel(*, validator_client,
  panel_models, metrics, project_dir, leaf_records, separation, report_claims=None)` @ 451-561 →
  `ValidatorVerdict{status: clean|vetoed|unavailable, veto_set, predicates[], panel_models,
  separation, evidence_fingerprint}`. `PredicateVerdict{predicate, metric_ref, violated, detail}` @
  35-42 (already the itemized-ledger shape for WS④). Min-aggregation veto; 5 fixed predicates
  (`provenance_present`, `not_all_constant`, `gpu_claim_plausible`, `rerun_agrees`,
  `report_claims_grounded`). It issues **one non-tool completion** over a truncated `leaf_records[:20]`
  blob (line 496) — this is *why* WS④'s agentic blind reviewer is a genuinely new capability (the
  panel can't open arbitrary files). `build_validator_client` @ `grader_transport.py:358` is
  **fail-closed** (raises rather than reuse the executor's lineage). Flags:
  `OPENRESEARCH_EXTERNAL_VALIDATOR`, `OPENRESEARCH_VALIDATOR_PANEL_N` (default 2). Sibling
  deterministic critic: `backend/agents/rlm/evidence_audit.py` (`OPENRESEARCH_EVIDENCE_AUDIT`).

### Native-cloud cell seams (WS/5.E — get these exactly right)
- `run_matrix(...)` @ `backend/agents/rlm/gpu_cell_runner.py:1273`. `train_cell.py` contract
  (`_run_cell_subprocess` @ 685-982): reads `OPENRESEARCH_CELL_PARAMS` (JSON of the cell),
  `OPENRESEARCH_CELL_OUTPUT_DIR`, argv `--cell-id`/`--output-dir`; honors
  `OPENRESEARCH_CELL_BATCH_SCALE`/`_GRAD_CHECKPOINT`; writes a **flat** `metrics.json`.
  `CUDA_VISIBLE_DEVICES` is harness-protected.
- **Execute-mode `command` seam:** a non-blank `cell["command"]` runs verbatim via `bash -lc`,
  cwd = cell dir, with `OUTPUT_DIR`/`OPENRESEARCH_CELL_ID` exported — how an authors' launcher runs
  unmodified. `cell["metrics_source"]` = execute-mode adapter (e.g. verl `{"kind":"verl",
  "log_glob":...,"success_rate_key":"val/success_rate"}`).
- **`cell["services"]`** = co-located auxiliary GPU services on a **disjoint** GPU slice via
  `_partition_cell_gpus` — the real hook for "serve vLLM alongside the trainer."
- **Multi-GPU: three distinct paths** — (a) legacy monolithic `commands.json` →
  `accelerate launch`+FSDP2, marker-gated (`primitives.py:_resolve_distributed_launch` @ 3712,
  `_DISTRIBUTED_MARKERS`); (b) GKE in-pod → plain `torchrun`
  (`docker/gke-cell-base/gke_cell_entrypoint.py:build_cell_launch_argv` @ 519, via
  `OPENRESEARCH_CELL_GPU_COUNT` set by `k8s_job_cell_runner.py:705-716`); (c) **mainstream cells
  route does NEITHER — a `>1`-GPU cell is model-parallel only (`device_map="auto"`), never
  launcher-rewritten** (`baseline_implementation.py:2667-2680`). A native multi-GPU-cell skill MUST
  state this. Canonical aggregate shape: `cell_matrix.py` `per_model[model_key][env][baseline]` (no
  `per_dataset` layer), `aggregate_cell_metrics` @ 826-997. Cloud env in `k8s_job_cell_runner.py`:
  `OPENRESEARCH_GCP_GCS_BUCKET` / `OPENRESEARCH_AZURE_STORAGE_ACCOUNT`+`_BLOB_CONTAINER`.

### Knowledge-grounding seams (WS②)
- **No external literature grounding exists today** — `arxiv_id`/`semanticscholar`/`openalex`/
  `crossref` appear nowhere in `backend/` outside the venv. `ArxivFetcher`
  (`backend/services/ingestion/intake/fetchers/arxiv.py`) downloads PDF/HTML bytes only, never the
  metadata API. `rubric_gen.py:59` bakes the paper's reported numbers into **hard rubric targets**
  — the reason input-side grounding matters. Output-side analog to mirror:
  `backend/agents/rlm/claim_grounding.py` + `report_claim_gate.py` (`OPENRESEARCH_REPORT_CLAIM_GATE`,
  caps verdict to `partial` on an ungrounded claimed number).
- **Upstream connectors to port 1:1** (Python, `httpx`/`requests`): `openscience-ref/backend/cli/
  src/science/connectors/literature/{arxiv,openalex,semantic-scholar,crossref,europepmc}.ts` — a
  tiny `search`/`fetch` protocol, all key-free (S2 optional `SEMANTIC_SCHOLAR_API_KEY` for higher
  limit). Semantic Scholar `fetch()` pulls `references.title,citations.title` (citation chaining).
  Cap summaries ~600 chars (upstream `snippet()`), cache via `primitive_cache`. **Wire inside
  `rubric_gen.py`/`understand_section` as a deterministic helper — NOT a new root primitive.**

### Self-improvement seams (WS③)
- `backend/agents/rlm/recipe_library.py`: `recipe_guidance_block` @ 544-584 selects
  `active[-INJECT_TOP_K:]` @ 565 (**recency, not efficacy**). `lesson_distiller.py` `occurrences`
  counter is a **perverse proxy** (climbs when a fix isn't working). Assembly chokepoint =
  `experience_memory.py:254` (lessons) / `:262` (recipes). Admission call sites =
  `run.py:2136,2282,4517` (`admit_recipe`). The usage→outcome loop stamps shown IDs here, correlates
  vs the **deterministic** `meets_target`/failure-recurrence (never an LLM grade — the red line).
  Existing structured capture: `failure_capsule.py` (failure path only), `context_map.py` (3
  primitives only) — neither captures assumptions/suggestions from a *successful* implement.

### Deliverable seams (WS④)
- `backend/agents/rlm/report.py`: `_render_markdown` @ ~2422 **never references**
  `report.validation` (field @ 199, stamped @ 1979-2006) or `report.evidence_bundle` (field @ 61,
  populated @ 1133-1152) → the evidence-report section is pure rendering of existing data.
- `evidence_bundle.py` sha256 receipt (mint/resolve coherence recheck @ 208-372); `claim_grounding.py`
  deterministic regex extraction. `leaf_actuator.py:emit_figure_sidecars` @ 428-480 writes grounded
  `fig_auto_*.json` (measured `per_model` only, guard @ 479) — today read **only** by the text-only
  grader; the figure pipeline renders these, never an agent PNG.
- **Blind reviewer scoping** reuses `backend/agents/runtime/claude_runtime.py` @ 81, 224-286
  (`allowed_tools` / `permission_mode` / `setting_sources=[]`). Invariants: blind to root reasoning;
  advisory/veto-only, never verdict-upgrading.

### Root-reliability seams (WS①)
- Per-provider tail keys off `role_models.resolve_root_model` / `models.RootModel` (+ the existing
  `prompt_addendum` hook). Split `system_prompt.py` into shared core + per-family tail. Content keyed
  to openresearch's OWN failure modes (`claude-oauth` premature-stop; `gpt-chat`/grok churn), NOT
  openscience's Claude-most-trusted mapping. Existing reliability band-aids to complement:
  `forced_iteration.py`, `lifecycle_driver.py`, `root_progress.infer_required_stage`,
  `OPENRESEARCH_REPAIR_MAX_ITERATIONS`. Progress journal complements `context_map.py`.

## 4. OpenScience reference layout (the source content)

- Clone: `/home/abheekp/openscience-ref` (persists on disk across the context clear). Apache-2.0
  (`LICENSE`/`NOTICE`).
- Skills: `backend/cli/skills/<category>/<name>/SKILL.md` (+ `references/`, `scripts/`). **292
  `SKILL.md` across 17 categories**: biology(43), chemistry(23), cloud-compute(10), coding(21),
  data-engineering(10), databases(33), document-parsing(1), llm-tools(30), ml-inference(9),
  ml-training(56), physics(23), quantum(4), research(9), scholar-evaluation(1), visualization(8),
  writing(10), other(6).
- SKILL.md format: YAML frontmatter (`name`,`description`,`category`,`tags` are the runtime-read
  fields; `version`/`author`/`license`/`dependencies` are documentation-only) + Markdown body
  (quick-start → workflows-with-checklists → when-to-use-vs-alternatives → common-issues →
  advanced-topics→references → resources). `license:` names the *documented tool's* license, not a
  grant over the prose. Two genuine third-party drops to preserve if copied: `data-engineering/
  markitdown` (MIT), `other/hugging-face-trackio` (Apache-2.0).
- Specialist workflow to seed the domain prompts from: `backend/cli/src/agent/prompt/ml.txt` (271
  lines: SCOPE→LITERATURE→DATA→DESIGN→TRAIN→EVALUATE→CRITIQUE-GATE→REPORT + a keyword→skill routing
  table + a compute-decision matrix + the blind critique/reviewer gate citing Feng et al. 2026,
  arXiv:2602.10177). Per-provider prompts: `backend/cli/src/session/prompt/{anthropic,beast,
  codex_header,gemini,qwen}.txt` (identical ~224-line preamble, divergent tails; `beast.txt` =
  anti-premature-stop reinforcement).

## 5. Which skills to seed FIRST (Release 1)

Vendor these (verbatim + surgical adaptation) into `backend/agents/rlm/skills/`:
- **ml-inference:** `vllm`, `sglang`, `tensorrt-llm`, `speculative-decoding`, `llama-cpp`.
- **ml-training:** `verl`, `grpo-rl-training`, `openrlhf`, `deepspeed`, `pytorch-fsdp`, `accelerate`,
  `megatron-core`, `peft`, `trl-fine-tuning`, `unsloth`, `flash-attention`, `bitsandbytes`,
  `lm-evaluation-harness`, `ml-benchmark-evaluation`, `hugging-face-evaluation`,
  `weights-and-biases`, `mlflow`, `ray-train`, `knowledge-distillation`, `model-merging`,
  `model-pruning`, `awq`, `gptq`, `torchtitan`, `nanogpt`, `transformer-lens`, `nnsight`.
- **cloud-compute:** `skypilot` (generic multi-cloud; **adapt away** Modal/Tinker/Lambda-specific
  ones or keep as reference — openresearch uses RunPod/GKE/AKS).
- **research/scholar/writing/viz (thin seed):** `research/{hypothesis-generation,
  scientific-critical-thinking,literature-review}`, `scholar-evaluation`, `writing/ml-paper-writing`,
  `visualization/{matplotlib,scientific-visualization}`.
- **Adaptation rule:** the DeepSpeed skill (and some others) are raw doc-scrapes with scraper
  artifacts (e.g. "Synthetic Sciencestion" where "Orchestration" belonged) — clean these on copy.
  Where openresearch owns a bespoke version (`_RL_SCAFFOLD_BLOCK`, `_SDAR_BASELINES_BLOCK`), the
  ported RL/GRPO skill should *point at* the harness helper, not duplicate a divergent recipe.

## 6. Release-1 implementation breakdown (delegate to Sonnet, Opus reviews each diff)

1. **Vendor skills** (§5) into `backend/agents/rlm/skills/<category>/<name>/` + write
   `backend/agents/rlm/skills/NOTICE` (Apache-2.0 attribution to Synthetic Sciences/OpenScience).
2. **`skill_catalog.py`** — pure loader (frontmatter → `{name,desc,category,tags}`), injection-phrase
   sanitization, memoized, fail-soft. Unit tests over real on-disk fixtures (no mocks).
3. **`skill_matcher.py`** — pure `match_skills(claim_map, env_spec, catalog) →
   {domain, skill_names, reasons}` over the skills' tags. Unit tests.
4. **`consult_skill` primitive** — full-body return + fuzzy "did-you-mean"; register in
   `PRIMITIVE_REGISTRY`; bump `tests/rlm/test_registry.py::EXPECTED` 18→19 + `CLAUDE.md` counts;
   advertise in `system_prompt.py` only when `OPENRESEARCH_SKILLS` on. Extend `detect_environment`
   to call `match_skills` + persist `rlm_state/skill_match.json`. Add `_skill_shortlist_block` into
   `_compute_constraint_guidance` (near `_load_paper_override`). Ship `scripts/` via
   `_HARNESS_CODE_HELPERS`.
5. **WS① per-provider prompt tails** (`OPENRESEARCH_PROVIDER_PROMPTS`) — split `system_prompt.py`
   core + per-family tail keyed off the resolved root token; content per openresearch failure modes.
6. **WS② literature connectors** (`backend/services/knowledge/connectors/`, `OPENRESEARCH_LITERATURE_GROUNDING`)
   + `literature_claim_gate.py` (`OPENRESEARCH_LITERATURE_CLAIM_GATE`, advisory `run_warning`), wired
   into `rubric_gen.py` as a deterministic helper.
7. **WS④ evidence-report section** (`OPENRESEARCH_EVIDENCE_REPORT_SECTION`) — render
   `evidence_bundle`+`validation`+`claim_grounding` into `report.py:_render_markdown`.
8. **Tests/discipline:** each item ships hermetic OFF (byte-identical: prompt unchanged, primitive
   count 18, `_compute_constraint_guidance`/`_render_markdown` unchanged) + ON tests. `ruff` clean;
   full `pytest` green. Flips **no** defaults.

Later releases (2-5): spec §7. Native cloud (§5.E seams above) is Release 2.

## 7. How we work (standing rules — from operator memory)

- **Opus designs + reviews every diff; Sonnet executes** (impl code included) against the tight
  spec. Use the **`/implement` skill** for implementation — **never** `implement_codex`/`codex:*`.
  Delegate trivial/mechanical work to Sonnet; keep Opus on orchestrator/debugging/architecture.
- **Default-OFF, byte-identical-when-off, TDD, hermetic ON+OFF tests, `ruff@0.15.16` clean.** No
  default-flip without **≥3 paired A/B runs + the grader-σ gate** (this program flips none).
- **Root-level elegant solutions** — one canonical abstraction + a guard test, not scattered
  patches (e.g. the loader/matcher are pure single-purpose modules; the critique gate REUSES
  `run_validation_panel`).
- **Git:** push **only to `deepinvent`** (`Deepinvent/scientific_article_generator`), never
  `origin/openresearch`, never on the default branch without branching. Current branch:
  `reconcile/grounded-self-improvement-on-main`. **No `Co-Authored-By`/AI-attribution trailer.**
  Author = local config (lolout1 / appradhann@gmail.com). No Conventional-Commit prefixes;
  descriptive present-tense headlines. Commit **infrequently** (milestones, not per-fix). **The
  spec + this handoff are currently UNCOMMITTED** — commit them with Release 1 (or when asked).

## 8. Pointers

- **Spec (source of truth):**
  `docs/superpowers/specs/2026-07-05-openscience-skill-library-and-harness-enhancements-design.md`
- **OpenScience clone:** `/home/abheekp/openscience-ref` (Apache-2.0).
- **openresearch:** `/home/abheekp/openresearch`, branch `reconcile/grounded-self-improvement-on-main`.
- **Memory:** `project_openscience_skill_port` (index in `MEMORY.md`); related
  [[project_lifecycle_driver]], [[project_reasoning_chat_root_guardrails]],
  [[feedback_delegation]], [[feedback_git_remote]].
- **Task list:** the session TaskList (#4 writing-plans, #5 execute Phase 1) tracks next steps —
  recreate if the context clear drops it.
```
