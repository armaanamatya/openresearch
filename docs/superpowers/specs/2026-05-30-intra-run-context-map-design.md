# Intra-run Context Map (PEEK-lite) — Design

**Date:** 2026-05-30
**Phase:** 8 of the RLM wedge-hardening-and-evolution plan
**Status:** Approved design, pre-implementation
**Flag:** `REPROLAB_CONTEXT_MAP` (default **off**)
**Source paper (infra inspiration only):** PEEK (arXiv 2605.19932) — a bounded orientation cache evaluated *on* RLM, reported 93–145 fewer iterations vs base RLM.

---

## 1. Goal & non-goals

**Goal.** Give the RLM root model a free, deterministic, intra-run orientation cache so it stops re-deriving the same paper facts through paid `rlm_query` / `llm_query` sub-calls within a single run. The map accumulates the structured outputs of the three orientation primitives (`understand_section`, `extract_hyperparameters`, `detect_environment`) into a small (≤8 KB; a measured full SDAR pass is ~3.6 KB), bounded JSON artifact the root can read in one cheap primitive call.

**Why this is not redundant with `primitive_cache.py`.** The existing content-addressed cache returns the *same per-slice* result for a *byte-identical* slice. It cannot aggregate across *different* slices. On a multi-section paper the root calls `understand_section` once per section; the cache holds three independent per-slice entries but never a unified view. The context map's one new capability is exactly that cross-slice union — a single `understand_section:datasets` entry holding *all* datasets seen across *all* sections. That aggregation is the value proposition, and the cache structurally cannot provide it.

**Non-goals (v1, explicit):**
- No LLM distiller / cartographer / evictor (PEEK's full pipeline — rejected; PEEK's own ablation shows a no-eviction deterministic map already beats base RLM, and three extra LLM calls/iteration to manage a 1 KB artifact is the opposite of the cost goal).
- No cross-run persistence — the map is keyed to a single run and dies with it. Cross-run learning is owned by the separate MUSE negative-lessons track (Phase 9).
- No DELETE primitive. Correction is handled structurally by the union model (§4), not by a root-driven delete.
- **The map is never the primary report source** (§6) — it feeds *navigation*, not the final report.

## 2. Architecture

One new module, two hook points, one new primitive, one prompt line, one flag. Every piece is reversible by unsetting the flag.

```
backend/agents/rlm/context_map.py        (NEW)   — owns runs/<id>/rlm_state/context_map.json
backend/agents/rlm/binding.py            (HOOK)  — write hook on the success path (~line 513)
backend/agents/rlm/primitives.py         (HOOK)  — new read_context_map() primitive + registry
backend/agents/rlm/system_prompt.py      (LINE)  — one instruction near understand_section guidance
```

### 2.1 Module: `context_map.py`
Owns the artifact `runs/<project_id>/rlm_state/context_map.json`. Pure file I/O plus a module-level `threading.Lock` for the read-modify-write. Fail-soft on every path — observability/persistence must never block or crash a run (mirrors the contract of `primitive_cache.py` exactly). Public API:

- `is_enabled() -> bool` — `False` unless `REPROLAB_CONTEXT_MAP` is a truthy value (`"1"`, `"on"`, `"true"`); default off.
- `record(project_dir: Path, primitive: str, result: dict, *, iteration: int | None = None) -> None` — extract fields from `result` per the §3 rules and union them into the map. No-op when disabled, when `primitive` not in the orientation allowlist, when `result` is not a dict, or on any I/O error.
- `read(project_dir: Path) -> dict` — return the full map object `{"version", "bytes", "entries": [...]}`. Returns `{"version": "v1", "bytes": 0, "entries": []}` on missing/corrupt file (fail-soft).

### 2.2 Write hook (binding.py)
In `wrap_primitive`'s `wrapped()` success path, beside the existing `_emit_supplemental(name, result, ctx, _emit_extra)` call (binding.py:513 — reached only when `failed` is `False`), add:

```python
try:
    from backend.agents.rlm import context_map as _cmap
    _cmap.record(ctx.project_dir, name, result, iteration=getattr(ctx, "iteration", None))
except Exception:  # noqa: BLE001 — context map MUST NOT break the run
    pass
```

This site is the single DRY chokepoint: it already fires on every non-failed primitive return, has `ctx` and `result` in hand, and is skipped for timeouts/failures (those carry `error`/`success=False` and take the `failed` branch). Cache-hit returns also pass through here, replaying the same value to the same field — deduped to a no-op (§4).

### 2.3 Read primitive (primitives.py)
`read_context_map(*, ctx: "RunContext") -> dict` — pure file I/O, mirrors `check_user_messages` one-for-one. Returns `context_map.read(ctx.project_dir)`. Registered in `PRIMITIVE_REGISTRY`, `PRIMITIVE_DESCRIPTIONS`, and given a `PRIMITIVE_TIMEOUT_S` entry of `30` (matching the other pure-I/O primitives). Fail-soft: never raises; returns the empty-map shape if anything goes wrong.

### 2.4 Prompt line (system_prompt.py)
A `_CONTEXT_MAP_SECTION` constant appended by `build_system_prompt` **only when `REPROLAB_CONTEXT_MAP` is enabled** (so the default path is never instructed to call an empty map — see §7). The instruction:

> Before re-deriving a known fact via `rlm_query` / `llm_query`, call `read_context_map()` — it accumulates the datasets, metrics, hyperparameters, and environment facts already extracted this run (each with provenance). Treat its entries as heuristic hints, not ground truth; a field may list several observed values across paper sections.

## 3. Field-extraction rules (per primitive)

`record(...)` reads only an allowlist of fields per primitive and ignores everything else (notably `_meta`, `outcome`, and the heavy `dockerfile` blob).

Fields are listed (and recorded) **valuable-first** so the incremental byte ceiling (§4) keeps the high-value entries when a call is large.

| Primitive | Field key | Source field | Element type | Union rule |
|---|---|---|---|---|
| `understand_section` | `understand_section:datasets` | `datasets` (list[dict]) | dict | union elements |
| `understand_section` | `understand_section:metrics` | `metrics` (list[dict]) | dict | union elements |
| `understand_section` | `understand_section:training_recipe` | `training_recipe` (dict) | dict | union whole dict as one element |
| `understand_section` | `understand_section:hardware_clues` | `hardware_clues` (list) | scalar/dict | union elements |
| `extract_hyperparameters` | `extract_hyperparameters:<slot>` | `optimizer`, `learning_rate`, `batch_size`, `epochs_or_steps`, `scheduler` | scalar | union distinct non-null scalars |
| `detect_environment` | `detect_environment:framework` | `framework` | scalar | union distinct non-null |
| `detect_environment` | `detect_environment:python_version` | `python_version` | scalar | union distinct non-null |

Skipped intentionally: `understand_section.ambiguities` (verbose — 7+ dicts per call — and the *least* fact-like field; it would consume the byte budget and crowd out real facts); `understand_section._meta`; `extract_hyperparameters.other_hparams` (free-form dict, unbounded — defer) and `_meta`; `detect_environment.dockerfile` (large, already on disk) and all other EnvironmentSpec fields.

Null/empty values are never written (a heuristic that returns `batch_size: null` contributes nothing).

## 4. Data shape, union, dedup, and bounding

`context_map.json` is one atomic-written JSON object:

```json
{
  "version": "v1",
  "bytes": 1180,
  "entries": [
    {
      "key": "understand_section:datasets",
      "primitive": "understand_section",
      "field": "datasets",
      "confidence": "heuristic",
      "values": [
        {"value": {"name": "ALFWorld", "...": "..."}, "dedup": "a1b2c3d4e5f6a1b2",
         "slice_hash": "ff00aa11", "iteration": 1, "ts": "2026-05-30T00:01:00Z"},
        {"value": {"name": "WebShop", "...": "..."}, "dedup": "99cc88dd77ee6655",
         "slice_hash": "bb22cc33", "iteration": 2, "ts": "2026-05-30T00:02:00Z"}
      ]
    },
    {
      "key": "extract_hyperparameters:batch_size",
      "primitive": "extract_hyperparameters",
      "field": "batch_size",
      "confidence": "heuristic",
      "values": [
        {"value": 8,  "dedup": "5d41402abc4b2a76", "slice_hash": "aa01", "iteration": 2, "ts": "..."},
        {"value": 16, "dedup": "6512bd43d9caa6e0", "slice_hash": "bb02", "iteration": 3, "ts": "..."}
      ]
    }
  ]
}
```

**Union model.** Each entry is keyed `primitive:field` and holds a deduplicated list of observed values:
- **List-valued source fields** (datasets, metrics, hardware_clues, ambiguities) — the list is *flattened* and each element is unioned individually. So `understand_section:datasets` accumulates `[ALFWorld, WebShop, Search-QA]` across SDAR's three environment sections rather than keeping only the last section's list.
- **Scalar/dict source fields** (batch_size, framework, training_recipe) — the whole value is one element. `extract_hyperparameters:batch_size` accumulates `[8, 16, 32]` across SDAR's three model sizes.

**Dedup.** Each element's `dedup` id is `sha256(json.dumps(element, sort_keys=True, default=str))[:16]`. Adding an element whose `dedup` already exists in the entry is a **no-op** (first-seen provenance is kept). This makes the write hook idempotent under cache-hit replays (identical slice → identical primitive output → identical elements → no growth) and across repeated identical re-runs.

**Provenance** lives per-value: `slice_hash` (`sha256` prefix of the originating `text_slice`/`method_spec`, when available), `iteration`, `ts`. The map carries no raw paper text — only the structured extracted values.

**Bounding (deterministic, refuse-new-keep-existing):**
- Max **entries** (distinct fields): **40**. A *new* field key is refused once the map holds 40 entries (keeps the earlier, foundational orientation facts; the root orients first). Logged, never raised.
- Max **values per entry**: **8**. A *new* element is refused once an entry holds 8 values. A dedup-hit on an existing element always succeeds (it does not grow the entry).
- **Byte ceiling, enforced incrementally: 8192 bytes (8 KB).** Within a single `record()` call, observations are added valuable-first (per `_FIELD_SPEC` order); the serialized size is checked after each addition, and the *first* value that would exceed the ceiling is undone and the rest of that call skipped. This is deliberately **not** an all-or-nothing rollback: an all-or-nothing ceiling let one verbose field shut out the whole map, including the small high-value entries. Incremental enforcement keeps what fits, valuable-first. **Why 8 KB, not the ~1.5 KB a PEEK-style snapshot uses:** this map is a `read_context_map()` *return value*, never injected into the prompt, so its size never touches the prompt cache — the cache-prefix size rationale does not apply. A measured full SDAR orientation pass (3 model sizes × 3 environments) is ~3.6 KB; at 2 KB the incremental ceiling silently truncated ~half the multi-section accumulation (the very loss the union keying exists to prevent, reappearing at the byte layer). 8 KB holds it with >2× headroom; the only cost of a larger cap is REPL tokens when the root reads it, far cheaper than the `rlm_query` re-derivation it replaces.

Writes are serialized by a module-level `threading.Lock` (the orientation primitives may run concurrently on threads in the run subprocess — system_prompt.py:312 encourages `ex.map(understand_section, slices)`), then persisted atomically via `tmp = path.with_suffix(".json.tmp"); tmp.write_text(...); os.replace(tmp, path)` (the established pattern, primitives.py:3994-3999).

## 5. Consumption

The root reads the map by calling `read_context_map()` — a single cheap primitive call returning the entries. It is *not* injected as a live REPL variable (RLM has no refresh-per-iteration variable mechanism; a primitive matches the existing `check_user_messages` model and is lower-risk). The §2.4 prompt line tells the root to consult it before spending an `rlm_query` / `llm_query` on a fact it may already hold. **The prompt line is itself flag-gated** — `build_system_prompt` appends it only when `REPROLAB_CONTEXT_MAP` is on (§7), so the default path is not instructed to call an empty map.

## 6. Contamination safety

The central hazard of any context map is a wrong cached fact poisoning downstream work. Four properties bound the blast radius:

1. **Additive, never clobbering.** The union model (§4) means the map *accumulates* observations and never silently replaces a real fact with a wrong one. The worst case is an *extra* spurious value listed alongside the real ones, each with provenance — the root sees both and can disambiguate. This is strictly safer than a "latest-wins" map, which would present a confidently-incomplete single value (exactly the artifact most likely to be copied verbatim).
2. **Map ≠ primary report source.** The final report is built from experiment evidence + `verify_against_rubric`, and Phase 3's evidence gate (`REPROLAB_EVIDENCE_GATE`) is the backstop that downgrades a `reproduced`/`partial` verdict lacking real experiment evidence. The map feeds the root's *navigation*. A wrong map value at worst wastes one iteration (the root acts on it; the experiment / verify step corrects it). It introduces **no new path to the report** that the orientation primitives didn't already have — surfacing a fact via `read_context_map` raises its salience, but the union model ensures what's surfaced is the *complete* set of observations, not a misleading singleton.
3. **Provenance + confidence framing.** Every value carries `source`/`slice_hash`/`confidence: heuristic`, and the prompt frames entries as hints, not ground truth.
4. **Intra-run scope.** A wrong value dies with the run. No future run of the same paper inherits it.

## 7. Flag & rollback

`REPROLAB_CONTEXT_MAP` — **default off**. A prototype behind a flag, consistent with the "every behavior change reversible" posture of this plan. `=on`/`=1`/`=true` opts in. When off: the write hook no-ops, `read_context_map()` returns the empty-map shape, **and the §2.4 prompt instruction is omitted** (`build_system_prompt` appends it only when the flag is on). Critically, the tool's `PRIMITIVE_DESCRIPTIONS` entry — which *does* ship in the auto-generated tool inventory regardless of the flag — is purely **declarative and self-disclosing** ("returns … ; returns an empty map unless `REPROLAB_CONTEXT_MAP` is enabled"), carrying **no imperative** to call it. The "call it before re-deriving a fact" instruction lives *only* in the gated `_CONTEXT_MAP_SECTION`. So with the flag off the root is never told to call the primitive and won't burn a REPL action on an empty read. The only off-state residue is the declarative description sitting in the cached prefix (~50 tokens, no LLM cost, no instructed calls). Rollback = unset the flag (no migration; the artifact is per-run and ephemeral).

## 8. Measurement

Validate with an A/B mirroring the accelerator runbook: `REPROLAB_CONTEXT_MAP=on` vs `off` across **≥3 paired SDAR runs**, comparing:
- **iteration count** (the PEEK metric — primary),
- **`rlm_query` + `llm_query` call counts** (expected to drop — the direct cost lever),
- **wall-clock**,
- **`final_report.json::rubric.overall_score`** (a guard — must not regress).

**Expectation-setting.** PEEK's 93–145-iteration headline was measured on base RLM with *no* primitive cache. ReproLab already ships `primitive_cache.py` caching these same primitives per-slice, so the map's incremental value is *over an existing cache* — expect a **smaller** delta, and do not read a modest improvement as failure. Make-default only if iteration and navigation-call counts drop with no score regression across the paired runs.

## 9. Testing (TDD)

- **`context_map.py` unit** — union accumulates distinct values; dedup-hit is a no-op (idempotent); list fields flatten-and-union elements; scalar fields accumulate distinct scalars; entry cap refuses new keys but keeps existing; value cap refuses new values but keeps existing; byte ceiling rolls back an over-budget mutation; atomic write; concurrent writes (two threads, same field) serialize without loss; corrupt/missing file → empty-map read (fail-soft); null/empty source values are skipped.
- **`read_context_map` primitive unit** — empty map → empty entries; populated map → entries; missing/corrupt file → empty shape (fail-soft); never raises.
- **Write-hook unit (binding)** — an orientation primitive's success writes entries; a non-orientation primitive (e.g. `run_experiment`) writes nothing; a failed orientation result (carries `error`) writes nothing; flag off → no writes; the hook never propagates an exception. **Plus an end-to-end test:** the *real* `understand_section` heuristic (not a synthetic dict) flowing through `build_custom_tools` populates the map — guards against a silent empty-map regression if the primitive's output shape drifts.
- **Prompt-gating unit (system_prompt)** — `build_system_prompt` *includes* the `read_context_map` instruction when the flag is on and *omits* it when off. This proves the default path stays inert rather than cementing an unconditional change.
- **Integration (SDAR clobber regression)** — call `extract_hyperparameters` returning `batch_size=8` then `batch_size=16` on two different slices; assert the map holds **both** under `extract_hyperparameters:batch_size`. Call `understand_section` with `datasets=[ALFWorld]` then `datasets=[WebShop]`; assert the map holds **both**. This is the exact failure the union model exists to prevent.
- **Egress contract** — `read_context_map`'s result flows through the normal `primitive_call` summary bounding in `wrap_primitive`; the artifact is a file, never an SSE payload, so no `sse_bridge` allowlist entry is needed.

## 10. Files

- **Create:** `backend/agents/rlm/context_map.py`
- **Create:** `tests/agents/rlm/test_context_map.py`
- **Create:** `tests/agents/rlm/test_read_context_map_primitive.py`
- **Create:** `tests/agents/rlm/test_context_map_write_hook.py`
- **Modify:** `backend/agents/rlm/binding.py` (write hook ~line 513)
- **Modify:** `backend/agents/rlm/primitives.py` (`read_context_map` + registry/descriptions/timeout)
- **Modify:** `backend/agents/rlm/system_prompt.py` (one prompt line)
- **Modify:** `CLAUDE.md` (document the flag under a new sub-section) and the master plan's implementation-status table.
