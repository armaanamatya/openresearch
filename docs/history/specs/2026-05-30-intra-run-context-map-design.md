<!-- doc-meta: status=current; last-verified=2026-06-07 -->
# Intra-run context map (PEEK-lite) — design

> **Status:** Implemented 2026-06-07 (`backend/agents/rlm/context_map.py`).
> Flag: `OPENRESEARCH_CONTEXT_MAP` (default off). Cited by `CLAUDE.md`.

## Problem

Within a single RLM run the root re-derives the same orientation facts
(datasets, metrics, hyperparameters, hardware, environment) across iterations —
each re-derivation is a full `understand_section` / `extract_hyperparameters` /
`detect_environment` primitive round-trip. A free, deterministic cache of what
was already learned lets the root skip the repeat.

## Design

A per-run JSON cache at `runs/<id>/rlm_state/context_map.json` shaped as
`{field: [values]}`.

- **Write hook** (`binding.py::wrap_primitive`, post-success): after a
  successful `understand_section` / `extract_hyperparameters` /
  `detect_environment`, `update_context_map` unions that primitive's structured
  output into the cache (scalar and list-of-scalar fields only; nested/empty
  values skipped). Union-per-field, deduped.
- **Read** (`read_context_map` primitive + `read_context_map()` helper): the
  root consults the cache before re-deriving. Registered in
  `PRIMITIVE_REGISTRY` / `PRIMITIVE_DESCRIPTIONS` and bound into `custom_tools`
  (so `tests/rlm/test_registry.py::EXPECTED` includes it).
- **Prompt**: a flag-gated `CONTEXT MAP` section in `system_prompt.py` tells the
  root to call `read_context_map()` before re-deriving — **only when the flag is
  on**.

## Bounds (deterministic)

`MAX_FIELDS=40`, `MAX_VALUES=8` per field, `MAX_BYTES=8192` serialized,
`MAX_VALUE_LEN=200` per value. The byte ceiling is enforced by dropping whole
fields (sorted by key, last first) until under `MAX_BYTES`.

## Invariants

- **Navigation aid only — never a report source.** The evidence gate
  (`OPENRESEARCH_EVIDENCE_GATE`) remains the backstop; nothing in the map may be
  cited as report evidence (a unit test asserts `report.py` never reads it).
- **Off-state is a strict no-op**: write no-ops, read returns `{}`, the prompt
  section is omitted.
- **Thread-safe** (a module lock guards the read-modify-write) and **fail-soft**
  (any error is swallowed — a broken orientation cache must never break a
  primitive call).

## Resolved open questions (implementer defaults)

- **8 KB ceiling = a hard cap on the serialized file** (not a per-write delta),
  with deterministic field-drop on overflow.
- **`read_context_map` is always registered** (returns `{}` when off) rather
  than registry-gated, to keep the bound-tool set stable across the flag.

## Tests

`tests/agents/rlm/test_context_map.py` (off-state, union, caps, byte ceiling,
fail-soft, primitive wiring) + the `test_registry.py` consistency assertion.
