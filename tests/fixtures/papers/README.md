# Paper archetype labels

One subdirectory per arxiv ID. Each holds `archetype.txt` containing exactly
one label, newline-terminated, no other content.

Allowed labels:
- `rl-agent`
- `nlp-eval`
- `cv-ablation`
- `optimization`
- `other`

Consumed by the GEPA driver (`scripts/optimize_prompts_gepa.py`) — archetype
flows into the reflective-dataset record as a trace field, never as a candidate
input. See `docs/superpowers/specs/2026-05-26-gepa-phase0-audit.md` §0.5.
