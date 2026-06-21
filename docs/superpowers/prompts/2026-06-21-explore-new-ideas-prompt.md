# Terminal prompt — explore NEW improvement ideas for OpenResearch

> Paste the block below into a fresh Claude Code session at the repo root. It runs a
> grounded, multi-angle exploration for **new, non-obvious** ideas — and is built to NOT
> rehash what's already cataloged. Reusable: run it whenever you want a fresh idea sweep.

---

```
Explore NEW improvement ideas for this codebase (OpenResearch: an RLM agent that
reproduces ML papers — ingest paper -> implement -> run on GPU -> score vs a rubric;
North Star = the SDAR paper 2605.15155 reproduced at the highest GROUNDED score).

DO NOT just re-derive the existing menu. First READ these so you find what's MISSING from
them, not what's already there:
- docs/superpowers/specs/2026-06-21-system-improvement-opportunities.md  (the 7-theme menu)
- docs/superpowers/plans/2026-06-21-dark-switches-plan.md                (T1/T6 in flight)
- docs/superpowers/prompts/2026-06-20-sdar-unification-megaprompt.md     (the SDAR goal)
- docs/superpowers/specs/2026-06-21-evidence-first-architecture-adr.md   (honesty design)
- CLAUDE.md + system_overview.md                                        (how/why)
Treat anything already in those as KNOWN; your job is the next layer.

METHOD (read-only research, then synthesize — zero GPU / zero live-API spend):
1. Dispatch parallel read-only research agents (superpowers:dispatching-parallel-agents),
   each on a DIFFERENT angle the existing menu under-covers, e.g.:
   - Failure-mode mining: read real run artifacts (runs/*/final_report.json,
     experiment_runs.jsonl, dashboard_events.jsonl, best_runs/*) and the failure_class
     taxonomy — what recurring failure are we NOT yet guarding?
   - Adversarial/hallucination red-team: how could a run still fake a high score given
     today's gates? (probe evidence_gate, claim_grounding, external_validator, leaf_scorer)
   - Cross-paper generality: where is the pipeline secretly SDAR-specific / hand-tuned and
     failing silently on arbitrary arXiv papers?
   - Agent-ergonomics: where does the root model waste iterations / get confused by the
     primitive surface, prompts, or error messages? (run.py, system_prompt.py, primitives)
   - Novel capabilities: what new primitive, signal, or loop (not just a fix) would raise
     the score ceiling? (e.g. retrieval over the Run/Test Bank, self-consistency, tool use)
   Each agent returns a TIGHT ranked list: idea -> file/mechanism anchor -> expected impact
   (score / honesty / cost / coverage) -> effort (S/M/L). Quality over quantity (~6-10 each).
2. Synthesize into ONE ranked catalog grouped by theme, DE-DUPED against the existing menu
   (explicitly note "already covered" vs "NEW"). Call out the 3 highest-leverage NEW ideas.
3. Write it to docs/superpowers/specs/2026-06-21-new-ideas-round-2.md and commit on a
   branch off the trunk (feat/bes-conversion-correctness), NOT main. Do NOT touch main.

RULES: ground every idea in code (file:line) — no vague suggestions; mark each NEW vs
already-in-the-menu; honesty-over-hype (an unverifiable idea is labeled speculative);
respect the repo discipline (default-OFF + fail-soft for any new flag; >=3 paired A/B before
flipping a default; no GPU/API spend to "verify"). End by asking which idea(s) to turn into
a spec + plan via superpowers:brainstorming -> writing-plans.

START by reading the 5 docs above, then dispatch the angle-agents.
```

---

## Why it's shaped this way
- **Anti-duplication:** it forces reading the existing 7-theme menu first and labeling each
  idea NEW vs already-covered, so you get the *next* layer instead of the same list.
- **New angles, not just fixes:** failure-mode mining from real artifacts, an adversarial
  red-team, cross-paper generality, agent-ergonomics, and net-new capabilities — angles the
  first sweep under-covered.
- **Grounded + safe:** read-only, code-anchored, zero spend, default-OFF discipline, lands on
  a branch off the trunk (never `main`).
