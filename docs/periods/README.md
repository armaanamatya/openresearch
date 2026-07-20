# Development timeline and consolidation contract

These period dossiers are the intended reading surface for people and LLMs.
They organize the project by the time in which a decision was made, instead of
by the accidental document type (plan, brainstorm, handoff, audit, or log).

## Non-loss rule

No historical context is deleted merely because it is inconvenient to read.
Before a source document can leave the repository, its information must be
captured in the relevant period dossier with enough detail to answer:

1. **What problem, failure, or opportunity was observed?** Include the
   affected subsystem and the evidence or run that established it.
2. **What was decided and why?** State alternatives, safety/default posture,
   and the constraint that drove the choice.
3. **What changed?** Name the code/configuration/artifact boundary, the
   implementation mechanism, and the source commit(s) or original record.
4. **What was the result?** Record validation performed, observed outcome,
   known limitations, and any remaining operator action.
5. **Who/what is the authority?** Attribute decisions to the dated source
   record and commits; use a named human owner only when the source explicitly
   provides one. Do not invent ownership.

The restored `docs/history/`, `docs/archive/`, `docs/audits/`,
`docs/runbooks/`, and related records remain the evidence corpus while this
consolidation is reviewed. They are not an invitation to start work from old
instructions: current code and the operating docs remain authoritative.

## How to use this set

Read the matching period first. Use its source ledger to follow a topic back
to the exact original plan, incident, audit, handoff, run artifact, or commit
when a decision needs full fidelity. A later compression may reduce the number
of source files only after the ledger has been checked for complete coverage.

- [2026-05 — foundation](2026-05.md)
- [2026-06 — execution and evidence](2026-06.md)
- [2026-07 — durable cloud and scheduler](2026-07.md)
