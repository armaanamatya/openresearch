// Tunable RLM-lab policy constants — values the UI uses to derive displayed
// state, kept out of component bodies so they are named and reviewable.

/**
 * Rubric scores at or below this are flagged as a possibly-degraded run (a run
 * that produced no measured metrics is capped here by the backend's rubric
 * verifier — the I7 cap). The UI shows a hedged "may be a degraded run" note.
 */
export const DEGRADED_SCORE_CAP = 0.35;
