/**
 * replay.ts — fixture replay harness for the RLM lab UI.
 *
 * Two call shapes:
 *   replayFixture("instant")              → returns the full fixture array immediately.
 *   replayFixture("timed", onUpdate)      → emits the fixture as a growing prefix via
 *                                           setInterval (~150 ms per event); returns a
 *                                           stop() function that cancels any pending timer.
 *
 * This is a plain harness module (no React). Used by:
 *   - vitest unit tests (instant mode)
 *   - the ?rlmFixture=1 dev path in lab-shell.tsx (timed mode)
 *   - Playwright e2e (either mode via the dev path)
 */

import type { RlmDashboardEvent } from "../../../lib/events/rlm-events";
import { rlmRunFixture } from "./__fixtures__/rlm-run.fixture";

const INTERVAL_MS = 150;

export function replayFixture(mode: "instant"): RlmDashboardEvent[];
export function replayFixture(
  mode: "timed",
  onUpdate: (events: RlmDashboardEvent[]) => void
): () => void;
export function replayFixture(
  mode: "instant" | "timed",
  onUpdate?: (events: RlmDashboardEvent[]) => void
): RlmDashboardEvent[] | (() => void) {
  if (mode === "instant") {
    return rlmRunFixture;
  }

  // timed mode — emit growing prefixes via setInterval
  let index = 0;
  const id = setInterval(() => {
    index += 1;
    const slice = rlmRunFixture.slice(0, index);
    onUpdate!(slice);
    if (index >= rlmRunFixture.length) {
      clearInterval(id);
    }
  }, INTERVAL_MS);

  return () => {
    clearInterval(id);
  };
}
