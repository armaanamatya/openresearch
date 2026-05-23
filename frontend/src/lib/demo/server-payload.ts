import "server-only";

import type { LiveDemoRunState } from "./demo-run-types";

// The 14-stage pipeline dashboard enrichment (buildLiveDemoDashboard) has
// been deleted with pipeline-dashboard.ts.  The RLM backend populates run
// state directly; enrichment is a no-op pass-through.
export async function enrichRunStateWithPayload(
  state: LiveDemoRunState
): Promise<LiveDemoRunState> {
  return state;
}
