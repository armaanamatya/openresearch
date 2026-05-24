import type { PrimitiveCallView } from "../../../hooks/use-rlm-run";

export interface PrimitiveErrorDetail {
  title: string;
  reason: string;
  recovery: string;
}

const RECOVERABLE_PRIMITIVES = new Set([
  "detect_environment",
  "build_environment",
  "plan_reproduction",
  "implement_baseline",
  "run_experiment",
]);

export function latestPrimitiveError(calls: PrimitiveCallView[]): PrimitiveCallView | null {
  for (let i = calls.length - 1; i >= 0; i--) {
    if (calls[i].status === "error") return calls[i];
  }
  return null;
}

function normalizeReason(summary: string | null): string {
  const raw = summary?.trim();
  if (!raw) return "Exception detail was not emitted for this primitive.";
  if (raw === "ValidationError") {
    return "Pydantic ValidationError. Field-level detail was not available on this event.";
  }
  if (raw.startsWith("ValidationError:")) {
    return raw.replace(/^ValidationError:\s*/, "Pydantic ValidationError: ");
  }
  return raw;
}

export function describePrimitiveError(call: PrimitiveCallView): PrimitiveErrorDetail {
  const recoverable = RECOVERABLE_PRIMITIVES.has(call.primitive);
  return {
    title: `ERROR · ${call.primitive}`,
    reason: normalizeReason(call.result_summary),
    recovery: recoverable
      ? "The root receives the full traceback and will retry, choose a fallback, or repair the next primitive on the next REPL turn."
      : "The root receives the full traceback and will decide whether to retry or continue on the next REPL turn.",
  };
}
