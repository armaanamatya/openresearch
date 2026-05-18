/**
 * Map a raw failure message from a halted run into a plain-English
 * summary + a remedy + (optionally) an actionable button.
 *
 * The orchestrator's failure messages come straight from Python stack
 * traces today — operators have to recognise WallClockExceeded /
 * AuthenticationError / ValidationError / BudgetExhausted to know what
 * to do next. This helper centralises that translation so the UI can
 * say "the LLM took too long — retry with extended timeout?" instead
 * of dropping a 30-line traceback into the panel.
 *
 * When `kind === "wall_clock"` the action calls /api/demo/resume with
 * executionMode=max — which the orchestrator's auto-resume-from-
 * checkpoint logic picks up at the last completed stage, so the user
 * doesn't lose every earlier agent's work.
 */

export type FailureKind =
  | "wall_clock"
  | "authentication"
  | "validation"
  | "budget"
  | "disk_full"
  | "rate_limit"
  | "docker_unreachable"
  | "unknown";

export interface FailureAction {
  /** Button label shown on the failure panel. */
  label: string;
  /** Endpoint the button POSTs to (with the current projectId). */
  endpoint: "resume" | null;
  /** Override body merged into the resume request. */
  overrides: Record<string, string>;
}

export interface FailureSummary {
  kind: FailureKind;
  /** One-line plain-English summary suitable for a heading. */
  headline: string;
  /** Two-to-three-line explanation of what went wrong and why. */
  explanation: string;
  /** Concrete next-step the operator should consider. */
  remedy: string;
  /** Optional button — null when no automated action makes sense. */
  action: FailureAction | null;
}

const WALL_CLOCK_RE = /WallClockExceeded:\s*Agent\s+'?([^'\s]+)'?\s+hit\s+wall_clock\s+cap\s+of\s+(\d+)/i;
const AUTH_RE = /AuthenticationError|Incorrect API key|invalid_api_key|401\b/i;
const VALIDATION_RE = /ValidationError|pydantic_core/i;
const BUDGET_RE = /BudgetExhausted|budget cap/i;
const DISK_RE = /no space left on device|ENOSPC/i;
const RATE_LIMIT_RE = /RateLimited|rate.limit|429\b|insufficient_quota/i;
const DOCKER_UNREACHABLE_RE = /Docker daemon is not reachable|Cannot connect to the Docker daemon|docker.*FileNotFoundError.*docker\.sock|Error while fetching server API version/i;

export function summariseFailure(rawError: string | null | undefined): FailureSummary | null {
  if (!rawError) return null;
  const error = rawError.trim();

  const wallMatch = error.match(WALL_CLOCK_RE);
  if (wallMatch) {
    const agentName = wallMatch[1];
    const capSeconds = Number(wallMatch[2]);
    const capMinutes = Math.round(capSeconds / 60);
    return {
      kind: "wall_clock",
      headline: `${agentName} ran out of time (${capMinutes}-minute cap)`,
      explanation:
        `The agent was still working when its per-agent wall-clock cap fired. ` +
        `Heavy stages on complex papers (baseline-implementation in particular) ` +
        `can need an hour or more — the default cap is tuned for the PPO demo.`,
      remedy:
        `Retry with executionMode=max (1-hour per-agent cap). The pipeline ` +
        `resumes from the last on-disk checkpoint, so earlier agents don't re-run.`,
      action: {
        label: "Retry with extended timeout",
        endpoint: "resume",
        overrides: { executionMode: "max" }
      }
    };
  }

  if (AUTH_RE.test(error)) {
    return {
      kind: "authentication",
      headline: "Provider authentication failed (401)",
      explanation:
        `An LLM provider rejected the credentials. If you only set up one ` +
        `provider (anthropic), an invalid OPENAI_API_KEY in shell env can ` +
        `kill runs when the fallback chain tries it.`,
      remedy:
        `Set REPROLAB_PROVIDER_FALLBACK_DISABLED=true in .env and restart the ` +
        `backend, or unset OPENAI_API_KEY in the backend's shell before starting it.`,
      action: null
    };
  }

  if (RATE_LIMIT_RE.test(error)) {
    return {
      kind: "rate_limit",
      headline: "Provider rate limit hit",
      explanation:
        `The LLM provider returned a 429 / quota-exhausted. The orchestrator's ` +
        `backoff retried but eventually gave up.`,
      remedy:
        `Wait a few minutes for the rate-limit window to refresh, then resume ` +
        `from the last checkpoint.`,
      action: {
        label: "Resume from last checkpoint",
        endpoint: "resume",
        overrides: {}
      }
    };
  }

  if (VALIDATION_RE.test(error)) {
    return {
      kind: "validation",
      headline: "An agent emitted malformed structured output",
      explanation:
        `The LLM returned JSON that didn't match a required Pydantic schema ` +
        `(e.g. an enum field carrying trailing rationale). Most of these are ` +
        `now caught and dropped fail-soft, but a stray one slipped through.`,
      remedy:
        `Resume from the last checkpoint — the next attempt usually emits valid ` +
        `JSON.`,
      action: {
        label: "Resume from last checkpoint",
        endpoint: "resume",
        overrides: {}
      }
    };
  }

  if (BUDGET_RE.test(error)) {
    return {
      kind: "budget",
      headline: "Run exceeded the configured budget",
      explanation:
        `The cost or wall-clock budget for the whole run was exhausted. The ` +
        `orchestrator stops rather than burning more compute.`,
      remedy:
        `Bump REPROLAB_RUN_BUDGET_USD or run-budget settings, restart the ` +
        `backend, and resume.`,
      action: null
    };
  }

  if (DOCKER_UNREACHABLE_RE.test(error)) {
    return {
      kind: "docker_unreachable",
      headline: "Docker daemon is not reachable from the backend",
      explanation:
        `The orchestrator subprocess couldn't open the Docker socket — usually ` +
        `this means the daemon isn't running, OR the default socket symlink at ` +
        `/var/run/docker.sock points at a runtime (OrbStack/colima) that isn't ` +
        `currently up while another runtime (Docker Desktop) is.`,
      remedy:
        `Start the runtime your shell uses (open Docker Desktop or OrbStack), ` +
        `OR set DOCKER_HOST to the live socket and restart the backend. On ` +
        `macOS Docker Desktop's socket is usually ~/.docker/run/docker.sock.`,
      action: null
    };
  }

  if (DISK_RE.test(error)) {
    return {
      kind: "disk_full",
      headline: "Docker ran out of disk space during environment build",
      explanation:
        `The Track 4 environment build attempted to install dependencies and ` +
        `Docker's storage layer filled up.`,
      remedy:
        `Run "docker system prune -af --volumes" to reclaim space, then resume.`,
      action: {
        label: "Resume after pruning",
        endpoint: "resume",
        overrides: {}
      }
    };
  }

  // Generic fall-through — we still parse the leading line for a headline so
  // the panel says something useful even when the failure isn't recognised.
  const firstLine = error.split(/\r?\n/, 1)[0]?.slice(0, 160) ?? "Run halted";
  return {
    kind: "unknown",
    headline: firstLine,
    explanation:
      `The orchestrator halted with an unrecognised failure type. The full ` +
      `error is in the technical details below.`,
    remedy:
      `Try resuming from the last checkpoint. If the failure repeats, inspect ` +
      `runner.stderr.log under the project's output directory.`,
    action: {
      label: "Resume from last checkpoint",
      endpoint: "resume",
      overrides: {}
    }
  };
}
