import { describe, expect, it } from "vitest";

import { summariseFailure } from "./failure-summary";

describe("summariseFailure", () => {
  it("returns null for empty input", () => {
    expect(summariseFailure(null)).toBeNull();
    expect(summariseFailure("")).toBeNull();
    expect(summariseFailure(undefined)).toBeNull();
  });

  it("recognises WallClockExceeded and offers retry-with-extended-timeout", () => {
    const raw =
      "WallClockExceeded: Agent 'baseline-implementation' hit wall_clock cap of 1200 after 1205.4s (1129 chars of partial output preserved)";
    const summary = summariseFailure(raw);
    expect(summary).not.toBeNull();
    expect(summary?.kind).toBe("wall_clock");
    expect(summary?.headline).toMatch(/baseline-implementation/);
    expect(summary?.headline).toMatch(/20-minute/);
    expect(summary?.action).not.toBeNull();
    expect(summary?.action?.endpoint).toBe("resume");
    expect(summary?.action?.overrides).toEqual({ executionMode: "max" });
  });

  it("recognises AuthenticationError 401 without offering automated retry", () => {
    const raw =
      "AuthenticationError: Error code: 401 - {'error': {'message': 'Incorrect API key provided: sk-svc...'}}";
    const summary = summariseFailure(raw);
    expect(summary?.kind).toBe("authentication");
    expect(summary?.headline).toMatch(/401|authentication/i);
    // Auth is a config error the UI can't fix automatically — no button.
    expect(summary?.action).toBeNull();
    expect(summary?.remedy).toMatch(/REPROLAB_PROVIDER_FALLBACK_DISABLED|OPENAI_API_KEY/);
  });

  it("recognises ValidationError and offers checkpoint resume", () => {
    const raw =
      "pydantic_core._pydantic_core.ValidationError: 1 validation error for ImprovementHypothesis";
    const summary = summariseFailure(raw);
    expect(summary?.kind).toBe("validation");
    expect(summary?.action?.endpoint).toBe("resume");
    expect(summary?.action?.overrides).toEqual({});
  });

  it("recognises BudgetExhausted without an automated action", () => {
    const summary = summariseFailure("BudgetExhausted: run budget cap of $5.00 exceeded");
    expect(summary?.kind).toBe("budget");
    expect(summary?.action).toBeNull();
  });

  it("recognises rate-limit / quota with a resume action", () => {
    const summary = summariseFailure("openai.RateLimitError: insufficient_quota");
    expect(summary?.kind).toBe("rate_limit");
    expect(summary?.action?.endpoint).toBe("resume");
  });

  it("recognises Docker-unreachable failures from the sandbox preflight", () => {
    const raw =
      "Sandbox preflight failed: Docker daemon is not reachable from this Python environment. ... " +
      "Original error: Error while fetching server API version: ('Connection aborted.', FileNotFoundError(2, 'No such file or directory'))";
    const summary = summariseFailure(raw);
    expect(summary?.kind).toBe("docker_unreachable");
    expect(summary?.remedy).toMatch(/Docker Desktop|OrbStack|DOCKER_HOST/);
    // No automated action — restart the daemon / fix env is operator work.
    expect(summary?.action).toBeNull();
  });

  it("recognises disk-full from Docker build", () => {
    const summary = summariseFailure(
      "docker: failed to register layer: no space left on device"
    );
    expect(summary?.kind).toBe("disk_full");
    expect(summary?.remedy).toMatch(/docker system prune/);
    expect(summary?.action?.endpoint).toBe("resume");
  });

  it("falls back to a generic summary for unrecognised errors but still offers resume", () => {
    const summary = summariseFailure("Something exploded mid-pipeline");
    expect(summary?.kind).toBe("unknown");
    expect(summary?.headline).toMatch(/Something exploded/);
    expect(summary?.action?.endpoint).toBe("resume");
  });
});
