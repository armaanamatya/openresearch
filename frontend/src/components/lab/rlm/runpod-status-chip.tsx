"use client";

import { useMemo } from "react";
import type { DemoRunpodStatusResponse, DemoSandboxMode } from "../../../lib/demo/demo-run-types";
import type { PrimitiveCallView, RlmRunState } from "../../../hooks/use-rlm-run";
import { useRunpodStatus } from "../../../hooks/use-runpod-status";
import styles from "./runpod-status-chip.module.css";

interface RunpodStatusChipProps {
  projectId: string;
  sandboxMode?: DemoSandboxMode | null;
  status: RlmRunState["status"];
  primitiveCalls: PrimitiveCallView[];
  nowMs: number | null;
}

interface ChipState {
  label: string;
  title: string;
  tone: "muted" | "info" | "warn" | "err";
  pulse: boolean;
}

function secsSince(timestamp: string, nowMs: number | null): number | null {
  if (nowMs === null) return null;
  const then = new Date(timestamp).getTime();
  if (Number.isNaN(then)) return null;
  return Math.max(0, Math.floor((nowMs - then) / 1000));
}

function hasLaterTerminator(calls: PrimitiveCallView[], index: number): boolean {
  const primitive = calls[index].primitive;
  for (let j = index + 1; j < calls.length; j++) {
    if (calls[j].primitive === primitive && calls[j].status !== "start") return true;
  }
  return false;
}

function deriveChipState({
  sandboxMode,
  status,
  primitiveCalls,
  nowMs,
}: RunpodStatusChipProps): ChipState {
  if (sandboxMode && sandboxMode !== "runpod") {
    return {
      label: `sandbox: ${sandboxMode}`,
      title: `This run is using the ${sandboxMode} sandbox; no RunPod pod will be created.`,
      tone: "muted",
      pulse: false,
    };
  }

  const runExperimentCalls = primitiveCalls.filter((c) => c.primitive === "run_experiment");
  for (let i = primitiveCalls.length - 1; i >= 0; i--) {
    const c = primitiveCalls[i];
    if (c.primitive !== "run_experiment" || c.status !== "start") continue;
    if (hasLaterTerminator(primitiveCalls, i)) continue;
    const secs = secsSince(c.timestamp, nowMs);
    return {
      label: `runpod: executing${secs === null ? "" : ` ${secs}s`}`,
      title:
        "RunPod pod work is active. OpenResearch creates the pod lazily at run_experiment, executes the generated commands, then destroys the pod.",
      tone: secs !== null && secs > 600 ? "warn" : "info",
      pulse: true,
    };
  }

  const lastRunExperiment = runExperimentCalls[runExperimentCalls.length - 1] ?? null;
  if (lastRunExperiment?.status === "ok") {
    return {
      label: "runpod: experiment complete",
      title:
        "The latest run_experiment completed. The pod should have been destroyed by the runtime cleanup path.",
      tone: "info",
      pulse: status === "running",
    };
  }
  if (lastRunExperiment?.status === "error") {
    return {
      label: "runpod: last experiment failed",
      title:
        "The latest run_experiment failed. The root REPL can still repair the baseline and retry on a later iteration.",
      tone: "warn",
      pulse: false,
    };
  }

  const hasBuiltEnvironment = primitiveCalls.some(
    (c) => c.primitive === "build_environment" && c.status === "ok",
  );
  if (hasBuiltEnvironment) {
    return {
      label: "runpod: ready at experiment",
      title:
        "The environment image is built. RunPod pods are intentionally not created until run_experiment fires.",
      tone: "info",
      pulse: status === "running",
    };
  }

  return {
    label: "runpod: not yet",
    title:
      "No RunPod pod is expected yet. OpenResearch creates pods lazily at run_experiment, after paper understanding, environment detection, planning, and baseline implementation.",
    tone: "muted",
    pulse: status === "running",
  };
}

function stateFromRemote(remote: DemoRunpodStatusResponse): ChipState {
  const tone =
    remote.status === "error"
      ? "err"
      : remote.status === "provisioning" || remote.status === "stopping"
      ? "warn"
      : remote.status === "ready" || remote.status === "executing"
      ? "info"
      : "muted";
  return {
    label: remote.label,
    title: remote.detail,
    tone,
    pulse: remote.status === "provisioning" || remote.status === "executing" || remote.status === "ready",
  };
}

export function RunpodStatusChip({
  projectId,
  sandboxMode,
  status,
  primitiveCalls,
  nowMs,
}: RunpodStatusChipProps) {
  const remote = useRunpodStatus(
    projectId,
    (sandboxMode ?? "runpod") === "runpod" && (status === "queued" || status === "running"),
  );
  const chip = useMemo(
    () =>
      remote
        ? stateFromRemote(remote)
        : deriveChipState({ projectId, sandboxMode, status, primitiveCalls, nowMs }),
    [remote, projectId, sandboxMode, status, primitiveCalls, nowMs],
  );

  const className = [
    styles.chip,
    chip.tone !== "muted" ? styles[chip.tone] : "",
    chip.pulse ? styles.pulse : "",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <span className={className} title={chip.title} aria-label={chip.title}>
      <span className={styles.dot} aria-hidden="true" />
      <span className={styles.label}>{chip.label}</span>
    </span>
  );
}
