"use client";

import type { LiveDemoRunState } from "@/lib/demo/demo-run-types";
import { useTopologyContext } from "@/lib/pipeline/topology-context";
import { stagesAfter, stagesDuring } from "@/lib/pipeline/topology-helpers";
import { issueText } from "./shared-helpers";

import "./gate-chips.css";

function gateChipState(
  gate: { passed?: boolean; status?: string; chipStatus?: string } | undefined,
  stage: string | null,
  passedStages: string[],
  runningStages: string[]
): { state: "pending" | "running" | "passed" | "caveat" | "failed"; label: string } {
  // Prefer the pre-normalized chipStatus from buildLiveDemoDashboard (handles
  // backend GateStatus enum values like "verified_with_caveats").
  const normalized = gate?.chipStatus;
  if (normalized === "caveat") return { state: "caveat", label: "caveat" };
  if (normalized === "failed") return { state: "failed", label: issueText("failed") };
  if (normalized === "passed" || (stage && passedStages.includes(stage))) {
    return { state: "passed", label: "passed" };
  }
  if (normalized === "running" || (stage && runningStages.includes(stage))) {
    return { state: "running", label: "checking" };
  }
  return { state: "pending", label: "pending" };
}

type GateKey = "gate_1" | "gate_2" | "gate_3";

export function GateChips({ run }: { run: LiveDemoRunState }) {
  const { topology, layout } = useTopologyContext();
  const gates = run.payload?.gates;
  const stage = run.payload?.summary.stage ?? null;

  return (
    <>
      {layout.gates.map((gate) => {
        const gateState = gates?.[gate.id as GateKey];
        const view = gateChipState(
          gateState,
          stage,
          stagesAfter(topology, gate.id),
          stagesDuring(topology, gate.id)
        );
        return (
          <div
            key={gate.id}
            className={`gate-chip gate-chip-${view.state}`}
            style={{ left: gate.x, top: gate.y }}
            title={gateState?.detail ?? undefined}
          >
            {gate.label} · {view.label}
          </div>
        );
      })}
    </>
  );
}
