import type { PipelineTopology } from "./topology";

/**
 * Stages that have happened by the time this gate has passed.
 *
 * Convention: `gate_<n>_passed` is the stage id that signals this gate
 * has been verified. Everything at or after that stage's order counts
 * as "passed".
 */
export function stagesAfter(topology: PipelineTopology, gateId: string): string[] {
  const passedStage = `${gateId}_passed`;
  const order = topology.stages.find((s) => s.id === passedStage)?.order ?? 0;
  return topology.stages.filter((s) => s.order >= order).map((s) => s.id);
}

/**
 * Stages during which this gate is being checked.
 *
 * Convention: the stage immediately before `gate_<n>_passed` is when
 * the gate is "running". E.g. `gate_1` is checked during `plan_created`.
 */
export function stagesDuring(topology: PipelineTopology, gateId: string): string[] {
  const passedStage = `${gateId}_passed`;
  const order = topology.stages.find((s) => s.id === passedStage)?.order ?? 0;
  const prev = topology.stages.find((s) => s.order === order - 1);
  return prev ? [prev.id] : [];
}

/**
 * Fraction of the pipeline complete given the current stage id.
 *
 * Replaces the hardcoded `PIPELINE_STAGES` array previously held in
 * `components/lab/node-config.ts` — the total is now derived from the
 * topology, so a backend stage change ripples through the UI without
 * frontend edits.
 */
export function stageProgressFromTopology(
  topology: PipelineTopology,
  stage: string | null | undefined
): number {
  if (!stage) return 0;
  const total = topology.stages.length;
  if (total === 0) return 0;
  const idx = topology.stages.find((s) => s.id === stage)?.order;
  return idx === undefined ? 0 : (idx + 1) / total;
}
