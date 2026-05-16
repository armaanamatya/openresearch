export type NodeKind = "source" | "agent" | "improvement_path" | "audit" | "report";
export type Tone = "accent" | "hermes" | "info" | "neutral";

export interface PipelineNode {
  id: string;
  kind: NodeKind;
  internal_label: string;
  demo_label: string;
  step: string;
  role: string;
  detail: string;
  icon: string;
  tone: Tone;
  agent_ids: string[];
  tour_caption?: string | null;
}

export interface PipelineEdge {
  source: string;
  target: string;
}

export interface PipelineGate {
  id: string;
  before_node: string;
  after_node: string;
  label: string;
}

export interface PipelineStage {
  id: string;
  order: number;
}

export interface PipelineTopology {
  nodes: PipelineNode[];
  edges: PipelineEdge[];
  gates: PipelineGate[];
  stages: PipelineStage[];
  improvement_path_ids: string[];
}
