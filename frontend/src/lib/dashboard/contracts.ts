export type PipelineStage = "plan" | "baseline" | "improvement" | "verification";

export type TrustLevel =
  | "primary"
  | "strong_secondary"
  | "secondary"
  | "weak";

export type TaskScope =
  | "private_to_parent"
  | "branch_shared"
  | "global_verified";

export type TaskStatus =
  | "created"
  | "context_prepared"
  | "running"
  | "artifact_submitted"
  | "verification_pending"
  | "verified"
  | "completed"
  | "failed"
  | "blocked_requires_human";

export interface Citation {
  source_id: string;
  trust_level: TrustLevel;
  content: string;
  source_kind: "paper_section" | "repo_file" | "github_issue" | "run_log" | "system";
}

export interface TaskRecord {
  task_id: string;
  parent_task_id: string | null;
  root_run_id: string;
  agent_id: string;
  title: string;
  summary: string;
  status: TaskStatus;
  stage: PipelineStage;
  scope: TaskScope;
  delegation_depth: number;
  assigned_budget: string;
  assigned_timeout: string;
  write_scope: string;
  output_record_ids: string[];
  artifacts: string[];
  structured_outputs: Record<string, unknown>;
  citations: Citation[];
  published_at: string;
}

interface BaseEvent {
  event: string;
  agent_id: string;
  task_id: string;
  parent_task_id?: string | null;
  timestamp: string;
  citations?: Citation[];
}

export interface AgentLifecycleEvent extends BaseEvent {
  event: "agent_started" | "agent_completed" | "agent_failed";
  status: TaskStatus;
  summary: string;
}

export interface AgentReasoningStepEvent extends BaseEvent {
  event: "agent_reasoning_step" | "rlm_query_executed" | "semantic_search_executed";
  step_type: "rlm_query" | "semantic_search" | "analysis" | "citation_review";
  query: string;
  context_segment: string;
  result: string;
}

export interface SharedStateUpdatedEvent extends BaseEvent {
  event: "shared_state_updated";
  record_type:
    | "task_result"
    | "assumption"
    | "decision"
    | "verification_status"
    | "artifact_index";
  scope: TaskScope;
  status: TaskStatus;
  structured_outputs: Record<string, unknown>;
}

export interface VerificationGateResultEvent extends BaseEvent {
  event: "verification_gate_result";
  gate: "plan" | "baseline" | "improvement";
  outcome: "pass" | "fail" | "caveat";
  details: string;
}

export interface ApprovalEvent extends BaseEvent {
  event: "approval_requested" | "approval_resolved";
  approval_label: string;
  state: "pending" | "approved" | "rejected";
  details: string;
}

export interface ContextEnrichmentEvent extends BaseEvent {
  event: "context_enrichment";
  variable_name: string;
  visibility: TaskScope;
  details: string;
}

export type DashboardEvent =
  | AgentLifecycleEvent
  | AgentReasoningStepEvent
  | SharedStateUpdatedEvent
  | VerificationGateResultEvent
  | ApprovalEvent
  | ContextEnrichmentEvent;

export interface DashboardSnapshot {
  run_id: string;
  project_name: string;
  current_stage: PipelineStage;
  stage_summary: string;
  tasks: TaskRecord[];
  events: DashboardEvent[];
}
