export type DashboardLiveEvent = {
  event: string;
  timestamp?: string;
  agentId?: string;
  agentLabel?: string;
  agent?: { id?: string; label?: string; status?: string; currentTask?: string };
  stepType?: string;
  title?: string;
  detail?: string;
  changeType?: string;
  fromAgentId?: string;
  toAgentId?: string;
  variableName?: string;
  summary?: string;
  panel?: { title?: string; summary?: string; overallStatus?: string };
};
