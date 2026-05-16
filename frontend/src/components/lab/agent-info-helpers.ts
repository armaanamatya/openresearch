import type { LiveDemoRunState } from "@/lib/demo/demo-run-types";

export function sourceTitle(run: LiveDemoRunState) {
  return run.sourceLabel || run.projectId;
}

export function sourcePdfUrl(run: LiveDemoRunState) {
  return `/api/demo/source-pdf?projectId=${encodeURIComponent(run.projectId)}`;
}

export function finalReportUrl(run: LiveDemoRunState) {
  return `/api/demo/final-report?projectId=${encodeURIComponent(run.projectId)}`;
}

export function formatBytes(bytes?: number | null) {
  if (!bytes || bytes <= 0) {
    return "n/a";
  }
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let index = 0;
  while (value >= 1024 && index < units.length - 1) {
    value /= 1024;
    index += 1;
  }
  return `${value.toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}

export function shortHash(value?: string | null) {
  return value ? value.slice(0, 12) : "pending";
}

export function verdictLabel(value?: string) {
  if (!value) {
    return "Pending";
  }
  return value
    .split("_")
    .map((part) => part.slice(0, 1).toUpperCase() + part.slice(1))
    .join(" ");
}
