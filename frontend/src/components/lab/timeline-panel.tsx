"use client";

import { useMemo, useState } from "react";

import type { TelemetryRecordPublic } from "@/lib/demo/demo-run-types";
import { formatDuration } from "@/lib/demo/progress";

interface TimelinePanelProps {
  telemetry: TelemetryRecordPublic[] | undefined;
}

type Filter = "all" | "errors";

/**
 * Compact timeline of agent invocations parsed from agent_telemetry.jsonl.
 * One card per invocation. No raw blobs — only the fields a developer
 * scanning for "what's slow / what failed" actually needs.
 */
export function TimelinePanel({ telemetry }: TimelinePanelProps) {
  const [filter, setFilter] = useState<Filter>("all");
  const records = useMemo(() => {
    if (!telemetry) return [];
    if (filter === "errors") {
      return telemetry.filter((r) => r.success === false || r.error_message);
    }
    return telemetry;
  }, [telemetry, filter]);

  if (!telemetry || telemetry.length === 0) {
    return (
      <section className="rounded-2xl border border-white/10 bg-stone-900/60 p-6">
        <header className="flex items-center justify-between">
          <p className="text-xs uppercase tracking-[0.3em] text-stone-400">
            Agent timeline
          </p>
          <span className="text-xs text-stone-500">No telemetry yet</span>
        </header>
      </section>
    );
  }

  const errorCount = telemetry.filter(
    (r) => r.success === false || r.error_message
  ).length;

  return (
    <section className="rounded-2xl border border-white/10 bg-stone-900/60">
      <header className="flex items-center justify-between border-b border-white/10 px-6 py-4">
        <div className="flex items-center gap-3">
          <p className="text-xs uppercase tracking-[0.3em] text-stone-400">
            Agent timeline
          </p>
          <span className="text-xs text-stone-500">
            {telemetry.length} record{telemetry.length === 1 ? "" : "s"}
          </span>
        </div>
        <div className="flex items-center gap-1 text-[11px]">
          <FilterChip active={filter === "all"} onClick={() => setFilter("all")}>
            All
          </FilterChip>
          <FilterChip
            active={filter === "errors"}
            onClick={() => setFilter("errors")}
            tone={errorCount > 0 ? "warn" : "neutral"}
          >
            Errors ({errorCount})
          </FilterChip>
        </div>
      </header>
      <ul className="max-h-[28rem] divide-y divide-white/5 overflow-auto">
        {records.map((record, idx) => (
          <li
            key={`${record.started_at ?? idx}-${record.agent_id ?? idx}`}
            className="px-6 py-3 text-sm"
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <StatusBadge success={record.success} />
                <span className="font-medium text-stone-100">
                  {record.agent_id ?? "(unknown agent)"}
                </span>
                {record.model ? (
                  <span className="rounded-full bg-white/5 px-2 py-0.5 text-[10px] uppercase tracking-wide text-stone-400">
                    {record.model}
                  </span>
                ) : null}
              </div>
              <div className="flex items-center gap-3 text-xs text-stone-400">
                {record.message_count !== undefined ? (
                  <span title="Messages exchanged with the model">
                    {record.message_count} msg
                  </span>
                ) : null}
                {record.output_chars !== undefined ? (
                  <span title="Output characters returned">
                    {record.output_chars} chars
                  </span>
                ) : null}
                <span>
                  {formatDuration(
                    record.duration_seconds !== undefined
                      ? Math.round(record.duration_seconds)
                      : null
                  )}
                </span>
              </div>
            </div>
            {record.error_message ? (
              <p className="mt-1 break-words text-xs text-rose-300">
                {record.error_message}
              </p>
            ) : null}
            <p className="mt-1 text-[11px] text-stone-500">
              {record.started_at ? new Date(record.started_at).toLocaleTimeString() : "—"}
              {record.finished_at
                ? ` → ${new Date(record.finished_at).toLocaleTimeString()}`
                : ""}
            </p>
          </li>
        ))}
      </ul>
    </section>
  );
}

function StatusBadge({ success }: { success: boolean | undefined }) {
  if (success === true) return <Dot color="bg-emerald-500" label="ok" />;
  if (success === false) return <Dot color="bg-rose-500" label="failed" />;
  return <Dot color="bg-zinc-500" label="unknown" />;
}

function Dot({ color, label }: { color: string; label: string }) {
  return (
    <span
      className={`inline-block h-2 w-2 rounded-full ${color}`}
      aria-label={label}
      title={label}
    />
  );
}

function FilterChip({
  children,
  active,
  onClick,
  tone = "neutral",
}: {
  children: React.ReactNode;
  active: boolean;
  onClick: () => void;
  tone?: "neutral" | "warn";
}) {
  const baseInactive =
    tone === "warn"
      ? "border-amber-400/30 bg-amber-500/10 text-amber-200 hover:bg-amber-500/15"
      : "border-white/10 bg-white/5 text-stone-300 hover:bg-white/10";
  return (
    <button
      type="button"
      onClick={onClick}
      className={`rounded-full border px-3 py-1 transition-colors ${
        active ? "border-sky-400/50 bg-sky-500/15 text-sky-200" : baseInactive
      }`}
    >
      {children}
    </button>
  );
}
