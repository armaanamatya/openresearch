"use client";

import { useEffect, useState } from "react";

import type {
  DashboardEvent,
  DashboardEventAdapter,
  DashboardSnapshot
} from "@/lib/events/contract";

import { applyDashboardEvent } from "./dashboard-state";

export function useDashboardState(adapter: DashboardEventAdapter): DashboardSnapshot {
  const [snapshot, setSnapshot] = useState(() => adapter.getSnapshot());

  useEffect(() => {
    const unsubscribe = adapter.subscribe((event: DashboardEvent) => {
      setSnapshot((current) => applyDashboardEvent(current, event));
    });

    void adapter.flush();

    return unsubscribe;
  }, [adapter]);

  return snapshot;
}
