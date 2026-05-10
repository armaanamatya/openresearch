import type {
  DashboardEvent,
  DashboardEventAdapter,
  DashboardEventListener,
  DashboardSnapshot
} from "./contract";

export interface LiveEventAdapter extends DashboardEventAdapter {
  /** Push a single event received from the SSE stream. */
  push(event: DashboardEvent): void;
}

/**
 * Creates a DashboardEventAdapter that receives events via push()
 * instead of replaying a static list. Used to bridge SSE
 * dashboard_event messages into the DashboardShell.
 */
export function createLiveEventAdapter(): LiveEventAdapter {
  const listeners = new Set<DashboardEventListener>();

  return {
    getSnapshot(): DashboardSnapshot {
      return {
        agents: [],
        reasoning: [],
        messages: [],
        citations: [],
        approvals: [],
        progress: [],
        dataPanels: [
          {
            id: "artifacts",
            title: "Artifacts",
            summary: "Agent outputs and context variables",
            items: []
          },
          {
            id: "assumptions",
            title: "Assumptions",
            summary: "Decisions and assumptions made during the run",
            items: []
          }
        ],
        hermesPanel: null,
        conceptCard: null
      };
    },
    subscribe(listener) {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
    async flush() {
      // No-op — events arrive via push()
    },
    push(event: DashboardEvent) {
      for (const listener of listeners) {
        listener(event);
      }
    }
  };
}
