import type {
  DashboardEvent,
  DashboardEventAdapter,
  DashboardEventListener,
  DashboardSnapshot
} from "./contract";
import { initialDashboardSnapshot, mockDashboardEvents } from "./mock-events";

interface MockEventAdapterOptions {
  events?: DashboardEvent[];
  snapshot?: DashboardSnapshot;
}

export function createMockEventAdapter(
  options: MockEventAdapterOptions = {}
): DashboardEventAdapter {
  const events = options.events ?? mockDashboardEvents;
  const snapshot = options.snapshot ?? initialDashboardSnapshot;
  const listeners = new Set<DashboardEventListener>();

  return {
    getSnapshot() {
      return snapshot;
    },
    subscribe(listener) {
      listeners.add(listener);

      return () => {
        listeners.delete(listener);
      };
    },
    async flush() {
      for (const event of events) {
        for (const listener of listeners) {
          listener(event);
        }
      }
    }
  };
}
