import { describe, expect, it } from "vitest";

import { createMockEventAdapter } from "./mock-event-adapter";

describe("createMockEventAdapter", () => {
  it("returns an initial snapshot and replays ordered events to subscribers", async () => {
    const adapter = createMockEventAdapter();
    const snapshot = adapter.getSnapshot();

    expect(snapshot.agents.length).toBeGreaterThan(0);

    const received: string[] = [];
    const unsubscribe = adapter.subscribe((event) => {
      received.push(event.event);
    });

    await adapter.flush();

    expect(received).toContain("agent_started");
    expect(received).toContain("agent_reasoning_step");
    unsubscribe();
  });
});
