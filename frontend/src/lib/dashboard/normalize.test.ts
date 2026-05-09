import { describe, expect, it } from "vitest";
import { sampleDashboardSnapshot } from "../../lib/dashboard/fixtures";
import { buildDashboardModel } from "../../lib/dashboard/normalize";

describe("buildDashboardModel", () => {
  it("derives parent and child topology from task records", () => {
    const model = buildDashboardModel(sampleDashboardSnapshot);

    expect(model.tasksById.task_042.parent_task_id).toBe("task_017");
    expect(model.topologyEdges).toEqual(
      expect.arrayContaining([{ from: "task_017", to: "task_042" }])
    );
  });

  it("surfaces citations and sorts feed items in reverse chronological order", () => {
    const model = buildDashboardModel(sampleDashboardSnapshot);

    expect(model.feed[0]?.event).toBe("verification_gate_result");
    expect(model.feed[0]?.citations.length).toBeGreaterThan(0);
    expect(model.citationsBySourceId.src_083?.trust_level).toBe("primary");
  });

  it("builds lineage and descendant views for task detail panels", () => {
    const model = buildDashboardModel(sampleDashboardSnapshot);
    const detail = model.detailsByTaskId.task_042;

    expect(detail.lineage.map((task) => task.task_id)).toEqual([
      "task_001",
      "task_017",
      "task_042"
    ]);
    expect(detail.descendants).toHaveLength(0);
  });
});
