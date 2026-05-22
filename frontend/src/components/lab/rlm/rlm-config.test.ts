import { describe, it, expect } from "vitest";
import { DEGRADED_SCORE_CAP } from "./rlm-config";

describe("rlm-config", () => {
  it("exposes the degraded-score cap as a documented constant", () => {
    expect(DEGRADED_SCORE_CAP).toBe(0.35);
  });
});
