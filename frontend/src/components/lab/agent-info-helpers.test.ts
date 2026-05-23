import { describe, expect, it } from "vitest";

import { shortHash } from "./agent-info-helpers";

// ------------------------------------------------------------------
// Item 5 — shortHash() returns "pending" when value is falsy
// agent-info-helpers.ts:30
// ------------------------------------------------------------------

describe("shortHash — item 5: falsy input behavior", () => {
  it("returns the first 12 chars of the hash when value is present", () => {
    expect(shortHash("abcdef1234567890abcdef")).toBe("abcdef123456");
  });

  it("returns empty string (not 'pending') when value is null", () => {
    // Must NOT return "pending" — that reads as if "pending" is the hash value
    expect(shortHash(null)).not.toBe("pending");
    expect(shortHash(null)).toBe("");
  });

  it("returns empty string (not 'pending') when value is undefined", () => {
    expect(shortHash(undefined)).not.toBe("pending");
    expect(shortHash(undefined)).toBe("");
  });

  it("returns empty string (not 'pending') when value is empty string", () => {
    expect(shortHash("")).not.toBe("pending");
    expect(shortHash("")).toBe("");
  });
});
