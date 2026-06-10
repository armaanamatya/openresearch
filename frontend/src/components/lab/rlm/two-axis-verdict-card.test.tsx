/**
 * Tests for TwoAxisVerdictCard — U17.
 *
 * Covers:
 *   1. Verdict → badge mapping (all fidelity × replication combinations)
 *   2. Graph data transform (plottableClaims, inferSeedCount)
 *   3. Schema gating (legacy report → card not rendered; schema_version >= 2 → rendered)
 *   4. Plain-language one-liner accuracy (especially faithful+contradicted = positive finding)
 *   5. Developer details collapsed by default
 *   6. isTwoAxisReport guard function
 *   7. Measured-vs-claimed graph renders correct testids
 */

import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import {
  TwoAxisVerdictCard,
  isTwoAxisReport,
  verdictOneLiner,
  plottableClaims,
  inferSeedCount,
} from "./two-axis-verdict-card";
import type { TwoAxisReport, PerClaimResult } from "./two-axis-verdict-card";

// ─── Factories ────────────────────────────────────────────────────────────────

function makeReport(
  overrides: Partial<TwoAxisReport> = {},
): TwoAxisReport {
  return {
    schema_version: 2,
    implementation_verdict: "faithful",
    replication_verdict: "replicated",
    reproducibility: {
      replication_credit: 1.0,
      fidelity_score: 0.9,
      per_claim: [],
      rationale: [],
    },
    ...overrides,
  };
}

function makeClaim(
  overrides: Partial<PerClaimResult> = {},
): PerClaimResult {
  return {
    claim_id: "claim-1",
    status: "replicated",
    credit: 1.0,
    reason: "matches paper",
    eligible: true,
    measured_mean: 0.95,
    ci_low: 0.90,
    ci_high: 0.98,
    claimed_value: 0.94,
    ...overrides,
  };
}

// ─── isTwoAxisReport guard ────────────────────────────────────────────────────

describe("isTwoAxisReport", () => {
  it("returns true for schema_version=2 report with implementation_verdict", () => {
    expect(isTwoAxisReport(makeReport())).toBe(true);
  });

  it("returns true for higher schema_version (e.g. 3)", () => {
    expect(isTwoAxisReport(makeReport({ schema_version: 3 }))).toBe(true);
  });

  it("returns false for schema_version=1 (legacy)", () => {
    const r = { schema_version: 1 } as Partial<TwoAxisReport>;
    expect(isTwoAxisReport(r)).toBe(false);
  });

  it("returns false for null", () => {
    expect(isTwoAxisReport(null)).toBe(false);
  });

  it("returns false for undefined", () => {
    expect(isTwoAxisReport(undefined)).toBe(false);
  });

  it("returns true for report with no schema_version but has implementation_verdict (early beta)", () => {
    const r: Partial<TwoAxisReport> = {
      implementation_verdict: "faithful",
      replication_verdict: "contradicted",
      reproducibility: { replication_credit: 0.3, fidelity_score: 0.9 },
    };
    expect(isTwoAxisReport(r)).toBe(true);
  });

  it("returns false for empty object", () => {
    expect(isTwoAxisReport({})).toBe(false);
  });
});

// ─── plottableClaims ──────────────────────────────────────────────────────────

describe("plottableClaims", () => {
  it("returns claims with eligible=true AND measured_mean AND claimed_value", () => {
    const claims: PerClaimResult[] = [
      makeClaim({ claim_id: "a" }),
      makeClaim({ claim_id: "b", eligible: false }),
      makeClaim({ claim_id: "c", measured_mean: null }),
      makeClaim({ claim_id: "d", claimed_value: null }),
    ];
    const result = plottableClaims(claims);
    expect(result.map((c) => c.claim_id)).toEqual(["a"]);
  });

  it("returns empty array for undefined input", () => {
    expect(plottableClaims(undefined)).toEqual([]);
  });

  it("returns empty array when all claims are ineligible", () => {
    const claims: PerClaimResult[] = [
      makeClaim({ eligible: false }),
      makeClaim({ eligible: false }),
    ];
    expect(plottableClaims(claims)).toEqual([]);
  });

  it("includes multiple eligible plottable claims", () => {
    const claims: PerClaimResult[] = [
      makeClaim({ claim_id: "a" }),
      makeClaim({ claim_id: "b" }),
      makeClaim({ claim_id: "c", eligible: false }),
    ];
    expect(plottableClaims(claims)).toHaveLength(2);
  });
});

// ─── inferSeedCount ───────────────────────────────────────────────────────────

describe("inferSeedCount", () => {
  it("counts eligible claims", () => {
    const claims: PerClaimResult[] = [
      makeClaim({ claim_id: "a", eligible: true }),
      makeClaim({ claim_id: "b", eligible: true }),
      makeClaim({ claim_id: "c", eligible: false }),
    ];
    expect(inferSeedCount(claims)).toBe(2);
  });

  it("returns null for empty array", () => {
    expect(inferSeedCount([])).toBeNull();
  });

  it("returns null for undefined", () => {
    expect(inferSeedCount(undefined)).toBeNull();
  });
});

// ─── verdictOneLiner ──────────────────────────────────────────────────────────

describe("verdictOneLiner", () => {
  it("faithful + replicated — positive", () => {
    const line = verdictOneLiner("faithful", "replicated");
    expect(line).toMatch(/faithfully reproduced/i);
    expect(line).toMatch(/replicated/i);
    expect(line).not.toMatch(/did not replicate/i);
  });

  it("faithful + contradicted — negative finding, NOT failure language", () => {
    const line = verdictOneLiner("faithful", "contradicted");
    expect(line).toMatch(/faithfully reproduced/i);
    expect(line).toMatch(/did NOT replicate/i);
    expect(line).toMatch(/negative finding/i);
    // Must NOT use "error" or "failed" framing
    expect(line.toLowerCase()).not.toMatch(/\bfailed\b/);
    expect(line.toLowerCase()).not.toMatch(/\berror\b/);
  });

  it("faithful + partially-replicated — right direction", () => {
    const line = verdictOneLiner("faithful", "partially-replicated");
    expect(line).toMatch(/partially replicated/i);
  });

  it("faithful + inconclusive", () => {
    const line = verdictOneLiner("faithful", "inconclusive");
    expect(line).toMatch(/inconclusive/i);
  });

  it("partial + contradicted — fidelity warning attached", () => {
    const line = verdictOneLiner("partial", "contradicted");
    expect(line).toMatch(/partially reproduced/i);
    expect(line).toMatch(/fidelity/i);
  });

  it("broken + any — unreliable language", () => {
    const line = verdictOneLiner("broken", "inconclusive");
    expect(line).toMatch(/no reliable verdict/i);
  });

  it("broken + contradicted — inconclusive due to code failures", () => {
    const line = verdictOneLiner("broken", "contradicted");
    expect(line).toMatch(/inconclusive/i);
    expect(line).toMatch(/code failure/i);
  });
});

// ─── Schema gating: legacy report → card not rendered ────────────────────────

describe("TwoAxisVerdictCard — schema gating", () => {
  it("does not render for a legacy report (schema_version=1)", () => {
    const legacyReport = {
      schema_version: 1,
      implementation_verdict: "faithful",
      replication_verdict: "replicated",
      reproducibility: { replication_credit: 1.0, fidelity_score: 0.9 },
    } as unknown as TwoAxisReport;
    const { container } = render(<TwoAxisVerdictCard report={legacyReport} />);
    expect(container.firstChild).toBeNull();
  });

  it("renders for schema_version=2", () => {
    render(<TwoAxisVerdictCard report={makeReport()} />);
    expect(screen.getByTestId("two-axis-verdict-card")).toBeInTheDocument();
  });

  it("renders for schema_version=3 (forward-compat)", () => {
    render(<TwoAxisVerdictCard report={makeReport({ schema_version: 3 })} />);
    expect(screen.getByTestId("two-axis-verdict-card")).toBeInTheDocument();
  });
});

// ─── Verdict → badge mapping ──────────────────────────────────────────────────

describe("TwoAxisVerdictCard — fidelity badge mapping", () => {
  it("shows fidelity-badge for faithful", () => {
    render(<TwoAxisVerdictCard report={makeReport({ implementation_verdict: "faithful" })} />);
    const badge = screen.getByTestId("fidelity-badge");
    expect(badge).toHaveAttribute("data-verdict", "faithful");
    expect(badge.textContent).toBe("faithful");
  });

  it("shows fidelity-badge for partial", () => {
    render(<TwoAxisVerdictCard report={makeReport({ implementation_verdict: "partial" })} />);
    const badge = screen.getByTestId("fidelity-badge");
    expect(badge).toHaveAttribute("data-verdict", "partial");
    expect(badge.textContent).toBe("partial");
  });

  it("shows fidelity-badge for broken", () => {
    render(<TwoAxisVerdictCard report={makeReport({ implementation_verdict: "broken" })} />);
    const badge = screen.getByTestId("fidelity-badge");
    expect(badge).toHaveAttribute("data-verdict", "broken");
    expect(badge.textContent).toBe("broken");
  });
});

describe("TwoAxisVerdictCard — replication badge mapping", () => {
  it("shows replication-badge for replicated", () => {
    render(<TwoAxisVerdictCard report={makeReport({ replication_verdict: "replicated" })} />);
    const badge = screen.getByTestId("replication-badge");
    expect(badge).toHaveAttribute("data-verdict", "replicated");
  });

  it("shows replication-badge for partially-replicated", () => {
    render(
      <TwoAxisVerdictCard
        report={makeReport({
          replication_verdict: "partially-replicated",
          reproducibility: { replication_credit: 0.75, fidelity_score: 0.9 },
        })}
      />,
    );
    const badge = screen.getByTestId("replication-badge");
    expect(badge).toHaveAttribute("data-verdict", "partially-replicated");
    expect(badge.textContent).toContain("partially replicated");
  });

  it("shows replication-badge for contradicted", () => {
    render(
      <TwoAxisVerdictCard
        report={makeReport({ replication_verdict: "contradicted" })}
      />,
    );
    const badge = screen.getByTestId("replication-badge");
    expect(badge).toHaveAttribute("data-verdict", "contradicted");
  });

  it("shows replication-badge for inconclusive", () => {
    render(
      <TwoAxisVerdictCard
        report={makeReport({ replication_verdict: "inconclusive" })}
      />,
    );
    const badge = screen.getByTestId("replication-badge");
    expect(badge).toHaveAttribute("data-verdict", "inconclusive");
  });
});

// ─── Plain-language one-liner (key: faithful+contradicted = finding) ──────────

describe("TwoAxisVerdictCard — one-liner rendering", () => {
  it("faithful+contradicted renders the negative-finding one-liner", () => {
    render(
      <TwoAxisVerdictCard
        report={makeReport({
          implementation_verdict: "faithful",
          replication_verdict: "contradicted",
        })}
      />,
    );
    const p = screen.getByTestId("verdict-one-liner");
    expect(p.textContent).toMatch(/faithfully reproduced/i);
    expect(p.textContent).toMatch(/did NOT replicate/i);
    expect(p.textContent).toMatch(/negative finding/i);
  });

  it("faithful+replicated renders a positive one-liner", () => {
    render(<TwoAxisVerdictCard report={makeReport()} />);
    const p = screen.getByTestId("verdict-one-liner");
    expect(p.textContent).toMatch(/replicated/i);
  });
});

// ─── Replication credit display ───────────────────────────────────────────────

describe("TwoAxisVerdictCard — replication credit", () => {
  it("shows replication credit value", () => {
    render(
      <TwoAxisVerdictCard
        report={makeReport({
          reproducibility: { replication_credit: 0.75, fidelity_score: 0.9 },
        })}
      />,
    );
    const credit = screen.getByTestId("replication-credit");
    expect(credit.textContent).toContain("0.75");
  });
});

// ─── Seed count display ───────────────────────────────────────────────────────

describe("TwoAxisVerdictCard — seed count", () => {
  it("shows seed count derived from eligible per_claim entries", () => {
    render(
      <TwoAxisVerdictCard
        report={makeReport({
          reproducibility: {
            replication_credit: 1.0,
            fidelity_score: 0.9,
            per_claim: [
              makeClaim({ claim_id: "c1", eligible: true }),
              makeClaim({ claim_id: "c2", eligible: true }),
              makeClaim({ claim_id: "c3", eligible: false }),
            ],
          },
        })}
      />,
    );
    const seedEl = screen.getByTestId("seed-count");
    expect(seedEl.textContent).toContain("2");
  });

  it("omits seed count when per_claim is empty", () => {
    render(
      <TwoAxisVerdictCard
        report={makeReport({
          reproducibility: { replication_credit: 1.0, fidelity_score: 0.9, per_claim: [] },
        })}
      />,
    );
    expect(screen.queryByTestId("seed-count")).toBeNull();
  });
});

// ─── Graph data transform + rendering ────────────────────────────────────────

describe("TwoAxisVerdictCard — measured-vs-claimed graph", () => {
  it("renders graph when eligible claims with measured+claimed values exist", () => {
    render(
      <TwoAxisVerdictCard
        report={makeReport({
          reproducibility: {
            replication_credit: 1.0,
            fidelity_score: 0.9,
            per_claim: [makeClaim({ claim_id: "c1" })],
          },
        })}
      />,
    );
    expect(screen.getByTestId("measured-vs-claimed-graph")).toBeInTheDocument();
    expect(screen.getByTestId("measured-dot-c1")).toBeInTheDocument();
    expect(screen.getByTestId("claimed-marker-c1")).toBeInTheDocument();
  });

  it("renders CI bar when ci_low and ci_high are present", () => {
    render(
      <TwoAxisVerdictCard
        report={makeReport({
          reproducibility: {
            replication_credit: 0.9,
            fidelity_score: 0.9,
            per_claim: [makeClaim({ claim_id: "c1", ci_low: 0.88, ci_high: 0.99 })],
          },
        })}
      />,
    );
    expect(screen.getByTestId("ci-bar-c1")).toBeInTheDocument();
  });

  it("shows no-plot notice when per_claim is empty", () => {
    render(
      <TwoAxisVerdictCard
        report={makeReport({
          reproducibility: {
            replication_credit: 1.0,
            fidelity_score: 0.9,
            per_claim: [],
          },
        })}
      />,
    );
    // Graph section not rendered at all (empty per_claim → graphSection guard)
    expect(screen.queryByTestId("measured-vs-claimed-graph")).toBeNull();
  });

  it("shows no-plot notice when all claims lack claimed_value or are ineligible", () => {
    render(
      <TwoAxisVerdictCard
        report={makeReport({
          reproducibility: {
            replication_credit: 0.5,
            fidelity_score: 0.7,
            per_claim: [
              makeClaim({ claim_id: "c1", claimed_value: null }),
              makeClaim({ claim_id: "c2", eligible: false }),
            ],
          },
        })}
      />,
    );
    // Graph section IS rendered (per_claim not empty), but MeasuredVsClaimedGraph
    // shows the no-plot notice because plottableClaims returns empty.
    expect(screen.getByTestId("no-plot-notice")).toBeInTheDocument();
  });
});

// ─── Developer details disclosure ────────────────────────────────────────────

describe("TwoAxisVerdictCard — developer details", () => {
  it("renders developer details disclosure when rationale is present", () => {
    render(
      <TwoAxisVerdictCard
        report={makeReport({
          reproducibility: {
            replication_credit: 1.0,
            fidelity_score: 0.9,
            rationale: ["Fidelity cert green.", "2 seeds with CI."],
          },
        })}
      />,
    );
    const details = screen.getByTestId("developer-details");
    expect(details).toBeInTheDocument();
    // Collapsed by default
    expect((details as HTMLDetailsElement).open).toBe(false);
  });

  it("renders developer details when per_claim reasons are present", () => {
    render(
      <TwoAxisVerdictCard
        report={makeReport({
          reproducibility: {
            replication_credit: 0.7,
            fidelity_score: 0.85,
            per_claim: [makeClaim({ claim_id: "c1", reason: "Matches within tolerance" })],
          },
        })}
      />,
    );
    expect(screen.getByTestId("developer-details")).toBeInTheDocument();
  });

  it("omits developer details when rationale and per_claim are both empty/absent", () => {
    render(
      <TwoAxisVerdictCard
        report={makeReport({
          reproducibility: {
            replication_credit: 1.0,
            fidelity_score: 0.9,
            rationale: [],
            per_claim: [],
          },
        })}
      />,
    );
    expect(screen.queryByTestId("developer-details")).toBeNull();
  });
});

// ─── ReportRail integration: card gated on schema_version ────────────────────

import { ReportRail } from "./report-rail";
import type { ReportRailProps } from "./report-rail";

const BASE_RUBRIC: ReportRailProps["rubric"] = {
  current: 0.8,
  baseline: 0.4,
  target: 0.7,
  series: [],
  areas: [],
  previousAreas: [],
  attributableCandidate: null,
};

describe("ReportRail — twoAxisReport integration", () => {
  it("does NOT render the verdict card when twoAxisReport is null (legacy run)", () => {
    render(
      <ReportRail
        status="completed"
        elapsedMs={3600000}
        report={null}
        rubric={BASE_RUBRIC}
        twoAxisReport={null}
      />,
    );
    expect(screen.queryByTestId("two-axis-verdict-card")).toBeNull();
  });

  it("does NOT render the verdict card when twoAxisReport is not provided", () => {
    render(
      <ReportRail
        status="completed"
        elapsedMs={3600000}
        report={null}
        rubric={BASE_RUBRIC}
      />,
    );
    expect(screen.queryByTestId("two-axis-verdict-card")).toBeNull();
  });

  it("renders the verdict card when valid twoAxisReport is provided", () => {
    const twoAxisReport = makeReport({
      implementation_verdict: "faithful",
      replication_verdict: "contradicted",
    });
    render(
      <ReportRail
        status="completed"
        elapsedMs={3600000}
        report={null}
        rubric={BASE_RUBRIC}
        twoAxisReport={twoAxisReport}
      />,
    );
    expect(screen.getByTestId("two-axis-verdict-card")).toBeInTheDocument();
    expect(screen.getByTestId("fidelity-badge")).toHaveAttribute("data-verdict", "faithful");
    expect(screen.getByTestId("replication-badge")).toHaveAttribute(
      "data-verdict",
      "contradicted",
    );
  });

  it("legacy verdict pill (run status) is still rendered alongside the card", () => {
    render(
      <ReportRail
        status="completed"
        elapsedMs={0}
        report={null}
        rubric={BASE_RUBRIC}
        twoAxisReport={makeReport()}
      />,
    );
    // Both the new card AND the existing status pill should be present
    expect(screen.getByTestId("two-axis-verdict-card")).toBeInTheDocument();
    expect(screen.getByText("Completed")).toBeInTheDocument();
  });
});
