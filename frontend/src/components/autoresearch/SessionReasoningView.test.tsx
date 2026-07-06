import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";

import { SessionReasoningView } from "./SessionReasoningView";
import styles from "./SessionReasoningView.module.css";
import { rlmRunFixture } from "../lab/rlm/__fixtures__/rlm-run.fixture";
import type { RlmDashboardEvent } from "@/lib/events/rlm-events";

const TEST_UUID = "019f34b0-afd6-7013-8db0-13170956e309";
const TEST_URL = "https://github.com/ZJU-REAL/SDAR";

// A legitimate, fully-sanitized iteration whose `response` text (the ONLY
// thing rendered from it) happens to embed a UUID-like id and a URL — this
// is what exercises the "mono id pill" / "maroon link" inline formatting.
const LINK_AND_ID_ITERATION: RlmDashboardEvent = {
  event: "repl_iteration",
  timestamp: "2026-07-06T00:20:00.000Z",
  iteration: 15,
  response: `Cloning the reference repo at ${TEST_URL} for project ${TEST_UUID}.`,
  code_blocks: [],
  sub_calls: 0,
  timing: 1.2,
};

const CORPUS_MARKER = "PAPER_CORPUS_LEAK_MARKER_zzz9142";

// Simulates a hypothetical backend bug that lets raw corpus text ride along
// on fields the sanitized IterationView/PrimitiveCallView shapes do NOT
// carry. IterationView only exposes {iteration, response, code_blocks,
// sub_calls, timing} and PrimitiveCallView only exposes {primitive, status,
// args_summary, result_summary, iteration, rubric_delta, timestamp} — an
// extra `context` property is only reachable if a renderer dumps the whole
// event object instead of picking named, sanitized fields.
const POISONED_ITERATION = {
  event: "repl_iteration",
  timestamp: "2026-07-06T00:20:01.000Z",
  iteration: 16,
  response: "Reviewing the environment spec.",
  code_blocks: [],
  sub_calls: 0,
  timing: 0.5,
  context: CORPUS_MARKER,
} as unknown as RlmDashboardEvent;

const POISONED_PRIMITIVE_CALL: RlmDashboardEvent = {
  event: "primitive_call",
  timestamp: "2026-07-06T00:20:02.000Z",
  primitive: "detect_environment",
  status: "ok",
  args_summary: { leaked: CORPUS_MARKER },
  result_summary: CORPUS_MARKER,
  iteration: 16,
  rubric_delta: null,
};

const EVENTS: RlmDashboardEvent[] = [
  ...rlmRunFixture,
  LINK_AND_ID_ITERATION,
  POISONED_ITERATION,
  POISONED_PRIMITIVE_CALL,
];

describe("SessionReasoningView", () => {
  it("renders the sanitized reasoning text from repl_iteration events", () => {
    render(<SessionReasoningView runId="prj_test" events={EVENTS} />);
    expect(screen.getByText(/Beginning paper comprehension/)).toBeInTheDocument();
  });

  it("renders a ReasoningChip per primitive_call with the right primitive name and status", () => {
    render(<SessionReasoningView runId="prj_test" events={EVENTS} />);
    const chips = screen.getAllByText("Understand section");
    expect(chips.length).toBeGreaterThanOrEqual(2);
    const statuses = chips.map((el) => el.closest("[data-status]")?.getAttribute("data-status"));
    expect(statuses).toContain("active"); // the "start" call
    expect(statuses).toContain("done"); // the "ok" call
  });

  it("renders a mono id pill for a UUID-like token embedded in the reasoning text", () => {
    render(<SessionReasoningView runId="prj_test" events={EVENTS} />);
    const idEl = screen.getByText(TEST_UUID);
    expect(idEl).toHaveClass(styles.idPill);
  });

  it("renders a maroon link for a URL embedded in the reasoning text", () => {
    render(<SessionReasoningView runId="prj_test" events={EVENTS} />);
    const link = screen.getByRole("link", { name: TEST_URL });
    expect(link).toHaveAttribute("href", TEST_URL);
    expect(link).toHaveClass(styles.link);
  });

  it("renders the rubric headline (best-of-run) score", () => {
    render(<SessionReasoningView runId="prj_test" events={EVENTS} />);
    expect(screen.getByText("0.53")).toBeInTheDocument();
  });

  it("never leaks a corpus marker planted on unsanitized event fields", () => {
    render(<SessionReasoningView runId="prj_test" events={EVENTS} />);
    expect(document.body.textContent).not.toContain(CORPUS_MARKER);
  });

  it("renders a fail-soft empty state when a run has no events yet", () => {
    render(<SessionReasoningView runId="prj_empty" events={[]} />);
    expect(screen.getByText(/waiting for the agent/i)).toBeInTheDocument();
  });

  it("renders the docked steering input", () => {
    render(<SessionReasoningView runId="prj_test" events={EVENTS} />);
    expect(screen.getByPlaceholderText(/send a message/i)).toBeInTheDocument();
  });
});
