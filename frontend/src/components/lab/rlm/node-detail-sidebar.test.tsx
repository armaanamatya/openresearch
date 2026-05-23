/* eslint-disable @typescript-eslint/no-explicit-any */
import { describe, it, expect, vi } from "vitest";
import { render, screen, act } from "@testing-library/react";
import { NodeDetailSidebar } from "./node-detail-sidebar";
import type { NodeDetailSidebarProps } from "./node-detail-sidebar";

// Minimal tree node factories
function makeNode(overrides: Partial<NodeDetailSidebarProps["node"]> = {}) {
  return {
    id: "n1",
    parentId: null,
    kind: "baseline" as const,
    title: "Test Node",
    iterationRange: [1, 1] as [number, number],
    ...overrides,
  } as NodeDetailSidebarProps["node"];
}

const noopSend = async () => {};

function baseProps(extra: Partial<NodeDetailSidebarProps> = {}): NodeDetailSidebarProps {
  return {
    node: makeNode(),
    iteration: null,
    primitiveCalls: [],
    paperMeta: "{}",
    projectId: "proj-1",
    chatMessages: [],
    onSendChat: noopSend,
    subRlms: [],
    iterationCount: 0,
    candidatesProposed: 0,
    candidatesPromoted: 0,
    ...extra,
  };
}

describe("NodeDetailSidebar", () => {
  it("renders the sidebar with data-testid", () => {
    render(<NodeDetailSidebar {...baseProps()} />);
    expect(screen.getByTestId("node-detail-sidebar")).toBeInTheDocument();
  });

  it("shows node title and kind badge when a node is provided", () => {
    render(<NodeDetailSidebar {...baseProps({ node: makeNode({ title: "MyNode", kind: "candidate" }) })} />);
    expect(screen.getByText("MyNode")).toBeInTheDocument();
    expect(screen.getByText("candidate")).toBeInTheDocument();
  });

  it("shows 'no node selected' when node is null", () => {
    render(<NodeDetailSidebar {...baseProps({ node: null })} />);
    expect(screen.getByText(/no node selected/i)).toBeInTheDocument();
  });

  it("collapses to a rail when toggle is clicked", async () => {
    render(<NodeDetailSidebar {...baseProps()} />);
    const collapseBtn = screen.getByRole("button", { name: /collapse/i });
    await act(async () => { collapseBtn.click(); });
    expect(screen.getByRole("button", { name: /expand/i })).toBeInTheDocument();
  });

  // ── Kind: paper ────────────────────────────────────────────────────────────
  it("kind=paper renders paperMeta fields from JSON", () => {
    render(
      <NodeDetailSidebar
        {...baseProps({
          node: makeNode({ kind: "paper", title: "Paper" }),
          paperMeta: JSON.stringify({ title: "Attention Is All You Need", pages: "11" }),
        })}
      />
    );
    expect(screen.getByText("title")).toBeInTheDocument();
    expect(screen.getByText("Attention Is All You Need")).toBeInTheDocument();
  });

  it("kind=paper renders plain-text paperMeta as prose", () => {
    render(
      <NodeDetailSidebar
        {...baseProps({
          node: makeNode({ kind: "paper", title: "Paper" }),
          paperMeta: "plain text meta",
        })}
      />
    );
    expect(screen.getByText("plain text meta")).toBeInTheDocument();
  });

  // ── Kind: work (comprehension) ─────────────────────────────────────────────
  it("kind=work shows comprehension primitives", () => {
    const calls = [
      {
        primitive: "understand_section",
        status: "ok" as const,
        args_summary: {},
        result_summary: "dict[6]",
        iteration: 1,
        rubric_delta: null,
        timestamp: "2026-05-21T00:00:00Z",
      },
      {
        primitive: "extract_hyperparameters",
        status: "ok" as const,
        args_summary: {},
        result_summary: "dict[18]",
        iteration: 2,
        rubric_delta: null,
        timestamp: "2026-05-21T00:00:08Z",
      },
    ];
    render(
      <NodeDetailSidebar
        {...baseProps({
          node: makeNode({ kind: "work", title: "Comprehension" }),
          primitiveCalls: calls,
        })}
      />
    );
    expect(screen.getByText("understand_section")).toBeInTheDocument();
    expect(screen.getByText("extract_hyperparameters")).toBeInTheDocument();
  });

  it("kind=work shows empty-state message when no comprehension calls", () => {
    render(
      <NodeDetailSidebar
        {...baseProps({
          node: makeNode({ kind: "work", title: "Comprehension" }),
          primitiveCalls: [],
        })}
      />
    );
    expect(
      screen.getByText(/no understand_section \/ extract_hyperparameters calls yet/i)
    ).toBeInTheDocument();
  });

  it("kind=work (environment phase) shows environment primitives", () => {
    const calls = [
      {
        primitive: "detect_environment",
        status: "ok" as const,
        args_summary: {},
        result_summary: "dict[7]",
        iteration: 4,
        rubric_delta: null,
        timestamp: "2026-05-21T00:00:00Z",
      },
    ];
    render(
      <NodeDetailSidebar
        {...baseProps({
          node: makeNode({ kind: "work", title: "Environment", phase: "environment" } as any),
          primitiveCalls: calls,
        })}
      />
    );
    expect(screen.getByText("detect_environment")).toBeInTheDocument();
  });

  // ── Kind: candidate ────────────────────────────────────────────────────────
  it("kind=candidate shows category, description, rubricDelta", () => {
    const node = makeNode({
      kind: "candidate",
      title: "warmup",
      round: 1,
      candidate: {
        id: "c1",
        title: "warmup",
        category: "optimizer",
        description: "Noam warmup schedule",
        reasoning: "stabilises training",
      },
      rubricDelta: 0.09,
    });
    render(<NodeDetailSidebar {...baseProps({ node })} />);
    expect(screen.getByText("optimizer")).toBeInTheDocument();
    expect(screen.getByText("Noam warmup schedule")).toBeInTheDocument();
    expect(screen.getByText(/\+0.09 rubric/)).toBeInTheDocument();
  });

  // ── Kind: subrlm ──────────────────────────────────────────────────────────
  it("kind=subrlm shows no sub-RLM detail placeholder when subRlms is empty", () => {
    render(
      <NodeDetailSidebar
        {...baseProps({
          node: makeNode({ kind: "subrlm", id: "subrlm-1", title: "Sub-RLM" }),
          subRlms: [],
        })}
      />
    );
    expect(screen.getByText(/no sub-RLM detail/i)).toBeInTheDocument();
  });

  it("kind=subrlm shows iteration response in now block alongside sub-RLM detail", () => {
    const subRlmEntry = {
      depth: 2,
      model: "claude-sonnet-4-6",
      prompt_preview: "Summarise the method section.",
      duration_ms: 1200,
      error: null,
      spawnedAt: "2026-05-21T00:00:00Z",
      completedAt: "2026-05-21T00:00:01.2Z",
    };
    render(
      <NodeDetailSidebar
        {...baseProps({
          node: makeNode({ kind: "subrlm", id: "subrlm-1", title: "Sub-RLM" }),
          subRlms: [subRlmEntry],
          iteration: {
            iteration: 13,
            response: "Running beam search sub-task.",
            code_blocks: [],
            sub_calls: 0,
            timing: null,
          },
        })}
      />
    );
    expect(screen.getByText("Running beam search sub-task.")).toBeInTheDocument();
  });

  // ── Chat panel ────────────────────────────────────────────────────────────
  it("renders the steering chat panel", () => {
    render(<NodeDetailSidebar {...baseProps()} />);
    expect(screen.getByTestId("steering-chat")).toBeInTheDocument();
  });

  it("renders chat messages", () => {
    const msgs = [
      { id: "m1", role: "user" as const, content: "hello", ts: "2026-05-21T00:00:00Z" },
      { id: "m2", role: "assistant" as const, content: "world", ts: "2026-05-21T00:00:01Z" },
    ];
    render(<NodeDetailSidebar {...baseProps({ chatMessages: msgs })} />);
    expect(screen.getByText("hello")).toBeInTheDocument();
    expect(screen.getByText("world")).toBeInTheDocument();
  });

  it("passes onSendChat to the chat panel (send button exists)", () => {
    const send = vi.fn().mockResolvedValue(undefined);
    render(<NodeDetailSidebar {...baseProps({ onSendChat: send })} />);
    expect(screen.getByRole("button", { name: /send/i })).toBeInTheDocument();
  });

  // ── Aggregate counters strip ───────────────────────────────────────────────
  it("aggregate counters strip renders with correct numbers", () => {
    render(
      <NodeDetailSidebar
        {...baseProps({
          iterationCount: 7,
          primitiveCalls: [
            { primitive: "understand_section", status: "ok", args_summary: {}, result_summary: null, iteration: 1, rubric_delta: null, timestamp: "2026-05-21T00:00:00Z" },
            { primitive: "detect_environment", status: "ok", args_summary: {}, result_summary: null, iteration: 2, rubric_delta: null, timestamp: "2026-05-21T00:00:01Z" },
          ],
          candidatesProposed: 3,
          candidatesPromoted: 1,
        })}
      />
    );
    const strip = screen.getByTestId("aggregate-counters");
    expect(strip).toBeInTheDocument();
    expect(strip).toHaveTextContent("7");      // iterations
    expect(strip).toHaveTextContent("2");      // primitive calls
    expect(strip).toHaveTextContent("3");      // proposed
    expect(strip).toHaveTextContent("1");      // promoted
  });

  // ── Kind: subrlm — enriched view ──────────────────────────────────────────
  it("kind=subrlm with matching SubRlmView shows model, depth, duration, prompt_preview", () => {
    const subRlmEntry = {
      depth: 2,
      model: "claude-sonnet-4-6",
      prompt_preview: "Summarise the method section in detail.",
      duration_ms: 1500,
      error: null,
      spawnedAt: "2026-05-21T00:00:00Z",
      completedAt: "2026-05-21T00:00:01.5Z",
    };
    render(
      <NodeDetailSidebar
        {...baseProps({
          node: makeNode({ kind: "subrlm", id: "subrlm-1", title: "Sub-RLM" }),
          subRlms: [subRlmEntry],
        })}
      />
    );
    expect(screen.getByText("claude-sonnet-4-6")).toBeInTheDocument();
    expect(screen.getByText("2")).toBeInTheDocument();     // depth
    expect(screen.getByText("1.5s")).toBeInTheDocument();  // duration
    expect(screen.getByText(/Summarise the method section in detail\./)).toBeInTheDocument();
  });

  // ── Kind: baseline — enriched view ────────────────────────────────────────
  it("kind=baseline with rubricScore shows score and status chip", () => {
    render(
      <NodeDetailSidebar
        {...baseProps({
          node: makeNode({ kind: "baseline", id: "baseline", title: "Baseline", rubricScore: 0.42 }),
        })}
      />
    );
    expect(screen.getByText(/Rubric score: 0\.42/)).toBeInTheDocument();
    expect(screen.getByText("partial")).toBeInTheDocument();
  });

  it("kind=baseline with high rubricScore shows pass chip", () => {
    render(
      <NodeDetailSidebar
        {...baseProps({
          node: makeNode({ kind: "baseline", id: "baseline", title: "Baseline", rubricScore: 0.75 }),
        })}
      />
    );
    expect(screen.getByText("pass")).toBeInTheDocument();
  });

  it("kind=baseline with low rubricScore shows fail chip", () => {
    render(
      <NodeDetailSidebar
        {...baseProps({
          node: makeNode({ kind: "baseline", id: "baseline", title: "Baseline", rubricScore: 0.2 }),
        })}
      />
    );
    expect(screen.getByText("fail")).toBeInTheDocument();
  });

  it("kind=baseline with no rubricScore shows placeholder", () => {
    render(
      <NodeDetailSidebar
        {...baseProps({
          node: makeNode({ kind: "baseline", id: "baseline", title: "Baseline" }),
        })}
      />
    );
    expect(screen.getByText(/no rubric score yet/i)).toBeInTheDocument();
  });
});
