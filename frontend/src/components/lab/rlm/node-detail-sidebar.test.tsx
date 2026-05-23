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
  it("kind=subrlm shows no-detail message when no iteration", () => {
    render(
      <NodeDetailSidebar
        {...baseProps({ node: makeNode({ kind: "subrlm", title: "Sub-RLM" }) })}
      />
    );
    expect(screen.getByText(/no iteration detail/i)).toBeInTheDocument();
  });

  it("kind=subrlm shows iteration response in now block", () => {
    render(
      <NodeDetailSidebar
        {...baseProps({
          node: makeNode({ kind: "subrlm", title: "Sub-RLM" }),
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
});
