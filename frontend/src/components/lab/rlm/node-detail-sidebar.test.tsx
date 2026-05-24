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

  it("shows structured primitive error detail with recovery guidance", () => {
    const calls = [
      {
        primitive: "plan_reproduction",
        status: "error" as const,
        args_summary: {},
        result_summary: "ValidationError: datasets.0 expected dict [type=dict_type]",
        iteration: 3,
        rubric_delta: null,
        timestamp: "2026-05-21T00:00:00Z",
      },
    ];
    render(
      <NodeDetailSidebar
        {...baseProps({
          node: makeNode({ kind: "baseline", title: "Baseline" }),
          primitiveCalls: calls,
        })}
      />
    );
    expect(screen.getByText("ERROR · plan_reproduction")).toBeInTheDocument();
    expect(screen.getByText("Reason")).toBeInTheDocument();
    expect(screen.getByText(/Pydantic ValidationError/)).toBeInTheDocument();
    expect(screen.getByText("Recovery")).toBeInTheDocument();
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

  // ── In-band hint rendering ────────────────────────────────────────────────
  it("renders amber hint dot when result_summary starts with '[hint] '", () => {
    const calls = [
      {
        primitive: "understand_section",
        status: "ok" as const,
        args_summary: {},
        result_summary: "[hint] dict[_meta, ambiguities, datasets, hardwa…]",
        iteration: 1,
        rubric_delta: null,
        timestamp: "2026-05-23T00:00:00Z",
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
    // The amber dot span is aria-hidden; find the hint container by its title attribute.
    const hintEl = screen.getByTitle("Harness hint: try rlm_query for this size");
    expect(hintEl).toBeInTheDocument();
    // The summary text (without the "[hint] " prefix) must appear.
    expect(hintEl.textContent).toContain("dict[");
  });

  it("does NOT render hint treatment for normal result_summary", () => {
    const calls = [
      {
        primitive: "understand_section",
        status: "ok" as const,
        args_summary: {},
        result_summary: "dict[ambiguities, datasets, hardware_clues, metrics]",
        iteration: 1,
        rubric_delta: null,
        timestamp: "2026-05-23T00:00:00Z",
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
    expect(screen.queryByTitle("Harness hint: try rlm_query for this size")).not.toBeInTheDocument();
    expect(screen.getByText(/dict\[ambiguities/)).toBeInTheDocument();
  });

  // ── GpuPlan badge ─────────────────────────────────────────────────────────

  it("renders GpuPlan badge when gpuPlan is provided", () => {
    const gpuPlan = {
      runpod_id: "NVIDIA_A100_SXM4_80GB",
      short_name: "a100_80",
      vram_gb: 80,
      gpu_count: 1,
      cloud_type: "COMMUNITY",
      sku_usd_per_hr: 1.89,
      total_usd_per_hr: 1.89,
      container_disk_gb: 40,
      volume_gb: 50,
      source: "paper" as const,
      requirements: {
        estimated_vram_gb: 72,
        paper_gpu_string: "A100 80GB",
        paper_gpu_count: 8,
        reasoning: "paper states 80GB",
        confidence: 0.85,
      },
      ladder_remaining: 2,
      resolved_at: "2026-05-23T00:00:01Z",
    };
    render(<NodeDetailSidebar {...baseProps({ gpuPlan })} />);
    const badge = screen.getByTestId("gpu-plan-badge");
    expect(badge).toBeInTheDocument();
    expect(screen.getByText("a100_80")).toBeInTheDocument();
    expect(screen.getByText("80 GB")).toBeInTheDocument();
    expect(screen.getByText("$1.89/hr")).toBeInTheDocument();
    // source !== "fallback" so no fallback tag
    expect(screen.queryByText("fallback")).not.toBeInTheDocument();
  });

  it("shows gpu_count multiplier when gpu_count > 1", () => {
    const gpuPlan = {
      runpod_id: "NVIDIA_A100_SXM4_80GB",
      short_name: "a100_80",
      vram_gb: 80,
      gpu_count: 4,
      cloud_type: "COMMUNITY",
      sku_usd_per_hr: 1.89,
      total_usd_per_hr: 7.56,
      container_disk_gb: 40,
      volume_gb: 50,
      source: "paper" as const,
      requirements: {
        estimated_vram_gb: 280,
        paper_gpu_string: "4x A100 80GB",
        paper_gpu_count: 4,
        reasoning: "paper states 4 GPUs",
        confidence: 0.9,
      },
      ladder_remaining: 0,
      resolved_at: "2026-05-23T00:00:01Z",
    };
    render(<NodeDetailSidebar {...baseProps({ gpuPlan })} />);
    expect(screen.getByText("×4")).toBeInTheDocument();
    expect(screen.getByText("$7.56/hr")).toBeInTheDocument();
  });

  it("shows fallback tag with reasoning title when source=fallback", () => {
    const gpuPlan = {
      runpod_id: "NVIDIA_RTX_4090",
      short_name: "rtx4090",
      vram_gb: 24,
      gpu_count: 1,
      cloud_type: "COMMUNITY",
      sku_usd_per_hr: 0.34,
      total_usd_per_hr: 0.34,
      container_disk_gb: 40,
      volume_gb: 50,
      source: "fallback" as const,
      requirements: {
        estimated_vram_gb: null,
        paper_gpu_string: null,
        paper_gpu_count: null,
        reasoning: "no hardware clues found; defaulting to RTX 4090",
        confidence: 0.1,
      },
      ladder_remaining: 3,
      resolved_at: "2026-05-23T00:00:01Z",
    };
    render(<NodeDetailSidebar {...baseProps({ gpuPlan })} />);
    const fallbackEl = screen.getByText("fallback");
    expect(fallbackEl).toBeInTheDocument();
    expect(fallbackEl).toHaveAttribute(
      "title",
      "no hardware clues found; defaulting to RTX 4090"
    );
  });

  it("does NOT render GpuPlan badge when gpuPlan is null", () => {
    render(<NodeDetailSidebar {...baseProps({ gpuPlan: null })} />);
    expect(screen.queryByTestId("gpu-plan-badge")).not.toBeInTheDocument();
  });

  it("does NOT render GpuPlan badge when gpuPlan is omitted (default)", () => {
    render(<NodeDetailSidebar {...baseProps()} />);
    expect(screen.queryByTestId("gpu-plan-badge")).not.toBeInTheDocument();
  });

  // ── Per-model metrics grid (Lane γ, 2026-05-23) ───────────────────────────

  it("renders per-model grid on baseline node when perModelMetrics has ≥2 models", () => {
    const perModelMetrics = {
      qwen3_1_7b: { alfworld_success_rate: 0.34, searchqa_accuracy: 0.42 },
      qwen2_5_3b: { alfworld_success_rate: 0.51, searchqa_accuracy: 0.55 },
    };
    render(
      <NodeDetailSidebar
        {...baseProps({
          node: makeNode({ kind: "baseline", id: "baseline", title: "Baseline" }),
          perModelMetrics,
        })}
      />
    );
    expect(screen.getByTestId("per-model-grid")).toBeInTheDocument();
    // Model column headers
    expect(screen.getByText("qwen2_5_3b")).toBeInTheDocument();
    expect(screen.getByText("qwen3_1_7b")).toBeInTheDocument();
    // Metric row label
    expect(screen.getByText("alfworld_success_rate")).toBeInTheDocument();
    // Value cells
    expect(screen.getByText("0.510")).toBeInTheDocument();
    expect(screen.getByText("0.340")).toBeInTheDocument();
  });

  it("does NOT render per-model grid when perModelMetrics is null", () => {
    render(
      <NodeDetailSidebar
        {...baseProps({
          node: makeNode({ kind: "baseline", id: "baseline", title: "Baseline" }),
          perModelMetrics: null,
        })}
      />
    );
    expect(screen.queryByTestId("per-model-grid")).not.toBeInTheDocument();
  });

  it("does NOT render per-model grid when perModelMetrics has only 1 model", () => {
    const perModelMetrics = {
      qwen3_1_7b: { acc: 0.34 },
    };
    render(
      <NodeDetailSidebar
        {...baseProps({
          node: makeNode({ kind: "baseline", id: "baseline", title: "Baseline" }),
          perModelMetrics,
        })}
      />
    );
    expect(screen.queryByTestId("per-model-grid")).not.toBeInTheDocument();
  });

  it("shows '—' for missing metrics in per-model grid", () => {
    // model_b is missing "recall" which model_a has
    const perModelMetrics = {
      model_a: { precision: 0.9, recall: 0.8 },
      model_b: { precision: 0.85 },
    };
    render(
      <NodeDetailSidebar
        {...baseProps({
          node: makeNode({ kind: "baseline", id: "baseline", title: "Baseline" }),
          perModelMetrics,
        })}
      />
    );
    expect(screen.getByTestId("per-model-grid")).toBeInTheDocument();
    // "—" for missing recall on model_b
    expect(screen.getByText("—")).toBeInTheDocument();
  });
});
