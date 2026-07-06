import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";

const pushMock = vi.fn();
const replaceMock = vi.fn();
vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: pushMock, replace: replaceMock, prefetch: vi.fn() }),
}));

import { RepoConfirm } from "./RepoConfirm";
import { markDevSessionSignedIn } from "@/lib/autoresearch/dev-session";

// Mirrors use-run.test.ts's helper — a terminal ("completed") run state so
// the poll/SSE effect early-returns and no EventSource/timer gets scheduled
// in jsdom.
function okRunStateResponse(projectId = "prj_test123") {
  return {
    ok: true,
    status: 202,
    json: async () => ({
      projectId,
      outputDir: `runs/${projectId}`,
      runMode: "rlm",
      status: "completed",
      payload: null,
      log: "",
    }),
  };
}

function failedRunStateResponse(message = "boom") {
  return {
    ok: false,
    status: 500,
    text: async () => JSON.stringify({ error: message }),
    json: async () => ({ error: message }),
  };
}

describe("RepoConfirm", () => {
  beforeEach(() => {
    window.sessionStorage.clear();
    window.localStorage.clear();
    pushMock.mockClear();
    replaceMock.mockClear();
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
    window.sessionStorage.clear();
    window.localStorage.clear();
  });

  it("renders the arXiv eyebrow with the paper id", () => {
    render(<RepoConfirm arxivId="2605.15155" initialRepoUrl="ZJU-REAL/SDAR" />);
    expect(screen.getByText(/arxiv 2605\.15155/i)).toBeInTheDocument();
  });

  it("renders a fail-soft fallback title when no title is supplied", () => {
    render(<RepoConfirm arxivId="2605.15155" initialRepoUrl="" />);
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(/reproduce this paper/i);
  });

  it("pre-fills the RepoField from the resolved repo suggestion", () => {
    render(<RepoConfirm arxivId="2605.15155" initialRepoUrl="ZJU-REAL/SDAR" />);
    expect(screen.getByLabelText("GitHub repository")).toHaveValue("ZJU-REAL/SDAR");
    expect(screen.getByText(/we found the code for this paper/i)).toBeInTheDocument();
  });

  it("renders blank + an honest fail-soft message when no repo was resolved", () => {
    render(<RepoConfirm arxivId="2605.15155" initialRepoUrl="" />);
    expect(screen.getByLabelText("GitHub repository")).toHaveValue("");
    expect(
      screen.getByText(/couldn.t automatically find a code repository/i)
    ).toBeInTheDocument();
  });

  it("opens a confirm dialog showing the $10 GPU cap when Start is clicked", () => {
    render(<RepoConfirm arxivId="2605.15155" initialRepoUrl="ZJU-REAL/SDAR" />);
    fireEvent.click(screen.getByRole("button", { name: /start autoresearch/i }));

    const dialog = screen.getByRole("dialog");
    expect(dialog).toBeInTheDocument();
    expect(dialog).toHaveTextContent("$10");
  });

  it("gates the launch behind the stubbed auth flag — disabled + no fetch when not signed in", () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    render(<RepoConfirm arxivId="2605.15155" initialRepoUrl="ZJU-REAL/SDAR" />);

    fireEvent.click(screen.getByRole("button", { name: /start autoresearch/i }));
    const confirmBtn = screen.getByRole("button", { name: /confirm .* launch/i });
    expect(confirmBtn).toBeDisabled();

    fireEvent.click(confirmBtn);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("closes without launching on Cancel", () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    render(<RepoConfirm arxivId="2605.15155" initialRepoUrl="ZJU-REAL/SDAR" />);

    fireEvent.click(screen.getByRole("button", { name: /start autoresearch/i }));
    fireEvent.click(screen.getByRole("button", { name: /cancel/i }));

    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("closes the dialog on Escape", () => {
    render(<RepoConfirm arxivId="2605.15155" initialRepoUrl="ZJU-REAL/SDAR" />);
    fireEvent.click(screen.getByRole("button", { name: /start autoresearch/i }));
    expect(screen.getByRole("dialog")).toBeInTheDocument();

    fireEvent.keyDown(document, { key: "Escape" });
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("launches with autonomous:true + the edited repoUrl and routes to /sessions/<runId> once signed in", async () => {
    markDevSessionSignedIn();
    const fetchMock = vi.fn().mockResolvedValue(okRunStateResponse("prj_launch1"));
    vi.stubGlobal("fetch", fetchMock);

    render(<RepoConfirm arxivId="2605.15155" initialRepoUrl="ZJU-REAL/SDAR" />);

    // Edit the field before confirming — the launcher should see the edit,
    // not the original pre-filled suggestion.
    fireEvent.change(screen.getByLabelText("GitHub repository"), {
      target: { value: "someone/else" },
    });

    fireEvent.click(screen.getByRole("button", { name: /start autoresearch/i }));
    const confirmBtn = screen.getByRole("button", { name: /confirm .* launch/i });
    expect(confirmBtn).toBeEnabled();

    fireEvent.click(confirmBtn);

    await waitFor(() => {
      expect(pushMock).toHaveBeenCalledWith("/sessions/prj_launch1");
    });

    const call = fetchMock.mock.calls.find((c: unknown[]) => c[0] === "/api/demo/arxiv");
    expect(call).toBeDefined();
    const body = JSON.parse(call![1].body as string);
    expect(body.autonomous).toBe(true);
    expect(body.repo_url).toBe("someone/else");
    expect(body.url).toBe("https://arxiv.org/abs/2605.15155");
  });

  it("keeps the dialog open and surfaces the error when the launch fails", async () => {
    markDevSessionSignedIn();
    const fetchMock = vi.fn().mockResolvedValue(failedRunStateResponse("boom"));
    vi.stubGlobal("fetch", fetchMock);

    render(<RepoConfirm arxivId="2605.15155" initialRepoUrl="ZJU-REAL/SDAR" />);
    fireEvent.click(screen.getByRole("button", { name: /start autoresearch/i }));
    fireEvent.click(screen.getByRole("button", { name: /confirm .* launch/i }));

    await waitFor(() => {
      expect(screen.getByText("boom")).toBeInTheDocument();
    });
    expect(screen.getByRole("dialog")).toBeInTheDocument();
    expect(pushMock).not.toHaveBeenCalled();
  });
});
