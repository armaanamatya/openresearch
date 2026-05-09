import { fireEvent, render, screen, waitFor } from "@testing-library/react";

import { LiveDemoClient } from "./live-demo-client";

describe("LiveDemoClient", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("passes the selected SDK provider when starting an SDK run", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        projectId: "ui_sdk_openai_demo_1",
        outputDir: "runs/ui_sdk_openai_demo_1",
        runMode: "sdk",
        llmProvider: "openai",
        status: "queued",
        payload: null,
        log: ""
      })
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<LiveDemoClient initialRun={null} />);

    fireEvent.change(screen.getByLabelText(/sdk provider/i), {
      target: { value: "openai" }
    });
    fireEvent.click(screen.getByRole("button", { name: /run sdk/i }));

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith("/api/demo?mode=sdk&provider=openai", {
        method: "POST"
      });
    });
  });
});
