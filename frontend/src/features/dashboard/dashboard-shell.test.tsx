import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { createMockEventAdapter } from "@/lib/events/mock-event-adapter";

import { DashboardShell } from "./dashboard-shell";

describe("DashboardShell", () => {
  it("renders the agent lab shell sections from mock state", () => {
    render(<DashboardShell />);

    expect(screen.getByRole("heading", { name: "Agent Lab" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Topology" })).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: "Reasoning Stream" })
    ).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Messages" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Citations" })).toBeInTheDocument();
  });

  it("applies streamed mock events to the visible dashboard state", async () => {
    const adapter = createMockEventAdapter();
    render(<DashboardShell adapter={adapter} />);

    await adapter.flush();

    await waitFor(() => {
      expect(screen.getByText(/Environment recommendation delivered/i)).toBeInTheDocument();
      expect(screen.getByText(/Draft Docker and package constraints added to shared context./i)).toBeInTheDocument();
    });
  });
});
