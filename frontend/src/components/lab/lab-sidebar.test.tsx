import { render, screen } from "@testing-library/react";
import { expect, it } from "vitest";

import { LabSidebar } from "./lab-sidebar";

it("renders Upload as the first navigation item", () => {
  render(<LabSidebar active="upload" recents={[]} />);

  const navItems = screen.getAllByRole("link", { name: /Upload|Lab|Library/i });
  // Filter to main nav items (exclude brand link and footer links that share href /lab)
  const mainNav = navItems.filter((el) => {
    const label = el.getAttribute("data-label");
    return label === "Upload" || label === "Lab" || label === "Library";
  });

  expect(mainNav[0]).toHaveAttribute("data-label", "Upload");
  expect(mainNav[0]).toHaveAttribute("href", "/lab?new=1");
});

it("marks Upload as active when active prop is 'upload'", () => {
  render(<LabSidebar active="upload" recents={[]} />);

  const uploadLink = screen.getByRole("link", { name: /Upload/ });
  expect(uploadLink.className).toContain("active");
});

it("does not mark Upload as active when a run is active", () => {
  render(<LabSidebar active="lab" recents={[]} />);

  const uploadLink = screen.getByRole("link", { name: /Upload/ });
  expect(uploadLink.className).not.toContain("active");
});
