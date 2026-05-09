# Mercury Landing Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current dashboard-first home route with a Mercury-inspired public landing page that sells OpenResearch as a system to reproduce, verify, and improve cutting-edge research.

**Architecture:** Keep the signed-in dashboard components intact, but swap the homepage over to a new presentational landing-page composition built from small section components and a static command-preview fixture. Move the visual system to Mercury-aligned tokens in global CSS, then add a landing-specific stylesheet for layout and section styling so the change stays isolated and maintainable.

**Tech Stack:** Next.js 16 App Router, React 19, TypeScript, global CSS, Vitest, Testing Library

---

## File Structure

### Existing files to modify

- `frontend/src/app/page.tsx`
  - Replace the direct `DashboardShell` render with the new landing-page assembly.
- `frontend/src/app/layout.tsx`
  - Update metadata to match the public landing page positioning.
- `frontend/src/app/globals.css`
  - Replace the warm dashboard-root tokens with Mercury tokens and shared page primitives.

### New files to create

- `frontend/src/app/landing.css`
  - Landing-page-only layout, section, and responsive styling.
- `frontend/src/components/landing/site-header.tsx`
  - Minimal translucent header with nav links and CTA.
- `frontend/src/components/landing/hero-section.tsx`
  - Full-height atmospheric hero with primary messaging and actions.
- `frontend/src/components/landing/transition-band.tsx`
  - Bridge section that pivots from mood to operational proof.
- `frontend/src/components/landing/workflow-section.tsx`
  - Five-stage Mercury-style interactive workflow list.
- `frontend/src/components/landing/command-preview.tsx`
  - Static preview of papers, topology, verification, and runtime surfaces.
- `frontend/src/components/landing/capability-grid.tsx`
  - Four capability rails for reproduce, verify, improve, and coordinate.
- `frontend/src/components/landing/final-cta.tsx`
  - Closing call-to-action block.
- `frontend/src/lib/landing/preview-fixtures.ts`
  - Static data for the command preview and workflow proof points.
- `frontend/src/test/landing-page.test.tsx`
  - Render-level tests for hero CTA, workflow stages, and preview labels.

## Task 1: Establish Mercury Tokens And Landing Page Test Harness

**Files:**
- Modify: `frontend/src/app/globals.css`
- Create: `frontend/src/test/landing-page.test.tsx`
- Modify: `frontend/src/app/layout.tsx`

- [ ] **Step 1: Write the failing landing-page test**

```tsx
import { render, screen } from "@testing-library/react";
import HomePage from "../app/page";

describe("landing page", () => {
  it("renders the primary Start building CTA", () => {
    render(<HomePage />);
    expect(
      screen.getByRole("link", { name: /start building/i })
    ).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm run test -- landing-page.test.tsx`
Expected: FAIL because `page.tsx` still renders the dashboard shell and does not include landing-page CTA content.

- [ ] **Step 3: Update metadata and global tokens**

Add the public-facing metadata in `frontend/src/app/layout.tsx`:

```tsx
export const metadata: Metadata = {
  title: "OpenResearch",
  description:
    "Reproduce, verify, and improve cutting-edge research with coordinated agents."
};
```

Replace the existing warm palette in `frontend/src/app/globals.css` with Mercury-aligned root tokens and shared primitives:

```css
:root {
  --color-mercury-blue: #5266eb;
  --color-ghost-blue: #cdddff;
  --color-deep-space: #171721;
  --color-midnight-slate: #1e1e2a;
  --color-graphite: #272735;
  --color-lead: #70707d;
  --color-starlight: #ededf3;
  --color-silver: #c3c3cc;
  --page-max-width: 1200px;
}

html,
body {
  background: var(--color-deep-space);
  color: var(--color-starlight);
}
```

- [ ] **Step 4: Run the focused test again**

Run: `npm run test -- landing-page.test.tsx`
Expected: FAIL, but now only because `page.tsx` still needs the landing-page structure.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/app/layout.tsx frontend/src/app/globals.css frontend/src/test/landing-page.test.tsx
git commit -m "test: add landing page harness"
```

## Task 2: Build The Page Skeleton And Hero Flow

**Files:**
- Modify: `frontend/src/app/page.tsx`
- Create: `frontend/src/app/landing.css`
- Create: `frontend/src/components/landing/site-header.tsx`
- Create: `frontend/src/components/landing/hero-section.tsx`
- Create: `frontend/src/components/landing/transition-band.tsx`

- [ ] **Step 1: Extend the failing test to cover the top-level sections**

```tsx
it("renders the hero headline and bridge copy", () => {
  render(<HomePage />);
  expect(
    screen.getByRole("heading", {
      name: /reproduce, verify, and improve cutting-edge research/i
    })
  ).toBeInTheDocument();
  expect(
    screen.getByText(/from paper abstractions to runnable systems/i)
  ).toBeInTheDocument();
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm run test -- landing-page.test.tsx`
Expected: FAIL because the hero, nav, and transition sections do not exist yet.

- [ ] **Step 3: Write the minimal landing skeleton**

Create `frontend/src/app/landing.css` and import it from `frontend/src/app/page.tsx`.

Assemble the page in `frontend/src/app/page.tsx`:

```tsx
import "./landing.css";
import { HeroSection } from "../components/landing/hero-section";
import { SiteHeader } from "../components/landing/site-header";
import { TransitionBand } from "../components/landing/transition-band";

export default function HomePage() {
  return (
    <main className="landing-page">
      <HeroSection header={<SiteHeader />} />
      <TransitionBand />
    </main>
  );
}
```

Use the approved content:

- headline: `Reproduce, verify, and improve cutting-edge research.`
- primary CTA: `Start building`
- secondary CTA: `View workflow`
- bridge copy: `From paper abstractions to runnable systems, every decision stays visible.`

- [ ] **Step 4: Run the focused test again**

Run: `npm run test -- landing-page.test.tsx`
Expected: PASS for the hero CTA and transition-band assertions.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/app/page.tsx frontend/src/app/landing.css frontend/src/components/landing/site-header.tsx frontend/src/components/landing/hero-section.tsx frontend/src/components/landing/transition-band.tsx frontend/src/test/landing-page.test.tsx
git commit -m "feat: add mercury landing page hero"
```

## Task 3: Add Workflow Section And Capability Rails

**Files:**
- Create: `frontend/src/components/landing/workflow-section.tsx`
- Create: `frontend/src/components/landing/capability-grid.tsx`
- Create: `frontend/src/lib/landing/preview-fixtures.ts`
- Modify: `frontend/src/app/page.tsx`
- Modify: `frontend/src/test/landing-page.test.tsx`

- [ ] **Step 1: Add failing tests for workflow stages and capability copy**

```tsx
it("renders all workflow stages", () => {
  render(<HomePage />);
  for (const label of ["Ingest", "Plan", "Execute", "Verify", "Improve"]) {
    expect(screen.getByRole("heading", { name: label })).toBeInTheDocument();
  }
});
```

```tsx
it("renders the four capability rails", () => {
  render(<HomePage />);
  for (const label of ["Reproduce", "Verify", "Improve", "Coordinate"]) {
    expect(screen.getByText(label)).toBeInTheDocument();
  }
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm run test -- landing-page.test.tsx`
Expected: FAIL because the workflow and capability sections are not rendered yet.

- [ ] **Step 3: Implement static content and section components**

Create `frontend/src/lib/landing/preview-fixtures.ts` with the workflow items and capability panel content:

```ts
export const workflowStages = [
  { step: "01", title: "Ingest", copy: "Parse papers, repositories, datasets, and references..." },
  { step: "02", title: "Plan", copy: "Route work to specialized agents..." }
];
```

Add `WorkflowSection` and `CapabilityGrid`, then include them in `page.tsx` after `TransitionBand`.

- [ ] **Step 4: Run the focused test again**

Run: `npm run test -- landing-page.test.tsx`
Expected: PASS for hero, workflow, and capability assertions.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/landing/workflow-section.tsx frontend/src/components/landing/capability-grid.tsx frontend/src/lib/landing/preview-fixtures.ts frontend/src/app/page.tsx frontend/src/test/landing-page.test.tsx
git commit -m "feat: add landing workflow and capability sections"
```

## Task 4: Build The Static Command Center Preview And Closing CTA

**Files:**
- Create: `frontend/src/components/landing/command-preview.tsx`
- Create: `frontend/src/components/landing/final-cta.tsx`
- Modify: `frontend/src/lib/landing/preview-fixtures.ts`
- Modify: `frontend/src/app/page.tsx`
- Modify: `frontend/src/test/landing-page.test.tsx`

- [ ] **Step 1: Add failing tests for preview content and final CTA**

```tsx
it("renders command preview labels from fixture data", () => {
  render(<HomePage />);
  expect(screen.getByText(/agent topology/i)).toBeInTheDocument();
  expect(screen.getByText(/verification events/i)).toBeInTheDocument();
  expect(screen.getByText(/runtime logs/i)).toBeInTheDocument();
});
```

```tsx
it("renders the closing build-from-frontier-research CTA", () => {
  render(<HomePage />);
  expect(
    screen.getByRole("heading", {
      name: /build from frontier research without losing the thread/i
    })
  ).toBeInTheDocument();
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm run test -- landing-page.test.tsx`
Expected: FAIL because the preview and final CTA sections do not exist yet.

- [ ] **Step 3: Implement the preview and closing block**

Expand `preview-fixtures.ts` with command-preview panels:

```ts
export const previewPanels = {
  leftRail: ["Diffusion Policy", "GraphSAGE Replica"],
  centerTitle: "Agent topology",
  rightTitle: "Verification events",
  runtimeTitle: "Runtime logs"
};
```

Render the preview as a curated static surface, not the live dashboard. Then append `FinalCta` at the bottom of `page.tsx`.

- [ ] **Step 4: Run the focused test again**

Run: `npm run test -- landing-page.test.tsx`
Expected: PASS for preview labels and final CTA content.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/landing/command-preview.tsx frontend/src/components/landing/final-cta.tsx frontend/src/lib/landing/preview-fixtures.ts frontend/src/app/page.tsx frontend/src/test/landing-page.test.tsx
git commit -m "feat: add landing command preview and final cta"
```

## Task 5: Polish Responsive Styling And Verify The Full Frontend

**Files:**
- Modify: `frontend/src/app/landing.css`
- Modify: `frontend/src/app/globals.css`
- Modify: `frontend/src/test/landing-page.test.tsx`

- [ ] **Step 1: Add a failing test for key section ordering**

```tsx
it("renders major sections in landing-page order", () => {
  render(<HomePage />);
  const hero = screen.getByRole("heading", {
    name: /reproduce, verify, and improve cutting-edge research/i
  });
  const workflow = screen.getByRole("heading", { name: "Ingest" });
  const finalCta = screen.getByRole("heading", {
    name: /build from frontier research without losing the thread/i
  });

  expect(hero.compareDocumentPosition(workflow)).toBeTruthy();
  expect(workflow.compareDocumentPosition(finalCta)).toBeTruthy();
});
```

- [ ] **Step 2: Run test to verify it fails or is inconclusive**

Run: `npm run test -- landing-page.test.tsx`
Expected: Either FAIL if structure is wrong, or PASS but reveal that responsive polish and semantic section wrappers still need adjustment. If it already passes, keep the test and proceed.

- [ ] **Step 3: Finish visual polish and responsive behavior**

In `landing.css`, complete:

- full-height hero layout
- translucent nav treatment
- large display typography
- stacked mobile CTA behavior
- workflow row hover states
- command preview grid collapse for tablet/mobile
- wide section spacing and border rhythm

Keep the implementation aligned with the Mercury rules:

- no shadows as primary elevation language
- no extra saturated colors
- pill CTAs only
- generous spacing

- [ ] **Step 4: Run the full frontend verification suite**

Run:

```bash
npm run test
npm run lint
npm run build
```

Expected:

- Vitest passes
- ESLint passes
- Next.js production build passes

- [ ] **Step 5: Commit**

```bash
git add frontend/src/app/landing.css frontend/src/app/globals.css frontend/src/test/landing-page.test.tsx
git commit -m "style: polish mercury landing page"
```
