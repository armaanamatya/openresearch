# Stellar Landing Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current dashboard homepage on `main` with the new Stellar-style white landing page hero, animated tab stage, video overlays, and logo rail.

**Architecture:** Build the new homepage as a small set of landing-specific components plus one client-side interactive tab stage. Replace the current dashboard render in `frontend/src/app/page.tsx`, move typography and animation primitives into `globals.css`, and drive the tab overlays from a single typed config module so the UI stays testable and compact.

**Tech Stack:** Next.js 16 App Router, React 19, TypeScript, Tailwind CSS utility classes, Lucide React, Vitest, Testing Library

---

## File Structure

### Existing files to modify

- `frontend/package.json`
  - Add `lucide-react` dependency.
- `frontend/src/app/layout.tsx`
  - Add the Inter font and update metadata for the new landing page.
- `frontend/src/app/page.tsx`
  - Replace the dashboard homepage with the new landing-page assembly.
- `frontend/src/app/globals.css`
  - Replace the old dashboard-specific global styling with the required font base and animation primitives.
- `frontend/src/test/workspace.test.ts`
  - Replace the placeholder workspace test with a real landing-page render suite or keep as a tiny smoke file if a dedicated landing test file is added.

### New files to create

- `frontend/src/components/landing/stellar-navigation.tsx`
  - Top navigation bar.
- `frontend/src/components/landing/stellar-hero.tsx`
  - Reviews badge, headline, subheading, CTA, and page assembly shell.
- `frontend/src/components/landing/stellar-tab-stage.tsx`
  - Client component with tab state, video stage, and conditional overlays.
- `frontend/src/components/landing/stellar-logo-rail.tsx`
  - Stylized company logo row.
- `frontend/src/lib/landing/stellar-tabs.ts`
  - Typed tab definitions, overlay content, and supporting display data.
- `frontend/src/test/landing-page.test.tsx`
  - Homepage behavior and render tests.

## Task 1: Establish Dependencies, Font, And Animation Primitives

**Files:**
- Modify: `frontend/package.json`
- Modify: `frontend/src/app/layout.tsx`
- Modify: `frontend/src/app/globals.css`
- Create: `frontend/src/test/landing-page.test.tsx`

- [ ] **Step 1: Write the failing homepage test**

```tsx
import { render, screen } from "@testing-library/react";
import HomePage from "../app/page";

it("renders the main Stellar headline", () => {
  render(<HomePage />);
  expect(screen.getByText(/work smarter\. move faster\./i)).toBeInTheDocument();
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm run test -- landing-page.test.tsx`
Expected: FAIL because `page.tsx` still renders the dashboard shell.

- [ ] **Step 3: Add dependency, Inter font, and animation CSS**

Update `frontend/package.json`:

```json
"dependencies": {
  "lucide-react": "latest"
}
```

Update `frontend/src/app/layout.tsx` to use `next/font/google`:

```tsx
import { Inter } from "next/font/google";

const inter = Inter({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"]
});
```

Apply the font class to `<body>` and update metadata for the new landing page.

Replace the current global dashboard styling in `frontend/src/app/globals.css` with:

- white base background
- `font-family: "Inter", sans-serif`
- the three custom animation keyframes/classes:
  - `fadeInUp`
  - `fadeInOverlay`
  - `fadeInDialog`

- [ ] **Step 4: Install dependency and rerun the focused test**

Run:

```bash
npm install
npm run test -- landing-page.test.tsx
```

Expected: test still fails, but only because the homepage has not been replaced yet.

- [ ] **Step 5: Commit**

```bash
git add frontend/package.json frontend/package-lock.json frontend/src/app/layout.tsx frontend/src/app/globals.css frontend/src/test/landing-page.test.tsx
git commit -m "test: add stellar landing page harness"
```

## Task 2: Replace The Homepage With Navigation And Hero

**Files:**
- Modify: `frontend/src/app/page.tsx`
- Create: `frontend/src/components/landing/stellar-navigation.tsx`
- Create: `frontend/src/components/landing/stellar-hero.tsx`
- Modify: `frontend/src/test/landing-page.test.tsx`

- [ ] **Step 1: Extend the failing test for nav and CTA**

```tsx
it("renders the navigation and primary CTA", () => {
  render(<HomePage />);
  expect(screen.getByText("Stellar.ai")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: /begin free trial/i })).toBeInTheDocument();
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm run test -- landing-page.test.tsx`
Expected: FAIL because the homepage still renders the old dashboard.

- [ ] **Step 3: Implement the static hero assembly**

Replace `frontend/src/app/page.tsx` with a landing-page assembly using:

- `StellarNavigation`
- `StellarHero`
- container classes matching the spec:
  - `bg-white`
  - `max-w-7xl`
  - `mx-auto`

Implement:

- nav with Lucide `Star` and `ChevronDown`
- review badge
- headline
- subheading
- `Begin Free Trial` CTA

All major blocks should:

- use `.animate-fade-in-up`
- start with inline `opacity: 0`
- use staggered inline `animationDelay`

- [ ] **Step 4: Rerun the focused test**

Run: `npm run test -- landing-page.test.tsx`
Expected: PASS for the nav/headline/CTA assertions.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/app/page.tsx frontend/src/components/landing/stellar-navigation.tsx frontend/src/components/landing/stellar-hero.tsx frontend/src/test/landing-page.test.tsx
git commit -m "feat: replace homepage with stellar hero"
```

## Task 3: Build The Interactive Tab Stage And Video Shell

**Files:**
- Create: `frontend/src/components/landing/stellar-tab-stage.tsx`
- Create: `frontend/src/lib/landing/stellar-tabs.ts`
- Modify: `frontend/src/components/landing/stellar-hero.tsx`
- Modify: `frontend/src/test/landing-page.test.tsx`

- [ ] **Step 1: Add failing tests for tabs and video**

```tsx
it("renders all four tabs", () => {
  render(<HomePage />);
  for (const label of ["Analyse", "Train", "Testing", "Deploy"]) {
    expect(screen.getByRole("button", { name: new RegExp(label, "i") })).toBeInTheDocument();
  }
});
```

```tsx
it("renders the autoplaying video stage", () => {
  render(<HomePage />);
  const video = screen.getByTestId("stellar-video");
  expect(video).toHaveAttribute("autoplay");
  expect(video).toHaveAttribute("muted");
  expect(video).toHaveAttribute("loop");
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm run test -- landing-page.test.tsx`
Expected: FAIL because the tab stage has not been implemented yet.

- [ ] **Step 3: Implement the client-side tab stage**

Create a client component with:

- `useState("analyse")`
- `setInterval` auto-cycle every 4 seconds
- cleanup on unmount
- mobile 2x2 tab layout
- desktop segmented row with dividers
- looping video element using the provided source

Keep the tab content driven by `stellar-tabs.ts`.

- [ ] **Step 4: Rerun the focused test**

Run: `npm run test -- landing-page.test.tsx`
Expected: PASS for tab and video assertions.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/landing/stellar-tab-stage.tsx frontend/src/lib/landing/stellar-tabs.ts frontend/src/components/landing/stellar-hero.tsx frontend/src/test/landing-page.test.tsx
git commit -m "feat: add stellar tab video stage"
```

## Task 4: Add Conditional Overlays And Logo Rail

**Files:**
- Create: `frontend/src/components/landing/stellar-logo-rail.tsx`
- Modify: `frontend/src/components/landing/stellar-tab-stage.tsx`
- Modify: `frontend/src/lib/landing/stellar-tabs.ts`
- Modify: `frontend/src/components/landing/stellar-hero.tsx`
- Modify: `frontend/src/test/landing-page.test.tsx`

- [ ] **Step 1: Add failing tests for overlay switching and logos**

```tsx
it("shows analyse overlay content by default", () => {
  render(<HomePage />);
  expect(screen.getByText(/set up your ai workspace/i)).toBeInTheDocument();
});
```

```tsx
it("renders the company logo rail", () => {
  render(<HomePage />);
  for (const label of ["INTERSCOPE", "SPOTIFY", "Nexera", "M3", "LAURA COLE", "vertex"]) {
    expect(screen.getByText(label)).toBeInTheDocument();
  }
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm run test -- landing-page.test.tsx`
Expected: FAIL because overlays/logos are not implemented yet.

- [ ] **Step 3: Implement the four overlays and logo rail**

Add overlay variants:

- Analyse
- Train
- Testing
- Deploy

Use the required animation classes:

- outer: `.animate-fade-in-overlay`
- inner: `.animate-slide-up-overlay`

Append the `StellarLogoRail` below the video stage.

- [ ] **Step 4: Rerun the focused test**

Run: `npm run test -- landing-page.test.tsx`
Expected: PASS for default overlay and logo assertions.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/landing/stellar-logo-rail.tsx frontend/src/components/landing/stellar-tab-stage.tsx frontend/src/lib/landing/stellar-tabs.ts frontend/src/components/landing/stellar-hero.tsx frontend/src/test/landing-page.test.tsx
git commit -m "feat: add stellar overlays and logo rail"
```

## Task 5: Final Polish, Tab Interaction Coverage, And Frontend Verification

**Files:**
- Modify: `frontend/src/components/landing/stellar-tab-stage.tsx`
- Modify: `frontend/src/components/landing/stellar-hero.tsx`
- Modify: `frontend/src/test/landing-page.test.tsx`
- Modify: `frontend/src/app/layout.tsx`

- [ ] **Step 1: Add a failing interaction test for tab switching**

```tsx
import userEvent from "@testing-library/user-event";

it("switches overlay content when a tab is clicked", async () => {
  const user = userEvent.setup();
  render(<HomePage />);
  await user.click(screen.getByRole("button", { name: /deploy/i }));
  expect(screen.getByText(/deploy to production/i)).toBeInTheDocument();
});
```

- [ ] **Step 2: Run test to verify it fails or reveals missing behavior**

Run: `npm run test -- landing-page.test.tsx`
Expected: FAIL if click switching is incomplete, or pass after minor polish if already correct.

- [ ] **Step 3: Finish polish and accessibility details**

Make sure:

- all animation delays match the spec
- buttons/controls are keyboard reachable
- active tab states are visually and semantically clear
- metadata aligns with the new public homepage
- layout remains clean on mobile and desktop

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
- Next build passes

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/landing/stellar-tab-stage.tsx frontend/src/components/landing/stellar-hero.tsx frontend/src/test/landing-page.test.tsx frontend/src/app/layout.tsx
git commit -m "style: polish stellar landing page"
```
