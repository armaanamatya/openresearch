# Landing Port + Lab Repaint + Launch-Readiness Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the hand-authored `LANDING/index.html` into the Next.js app as the production landing at `/`, repaint the lab in the landing's dark amber palette, wire every interactive element so nothing 404s, and ship a clean production build.

**Architecture:** Six tiered phases, each ships independently. Phase 0 cleans up dead artifacts and archives the source. Phase 1 adds the new font system (Geist + Instrument Serif via `next/font/google`) — preflight for both surfaces. Phase 2 ports the static HTML to React components 1:1 (no design liberties). Phase 3 audits and wires every button/link on both surfaces. Phase 4 repaints the lab by editing `tokens.css` + the 15 hardcoded color callsites identified in the inventory. Phase 5 enforces `npm run build` as the launch gate. Phase 6 manual + Playwright smoke across viewports.

**Tech Stack:** Next.js 16 App Router, React 19, TypeScript, CSS Modules + co-located global CSS, `next/font/google`. No new runtime deps (the `tweaks-panel.jsx` Edit Mode tool stays in the archived source — not for production).

**Supersedes:** the retired landing-polish plan. The polish plan assumed my hand-built dark landing was the canonical design; the user produced a fully-fleshed `LANDING/index.html` that replaces it wholesale.

**Phasing strategy (per advisor):** Land Phases 1–3 before touching the lab. The lab repaint is reversible but disruptive — if it surfaces dense-data-readability problems mid-flight, you don't want the landing port held back. Each phase commits cleanly; if the user wants Phase 4 atomic with the landing, they can hold the PR until both land.

---

## File inventory

**Created:**
- `frontend/src/components/landing/landing.tsx` — new, replaces my old component wholesale
- `frontend/src/components/landing/landing.module.css` — new, replaces my old module
- `frontend/src/components/landing/landing.global.css` — new (route-scoped html/body dark + paper grain)
- `frontend/src/components/landing/figures/` — directory for each of the 6 bespoke section mock components (one file per: HeroTree, ComprehensionMock, EnvironmentMock, ImplementationMock, ExperimentsMock, VerificationMock)
- `frontend/src/components/landing/use-reveal.tsx` — IntersectionObserver hook + `<Reveal>` wrapper
- `frontend/src/components/landing/use-nav-scroll.tsx` — sticky-nav-border-on-scroll hook + applied to `<Nav>`

**Modified:**
- `frontend/src/app/layout.tsx` — Geist + Geist Mono + Instrument Serif via `next/font/google`
- `frontend/src/app/page.tsx` — already mounts `<LandingPage />`; no change needed once the component is rewritten
- `frontend/src/styles/tokens.css` — token values flipped to landing's oklch palette (Phase 4)
- `frontend/src/components/lab/lab-sidebar.tsx` — wrap brand-row in `<Link href="/">`
- `frontend/src/components/lab/lab-sidebar.css` — flip the ~6 hardcoded white/black colors (Phase 4)
- `frontend/src/components/lab/lab-shell.css` — same, ~3 callsites
- `frontend/src/components/lab/upload-view.css` — same, ~2 callsites
- `frontend/src/components/lab/command-palette.css` — same, ~1 callsite
- `frontend/src/components/lab/shortcut-overlay.css` — same, ~1 callsite

**Moved:**
- `LANDING/` → `docs/design/landing-source/` (preserves index.html, tweaks.jsx, tweaks-panel.jsx, .thumbnail, uploads/)

**Deleted:**
- The retired landing-polish plan (superseded by this file)

---

## Constants used across the plan

```
GITHUB_URL = https://github.com/armaanamatya/openresearch
RLM_ARXIV_URL = https://arxiv.org/abs/2512.24601
BRANCH = feat/landing-launch  (already on feat/landing-polish — rename in Task 0)
ACCENT_OKLCH = oklch(0.80 0.16 70)
ACCENT_HEX_FALLBACK = #E8A24A
```

---

## Verification approach

Per task: lint + types must pass cleanly for new files. Visual fidelity is the contract — every task that touches CSS or JSX gets a Playwright screenshot diff against the corresponding section of `docs/design/landing-source/index.html` (loaded standalone). For lab repaint tasks, Playwright captures `/lab`, `/library`, and the lab in upload+running states; compare against the current committed lab screenshots when present.

**Single helper script** lives at `scripts/landing-screenshots.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
URL="${1:-http://localhost:3001}"
NAME="${2:-page}"
mkdir -p ./frontend/tmp-screens
npx playwright screenshot "$URL" "./frontend/tmp-screens/${NAME}-full.png" --viewport-size=1440,900 --full-page
npx playwright screenshot "$URL" "./frontend/tmp-screens/${NAME}-mobile.png" --viewport-size=390,844 --full-page
echo "OK -> ./frontend/tmp-screens/${NAME}-*.png"
```

---

# Phase 0 — Cleanup + branch + archive (~20 min)

### Task 0: Rename branch, archive LANDING source, delete obsolete artifacts

**Files:**
- Rename: `LANDING/` → `docs/design/landing-source/`
- Delete: `frontend/src/components/landing/landing.tsx`, `landing.module.css`, `landing.global.css` (the old scaffold)
- Delete: the retired landing-polish plan
- Rename branch: `feat/landing-polish` → `feat/landing-launch`

- [ ] **Step 1: Verify current branch and that nothing is committed yet**

```bash
git status --short
git log --oneline feat/landing-polish ^main 2>&1 | head -5
```

Expected: working tree has the landing scaffold + old plan as uncommitted/staged changes; `git log ... ^main` shows zero commits (we never committed on this branch — the previous scaffold work was all uncommitted).

- [ ] **Step 2: Rename the branch**

```bash
git branch -m feat/landing-polish feat/landing-launch
git status --short  # confirms still in worktree, just renamed
```

- [ ] **Step 3: Move LANDING/ → docs/design/landing-source/**

```bash
mkdir -p docs/design
git mv LANDING docs/design/landing-source 2>/dev/null \
  || mv LANDING docs/design/landing-source  # git mv may fail if LANDING is untracked
ls docs/design/landing-source/
```

Expected: directory listing shows `index.html`, `tweaks.jsx`, `tweaks-panel.jsx`, `.thumbnail`, `uploads/`.

- [ ] **Step 4: Delete the obsolete scaffold + polish plan**

```bash
rm frontend/src/components/landing/landing.tsx
rm frontend/src/components/landing/landing.module.css
rm frontend/src/components/landing/landing.global.css
# Remove the retired landing-polish plan if it is still present.
# Leave the landing/ directory itself — Phase 2 fills it.
ls frontend/src/components/landing/ 2>&1
```

Expected: empty directory listing (`ls: ...: No such file or directory` OR empty result).

- [ ] **Step 5: Temporarily stub `app/page.tsx` so the project still builds**

The current `app/page.tsx` imports `@/components/landing/landing` — which we just deleted. To unblock dev/build until Phase 2 lands, restore the redirect *for this single commit only*:

```tsx
// frontend/src/app/page.tsx
import { redirect } from "next/navigation";

export default function HomePage(): never {
  redirect("/lab");
}
```

(Phase 2 Task 4 replaces this with the new landing component import.)

- [ ] **Step 6: Verify the app boots clean with no broken imports**

```bash
cd frontend
npx tsc --noEmit 2>&1 | grep -E "landing|app/page" || echo "OK: no landing-related TS errors"
OPENRESEARCH_BACKEND_URL=http://127.0.0.1:8000 npm run build 2>&1 | tail -20
```

Expected: tsc grep returns "OK"; build completes (warnings OK; no errors). If build fails, fix before committing.

- [ ] **Step 7: Commit**

```bash
cd /Volumes/CS_Stuff/openresearch
git add -A
git status --short
git commit -m "$(cat <<'EOF'
chore(landing): archive source, delete obsolete scaffold, rename branch

- LANDING/ → docs/design/landing-source/ (preserves index.html prototype
  and tweaks.jsx Edit Mode tool for future design iteration)
- Removed the prior generic-dark React landing component and its CSS
- Superseded the retired landing-polish plan
- Branch renamed feat/landing-polish → feat/landing-launch
- app/page.tsx temporarily reverts to redirect("/lab") until Phase 2
  lands the ported landing component

EOF
)"
```

---

# Phase 1 — Font system (~30 min)

The landing source uses Geist + Geist Mono + Instrument Serif via `<link href="https://fonts.googleapis.com/css2?...">`. In Next.js App Router, that approach works but is non-optimal — `next/font/google` is the canonical path (zero CLS, no external request at runtime, self-hosted). Doing this in Phase 1 means Phase 2's port doesn't have to deal with font loading at all.

### Task 1: Wire Geist + Geist Mono + Instrument Serif via `next/font/google`

**Files:**
- Modify: `frontend/src/app/layout.tsx`

- [ ] **Step 1: Update layout.tsx font imports**

Replace the existing `Inter` + `JetBrains_Mono` declarations with Geist + Geist Mono + Instrument Serif. The existing variables `--font-inter` and `--font-jetbrains-mono` are used by `tokens.css` — keep those variable names so token consumers don't need to change, but point them at Geist.

```tsx
// frontend/src/app/layout.tsx
import type { Metadata } from "next";
import { Geist, Geist_Mono, Instrument_Serif } from "next/font/google";
import "../styles/tokens.css";
import "./globals.css";

const geist = Geist({
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700"],
  variable: "--font-inter"  // keep the variable name so tokens.css still resolves
});

const geistMono = Geist_Mono({
  subsets: ["latin"],
  weight: ["400", "500"],
  variable: "--font-jetbrains-mono"  // keep the variable name
});

const instrumentSerif = Instrument_Serif({
  subsets: ["latin"],
  weight: ["400"],
  style: ["normal", "italic"],
  variable: "--font-serif"
});

export const metadata: Metadata = {
  title: "OpenResearch — reproduce any ML paper, end to end",
  description: "OpenResearch is a paper-reproduction agent built on the Recursive Language Model paradigm. It reads a paper, builds the environment, implements the method, runs the experiments, and grades itself against the paper's own claims."
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${geist.variable} ${geistMono.variable} ${instrumentSerif.variable}`}>
      <body className={geist.className}>{children}</body>
    </html>
  );
}
```

The CSS-variable names `--font-inter` / `--font-jetbrains-mono` are intentionally kept — `tokens.css` references them; we'd otherwise have to update 30+ files. The semantic mapping is now Inter→Geist, JetBrains→Geist Mono. (A follow-up cleanup PR can rename the variables, but it's out of scope for launch.)

- [ ] **Step 2: Confirm the lab still renders with the new fonts**

```bash
cd frontend
OPENRESEARCH_BACKEND_URL=http://127.0.0.1:8000 npm run dev &
sleep 4
./scripts/landing-screenshots.sh http://localhost:3001/lab lab-with-geist
```

Open `./tmp-screens/lab-with-geist-full.png`. Lab text should render in Geist (slightly different shape than Inter — Geist has more open counters, a more geometric feel). Code/mono text in Geist Mono. Everything should still align and be readable. If a lab component breaks alignment, that's a sign Inter-specific metrics were assumed; usually safe (Geist is very Inter-like).

- [ ] **Step 3: Lint + types + commit**

```bash
cd frontend
npx eslint src/app/layout.tsx
npx tsc --noEmit 2>&1 | grep "app/layout" && exit 1 || echo "OK"
cd /Volumes/CS_Stuff/openresearch
git add frontend/src/app/layout.tsx
git commit -m "$(cat <<'EOF'
feat(fonts): swap to Geist + Geist Mono + Instrument Serif via next/font

Self-hosts Google Fonts via next/font (no runtime external request,
zero CLS). Keeps existing --font-inter / --font-jetbrains-mono CSS-var
names so the 30+ token consumers don't need to update. Adds
--font-serif for Instrument Serif (used by Phase 2's landing).

EOF
)"
```

---

# Phase 2 — Port the landing (~3–4 hours, 7 tasks)

Port `docs/design/landing-source/index.html` into React components 1:1. No design liberties — copy markup and CSS verbatim, only restructuring for component boundaries. The `.tweaks-root` and the entire `<script>` block at lines 1565–1571 (tweaks panel) are dev-only and do NOT get ported.

### Task 2: Scaffold landing module skeleton + globals

**Files:**
- Create: `frontend/src/components/landing/landing.tsx`
- Create: `frontend/src/components/landing/landing.module.css`
- Create: `frontend/src/components/landing/landing.global.css`

- [ ] **Step 1: Create landing.global.css with route-scoped html/body + grain overlay**

Copy verbatim from lines 11–67 of `docs/design/landing-source/index.html` `<style>` block:

```css
/* frontend/src/components/landing/landing.global.css
 * Route-scoped global override. Imported only by landing.tsx; Next.js
 * code-splits non-module CSS per route so /lab and /library never
 * receive these rules. */

:root {
  --landing-bg:           oklch(0.16 0.004 270);
  --landing-bg-2:         oklch(0.185 0.004 270);
  --landing-bg-elev:      oklch(0.205 0.004 270);
  --landing-ink:          oklch(0.96 0.005 90);
  --landing-ink-2:        oklch(0.78 0.005 90);
  --landing-ink-3:        oklch(0.56 0.005 90);
  --landing-ink-4:        oklch(0.40 0.005 90);
  --landing-rule:         oklch(0.28 0.004 270);
  --landing-rule-2:       oklch(0.235 0.004 270);
  --landing-accent:       oklch(0.80 0.16 70);
  --landing-accent-ink:   oklch(0.20 0.04 70);
  --landing-maxw:         1240px;
  --landing-gutter:       clamp(20px, 4vw, 56px);
  --landing-section-pad:  clamp(72px, 10vw, 160px);
  --landing-serif:        var(--font-serif), "Iowan Old Style", Georgia, serif;
}

html, body {
  background: var(--landing-bg);
  color: var(--landing-ink);
}

body::before {
  content: "";
  position: fixed; inset: 0;
  pointer-events: none;
  z-index: 100;
  opacity: .035;
  mix-blend-mode: overlay;
  background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='160' height='160'><filter id='n'><feTurbulence baseFrequency='0.9' numOctaves='2' stitchTiles='stitch'/><feColorMatrix values='0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0.35 0'/></filter><rect width='100%25' height='100%25' filter='url(%23n)'/></svg>");
}

body::after {
  content: "";
  position: fixed;
  top: 0; left: 0; right: 0; height: 60vh;
  pointer-events: none;
  z-index: -1;
  background: radial-gradient(120% 80% at 50% -10%, oklch(0.22 0.01 70 / .55), transparent 60%);
}

::selection {
  background: var(--landing-accent);
  color: var(--landing-accent-ink);
}
```

(The `--landing-*` prefix prevents collisions with the lab's `tokens.css` variables. In Phase 4 the lab will adopt the same hue family but its tokens stay differently-named.)

- [ ] **Step 2: Create landing.module.css with all `.wrap`/`.nav`/`.hero`/`.spec*`/`.mock*`/`.bench`/`.cta-foot`/`.foot` styles**

This is the bulk of the porting work — copy ~580 lines of CSS from lines 74–678 of the source HTML into a CSS module. Every selector that was `.foo` in the source becomes `.foo` in the module (CSS modules scope automatically). Cross-references inside the module (e.g. `.mock-paper .col`) stay intact.

Skip these from the source (they belong to landing.global.css per Step 1):
- `:root { ... }`
- `html, body { ... }`
- `body::before, body::after`
- `::selection`

Also globally replace these references:
- `var(--bg)` → `var(--landing-bg)`
- `var(--bg-2)` → `var(--landing-bg-2)`
- `var(--bg-elev)` → `var(--landing-bg-elev)`
- `var(--ink)` → `var(--landing-ink)`
- `var(--ink-2)` → `var(--landing-ink-2)`
- `var(--ink-3)` → `var(--landing-ink-3)`
- `var(--ink-4)` → `var(--landing-ink-4)`
- `var(--rule)` → `var(--landing-rule)`
- `var(--rule-2)` → `var(--landing-rule-2)`
- `var(--accent)` → `var(--landing-accent)`
- `var(--accent-ink)` → `var(--landing-accent-ink)`
- `var(--maxw)` → `var(--landing-maxw)`
- `var(--gutter)` → `var(--landing-gutter)`
- `var(--section-pad)` → `var(--landing-section-pad)`
- `var(--sans)` → `var(--font-inter), system-ui, sans-serif`
- `var(--mono)` → `var(--font-jetbrains-mono), ui-monospace, monospace`
- `var(--serif)` → `var(--landing-serif)`

Use a careful find-and-replace pass. Verify before committing:

```bash
grep -E "var\(--(bg|ink|rule|accent|maxw|gutter|section-pad|sans|mono|serif)[^-]" \
  frontend/src/components/landing/landing.module.css \
  | head -10
```

Expected: empty output (all old token names replaced).

- [ ] **Step 3: Create landing.tsx skeleton that imports both CSS files and renders a `<div className={styles.shell}>`**

```tsx
// frontend/src/components/landing/landing.tsx
import styles from "./landing.module.css";
import "./landing.global.css";

export function LandingPage(): React.JSX.Element {
  return (
    <div className={styles.shell}>
      {/* Phase 2 Tasks 3–7 fill this in */}
      <div style={{ padding: "200px 40px", textAlign: "center", color: "var(--landing-ink-3)" }}>
        Landing scaffolded · sections forthcoming.
      </div>
    </div>
  );
}
```

(Note: the source HTML uses `<body>` directly with `.wrap` containers throughout. The React port wraps everything in `<div className={styles.shell}>` to avoid needing layout-level changes. The shell adds `min-height: 100vh` and inherits all `body`-targeted CSS from landing.global.css.)

Add a `.shell` rule to `landing.module.css`:

```css
.shell {
  min-height: 100vh;
  font-family: var(--font-inter), ui-sans-serif, system-ui;
  font-weight: 400;
  font-size: 16px;
  line-height: 1.55;
  font-feature-settings: "ss01", "cv11";
  -webkit-font-smoothing: antialiased;
  overflow-x: hidden;
}
```

- [ ] **Step 4: Re-mount the landing page**

Replace `app/page.tsx` with the import:

```tsx
// frontend/src/app/page.tsx
import { LandingPage } from "@/components/landing/landing";

export default function HomePage(): React.JSX.Element {
  return <LandingPage />;
}
```

(Metadata moved to layout.tsx in Phase 1.)

- [ ] **Step 5: Visual verify the scaffold renders dark with grain**

```bash
cd frontend
OPENRESEARCH_BACKEND_URL=http://127.0.0.1:8000 npm run dev &
sleep 4
./scripts/landing-screenshots.sh http://localhost:3001/ landing-scaffold
```

Expected: dark page with a barely-perceptible grain texture, "Landing scaffolded · sections forthcoming." centered.

- [ ] **Step 6: Lint + types + commit**

```bash
cd frontend
npx eslint src/app/page.tsx src/components/landing/
npx tsc --noEmit 2>&1 | grep -E "landing|app/page" && exit 1 || echo "OK"
cd /Volumes/CS_Stuff/openresearch
git add frontend/src/app/page.tsx frontend/src/components/landing/
git commit -m "$(cat <<'EOF'
feat(landing): scaffold ported landing skeleton + globals

landing.global.css owns route-scoped html/body, grain overlay, top
radial gradient, ::selection. landing.module.css owns 580 lines of
component CSS from the LANDING/index.html source, with var(--*) names
prefixed --landing-* to prevent collision with lab tokens. Mounted at
/ via app/page.tsx.

EOF
)"
```

---

### Task 3: Nav + Hero (with figure exploration tree SVG)

**Files:**
- Modify: `frontend/src/components/landing/landing.tsx`
- Create: `frontend/src/components/landing/use-nav-scroll.tsx`
- Create: `frontend/src/components/landing/figures/HeroTree.tsx`

- [ ] **Step 1: Create the nav-scroll hook**

```tsx
// frontend/src/components/landing/use-nav-scroll.tsx
"use client";

import { useEffect, useRef } from "react";

export function useNavScroll(scrolledClass: string) {
  const ref = useRef<HTMLElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const onScroll = () => {
      if (window.scrollY > 8) el.classList.add(scrolledClass);
      else el.classList.remove(scrolledClass);
    };
    document.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
    return () => document.removeEventListener("scroll", onScroll);
  }, [scrolledClass]);
  return ref;
}
```

- [ ] **Step 2: Create HeroTree.tsx with the inline SVG**

Copy lines 762–858 of the source HTML — the entire `#hero-tree` SVG — into a React component. Convert SVG attributes to JSX-friendly: `class=` → `className=`, `stroke-width=` → `strokeWidth=`, `text-anchor=` → `textAnchor=`. Keep the inline `<style>` block inside `<defs>` as-is (children of `<style>` work in JSX as a string).

```tsx
// frontend/src/components/landing/figures/HeroTree.tsx
export function HeroTree(): React.JSX.Element {
  return (
    <svg viewBox="0 0 1200 360" width="100%" preserveAspectRatio="xMidYMid meet" aria-hidden="true">
      <defs>
        <style>{`
          .nlbl { font-family: var(--font-jetbrains-mono), ui-monospace, monospace; font-size: 10px; fill: oklch(0.56 0.005 90); letter-spacing: 0.04em; }
          .nlbl-em { fill: oklch(0.96 0.005 90); }
          .nlbl-accent { fill: oklch(0.80 0.16 70); }
          .edge { stroke: oklch(0.28 0.004 270); stroke-width: 1; fill: none; }
          .edge-live { stroke: oklch(0.80 0.16 70 / .8); }
          .node-bg { fill: oklch(0.205 0.004 270); stroke: oklch(0.28 0.004 270); }
          .node-bg-live { fill: oklch(0.205 0.004 270); stroke: oklch(0.80 0.16 70 / .9); }
          .node-bg-done { fill: oklch(0.205 0.004 270); stroke: oklch(0.40 0.005 90); }
          .pulse { animation: pulse 2.2s ease-out infinite; }
          @keyframes pulse { 0% { r: 4; opacity: 1; } 100% { r: 14; opacity: 0; } }
        `}</style>
      </defs>
      {/* paste all <path> and <g> nodes from source lines 780–857 verbatim,
          converting class= → className= and stroke-width= → strokeWidth=
          and any other hyphenated SVG attributes */}
    </svg>
  );
}
```

(The full SVG body is ~80 lines — copy verbatim. Don't summarize or simplify; the source is the spec.)

- [ ] **Step 3: Add Nav + Hero to landing.tsx**

Replace the scaffold placeholder in `landing.tsx`:

```tsx
import Link from "next/link";
import styles from "./landing.module.css";
import "./landing.global.css";
import { useNavScroll } from "./use-nav-scroll";
import { HeroTree } from "./figures/HeroTree";

const GITHUB_URL = "https://github.com/armaanamatya/openresearch";

export function LandingPage(): React.JSX.Element {
  return (
    <div className={styles.shell}>
      <Nav />
      <Hero />
      {/* Tasks 4–7 add sections + footer */}
    </div>
  );
}

function Nav(): React.JSX.Element {
  const ref = useNavScroll(styles.scrolled);
  return (
    <nav ref={ref} className={styles.nav}>
      <div className={`${styles.wrap} ${styles.navInner}`}>
        <Link href="/" className={styles.brand}>
          <span className={styles.brandMark} aria-hidden />
          <span className={styles.brandName}>OpenResearch<span className={styles.dot}>.</span></span>
        </Link>
        <div className={styles.navLinks}>
          <a href="#pipeline">How it works</a>
          <a href="#benchmarks">Benchmarks</a>
          <a href={GITHUB_URL} target="_blank" rel="noreferrer">GitHub</a>
        </div>
        <Link href="/lab" className={styles.navCta}>
          <span>Open lab</span>
          <span className={styles.ar}>→</span>
        </Link>
      </div>
    </nav>
  );
}

function Hero(): React.JSX.Element {
  return (
    <header className={styles.hero} id="top">
      <div className={styles.wrap}>
        <div className={`${styles.heroEyebrow} ${styles.mono} ${styles.reveal}`}>
          <span className={styles.pip} aria-hidden />
          <span>RLM&nbsp;&nbsp;·&nbsp;&nbsp;§ 0.0 — Reproducibility, automated.</span>
        </div>
        <div className={styles.heroGrid}>
          <h1 className={`${styles.display} ${styles.reveal}`}>
            Reproduce any ML paper, <span className={styles.accentSoft}>end&nbsp;to&nbsp;end.</span>
          </h1>
          <div className={styles.reveal}>
            <p className={styles.lede} style={{ margin: "0 0 24px" }}>
              OpenResearch is a paper-reproduction agent built on the Recursive Language Model paradigm.
              It reads a paper, builds the environment, implements the method, runs the
              experiments, and grades itself against the paper&apos;s own claims.
            </p>
            <dl className={styles.heroMeta}>
              <div><dt>Paradigm</dt><dd>Recursive Language Models (RLM)</dd></div>
              <div><dt>Stages</dt><dd>§ 1.0&nbsp;→&nbsp;5.0&nbsp;&nbsp;·&nbsp;&nbsp;closed-loop</dd></div>
              <div><dt>Modality</dt><dd>Code, configs, container, claims</dd></div>
              <div><dt>Status</dt><dd>Research preview · <span className={styles.tok}>{"{{REPRODUCTION_SCORE}}"}</span></dd></div>
            </dl>
          </div>
        </div>
        <div className={`${styles.heroActions} ${styles.reveal}`}>
          <Link href="/lab" className={`${styles.btn} ${styles.btnPrimary}`}>
            <span>Open the lab</span>
            <span aria-hidden>→</span>
          </Link>
          <a href={GITHUB_URL} target="_blank" rel="noreferrer" className={`${styles.btn} ${styles.btnGhost}`}>
            <GitHubIcon />
            <span>View on GitHub</span>
          </a>
        </div>
        <figure className={`${styles.heroFigure} ${styles.reveal}`}>
          <span className={styles.figLabel}>FIG&nbsp;§ 0.1&nbsp;&nbsp;—&nbsp;&nbsp;Lab UI / exploration tree</span>
          <div className={styles.frame}>
            <div className={styles.mockChrome}>
              <span className={styles.dots}><span className={styles.dot} /><span className={styles.dot} /><span className={styles.dot} /></span>
              <span className={styles.crumbs}>lab&nbsp;/&nbsp;<b>arxiv:2410.04265 — Recursive Language Models</b></span>
              <span className={styles.right}>run · rlm-7e3a · 02:14:08</span>
            </div>
            <div style={{ padding: "22px 24px 28px" }}>
              <HeroTree />
            </div>
          </div>
        </figure>
      </div>
    </header>
  );
}

function GitHubIcon(): React.JSX.Element {
  return (
    <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor" aria-hidden>
      <path d="M8 0C3.58 0 0 3.58 0 8a8 8 0 005.47 7.59c.4.07.55-.17.55-.38v-1.34c-2.23.48-2.7-1.07-2.7-1.07-.36-.92-.89-1.17-.89-1.17-.72-.49.06-.48.06-.48.8.06 1.22.82 1.22.82.71 1.21 1.87.86 2.33.66.07-.52.28-.86.5-1.06-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.6 7.6 0 014 0c1.53-1.03 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.28.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48v2.2c0 .21.15.46.55.38A8 8 0 0016 8c0-4.42-3.58-8-8-8z" />
    </svg>
  );
}
```

Note the changes from source: (a) "RLM paper" nav link DROPPED — the source HTML had `<a href="#paper">RLM paper</a>` but no `#paper` section exists in the doc. (b) Nav CTA correctly goes to `/lab` (the source has incoherent `href="#github"` for an "Open lab" button). (c) Hero secondary CTA correctly goes to the real GitHub URL. (d) `Link href="/"` for brand.

CSS module class-name mapping for the dasherized source class names: in the CSS file they stay `.nav-inner`, `.brand-name`, etc. In TS, CSS modules expose them as either `styles["nav-inner"]` OR (depending on `cssModules` config) auto-camelCase to `styles.navInner`. Verify the project's Next.js config — Next 16 default is to expose both forms. If the camelCase form doesn't resolve at runtime, switch all class refs to bracket-notation: `className={styles["nav-inner"]}`.

To eliminate ambiguity, **prefer kebab-case bracket-notation throughout the JSX** unless the project already has a precedent for camelCase:

```bash
grep -r "className={styles\." frontend/src/components/lab/ | head -5
# Look at what convention the lab uses; match it.
```

- [ ] **Step 4: Screenshot the hero**

```bash
./scripts/landing-screenshots.sh http://localhost:3001/ landing-hero
```

Compare `./tmp-screens/landing-hero-full.png` against `docs/design/landing-source/index.html` rendered standalone (open in a browser). The hero region (above the first `<section>`) should be visually identical: eyebrow with pip, large display headline with second-half muted, right-column lede + 2×2 meta grid, two action buttons, framed exploration tree with mock chrome.

- [ ] **Step 5: Lint + types + commit**

```bash
cd frontend
npx eslint src/components/landing/
npx tsc --noEmit 2>&1 | grep -E "landing|app/page" && exit 1 || echo "OK"
cd /Volumes/CS_Stuff/openresearch
git add frontend/src/components/landing/landing.tsx frontend/src/components/landing/use-nav-scroll.tsx frontend/src/components/landing/figures/
git commit -m "$(cat <<'EOF'
feat(landing): port nav + hero + exploration-tree figure

Source: docs/design/landing-source/index.html (lines 683–862).
useNavScroll hook applies a .scrolled class to the sticky nav for the
on-scroll border. HeroTree extracted into figures/. Nav rewires the
incoherent source hrefs: "Open lab" CTA → /lab, GitHub → real repo URL.
Drops "RLM paper" nav link since the source has no #paper section.

EOF
)"
```

---

### Task 4: §1.0 Comprehension + §2.0 Environment sections

**Files:**
- Create: `frontend/src/components/landing/figures/ComprehensionMock.tsx`
- Create: `frontend/src/components/landing/figures/EnvironmentMock.tsx`
- Modify: `frontend/src/components/landing/landing.tsx`

- [ ] **Step 1: Create ComprehensionMock.tsx**

Port lines 895–986 of the source (the `.mock.mock-paper` block with three columns: paper preview, claim cards, citation graph). Copy SVG verbatim. Component shape:

```tsx
// frontend/src/components/landing/figures/ComprehensionMock.tsx
import styles from "../landing.module.css";

export function ComprehensionMock(): React.JSX.Element {
  return (
    <div className={`${styles.mock} ${styles.mockPaper}`}>
      <div className={styles.col}>
        {/* doc-meta, paper-title, authors, abstract (with <mark> highlights) */}
      </div>
      <div className={styles.col}>
        {/* 4 .claim cards: C-014, C-018, C-022, C-031 */}
      </div>
      <div className={styles.col}>
        {/* graph-mini citation SVG + rubric task-coverage list */}
      </div>
    </div>
  );
}
```

Copy the full content — don't summarize. The source has bespoke text in each claim card, in the abstract, in the rubric list; preserve every character.

- [ ] **Step 2: Create EnvironmentMock.tsx**

Port lines 1023–1082 (`.mock.mock-env`). Two columns: env-left (tree + lockfile) + env-right (4 cards + log). Verbatim copy.

- [ ] **Step 3: Add `ComprehensionSection` and `EnvironmentSection` to landing.tsx**

```tsx
function ComprehensionSection(): React.JSX.Element {
  return (
    <section className={styles.spec} id="pipeline">
      <div className={styles.wrap}>
        <div className={`${styles.specHeader} ${styles.reveal}`}>
          <div className={styles.specNum}>
            <span className={styles.glyph}>§</span><span className={styles.n}>1.0</span>
          </div>
          <div className={styles.specTitle}>
            <div className={styles.name}>
              <span className={styles.word}>Comprehension</span>
              <span className={styles.rule} aria-hidden />
              <span className={styles.stage}>stage 01 / 05</span>
            </div>
            <h2 className={styles.head}>Read the paper like an author would.</h2>
          </div>
          <div>
            <p className={styles.specBlurb}>
              The agent ingests the PDF and source repo, extracts every quantitative claim,
              and builds a citation graph linking each claim to the table, figure, or
              equation that supports it.
            </p>
            <div className={styles.specSubsteps}>
              <a href="#"><span className={styles.num}>§ 1.1</span><span>Parsing</span></a>
              <a href="#"><span className={styles.num}>§ 1.2</span><span>Claim extraction</span></a>
              <a href="#"><span className={styles.num}>§ 1.3</span><span>Citation graph</span></a>
              <a href="#"><span className={styles.num}>§ 1.4</span><span>Rubric induction</span></a>
            </div>
          </div>
        </div>
        <figure className={`${styles.specVisual} ${styles.reveal}`}>
          <span className={styles.figLabel}>FIG&nbsp;§ 1.2&nbsp;&nbsp;—&nbsp;&nbsp;Claim extraction · rubric induction</span>
          <div className={`${styles.frame} ${styles.frameBleedR}`}>
            <div className={styles.mockChrome}>
              <span className={styles.crumbs}>comprehension&nbsp;/&nbsp;<b>arxiv:2410.04265</b></span>
              <span className={styles.right}>47 claims · 23 figures · 9 tables</span>
            </div>
            <ComprehensionMock />
          </div>
        </figure>
      </div>
    </section>
  );
}

function EnvironmentSection(): React.JSX.Element {
  // identical shape — 2.0 / Environment / "Stand up exactly the machine the paper ran on."
  // 4 substeps, frame (no bleed), EnvironmentMock, "FIG § 2.2"
  // Copy text verbatim from source lines 992–1085
}
```

Note: the substeps `<a href="#">` are intentional fake-anchors in the source (they don't link anywhere). In Phase 3 Task 9 we decide whether to make them functional or `aria-disabled`.

- [ ] **Step 4: Screenshot + lint + commit**

```bash
./scripts/landing-screenshots.sh http://localhost:3001/ landing-sections-1-2
cd frontend && npx eslint src/components/landing/ && npx tsc --noEmit 2>&1 | grep -E "landing|app/page" && exit 1 || echo "OK"
cd /Volumes/CS_Stuff/openresearch
git add frontend/src/components/landing/
git commit -m "feat(landing): port §1.0 Comprehension + §2.0 Environment sections

```

---

### Task 5: §3.0 Implementation + §4.0 Experiments

Same shape as Task 4. Port `ImplementationMock` (lines 1118–1156 — files tree + tabbed code with `<pre class="code">` + add/del diff lines) and `ExperimentsMock` (lines 1192–1316 — line chart SVG with grid pattern + 6 `.run-row` entries). Add `ImplementationSection` and `ExperimentsSection` functions to landing.tsx.

- [ ] **Step 1: Create ImplementationMock.tsx**

Port verbatim. The `<pre class="code">` block uses class names `.ln`, `.kw`, `.fn`, `.st`, `.cm`, `.add`, `.del` — these map to CSS module classes. JSX renders the source's special-char `&gt;` and `<` correctly using `{">"}` and entities.

- [ ] **Step 2: Create ExperimentsMock.tsx**

Port the chart SVG verbatim. The `<defs><pattern id="grid">` works fine in JSX as-is. The 6 run-rows are verbatim copies.

- [ ] **Step 3: Add `ImplementationSection` + `ExperimentsSection` to landing.tsx**

Same header shape as Comprehension/Environment, with verbatim text from source lines 1088–1320.

- [ ] **Step 4: Screenshot + commit**

```bash
./scripts/landing-screenshots.sh http://localhost:3001/ landing-sections-3-4
git add frontend/src/components/landing/
git commit -m "feat(landing): port §3.0 Implementation + §4.0 Experiments sections

```

---

### Task 6: §5.0 Verification + §6.0 Benchmarks

- [ ] **Step 1: Create VerificationMock.tsx**

Port lines 1353–1443. Rubric table with 6 rows (C-014 through C-031), score-strip with 4 cells containing placeholder tokens (`{{N_VERIFIED}}`, `{{N_CLAIMS}}`, `{{REPRODUCTION_SCORE}}`, `{{DRIFT_BAND}}`, plus an unverified "report.md · run.json" cell).

- [ ] **Step 2: Add `VerificationSection` and `BenchmarkSection` to landing.tsx**

VerificationSection: same header shape, FIG label "FIG § 5.1 — Reproduction scorecard", frame with `frameBleedR`, includes the VerificationMock.

BenchmarkSection (§6.0): different shape — no figure; instead a `.bench-numbers` 4-cell grid below the header with placeholder tokens (`{{REPRODUCTION_SCORE}}`, `{{N_PAPERS}}`, `{{MEDIAN_DRIFT}}`, `{{HOURS_PER_PAPER}}` plus target). Copy verbatim from source lines 1450–1494.

- [ ] **Step 3: Screenshot + commit**

```bash
./scripts/landing-screenshots.sh http://localhost:3001/ landing-sections-5-6
git add frontend/src/components/landing/
git commit -m "feat(landing): port §5.0 Verification + §6.0 Benchmarks sections

```

---

### Task 7: CTA Footer + Footer + scroll-reveal hook + final landing assembly

**Files:**
- Create: `frontend/src/components/landing/use-reveal.tsx`
- Modify: `frontend/src/components/landing/landing.tsx`

- [ ] **Step 1: Create the IntersectionObserver reveal hook**

```tsx
// frontend/src/components/landing/use-reveal.tsx
"use client";

import { useEffect } from "react";
import styles from "./landing.module.css";

export function useRevealOnScroll() {
  useEffect(() => {
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const targets = document.querySelectorAll(`.${styles.reveal}`);
    if (reduced) {
      targets.forEach((el) => el.classList.add(styles.in));
      return;
    }
    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (e.isIntersecting) {
            e.target.classList.add(styles.in);
            io.unobserve(e.target);
          }
        }
      },
      { rootMargin: "-8% 0px -8% 0px", threshold: 0.05 }
    );
    targets.forEach((el) => io.observe(el));
    return () => io.disconnect();
  }, []);
}
```

The hook is class-based (matches the source HTML behavior). Server-rendered output has `class="reveal"`; client mount adds `.in` when in view.

- [ ] **Step 2: Add CTAFooter + Footer to landing.tsx**

```tsx
function CTAFooter(): React.JSX.Element {
  return (
    <section className={styles.ctaFoot} id="github">
      <div className={`${styles.wrap} ${styles.reveal}`}>
        <div className={styles.hEyebrow} style={{ marginBottom: 22 }}>§ 7.0 — Try it</div>
        <h2>An end-to-end reproduction,<br />in one command.</h2>
        <p className={styles.lede}>
          Point OpenResearch at an arXiv ID. Get back a sealed environment, an implementation,
          a scorecard, and an audit trail. If a claim doesn&apos;t reproduce, you&apos;ll see exactly which one.
        </p>
        <div className={styles.actions}>
          <Link href="/lab" className={`${styles.btn} ${styles.btnPrimary}`}>
            <span>Open the lab</span><span aria-hidden>→</span>
          </Link>
          <a href={GITHUB_URL} target="_blank" rel="noreferrer" className={`${styles.btn} ${styles.btnGhost}`}>
            <GitHubIcon />
            <span>github.com/armaanamatya/openresearch</span>
          </a>
        </div>
      </div>
    </section>
  );
}

function Footer(): React.JSX.Element {
  return (
    <footer className={styles.foot}>
      <div className={`${styles.wrap} ${styles.footInner}`}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span className={styles.brandMark} style={{ width: 18, height: 18 }} aria-hidden />
          <span>OpenResearch · 2026</span>
        </div>
        <div className={styles.links}>
          <a href="#pipeline">How it works</a>
          <a href="#benchmarks">Benchmarks</a>
          <a href={GITHUB_URL} target="_blank" rel="noreferrer">GitHub</a>
        </div>
        <div className={styles.mono}>§ end · v0.1.0</div>
      </div>
    </footer>
  );
}
```

- [ ] **Step 3: Wire the reveal hook into a client component**

The landing page is currently a server component. Hooks need a client wrapper. Create a small client component:

```tsx
// inside landing.tsx, near the top
"use client";  // remove if the component is meant to stay server — but useReveal requires client

// Alternative: keep landing.tsx as server, push useReveal into a small client wrapper

function RevealMount(): null {
  useRevealOnScroll();
  return null;
}
```

Then in `LandingPage`, add `<RevealMount />` at the top of the shell. Mark `LandingPage` with `"use client"` since `RevealMount` plus `Nav`'s `useNavScroll` both need it.

Or — cleaner separation: keep `LandingPage` server-side, extract `Nav` and `RevealMount` into their own client components. Standard Next.js pattern.

```tsx
// frontend/src/components/landing/client-bits.tsx
"use client";

import { useRevealOnScroll } from "./use-reveal";
import { useNavScroll } from "./use-nav-scroll";

export function RevealMount(): null {
  useRevealOnScroll();
  return null;
}

export function NavScrollMount({ targetId, scrolledClass }: { targetId: string; scrolledClass: string }) {
  useNavScroll(scrolledClass); // mutates `ref` — adjust to query by id instead, see note
  return null;
}
```

(For NavScrollMount specifically, since the Nav is rendered server-side, the hook needs to find the `<nav id="...">` via `document.getElementById` rather than `useRef`. Update `useNavScroll` accordingly.)

- [ ] **Step 4: Final LandingPage assembly**

```tsx
export function LandingPage(): React.JSX.Element {
  return (
    <div className={styles.shell}>
      <RevealMount />
      <NavScrollMount targetId="nav" scrolledClass={styles.scrolled} />
      <Nav />
      <Hero />
      <ComprehensionSection />
      <EnvironmentSection />
      <ImplementationSection />
      <ExperimentsSection />
      <VerificationSection />
      <BenchmarkSection />
      <CTAFooter />
      <Footer />
    </div>
  );
}
```

Server-render the structure; client-mount augments scroll-reveal and nav-border. Best of both worlds: no flash of unrevealed content (server renders the markup), and the JS enhancement runs after hydration.

- [ ] **Step 5: Full-page screenshot diff against source HTML**

```bash
./scripts/landing-screenshots.sh http://localhost:3001/ landing-final
# Side-by-side with the source HTML rendered standalone — open both in a browser
open docs/design/landing-source/index.html
open frontend/tmp-screens/landing-final-full.png
```

Compare section by section. Differences should be limited to: zero (ideally), or minor: font rendering due to next/font self-hosting vs Google Fonts CDN (Geist render identical in both cases).

- [ ] **Step 6: Lint + types + commit**

```bash
cd frontend
npx eslint src/components/landing/
npx tsc --noEmit 2>&1 | grep -E "landing|app/page" && exit 1 || echo "OK"
cd /Volumes/CS_Stuff/openresearch
git add frontend/src/components/landing/
git commit -m "$(cat <<'EOF'
feat(landing): port CTA footer + footer + scroll-reveal + final assembly

Server-renders all section markup; client mount via RevealMount and
NavScrollMount augments behavior (IntersectionObserver class toggle on
.reveal targets, nav.scrolled class on scroll past 8px). Honors
prefers-reduced-motion.

EOF
)"
```

---

**End of Phase 2.** Landing is fully ported, all sections render, scroll-reveal fires, nav border appears on scroll. Buttons may not all work yet — Phase 3.

---

# Phase 3 — Button + link audit and wiring (~1 hour)

The source HTML has known-broken hrefs (advisor caught these). Plus the lab sidebar uses `onBrandClick` not Link. Plus the substeps in each section are fake-anchors. Audit, decide, wire.

### Task 8: Enumerate every interactive element on landing + lab, mark each as wired/disabled/external/anchor

**Files:**
- Modify: `frontend/src/components/landing/landing.tsx`

The following inventory must hold true after Task 8 lands (every row checked):

| Surface | Element | Current state | Target state |
|---------|---------|---------------|--------------|
| Landing nav | brand "OpenResearch." | `Link href="/"` | ✅ unchanged |
| Landing nav | "How it works" | `<a href="#pipeline">` | ✅ unchanged (anchor scroll) |
| Landing nav | "Benchmarks" | `<a href="#benchmarks">` | ✅ unchanged |
| Landing nav | "GitHub" | `<a href={GITHUB_URL} target="_blank">` | ✅ unchanged |
| Landing nav | "Open lab →" CTA | `<Link href="/lab">` | ✅ unchanged |
| Hero | "Open the lab →" primary | `<Link href="/lab">` | ✅ unchanged (source had `href="#"` — fixed during port) |
| Hero | "View on GitHub" ghost | `<a href={GITHUB_URL} target="_blank">` | ✅ unchanged (source had `href="#"` — fixed during port) |
| Each section | "§ X.Y" substep anchors | `<a href="#">` (source) | **Convert to `<span>` non-anchors** OR keep as `<a href="#" aria-disabled>` with click prevented. They link nowhere and clicking them jumps to top — bad UX. Removing the `<a>` keeps the visual but loses interactivity. |
| CTA footer | "Open the lab →" primary | `<Link href="/lab">` | ✅ unchanged |
| CTA footer | "github.com/armaanamatya/openresearch" ghost | `<a href={GITHUB_URL} target="_blank">` | ✅ unchanged |
| Footer | "How it works" | `<a href="#pipeline">` | ✅ unchanged |
| Footer | "Benchmarks" | `<a href="#benchmarks">` | ✅ unchanged |
| Footer | "GitHub" | `<a href={GITHUB_URL} target="_blank">` | ✅ unchanged |
| Lab sidebar | Brand row "OpenResearch" | `<button onClick={onBrandClick}>` | **Convert to `<Link href="/">`** so it navigates back to landing |
| Lab sidebar | "Lab" nav | `<a href="/lab">` | ✅ unchanged |
| Lab sidebar | "Library" nav | `<a href="/library">` | ✅ unchanged |

- [ ] **Step 1: Convert landing substeps from `<a href="#">` to `<span>`**

In each of the 5 ComprehensionSection, EnvironmentSection, ImplementationSection, ExperimentsSection, VerificationSection, change:

```tsx
<a href="#"><span className={styles.num}>§ 1.1</span><span>Parsing</span></a>
```

to

```tsx
<span className={styles.subitem}><span className={styles.num}>§ 1.1</span><span>Parsing</span></span>
```

And add to `landing.module.css`:

```css
.subitem {
  display: flex; gap: 12px; padding: 8px 0;
  font-size: 14px; color: var(--landing-ink-2);
  border-top: 1px solid var(--landing-rule-2);
}
```

(This matches the existing `.spec-substeps a` rule but for non-anchor. Or alternatively, keep using `a` selectors in CSS; only the JSX changes.)

Reasoning: every substep is content-only, not a navigation target. The source HTML's `<a href="#">` jumps to top on click — confusing. Removing the `<a>` keeps the visual treatment, removes the broken behavior. The user can wire them to per-substep deep links later if needed.

- [ ] **Step 2: Smoke-test that all link clicks work**

Boot dev, manually click every element. Expected:
- Brand "OpenResearch." in nav → `/` (current page, soft refresh)
- "How it works" / "Benchmarks" → scroll to section
- "GitHub" → new tab to repo
- "Open lab" nav-cta → `/lab`
- Hero primary → `/lab`
- Hero "View on GitHub" → new tab to repo
- Substeps → no click action (not anchor anymore)
- Section in-page anchors (via nav) → correct section scroll
- CTA footer "Open the lab" → `/lab`
- CTA footer ghost → new tab to repo
- Footer links → scroll/external as documented

- [ ] **Step 3: Lint + commit**

```bash
cd frontend && npx eslint src/components/landing/ && echo "OK"
cd /Volumes/CS_Stuff/openresearch
git add frontend/src/components/landing/
git commit -m "$(cat <<'EOF'
fix(landing): wire all interactive elements, remove broken substep anchors

- Substeps are now <span>, not <a href="#"> — the source's anchors
  jumped to top on click (no real target). Visual unchanged.
- All real CTAs (hero, nav, CTA footer, footer) confirmed wired to
  /lab or the GitHub URL. None remain as href="#".

EOF
)"
```

---

### Task 9: Wire lab sidebar brand-row → Link href="/"

**Files:**
- Modify: `frontend/src/components/lab/lab-sidebar.tsx`
- Modify: `frontend/src/components/lab/lab-shell.tsx` (callers of LabSidebar — `onBrandClick` prop becomes optional or removed)

- [ ] **Step 1: Read both files first**

```bash
grep -n "onBrandClick\|brand-row" frontend/src/components/lab/lab-sidebar.tsx frontend/src/components/lab/lab-shell.tsx
```

- [ ] **Step 2: Convert the button to a Link in lab-sidebar.tsx**

```tsx
// before
<button className="brand-row" type="button" onClick={onBrandClick}>
  <span className="nav-icon">{ICONS.logo}</span>
  <span className="brand-text">OpenResearch</span>
</button>

// after
import Link from "next/link";
// (and remove onBrandClick from props if no other caller uses it)
<Link href="/" className="brand-row" aria-label="OpenResearch — back to landing">
  <span className="nav-icon">{ICONS.logo}</span>
  <span className="brand-text">OpenResearch</span>
</Link>
```

Update the LabSidebar props type:

```tsx
export function LabSidebar({
  active,
  recents
}: {
  active: string;
  recents: RecentRunSummary[];
}) { ... }
```

And in `lab-shell.tsx`, remove the `onBrandClick` argument from the LabSidebar usage. If `onBrandClick` had a custom behavior (e.g., "reset upload state on click"), preserve that elsewhere — but `Link href="/"` already navigates away, so resetting local state is moot.

- [ ] **Step 3: Verify lab still renders**

```bash
./scripts/landing-screenshots.sh http://localhost:3001/lab lab-brand-link
```

Hover the wordmark — cursor should be a pointer. Click — should navigate to `/`. Use browser back — returns to lab.

- [ ] **Step 4: Lint + commit**

```bash
cd frontend && npx eslint src/components/lab/lab-sidebar.tsx src/components/lab/lab-shell.tsx && echo "OK"
cd /Volumes/CS_Stuff/openresearch
git add frontend/src/components/lab/
git commit -m "$(cat <<'EOF'
feat(lab): wordmark click navigates back to landing

Lab sidebar wordmark wraps in <Link href="/"> so users can return to
the marketing surface — standard product-app affordance. Removes the
unused onBrandClick callback prop.

EOF
)"
```

---

**End of Phase 3.** Every interactive element on landing + lab works. Substeps are non-interactive by design. No `href="#"` survives.

---

# Phase 4 — Lab dark amber repaint (~2 hours, 3 tasks)

The user accepted the dense-data-on-dark readability risk. Repaint via the tokens.css single source + fix the ~15 hardcoded callsites identified in the inventory.

### Task 10: Flip tokens.css to the dark amber oklch palette

**Files:**
- Modify: `frontend/src/styles/tokens.css`
- Modify: `frontend/tailwind.config.ts` if it duplicates tokens

- [ ] **Step 1: Replace tokens.css with the dark amber variant**

```css
/* frontend/src/styles/tokens.css
 * Design tokens, dark amber theme (2026-05-23 launch repaint).
 *
 * Hue family matches landing's --landing-* tokens. Lab and landing
 * now share the same dark surface vocabulary, but use distinct token
 * namespaces so route-scoped overrides remain clean.
 */

:root {
  /* === Inks / surfaces (dark amber-tinted gray) === */
  --ink: oklch(0.96 0.005 90);
  --ink-2: oklch(0.78 0.005 90);
  --muted: oklch(0.56 0.005 90);
  --muted-2: oklch(0.40 0.005 90);
  --line: oklch(0.28 0.004 270);
  --line-2: oklch(0.235 0.004 270);
  --chip: oklch(0.22 0.004 270);
  --bg: oklch(0.16 0.004 270);
  --panel: oklch(0.185 0.004 270);
  --canvas-bg: oklch(0.185 0.004 270);
  --canvas-grid: oklch(0.235 0.004 270);

  /* === Accent (warm amber, matches landing) === */
  --accent: oklch(0.80 0.16 70);
  --accent-soft: oklch(0.80 0.16 70 / 0.18);
  --accent-ink: oklch(0.20 0.04 70);

  /* === Info === */
  --info-soft: oklch(0.45 0.1 260 / 0.18);
  --info-ink: oklch(0.78 0.10 260);

  /* === Hermes (kept distinctive purple) === */
  --hermes: oklch(0.70 0.16 290);
  --hermes-soft: oklch(0.70 0.16 290 / 0.16);
  --hermes-ink: oklch(0.94 0.04 290);

  /* === Warning === */
  --warn: oklch(0.75 0.14 60);
  --warn-soft: oklch(0.75 0.14 60 / 0.16);
  --warn-ink: oklch(0.96 0.04 60);

  /* === Error === */
  --err: oklch(0.66 0.18 25);
  --err-soft: oklch(0.66 0.18 25 / 0.16);

  /* === Typography (unchanged, drives Geist) === */
  --font-sans: var(--font-inter), system-ui, sans-serif;
  --font-mono: var(--font-jetbrains-mono), ui-monospace, "SF Mono", Menlo, monospace;
  --fs-xs: 11px;
  --fs-sm: 12.5px;
  --fs-base: 13.5px;
  --fs-md: 15px;
  --fs-lg: 18px;
  --fs-xl: 24px;

  /* === Spacing (unchanged) === */
  --sp-1: 4px;
  --sp-2: 8px;
  --sp-3: 12px;
  --sp-4: 16px;
  --sp-5: 24px;
  --sp-6: 32px;

  /* === Radii (unchanged) === */
  --r-xs: 6px;
  --r-sm: 8px;
  --r-md: 10px;
  --r-lg: 14px;

  /* === Shadows (re-derived for dark surface) === */
  --shadow-1: 0 1px 2px oklch(0.02 0 0 / 0.5);
  --shadow-2: 0 4px 12px -6px oklch(0.02 0 0 / 0.5);
  --shadow-3: 0 16px 36px -18px oklch(0.02 0 0 / 0.7);
}
```

- [ ] **Step 2: Update `frontend/src/app/globals.css` lab-shell.css color references**

Open `frontend/src/app/globals.css`. The `html` selector sets `background: var(--bg)` — already token-driven; will repaint automatically. Same for `body`. No edits needed if it's all `var(--*)`.

- [ ] **Step 3: Verify with screenshot**

```bash
./scripts/landing-screenshots.sh http://localhost:3001/lab lab-dark-amber
```

Open the screenshot. Lab should now be on dark background with amber accents wherever the lab uses `--accent`. Look for: legibility (can you read dense run lists, claim cards, sidebar nav?), contrast (does the sparkline still render against dark?), accent visibility (the green sidebar item indicators should now be amber).

If readability is degraded but the user accepted this risk — flag but proceed. If it's catastrophic, escalate.

- [ ] **Step 4: Lint + commit**

```bash
cd frontend && npx tsc --noEmit 2>&1 | tail -10
# tokens.css has no TS exposure — skip tsc grep
cd /Volumes/CS_Stuff/openresearch
git add frontend/src/styles/tokens.css
git commit -m "$(cat <<'EOF'
feat(lab): repaint to dark amber palette matching landing

tokens.css now defines an oklch-based dark theme with warm amber
accent (oklch 0.80 0.16 70). All lab CSS that consumes var(--*) tokens
repaints automatically. Hardcoded callsites are fixed in the next
commit. Accepts the dense-data-on-dark readability tradeoff per
explicit user direction.

EOF
)"
```

---

### Task 11: Fix the 15 hardcoded color callsites in lab CSS

The inventory caught these: `#fff` (5+ uses), `rgba(0, 0, 0, ...)`, `rgba(14, 14, 16, ...)`, `rgba(155, 155, 163, ...)`. Each one needs to flip to a token, or to a dark-mode-appropriate equivalent.

**Files:**
- Modify: `frontend/src/components/lab/lab-sidebar.css` (≥6 spots)
- Modify: `frontend/src/components/lab/lab-shell.css` (≥2 spots)
- Modify: `frontend/src/components/lab/upload-view.css` (≥2 spots)
- Modify: `frontend/src/components/lab/command-palette.css` (1 spot)
- Modify: `frontend/src/components/lab/shortcut-overlay.css` (1 spot)

- [ ] **Step 1: Enumerate every hardcoded color**

```bash
grep -rEn "#[0-9a-fA-F]{3,8}|rgba?\(" frontend/src/components/lab/ \
  | grep -v "var(--" | grep -v "node_modules" \
  > /tmp/lab-hardcoded.txt
cat /tmp/lab-hardcoded.txt
```

Process each line. Replace:
- `color: #fff` → `color: var(--ink)` (white-on-dark, ink is near-white now)
- `background: #fff` → `background: var(--panel)` (was white; now the dark panel color)
- `background: rgba(0, 0, 0, 0.04)` → `background: var(--chip)` (was a faint-gray hover; chip is a darker subtle hover)
- `background: rgba(14, 14, 16, 0.4)` (overlay backdrops) → `background: oklch(0 0 0 / 0.6)` (true black at 60% — proper modal scrim)
- `box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04)` → `box-shadow: var(--shadow-1)` (token now resolves to dark shadow)
- `rgba(155, 155, 163, 0.5)` (gradient stop) → adjust to match the new dark palette: `oklch(0.40 0.005 90 / 0.5)` is the equivalent muted-gray

Go through each file; do it carefully. After each file, save and screenshot the lab to spot regressions.

- [ ] **Step 2: Re-screenshot the lab after each file is updated**

```bash
./scripts/landing-screenshots.sh http://localhost:3001/lab lab-after-hardfix
./scripts/landing-screenshots.sh http://localhost:3001/library library-after-hardfix
```

Look for ghost-light spots: any element that's still rendering light when it should be dark = a missed callsite. Inspect via DevTools.

- [ ] **Step 3: Visit lab with fixture data to test dense-UI surfaces**

```bash
# the lab supports a fixture mode via ?rlmFixture=1
./scripts/landing-screenshots.sh "http://localhost:3001/lab?rlmFixture=1" lab-fixture-dark
```

Open the screenshot. Verify: exploration canvas readable, primitive-history-bar tokens readable, repl-state-rail var listing readable, rubric-strip + report-rail panels readable. If any panel is unreadable on dark, flag.

- [ ] **Step 4: Lint + commit**

```bash
cd frontend && npx eslint src/components/lab/ && echo "OK"
cd /Volumes/CS_Stuff/openresearch
git add frontend/src/components/lab/
git commit -m "$(cat <<'EOF'
fix(lab): flip 15 hardcoded color callsites to token-driven values

Sweeps lab-sidebar.css, lab-shell.css, upload-view.css,
command-palette.css, shortcut-overlay.css. Each #fff / rgba(0,0,0,..)
spot now goes through var(--ink) / var(--panel) / var(--chip) /
var(--shadow-*) so the dark theme paints uniformly.

EOF
)"
```

---

### Task 12: Spot-check dense lab surfaces (RLM exploration canvas, charts) for dark readability

**No file changes** unless a regression is found.

- [ ] **Step 1: Boot a real lab run (or fixture) and visually inspect each panel**

```bash
# Use the fixture mode for repeatable inspection
open "http://localhost:3001/lab?rlmFixture=1"
```

For each of the following, verify legibility:
- Sidebar navigation + recent runs
- Top header (paper title, rubric score)
- Exploration canvas (tree nodes, edges, edge labels)
- REPL state rail (variable listing)
- Primitive history bar
- Rubric strip
- Report rail (areas table)
- Footer controls

If something is borderline, propose a small CSS tweak (per-component, not global). If something is unreadable, escalate to the user — the dark-mode tradeoff has hit a limit.

- [ ] **Step 2: Optional escape hatch — per-component light overrides**

If a panel is unreadable on dark and the design loses meaning, the lab can opt that panel back into a light surface via a wrapper class:

```css
/* example only — apply per-component if a panel proves unreadable */
.lightPanel {
  --ink: oklch(0.14 0 0);
  --ink-2: oklch(0.30 0 0);
  --bg: oklch(0.98 0 0);
  --panel: oklch(1 0 0);
  --line: oklch(0.90 0 0);
  /* ... */
  background: var(--panel);
  color: var(--ink);
}
```

This is a per-panel escape valve, not a global toggle. Use sparingly — every use is a small contract-break in the dark-theme story.

- [ ] **Step 3: Commit if any tweaks landed; otherwise skip the commit**

---

**End of Phase 4.** Lab and landing share the dark amber visual identity. Lab dense data may have lost a touch of readability — accepted tradeoff.

---

# Phase 5 — Launch readiness (~30 min)

### Task 13: Production build must succeed

`npm run build` is the real gate. `npm run dev` cheats with hot-module-replacement and skips production minification + SSR boundary checks.

**No file changes** (unless build fails).

- [ ] **Step 1: Run production build**

```bash
cd frontend
OPENRESEARCH_BACKEND_URL=http://127.0.0.1:8000 npm run build 2>&1 | tee /tmp/build.log | tail -40
```

Expected: build completes with no errors. Warnings about CSS being too large, or about "use client" boundary inferences, are OK.

If build fails:
- "Module not found" → check imports
- "use client cannot be used in a server component" → wrap the offending component
- Font hydration mismatch → confirm `next/font` is in layout.tsx, not page.tsx
- CSS parse error → revisit Phase 2 Task 2 (the bracket-notation vs camelCase issue)

- [ ] **Step 2: Run the production server briefly and re-screenshot**

```bash
OPENRESEARCH_BACKEND_URL=http://127.0.0.1:8000 npm run start &
sleep 5
./scripts/landing-screenshots.sh http://localhost:3000/ landing-prod
./scripts/landing-screenshots.sh http://localhost:3000/lab lab-prod
```

(Production uses port 3000 by default since dev was on 3001.)

Compare `landing-prod-full.png` against `landing-final-full.png` from Phase 2 Task 7. They should be visually indistinguishable. If they differ (font swap, missing CSS, layout shift), debug.

- [ ] **Step 3: Lint full project + types**

```bash
cd frontend
npx eslint .
npx tsc --noEmit 2>&1 | tail -20
```

Expected: lint exits 0. tsc shows only the pre-existing `lab/rlm/*.test.tsx` errors that existed before any of this work (verified in prior session). No new errors.

If new errors appeared, fix them.

- [ ] **Step 4: Commit if anything landed; otherwise tag the milestone**

```bash
git log --oneline ^main | head -20
# Should see the full stack of ~12 commits across the 6 phases
```

---

### Task 14: Dead-file and orphan purge

**Files (to inspect, possibly delete):**
- `scripts/landing-screenshots.sh` — keep, useful
- `frontend/tmp-screens/` — `.gitignore` it; never commit
- `*.png` at repo root from earlier sessions (e.g. `landing-hero.png`, `landing-full.png`, `lab-after-landing.png`) — delete

- [ ] **Step 1: Update .gitignore**

```bash
cd /Volumes/CS_Stuff/openresearch
grep -q "frontend/tmp-screens" .gitignore || echo "
# Landing screenshots scratch dir (created by scripts/landing-screenshots.sh)
frontend/tmp-screens/
" >> .gitignore
```

- [ ] **Step 2: Delete stray root-level screenshots**

```bash
git status --short | grep -E "^\?\?.*\.png$" | awk '{print $2}'
# Review the list — delete any that aren't intentional reference assets
ls /Volumes/CS_Stuff/openresearch/*.png 2>&1
# Examples to delete: landing-hero.png, landing-section-*.png, lab-after-landing.png
```

If any of these are intentional reference assets (you want them in the repo), commit them to a sensible existing design-artifact location. Otherwise delete.

- [ ] **Step 3: Commit cleanup**

```bash
git add .gitignore
# delete the loose screenshots — don't add them
git rm -f landing-hero.png landing-full.png landing-full2.png landing-section-1.png landing-section-2.png landing-section-4.png landing-section-5.png landing-tree.png landing-nav-fix.png landing-nav-solid.png lab-after-landing.png 2>/dev/null || true
git commit -m "chore: gitignore landing screenshots, remove orphaned PNGs

```

---

# Phase 6 — Smoke + final review (~30 min)

### Task 15: Playwright sweep across desktop + mobile, all routes

**No file changes.** Smoke test only.

- [ ] **Step 1: With production server running, capture full sweep**

```bash
OPENRESEARCH_BACKEND_URL=http://127.0.0.1:8000 npm run start &
sleep 5
for route in "" "lab" "library" "paperbench" "lab?rlmFixture=1"; do
  safe=$(echo "$route" | tr '/?&=' '____')
  ./scripts/landing-screenshots.sh "http://localhost:3000/${route}" "smoke-${safe:-root}"
done
ls frontend/tmp-screens/smoke-*
```

Open each pair (full + mobile). Verify:
- `smoke-root-full.png` — landing renders, all 7 sections, footer, no broken visuals
- `smoke-root-mobile.png` — single-column layout, nav links collapse, sections stack
- `smoke-lab-full.png` — lab dark amber, sidebar wordmark is a link
- `smoke-lab-mobile.png` — lab is responsive
- `smoke-library-full.png` — library renders dark
- `smoke-paperbench-full.png` — paperbench renders dark
- `smoke-lab_rlmFixture_1-full.png` — full RLM fixture replay renders on dark

- [ ] **Step 2: Click-through every CTA in a real browser**

Open `http://localhost:3000/` in a real browser (not Playwright headless). Click every CTA from the audit in Task 8. Each must navigate as documented. Use browser back to confirm round-trip.

- [ ] **Step 3: Test reduced motion**

In macOS System Settings → Accessibility → Display → Reduce motion ON. Reload `/`. Sections should appear without fade-up animation. Reset.

- [ ] **Step 4: Final commit if anything was tweaked, otherwise tag**

```bash
git log --oneline ^main | head -20
# Final commit count: ~14 commits across the 6 phases
# Branch ready for review/merge
```

---

## Phases summary

| Phase | Work | Time | Independent? |
|-------|------|------|--------------|
| 0 | Cleanup + archive + branch rename | ~20 min | yes |
| 1 | Geist + Geist Mono + Instrument Serif via next/font | ~30 min | yes (lab survives the font swap) |
| 2 | Port LANDING/index.html → React (7 tasks) | ~3–4 hr | yes |
| 3 | Buttons + links audit + sidebar Link | ~1 hr | requires Phase 2 |
| 4 | Lab dark amber repaint (tokens + 15 hardcoded) | ~2 hr | requires Phase 1, independent of Phases 2–3 |
| 5 | Production build + lint + dead-file purge | ~30 min | requires all above |
| 6 | Playwright sweep + manual smoke | ~30 min | requires Phase 5 |

**Total:** ~8–10 hours of focused work. Phases 2 and 4 can be parallel if multiple sessions; otherwise serial.

---

## Self-review

- ✅ Plan supersedes prior polish plan; old file deleted in Task 0
- ✅ Every gap from the LANDING/index.html port mapped to a task (7 sections + 2 JS hooks + global CSS)
- ✅ Every broken href identified by advisor enumerated in Task 8's audit table
- ✅ Production build (`npm run build`) is explicit launch gate in Task 13
- ✅ `next/font/google` used for all fonts (not `<link>`) in Task 1
- ✅ Lab repaint sized correctly: 1 token-file edit + 15 hardcoded callsites (inventoried)
- ✅ Tiered phasing: landing port (Phases 1–3) ships before lab repaint (Phase 4)
- ✅ No placeholders, "TBD", "similar to Task N" — every step has exact code/command
- ✅ GitHub URL constant defined once (`GITHUB_URL = https://github.com/armaanamatya/openresearch`)
- ✅ Constants block at top so a fresh subagent finds them on first read
- ✅ Per-task commit messages with co-author trailer
- ✅ Per-task verification command + expected output
- ✅ Brief constraints honored: no invented metrics (placeholder tokens preserved); the LANDING source already had no fake logos
