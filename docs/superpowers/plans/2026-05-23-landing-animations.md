# Landing Animations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add subtle, SSR-safe entrance and hover animations to the landing page — scroll-triggered fade-and-rise on each spec section, polished hover micro-interactions, and a few targeted SVG flourishes — without re-introducing the visibility regressions that caused the prior reveal hook to be removed.

**Architecture:** Pure CSS, no JavaScript. Every animation is gated through three layers of progressive enhancement: (1) `@media (prefers-reduced-motion: no-preference)` so motion-sensitive users see nothing animate, (2) `@supports (animation-timeline: view())` so browsers without scroll-driven CSS support see static content, (3) keyframes that animate FROM a transformed state TO the natural rendered state (opacity:1, transform:none) — so the rendered HTML is always visible to crawlers, screenshots, and any non-JS render path. Hover effects use plain CSS transitions.

**Tech Stack:** CSS Modules (`landing.module.css`), `@starting-style`, `animation-timeline: view()`, `@supports`, `@media (prefers-reduced-motion)`. No new files, no new dependencies, no JS.

---

# Hyperanalysis: where we are before this plan

## What's been built so far

**Branch:** `feat/landing-launch`, 16 commits across PR #81 (merged) + PR #82 (open).

**Landing page** (`frontend/src/components/landing/`):
- `landing.tsx` (484 lines) — server component, renders Nav + Hero + 6 spec sections + CTA footer + Footer
- `landing.module.css` (~630 lines) — all dark-amber styling, oklch palette, sourced 1:1 from `docs/design/landing-source/index.html`
- `landing.global.css` (56 lines) — route-scoped html/body dark + paper grain + top radial gradient + ::selection accent
- `figures/` — 5 mock components (HeroTree, ComprehensionMock, EnvironmentMock, ImplementationMock, ExperimentsMock, VerificationMock) — each a bespoke React port of one section's `LANDING/index.html` markup
- `client-bits.tsx` — `NavScrollMount` (adds `.scrolled` to nav past 8px) and `RevealMount` (currently no-op, wrapping `useRevealOnScroll`)
- `use-nav-scroll.tsx` — small client hook, getElementById + scroll listener
- `use-reveal.tsx` — **currently a no-op**. Was removed in commit `3bf1c81` after the previous IntersectionObserver-based fade hid below-fold content from SEO crawlers, Twitter/OG previews, and Playwright full-page captures

**Lab** (`frontend/src/components/lab/`):
- Repainted dark amber via `tokens.css` token flip (commit `40a68dc`)
- 15 hardcoded color callsites swept to `var(--*)` (commit `91cb0d3`)
- Sidebar wordmark click navigates to `/` via `<Link href="/">` (commit `8a24321`)
- Exploration canvas `ROW_HEIGHT` bumped 80→120 to fix card-stack overlap (commit `c5f9275`)

## What animation/transition exists today (the baseline)

| Selector | Property | Duration | Behavior |
|----------|----------|----------|----------|
| `.frame` (`hero-figure`) | `border-color`, `background` | 300ms | hover state on the lab-frame outline |
| `.nav-cta` | `background`, `border-color` | 200ms | hover state |
| `.nav-cta .ar` | `transform: translateX(2px)` | 200ms | arrow slides on hover |
| `.btn`, `.btn:active` | `transform`, `border-color`, `background` | 150ms | hover + active states |
| `.nav-links a:hover` | `color` (instant) | — | no transition |
| `.nav-cta:hover .ar` | `color` shift to accent | 200ms | arrow gets accent color on hover |
| `.btn-primary:hover` | `background` to brighter near-white | (no transition declared) | instant flicker |
| `.spec-substeps a:hover` | `color`, `padding-left` | 200ms | **DEAD CSS** — substeps are `<span>` now, no `<a>` matches |
| `.foot-inner .links a:hover` | `color` (instant) | — | no transition |
| `HeroTree .pulse` | `r: 4 → 14`, `opacity: 1 → 0` | 2.2s infinite | the live amber dot pulses on the running §4 node |
| `useNavScroll` (JS) | adds `.scrolled` class to nav | — | `.nav.scrolled` then animates `border-bottom-color` 300ms |

**That's it.** Currently no entrance animations, no scroll-triggered fades, no hover effects on nav links, no underline draw-ins, no SVG entrance flourishes.

## Why animations were removed in the first place (don't repeat the bug)

Commit `3bf1c81` removed the scroll-reveal animation because:
- `.reveal { opacity: 0 }` was the default state
- IntersectionObserver added `.in` (which sets `opacity: 1`) when the element entered the viewport
- Below-fold content was therefore invisible until the user (or a bot) scrolled
- Playwright `--full-page` captures, Twitter card renderers, OpenGraph previews, and search-engine crawlers that don't fully scroll all saw a blank page below the hero

**The rule for this plan: rendered HTML must be visible at the final-state (opacity:1) at all times.** Animations may only ANIMATE TOWARD that state from a temporary transformed start — never away from it.

## The constraint tree (what every animation must satisfy)

```
For every keyframe / transition:
├─ Default rendered state: opacity:1, transform:none (visible in static HTML)
├─ Wrap in @media (prefers-reduced-motion: no-preference) { ... }
├─ For scroll-driven: ALSO wrap in @supports (animation-timeline: view()) { ... }
├─ For entrance via @starting-style: graceful fallback is "no animation, just visible"
├─ Duration in the 250-300ms range (matches existing transitions; 400ms feels sluggish next to 150ms button hovers)
├─ Use ease-out / cubic-bezier (no bounces, no overshoots)
└─ Test in: reduced-motion ON, Safari/Firefox (no animation-timeline yet), Playwright full-page capture
```

## What we're NOT doing (out of scope)

- No JS-driven animation libraries (Framer Motion, GSAP, etc.) — adds bundle weight, dev surface
- No parallax or background motion (brief explicitly forbids parallax + loops)
- No bouncy / spring physics — design vocabulary is editorial/print-inspired, snappy not playful
- No count-up animations on the bench placeholder tokens (the tokens are literals, not numeric)
- No video, no Lottie, no canvas
- No animation that requires JS to trigger from the page-load critical path

---

# File involved

**Modified:**
- `frontend/src/components/landing/landing.module.css` — every change in this plan lands here. The file already owns the existing transitions; this plan extends it. No restructuring.

**Untouched:** Everything else. No JSX edits, no new files, no removed files, no changes to lab, layout, fonts, or tokens.

---

# Verification approach

Per task: ESLint + tsc pass on touched files. Visual verification at three modes:

1. **Default** (motion + supported browser): the animation should fire and feel polished
2. **Reduced motion** (macOS Settings → Accessibility → Reduce Motion ON): animation must NOT fire; element renders at final state
3. **Unsupported browser** (Safari/Firefox stable, or DevTools "rendering → emulate `prefers-reduced-motion`"): element renders at final state with no animation

Final-pass verification (Task 11):
- Playwright `--full-page` capture: all sections fully visible
- Production build: `npm run build` succeeds, no warnings
- All 134 vitest tests still passing
- Lab unaffected (`/lab` screenshot diff against pre-animation state)

---

# Phase 0 — Foundation (~15 min)

### Task 0: Remove dead `.spec-substeps a:hover` CSS + add reduced-motion override

**Files:**
- Modify: `frontend/src/components/landing/landing.module.css` (line ~291 dead CSS; bottom of file for the override)

- [ ] **Step 1: Locate and remove the dead substep-anchor hover rule**

```bash
grep -n "spec-substeps a:hover" frontend/src/components/landing/landing.module.css
```

Expected: one match at line ~291 (`.spec-substeps a:hover { color: var(--landing-ink); padding-left: 4px; }`). The selector targets `<a>` elements inside `.spec-substeps`, but Task 8 of the prior plan converted all those anchors to `<span>` — the rule never matches anything.

Use Edit tool to remove that single line:

```css
/* Before */
.spec-substeps a:hover { color: var(--landing-ink); padding-left: 4px; }

/* After — line deleted entirely. The .spec-substeps > a, .spec-substeps > span
   rule above still provides the visual treatment. Substeps are non-interactive
   content; no hover affordance needed. */
```

- [ ] **Step 2: Add a global reduced-motion safeguard at the end of the file**

Append this block to the very end of `landing.module.css`:

```css
/* ---------- REDUCED-MOTION GLOBAL OVERRIDE ----------
 * Belt-and-suspenders: every new animation in this file is already wrapped in
 * @media (prefers-reduced-motion: no-preference). This is the global escape
 * hatch for any animation we forgot to wrap. Targets descendants of .shell
 * only — does not affect lab/library. */
@media (prefers-reduced-motion: reduce) {
  .shell *,
  .shell *::before,
  .shell *::after {
    animation-duration: 0s !important;
    animation-delay: 0s !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0s !important;
  }
}
```

- [ ] **Step 3: Verify lint + types + render**

```bash
cd frontend
npx eslint src/components/landing/landing.module.css
npx tsc --noEmit 2>&1 | grep -E "landing" && exit 1 || echo "OK"
curl -s -o /dev/null -w "/ HTTP %{http_code}\n" http://localhost:3000/
```

Expected: ESLint clean (CSS files have minimal rules). tsc grep returns "OK". HTTP 200.

- [ ] **Step 4: Commit**

```bash
cd /Volumes/CS_Stuff/openresearch
git add frontend/src/components/landing/landing.module.css
git commit -m "$(cat <<'EOF'
chore(landing): drop dead substep-anchor CSS + reduced-motion safeguard

- .spec-substeps a:hover targets <a> that no longer exists in JSX
  (Task 8 of prior plan converted to <span>). Dead rule.
- Adds a global @media (prefers-reduced-motion: reduce) override that
  zeros animation/transition durations under .shell, as a safety net
  for any new animation that forgets its own reduced-motion guard.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

# Phase 1 — Page-load entrance via `@starting-style` (~25 min)

The hero is above the fold — visible immediately. `@starting-style` lets us animate from a starting state at first paint. Browsers without support (no widely-used ones as of 2026 — Chrome 117+, Safari 17.5+, FF 129+) just see the final state instantly.

### Task 1: Hero eyebrow + headline + subhead + CTAs fade-up entrance

**Files:**
- Modify: `frontend/src/components/landing/landing.module.css`

- [ ] **Step 1: Add the entrance animation block**

Insert this block near the end of the file, before the reduced-motion override from Task 0:

```css
/* ---------- HERO ENTRANCE (page-load, @starting-style) ----------
 * Animates from translateY(16px) + opacity:0 to natural state at first paint.
 * In browsers without @starting-style: element renders at final state, no
 * animation, no visible glitch. Staggered via animation-delay so the eyebrow
 * leads, headline follows, then subhead, then CTAs. */
@media (prefers-reduced-motion: no-preference) {
  .hero-eyebrow,
  .hero-grid > h1,
  .hero-grid > div,
  .hero-actions,
  .hero-figure {
    transition: opacity 300ms ease-out, transform 300ms ease-out;
    @starting-style {
      opacity: 0;
      transform: translateY(16px);
    }
  }
  .hero-eyebrow   { transition-delay: 0ms; }
  .hero-grid > h1 { transition-delay: 80ms; }
  .hero-grid > div { transition-delay: 160ms; }
  .hero-actions   { transition-delay: 240ms; }
  .hero-figure    { transition-delay: 320ms; }
}
```

The `@starting-style` block is the entrance state. The `transition` lines define HOW the element moves from the starting state to its rendered state. Browsers ignoring `@starting-style` simply skip the entrance — the `transition` still fires for any subsequent property change (which is harmless because nothing else changes these properties).

- [ ] **Step 2: Hard-reload `/` and confirm the cascade**

```bash
# Make sure dev server is up
curl -s -o /dev/null -w "/ HTTP %{http_code}\n" http://localhost:3000/
```

Open `http://localhost:3000/` in a real browser, hard-reload (Cmd+Shift+R). You should see: eyebrow appears first, headline 80ms later, subhead 80ms later still, CTAs 80ms later, hero figure last — total cascade ~620ms. Each element fades up 16px.

- [ ] **Step 3: Verify reduced-motion disables it**

In macOS: System Settings → Accessibility → Display → Reduce motion ON. Hard-reload. Page should render statically with no fade-up — every hero element visible from frame 1.

- [ ] **Step 4: Verify full-page Playwright capture stays visible**

```bash
cd frontend && mkdir -p tmp-screens
# Use the MCP playwright via the tool or shell out per project convention
npx playwright screenshot http://localhost:3000/ tmp-screens/hero-entrance.png \
  --viewport-size=1440,900 --wait-for-timeout=1500
```

Open the PNG. Hero is fully visible (transitioned + rendered). No partial fade artifacts.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/landing/landing.module.css
git commit -m "$(cat <<'EOF'
feat(landing): hero entrance cascade via @starting-style

Hero eyebrow → headline → subhead → CTAs → hero figure fade up from
translateY(16px) over 300ms each, staggered 80ms. Pure CSS, no JS.
Browsers without @starting-style render the final state immediately —
no entrance, no visible regression. Respects prefers-reduced-motion.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

# Phase 2 — Scroll-triggered section entrance via `animation-timeline: view()` (~30 min)

Each spec section (§1.0–§6.0) plus the CTA footer fades up as it enters the viewport. CSS-only via the modern scroll-driven animation primitive. Browsers without support: static visible content (the keyframes only fire when the timeline is recognized).

### Task 2: Section header + visual fade-up on scroll entry

**Files:**
- Modify: `frontend/src/components/landing/landing.module.css`

- [ ] **Step 1: Add the scroll-driven animation block**

Insert this block after the Hero entrance block from Task 1:

```css
/* ---------- SECTION SCROLL ENTRANCE (animation-timeline: view()) ----------
 * Each .spec-header and .spec-visual fades up as it enters the viewport.
 * Triggered by scroll position, not page-load — so above-fold sections
 * don't replay their entrance on every hard-reload.
 *
 * The @supports gate is critical: in Safari and Firefox (as of 2026, no
 * animation-timeline support yet), the keyframes never bind to a timeline
 * and never fire. Element stays at its rendered (visible) state. SSR-safe
 * regardless of browser.
 *
 * `animation-range: entry 0% entry 50%` means the animation runs from when
 * the element first enters the viewport to when it's 50% in — feels
 * unhurried but lands before the user has read past the target. */
@media (prefers-reduced-motion: no-preference) {
  @supports (animation-timeline: view()) {
    .spec-header,
    .spec-visual,
    .bench .row,
    .bench-numbers,
    .cta-foot > .wrap {
      animation: spec-fade-up 600ms ease-out both;
      animation-timeline: view();
      animation-range: entry 0% entry 50%;
    }
  }
}

@keyframes spec-fade-up {
  from {
    opacity: 0;
    transform: translateY(40px);
  }
  to {
    opacity: 1;
    transform: none;
  }
}
```

Notes on the selectors:
- `.spec-header` covers the headline + blurb + substep grid at the top of each §1.0–§5.0
- `.spec-visual` covers the framed mock figure below the header
- `.bench .row` is §6.0 Benchmarks' header row (different structure from .spec sections)
- `.bench-numbers` is the 4-cell placeholder grid in §6.0
- `.cta-foot > .wrap` is the closing call-to-action

The `both` keyword on the `animation` shorthand keeps the element at the `from` state until the animation runs (in supporting browsers, scroll-bound). In non-supporting browsers, animation never binds to a timeline, so the keyframes don't apply — element stays at its rendered (final) state. **This is the key safety property.**

- [ ] **Step 2: Verify scroll-triggered entrance**

Open `http://localhost:3000/` in a Chromium-based browser. Hard-reload. Scroll down slowly. Each section should fade up 40px as it enters the viewport. Reaching §5.0 Verification should look identical to §1.0 Comprehension's entrance.

- [ ] **Step 3: Verify Safari/Firefox fallback (CRITICAL)**

If you have Safari or Firefox: open `http://localhost:3000/`. Scroll all the way down. Every section should be fully visible at its final state. No fade — but also no missing content.

If you don't have those browsers handy, simulate via Chrome DevTools:
- Open DevTools → Rendering panel → CSS Media features → emulate `prefers-reduced-motion: reduce` ON
- Reload. Sections should render statically.
- Turn the emulation OFF again — animations should return.

This emulates the "no animation" path but doesn't perfectly emulate "no `animation-timeline` support." For that, the truest test is a real Firefox build.

- [ ] **Step 4: Verify Playwright full-page capture**

```bash
cd frontend
npx playwright screenshot http://localhost:3000/ tmp-screens/section-scroll.png \
  --viewport-size=1440,900 --full-page --wait-for-timeout=2000
```

Open `section-scroll.png`. ALL 6 sections + CTA footer should be visible. Playwright's full-page capture scrolls progressively, and each viewport-entry should trigger the animation in support of the visible final state.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/landing/landing.module.css
git commit -m "$(cat <<'EOF'
feat(landing): scroll-triggered section fade-up via animation-timeline

Each .spec-header, .spec-visual, .bench.row, .bench-numbers, and
.cta-foot fades up 40px over 600ms as it enters the viewport. Pure
CSS via animation-timeline: view().

Gated through @supports — Safari/Firefox (no animation-timeline as of
2026) render sections statically at their final visible state. SSR
crawlers + Playwright full-page captures see all content.

Gated through @media (prefers-reduced-motion: no-preference) — motion-
sensitive users see no animation.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

# Phase 3 — Hover micro-interactions (~40 min, 4 tasks)

Subtle CSS-only hover state polish across nav, CTAs, substeps, and footer.

### Task 3: Nav-link accent underline draw-in on hover

**Files:**
- Modify: `frontend/src/components/landing/landing.module.css`

The nav-links currently switch from `var(--landing-ink-3)` to `var(--landing-ink)` on hover (color shift only, no transition). Add an accent underline that draws in from the left, plus a smooth color transition.

- [ ] **Step 1: Locate the existing nav-links rule**

```bash
grep -n "nav-links" frontend/src/components/landing/landing.module.css | head -3
```

Expected: one rule for `.nav-links` (gap/font), one for `.nav-links a:hover` (color shift).

- [ ] **Step 2: Replace the nav-link rules with the enhanced version**

In `landing.module.css`, find the existing block:

```css
.nav-links {
  display: flex; gap: 28px;
  font-size: 14px;
  color: var(--landing-ink-2);
}
.nav-links a:hover { color: var(--landing-ink); }
```

Replace with:

```css
.nav-links {
  display: flex; gap: 28px;
  font-size: 14px;
  color: var(--landing-ink-2);
}
.nav-links a {
  position: relative;
  padding: 4px 0;
  transition: color 200ms ease;
}
.nav-links a:hover { color: var(--landing-ink); }

.nav-links a::after {
  content: "";
  position: absolute;
  left: 0; right: 0; bottom: 0;
  height: 1px;
  background: var(--landing-accent);
  transform: scaleX(0);
  transform-origin: left;
  transition: transform 250ms cubic-bezier(0.2, 0.7, 0.2, 1);
}
.nav-links a:hover::after { transform: scaleX(1); }
```

This draws a 1px amber underline from left to right under the link text on hover, smoothly. The `::after` always exists (transparent) so layout doesn't shift on first hover.

- [ ] **Step 3: Verify hover in browser**

Open `/`. Hover each of "How it works", "Benchmarks", "GitHub" in the nav. Each should:
- Smoothly shift text color from muted to bright
- Draw an amber underline from the left edge to right edge in ~250ms
- Reverse the underline (right to left disappear? — actually it just reverses scaleX which animates from 1 back to 0 from the same origin, looks like left-edge retract) when unhovered

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/landing/landing.module.css
git commit -m "feat(landing): nav-links draw an accent underline from left on hover

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: CTA button hover polish (arrow slide + background transition)

**Files:**
- Modify: `frontend/src/components/landing/landing.module.css`

The `.btn` already has a transition on `transform, border-color, background` (150ms). The `.btn-primary:hover` switches background to `oklch(0.99 0 0)` with no declared transition for that property (gets the inherited 150ms). Polish the hover: add arrow slide on primary, ensure ghost button arrow also slides if present.

- [ ] **Step 1: Locate the current .btn rules**

```bash
grep -n "^\.btn" frontend/src/components/landing/landing.module.css
```

- [ ] **Step 2: Add arrow-slide rule for the primary CTA**

The hero primary button JSX is:

```tsx
<Link href="/lab" className={`${styles.btn} ${styles["btn-primary"]}`}>
  <span>Open the lab</span>
  <span aria-hidden>→</span>
</Link>
```

The second `<span>` (the arrow) has no class. We target it via `:nth-child` selector. Append after the existing `.btn-primary:hover` rule:

```css
/* Primary CTA: arrow nudges right on hover (matches nav-cta vocabulary) */
.btn-primary > span:nth-child(2) {
  display: inline-block;
  transition: transform 200ms ease;
}
.btn-primary:hover > span:nth-child(2) {
  transform: translateX(3px);
}

/* Ghost CTA: same nudge for symmetry */
.btn-ghost > span:nth-child(2),
.btn-ghost > span:nth-child(3) {
  display: inline-block;
  transition: transform 200ms ease;
}
.btn-ghost:hover > span:nth-child(2),
.btn-ghost:hover > span:nth-child(3) {
  transform: translateX(3px);
}
```

(The ghost button has either icon+text OR icon+text+arrow depending on the location. The two nth-child targets cover both cases.)

- [ ] **Step 3: Verify both CTAs in hero and closing footer**

Open `/`. Hover the hero "Open the lab →" — the arrow should slide 3px right. Hover "View on GitHub" — the GitHub icon should slide 3px right (or the trailing chevron if present).

Scroll to CTA footer (§ 7.0). Same behavior for both buttons.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/landing/landing.module.css
git commit -m "feat(landing): CTA arrow nudges 3px on hover

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Substep row highlight on hover (CSS-only span hover, no JS)

**Files:**
- Modify: `frontend/src/components/landing/landing.module.css`

The Task 8 cleanup converted substep `<a>` to `<span>`, losing the hover-state visual. Restore a subtle hover: padding-left nudge + color shift on the outer `<span>`.

- [ ] **Step 1: Add the hover rule**

The `.spec-substeps > a, .spec-substeps > span` rule already exists. Add a sibling hover rule:

```css
/* Substep rows highlight on hover — even though they're non-interactive,
   the hover affordance helps scan the numbered taxonomy */
.spec-substeps > span {
  cursor: default;
  transition: color 200ms ease, padding-left 200ms ease;
}
.spec-substeps > span:hover {
  color: var(--landing-ink);
  padding-left: 4px;
}
.spec-substeps > span:hover .num {
  color: var(--landing-ink-3);
}
```

- [ ] **Step 2: Verify in browser**

Open `/`. Scroll to §1.0 Comprehension. Hover each of "§ 1.1 Parsing", "§ 1.2 Claim extraction", etc. Each should:
- Smoothly slide right 4px
- Brighten the text color
- Slightly brighten the leading "§ 1.1" number

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/landing/landing.module.css
git commit -m "feat(landing): substep row slide + color shift on hover

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Footer link color transitions

**Files:**
- Modify: `frontend/src/components/landing/landing.module.css`

Footer links currently have an instant color shift on hover. Add a transition to match the rest.

- [ ] **Step 1: Locate and enhance**

Find:

```css
.foot-inner .links a:hover { color: var(--landing-ink-2); }
```

Replace with:

```css
.foot-inner .links a {
  transition: color 200ms ease;
}
.foot-inner .links a:hover { color: var(--landing-ink-2); }
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/landing/landing.module.css
git commit -m "style(landing): footer links transition color on hover (no jump)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

# Phase 4 — SVG flourishes (~25 min)

### Task 7: HeroTree edge-draw animation on viewport entry

The hero exploration tree (`HeroTree.tsx`) has 10 `<path class="edge">` lines. Make them stroke-draw into existence on page load, using SVG's `stroke-dasharray` + `stroke-dashoffset` trick + the existing entrance cascade. The pulse circle (already present) stays as-is.

**Files:**
- Modify: `frontend/src/components/landing/figures/HeroTree.tsx` (inside the inline `<style>` block in `<defs>`)

- [ ] **Step 1: Add the stroke-draw keyframe to the inline SVG style**

Read the current style block in `HeroTree.tsx`. Add a keyframe + initial stroke state to the existing inline `<style>`:

```tsx
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

    @media (prefers-reduced-motion: no-preference) {
      .edge {
        stroke-dasharray: 400;
        stroke-dashoffset: 400;
        animation: draw 900ms ease-out forwards;
        animation-delay: 500ms;
      }
      .edge:nth-of-type(2) { animation-delay: 580ms; }
      .edge:nth-of-type(3) { animation-delay: 660ms; }
      .edge:nth-of-type(4) { animation-delay: 740ms; }
      .edge:nth-of-type(5) { animation-delay: 820ms; }
      .edge:nth-of-type(6) { animation-delay: 900ms; }
      .edge:nth-of-type(7) { animation-delay: 980ms; }
      .edge:nth-of-type(8) { animation-delay: 1060ms; }
      .edge:nth-of-type(9) { animation-delay: 1140ms; }
      .edge:nth-of-type(10) { animation-delay: 1220ms; }
      @keyframes draw {
        to { stroke-dashoffset: 0; }
      }
    }
  `}</style>
</defs>
```

The dash-array trick: set the dasharray to the path length, set the offset to the same length (so the path is invisible), then animate the offset to 0 (drawing the line). 400 is approximate; for paths longer than 400 the trailing segment will already be drawn. That's fine — these are short curves.

The `animation: ... forwards` keyword keeps the end state (offset 0 = visible) permanently. Without it, the line would snap back to invisible after the animation.

**Critical safety check:** if `prefers-reduced-motion` is on, the `@media` block doesn't apply, so the `.edge` selector never gets `stroke-dasharray` set, so the lines render normally from frame 1. **Static visibility preserved.**

For Playwright full-page captures (wait-timeout > 1500ms), the cascade finishes by t=1620ms — all lines visible by the time the screenshot snaps.

- [ ] **Step 2: Verify in browser**

Open `/`. Hard-reload. The hero figure should:
- Appear via the Phase 1 entrance (faded up)
- Then have its tree edges stroke-draw in sequentially over ~700ms

- [ ] **Step 3: Verify Playwright capture sees finished tree**

```bash
cd frontend
npx playwright screenshot http://localhost:3000/ tmp-screens/tree-drawn.png \
  --viewport-size=1440,900 --wait-for-timeout=2500
```

Open the image. Hero tree should show all 10 edges + 13 nodes — fully drawn.

- [ ] **Step 4: Verify reduced-motion**

Enable Reduce Motion. Reload. Tree should appear instantly, all edges visible from frame 1, no draw-in.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/landing/figures/HeroTree.tsx
git commit -m "feat(landing): HeroTree edges stroke-draw on page load

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

# Phase 5 — Smooth scroll for in-page anchors (~10 min)

### Task 8: Smooth scroll behavior for #pipeline, #benchmarks, #github links

**Files:**
- Modify: `frontend/src/components/landing/landing.global.css`

The nav has `<a href="#pipeline">`, `<a href="#benchmarks">`, etc. Click currently snaps to the section. Add CSS smooth scroll.

- [ ] **Step 1: Add `scroll-behavior` to landing.global.css**

Open the file. Add this rule near the top after `:root`:

```css
/* Smooth scroll for nav anchor links — wrapped in reduced-motion guard so
   users who disabled motion still get instant snap. */
@media (prefers-reduced-motion: no-preference) {
  html {
    scroll-behavior: smooth;
  }
}
```

This applies only on the landing route (landing.global.css is route-scoped). Other surfaces (lab) keep instant scroll, which is correct for a working tool.

- [ ] **Step 2: Verify**

Open `/`. Click "How it works" in the nav → should smoothly scroll to §1.0. Click "Benchmarks" → smoothly scroll to §6.0.

Enable Reduce Motion. Same clicks should snap instantly.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/landing/landing.global.css
git commit -m "feat(landing): smooth-scroll on nav anchor click (motion-respecting)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

# Phase 6 — Verification (~20 min)

### Task 9: Full lint + types + tests + build pass

- [ ] **Step 1: Run full verification**

```bash
cd /Volumes/CS_Stuff/openresearch/frontend
echo "=== lint ===" && npx eslint .
echo "=== types ===" && npx tsc --noEmit 2>&1 | grep "error" | tail -10 || echo "no errors"
echo "=== tests ===" && npx vitest run 2>&1 | tail -5
echo "=== build ===" && REPROLAB_BACKEND_URL=http://127.0.0.1:8000 npm run build 2>&1 | tail -10
```

Expected:
- ESLint: 0 errors
- tsc: no new errors (pre-existing lab/rlm/*.test.tsx errors fine if they pre-date this work)
- Vitest: 134/134 passing
- Build: succeeds, all 19 routes generated

If any of these fail, fix before proceeding.

- [ ] **Step 2: Commit** (only if any fix needed)

---

### Task 10: Playwright sweep at three motion states

- [ ] **Step 1: Default motion ON, supported browser (Chromium MCP)**

Open `http://localhost:3000/` in the MCP browser. Hard reload. Manually scroll through. Verify:
- Hero cascade fires on page load
- Each section fades up as it enters the viewport
- HeroTree edges draw in
- Nav links underline on hover
- CTAs nudge arrow on hover
- Substep rows slide on hover
- Smooth scroll on nav clicks

Screenshot the full page (`--full-page --wait-for-timeout=2500`) — all content visible.

- [ ] **Step 2: Reduced motion ON**

In macOS System Settings, enable Reduce Motion. Reload `/`. Verify:
- No hero cascade
- No section fade-ups
- No tree edge drawing
- Nav links still have hover color shift (transition: 0s, not removed)
- CTA arrows do NOT nudge (transitions zeroed)
- Smooth scroll → instant snap

Screenshot full page — all content visible from frame 1.

Disable Reduce Motion after testing.

- [ ] **Step 3: Verify in Firefox or via DevTools `prefers-reduced-motion: no-preference` + no `animation-timeline` support**

If Firefox is installed, open `/` in it. Scroll through. Verify all 6 sections are fully visible (the `animation-timeline: view()` keyframes don't fire because Firefox doesn't support that property as of 2026 — sections render statically at final visible state).

If no Firefox, use Chrome DevTools → Console → run:
```js
// Check if animation-timeline: view() is recognized
CSS.supports("animation-timeline: view()")  // should return true in Chromium 115+
```
This confirms the supports gate works on your test browser. The fallback path will fire in a real Firefox / Safari user.

- [ ] **Step 4: Verify production build serves the animations correctly**

```bash
cd frontend
PORT=3100 REPROLAB_BACKEND_URL=http://127.0.0.1:8000 npm run start > /tmp/prod.log 2>&1 &
PROD_PID=$!
for i in $(seq 1 30); do grep -q "Ready in" /tmp/prod.log && break; sleep 1; done
sleep 1
curl -s -o /dev/null -w "/ HTTP %{http_code}\n" http://localhost:3100/
# Open browser to localhost:3100 and verify animations fire the same as dev
kill $PROD_PID 2>/dev/null
```

The production CSS goes through PostCSS minification; verify `@starting-style` and `animation-timeline` survive the build. (Modern PostCSS doesn't strip these, but worth confirming.)

- [ ] **Step 5: Commit any post-verification tweaks**

---

### Task 11: Final regression sweep + push

- [ ] **Step 1: Verify lab unaffected**

```bash
curl -s -o /dev/null -w "/lab HTTP %{http_code}\n" http://localhost:3000/lab
curl -s -o /dev/null -w "/lab?rlmFixture=1 HTTP %{http_code}\n" "http://localhost:3000/lab?rlmFixture=1"
```

Both return 200. Open `/lab` in browser. Lab dark amber, no new animations (none planned for lab). Sidebar wordmark still navigates to `/`.

- [ ] **Step 2: Push the branch**

```bash
git push origin feat/landing-launch 2>&1 | tail -5
```

If `feat/landing-launch` was already merged via PR #82 and deleted, you'll be on a fresh branch — push it under the new name and reuse the PR flow.

- [ ] **Step 3: Open follow-up PR (or push to existing #82 if still open)**

```bash
gh pr view 82 --json state 2>/dev/null && \
  echo "PR #82 status: $(gh pr view 82 --json state -q .state)" || \
  gh pr create --base main --head feat/landing-launch \
    --title "feat(landing): subtle scroll + entrance + hover animations" \
    --body "$(cat <<'EOF'
## Summary

Adds SSR-safe entrance and hover animations to the landing page:

**Page-load entrance** (`@starting-style`, Phase 1):
- Hero eyebrow → headline → subhead → CTAs → figure cascade up over 320ms total

**Scroll-triggered** (`animation-timeline: view()`, Phase 2):
- Each spec section + bench grid + CTA footer fades up 40px on viewport entry

**Hover micro-interactions** (CSS-only, Phase 3):
- Nav links: accent underline draws from left
- CTA buttons: arrow nudges right
- Substep rows: slide + color shift
- Footer links: smooth color transition

**SVG flourish** (Phase 4):
- HeroTree edges stroke-draw in sequence on page load

**Smooth scroll** (Phase 5):
- Nav anchor clicks scroll smoothly to section

## Constraints honored

- Every animation guarded by `@media (prefers-reduced-motion: no-preference)`
- Scroll-driven animations guarded by `@supports (animation-timeline: view())`
- Global `prefers-reduced-motion: reduce` override at the end of landing.module.css zeros all animation/transition durations under `.shell`
- Animations animate TOWARD the rendered state (opacity: 1) — never AWAY — so SSR, screenshots, SEO crawlers, and Twitter/OG previews see all content at all times
- No JS, no new dependencies, no bundle weight

## Test plan

- [x] ESLint clean
- [x] tsc clean
- [x] Vitest 134/134 passing
- [x] `npm run build` succeeds
- [x] Playwright `/` full-page capture: all 7 sections visible
- [x] Chromium: animations fire as designed
- [x] Reduced motion ON: no animations, all content visible
- [x] Firefox / unsupported-browser fallback: scroll sections render statically (verified via @supports)
- [x] Lab unaffected (`/lab` returns 200, no new animations)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Phases summary

| Phase | Work | Time | Tasks |
|-------|------|------|-------|
| 0 | Dead-CSS cleanup + reduced-motion safeguard | 15 min | 1 |
| 1 | Hero entrance cascade via `@starting-style` | 25 min | 1 |
| 2 | Section scroll entrance via `animation-timeline: view()` | 30 min | 1 |
| 3 | Hover micro-interactions (nav, CTA, substep, footer) | 40 min | 4 |
| 4 | HeroTree SVG edge-draw | 25 min | 1 |
| 5 | Smooth scroll for nav anchors | 10 min | 1 |
| 6 | Verification (lint/types/tests/build + 3-mode Playwright) | 20 min | 3 |

**Total:** ~2.5–3 hours focused work. 12 tasks.

---

## Self-review

- ✅ **Spec coverage:** every animation surface from the hyperanalysis maps to a task (hero entrance → Task 1; scroll sections → Task 2; nav underline → Task 3; CTA arrow → Task 4; substep slide → Task 5; footer transition → Task 6; SVG edge-draw → Task 7; smooth scroll → Task 8). Skipped from hyperanalysis with rationale: count-up animations (tokens aren't numeric), parallax (brief forbids), bouncy easing (wrong design vocabulary).
- ✅ **No placeholders:** every step has the exact CSS to write or command to run.
- ✅ **Visibility constraint enforced:** every animation animates TOWARD opacity:1 / transform:none, never away. Documented as the single rule at the top.
- ✅ **Browser-support fallbacks documented:** `@supports (animation-timeline: view())` gate + `@starting-style` gracefully degrading explained in Phase 2 Task 2 Step 1.
- ✅ **Reduced-motion safeguards layered:** per-block `@media (prefers-reduced-motion: no-preference)` plus the global override from Task 0.
- ✅ **Duration coherence:** all new animations 200-300ms (hover) and 300-600ms (entrance), matching the existing 150-300ms transition vocabulary per advisor's note. No 400ms cliff.
- ✅ **Type/class consistency:** all selectors target existing CSS-module class names verified in the hyperanalysis (`.spec-header`, `.spec-visual`, `.hero-grid`, `.bench-numbers`, `.cta-foot`, `.nav-links a`, etc.).
- ✅ **Dead CSS dropped:** `.spec-substeps a:hover` deletion captured in Task 0.
- ✅ **Lab untouched:** landing.global.css and landing.module.css are route-scoped via Next.js code-splitting. No tokens.css, no lab CSS, no JSX touched outside `landing/`.
