# Mercury-Inspired Landing Page Design

## Overview

This spec defines a new public landing page for OpenResearch that uses the "Mercury" style reference as its visual system. The page should feel like a twilight command center: cinematic first, then operational. Its purpose is to introduce OpenResearch as a system to reproduce, verify, and improve cutting-edge research, then move visitors toward a single primary call to action: `Start building`.

The landing page is not the signed-in dashboard. It is a public-facing entrypoint that borrows the emotional tone of the dashboard without inheriting live application complexity.

## Product Goal

The page should make a mixed technical audience of researchers, founders, and independent builders believe three things quickly:

1. OpenResearch is serious infrastructure for frontier technical work.
2. The product is designed around traceability and verification, not vague AI automation.
3. Visitors can move from paper to working system through a visible, agent-driven workflow.

## Core Positioning

### Primary Promise

`Reproduce, verify, and improve cutting-edge research.`

### Supporting Message

`Turn papers into executable systems with coordinated agents, traceable experiments, and evidence-backed iteration.`

### Primary CTA

`Start building`

### Secondary CTA

`View workflow`

## Experience Strategy

The experience should follow the same emotional arc as Mercury:

1. Start with atmosphere and ambition.
2. Transition into a severe, text-led operational interface.
3. End with a grounded, product-trusting closing action.

The page should not read like a generic AI startup homepage. It should feel closer to a research operations deck or mission-control environment: quiet, precise, and authoritative.

## Visual System

### Mood

The design should feel like a mountain-top command center at dusk: expansive, calm, and focused. The page should open with a sense of scale and solitude, then resolve into a functional surface that communicates control and technical depth.

### Color

Use the Mercury palette directly:

- `--color-deep-space: #171721`
- `--color-midnight-slate: #1e1e2a`
- `--color-graphite: #272735`
- `--color-lead: #70707d`
- `--color-starlight: #ededf3`
- `--color-silver: #c3c3cc`
- `--color-mercury-blue: #5266eb`
- `--color-ghost-blue: #cdddff`

Rules:

- Reserve `Mercury Blue` for primary actions and active states only.
- Use `Starlight` for primary text and `Silver` for secondary text.
- Use thin `Lead` borders and dividers instead of shadows.
- Do not introduce additional saturated colors.

### Typography

Typography should carry most of the page's authority:

- Headings use `arcadiaDisplay` or the configured fallback stack.
- Body, labels, navigation, and CTA text use `arcadia` or the configured fallback stack.
- Favor light-to-medium weights and wide spacing over bold type.

Tone rules:

- restrained
- technical
- unsentimental
- high contrast

### Motion

Motion should be sparse and meaningful:

- hero content fades in
- workflow rows stagger slightly into view
- hover states brighten subtly or reveal a thin blue status accent

Avoid:

- bounce
- heavy parallax
- decorative animation loops

## Page Architecture

The landing page should be composed of six sections in order.

### 1. Hero

The hero occupies the full viewport height.

Contents:

- minimal top navigation
- centered headline
- short supporting paragraph
- primary CTA `Start building`
- secondary CTA `View workflow`

Visual behavior:

- full-bleed atmospheric background image
- strong dark overlay and vignette
- centered copy block with wide breathing room

Suggested hero headline:

`Reproduce, verify, and improve cutting-edge research.`

Suggested subhead:

`Turn papers into executable systems with coordinated agents, traceable experiments, and evidence-backed iteration.`

### 2. Transition Band

A narrow bridge section directly under the hero that shifts the page from aspiration into operation.

Suggested line:

`From paper abstractions to runnable systems, every decision stays visible.`

Support this with three compact proof points:

- `Paper -> Plan -> Runtime -> Verification`
- `Multi-agent delegation with traceable outputs`
- `Artifacts, citations, and experiment history included`

### 3. Command Workflow

This section explains the full product loop as a clear, staged sequence. It should use Mercury-style interactive list rows with strong typographic hierarchy and bottom-border separation.

Workflow stages:

1. `Ingest`
   Parse papers, repositories, datasets, and references into a shared research workspace.
2. `Plan`
   Route work to specialized agents with explicit scope, dependencies, and execution intent.
3. `Execute`
   Run experiments, produce artifacts, and coordinate tool-driven implementation.
4. `Verify`
   Inspect citations, outputs, logs, and provenance before accepting results.
5. `Improve`
   Branch from baselines, compare alternatives, and push beyond reproduction.

Interaction behavior:

- each row is full-width
- bottom border only
- large display-style title
- subtle hover shift
- optional small active indicator in `Mercury Blue`

### 4. Command Center Preview

This section shows the product surface without turning the landing page into the app itself.

It should be a curated static mock, not live application state.

Recommended composition:

- left rail: papers or active projects
- center: agent topology and task flow
- right rail: citations, verification events, and artifact summaries
- lower strip or inset panel: runtime logs and experiment statuses

Purpose:

- make the product feel real
- communicate system depth
- visually connect the landing page to the signed-in application

This preview should feel cleaner than a real dashboard while remaining plausibly operational.

### 5. Capability Rails

This section explains the product in four compact, text-first panels.

Panels:

- `Reproduce`
  Convert papers into executable plans, environments, and task trees.
- `Verify`
  Inspect source grounding, outputs, logs, and citations before trusting results.
- `Improve`
  Fork experiments, compare strategies, and iterate beyond baseline replication.
- `Coordinate`
  Use multiple specialized agents without losing scope, context, or provenance.

Presentation rules:

- thin borders
- dark surfaces
- no icon-heavy treatment
- concise, technical copy

### 6. Closing CTA

The final section should be calmer than the hero and more operational in tone.

Suggested headline:

`Build from frontier research without losing the thread.`

Suggested supporting copy:

`Go from paper to reproducible system with agents that plan, execute, and verify in public view.`

Primary action:

- `Start building`

## Navigation

The top navigation should be minimal and slightly translucent over the hero.

Recommended items:

- `Workflow`
- `Command center`
- `Verification`
- `Docs`

The far-right CTA should use the Mercury secondary pill style or a condensed primary state depending on visual balance.

## Content Voice

The copy should feel:

- technical
- measured
- concrete
- quietly confident

Use vocabulary like:

- reproduce
- verify
- artifacts
- citations
- provenance
- runtime
- delegation

Avoid:

- "10x"
- "revolutionary"
- "supercharge"
- generic "AI-powered" filler

## Implementation Shape

This design should fit the current Next.js frontend while remaining maintainable.

Recommended file structure:

- `frontend/src/app/page.tsx`
- `frontend/src/components/landing/site-header.tsx`
- `frontend/src/components/landing/hero-section.tsx`
- `frontend/src/components/landing/transition-band.tsx`
- `frontend/src/components/landing/workflow-section.tsx`
- `frontend/src/components/landing/command-preview.tsx`
- `frontend/src/components/landing/capability-grid.tsx`
- `frontend/src/components/landing/final-cta.tsx`

## Styling Approach

Use global tokens plus a landing-specific stylesheet.

Recommended split:

- `frontend/src/app/globals.css`
  - Mercury tokens
  - shared typography variables
  - spacing and surface tokens
- dedicated landing stylesheet
  - section layout
  - component-specific visual treatment
  - responsive behavior

This keeps the design system reusable while preventing the landing page from overloading the global app styles.

## Data Strategy For Preview Content

The command preview should be driven by static fixture data, not live dashboard state.

Reasons:

- deterministic rendering
- easier visual polish
- simpler tests
- no accidental coupling to runtime logic

If pseudo-live behavior is desired later, it should be layered onto the fixture-driven design without changing the page architecture.

## Responsive Design

### Desktop

- full cinematic hero
- wide preview composition
- strong horizontal whitespace

### Tablet

- stacked workflow rhythm
- slightly compressed preview layout
- preserved section spacing

### Mobile

- dramatic but simpler hero
- stacked CTA layout if needed
- simplified command preview as vertical panels
- preserved mood rather than a mechanical desktop collapse

## Testing Scope

Implementation should focus on lightweight presentational confidence:

- landing page renders
- hero CTA is present
- all five workflow steps render
- command preview fixture labels render
- key sections appear in order

No runtime event wiring or live state testing should be required for this landing page.

## Success Criteria

The landing page is successful if:

1. it clearly presents OpenResearch as a technical system for reproduction, verification, and improvement
2. it feels visually distinct from generic AI startup sites
3. it uses the Mercury reference faithfully without copying it mechanically
4. it makes the signed-in product feel like the natural next step
5. it drives visitors toward `Start building` with a single, focused action hierarchy
