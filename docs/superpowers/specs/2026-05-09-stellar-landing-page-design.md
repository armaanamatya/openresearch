# Stellar Landing Page Design

## Overview

This spec replaces the current homepage with a new public landing page inspired by the provided "Stellar.ai" reference. The page is a bright, high-clarity hero-led experience built in React with Tailwind CSS and Lucide React icons. It should feel fast, modern, and product-confident, with a strong first-screen interaction centered on a tabbed video stage and contextual overlays.

This design fully replaces the current homepage and supersedes the previously discussed Mercury landing-page direction for `main`.

## Product Goal

The page should communicate that OpenResearch helps people work faster with powerful AI-driven workflows, while feeling polished and conversion-ready from the first screen.

The homepage should:

1. present a modern, premium AI product aesthetic
2. drive the primary call to action `Begin Free Trial`
3. demonstrate product motion through tab-driven overlays on a looping video
4. keep the experience lightweight, focused, and centered around the hero

## Experience Strategy

The page is intentionally concentrated around one dominant hero system rather than a long narrative site. The visitor should be able to understand the promise, interact with the tabs, see different product moments, and immediately know what action to take next.

The experience arc is:

1. clean trust-building top navigation
2. social proof via the review badge
3. bold value proposition via the two-line headline
4. CTA-driven conversion moment
5. hands-on product flavor via the tabbed video stage
6. credibility reinforcement via the company logo rail

## Visual System

### Background And Layout

- page background: `bg-white`
- page content centered with `max-w-7xl mx-auto`
- generous whitespace throughout
- layout should feel balanced and minimal, not dense

### Typography

Use `Inter` imported from Google Fonts with weights:

- 400
- 500
- 600
- 700

Set the body font to:

```css
font-family: "Inter", sans-serif;
```

Typography behavior:

- headline uses large, airy sizes with normal weight
- navigation and labels use smaller, restrained sizes
- supporting copy uses gray text with strong readability
- the second headline line uses a black-to-gray gradient fill

### Color

This page uses a mostly neutral palette:

- white page background
- black for primary text and CTAs
- gray for supporting text and surfaces
- no custom accent theme beyond the overlay cards' internal status colors

### Motion

Add the following custom animations in `index.css` or the app-global CSS entrypoint:

```css
@keyframes fadeInUp {
  from {
    opacity: 0;
    transform: translateY(30px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.animate-fade-in-up {
  animation: fadeInUp 0.6s ease-out forwards;
}

@keyframes fadeInOverlay {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}

.animate-fade-in-overlay {
  animation: fadeInOverlay 0.4s ease-out forwards;
}

@keyframes fadeInDialog {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}

.animate-slide-up-overlay {
  animation: fadeInDialog 0.5s ease-out forwards;
  transform: translate(-50%, -50%);
}
```

Every major section should:

- start with inline `opacity: 0`
- use `.animate-fade-in-up`
- receive a staggered inline `animationDelay`
- begin at `0.1s` and increment by `0.1s`

Overlay cards should use:

- outer layer: `.animate-fade-in-overlay`
- inner card: `.animate-slide-up-overlay`

## Tailwind

Use the default Tailwind config with standard content paths only. Do not add theme extensions for this page.

## Page Structure

The landing page consists of five major parts:

1. navigation
2. hero copy stack
3. interactive tab bar
4. video and overlay stage
5. company logo rail

## Navigation

Animation delay: `0.1s`

Container:

- `px-6 py-4 flex items-center justify-between max-w-7xl mx-auto`

Left side:

- Lucide `Star` icon
  - `w-5 h-5`
  - filled black
- `Stellar.ai`
  - `text-lg font-semibold`

Center navigation:

- hidden on mobile
- `hidden md:flex items-center gap-8`
- items:
  - `Solutions` with `ChevronDown`
  - `For Teams` with `ChevronDown`
  - `About Us`
  - `Learn Hub`
- text style:
  - `text-sm text-gray-700 hover:text-black`

Right side:

- `Login`
  - `text-sm text-gray-700`
- `Get started free`
  - `bg-black text-white px-5 py-2.5 rounded-full text-sm font-medium hover:bg-gray-800 transition-colors`

## Hero Section

Container:

- `px-6 pt-24 pb-32 max-w-7xl mx-auto text-center`

### Reviews Badge

Animation delay: `0.2s`

Structure:

- `inline-flex items-center gap-2 mb-8`
- bordered square
  - `w-6 h-6 border border-gray-300 rounded`
  - contains a filled `Star` icon
- text:
  - `4.9 rating from 18.3K+ users`
  - `text-sm font-medium text-black`

### Main Heading

Animation delay: `0.3s`

Styles:

- `text-6xl md:text-7xl lg:text-[80px]`
- `font-normal`
- `leading-[1.1]`
- `tracking-tight`
- `mb-5`

Content:

Line 1:

`Work Smarter. Move Faster.`

Line 2:

`AI Powers You Up.`

Line 2 styles:

- `bg-gradient-to-r from-black via-gray-500 to-gray-400`
- `bg-clip-text`
- `text-transparent`

### Subheading

Animation delay: `0.4s`

Styles:

- `text-lg md:text-xl text-gray-600 mb-8 max-w-2xl mx-auto`

Text:

`Intelligent automation syncs with the tools you love to streamline tasks, boost output, and save time.`

### Primary CTA

Animation delay: `0.5s`

Styles:

- `bg-black text-white px-8 py-3 rounded-full text-base font-medium hover:bg-gray-800 transition-colors mb-12`

Text:

`Begin Free Trial`

## Tab Bar

Animation delay: `0.6s`

Container:

- centered
- `bg-gray-100 rounded-lg p-1`

State:

- `useState("analyse")`
- auto-cycle every 4 seconds using `setInterval`

Tabs:

- `Analyse` with `BarChart3`
- `Train` with `BookOpen`
- `Testing` with `Users`
- `Deploy` with `Rocket`

### Mobile

- `md:hidden`
- 2x2 grid
- active tab:
  - `bg-white text-black shadow-sm`
- inactive tab:
  - `text-gray-600`

### Desktop

- `hidden md:flex`
- horizontal row
- vertical dividers:
  - `w-px h-5 bg-gray-300`

## Video And Overlay Stage

Animation delay: `0.7s`

Container:

- `relative rounded-3xl overflow-hidden h-[400px] md:h-[500px]`

Video:

- `autoPlay`
- `loop`
- `muted`
- `playsInline`
- `w-full h-full object-cover`

Source:

`https://d8j0ntlcm91z4.cloudfront.net/user_38xzZboKViGWJOttwIXH07lWA1P/hf_20260319_165750_358b1e72-c921-48b7-aaac-f200994f32fb.mp4`

## Overlay Cards

Only one overlay is visible at a time based on the active tab.

### Analyse

Show:

- `Set Up Your AI Workspace`
- purple progress bar at `25%`
- 4-step wizard layout

### Train

Show:

- `AI Model Training`
- orange progress at `67%`
- 4 metrics

### Testing

Show:

- `Test Suite Results`
- green success treatment
- `127/127 tests`

### Deploy

Show:

- `Deploy to Production`
- 4 checklist items
- `Deploy Now` button

All overlays should be driven from a single tab configuration object rather than hardcoded as separate branches everywhere.

## Company Logos

Animation delay: `0.8s`

Container:

- `mt-24`
- centered flex layout

Display these brand marks as stylized text treatments:

- `INTERSCOPE`
- `SPOTIFY`
- `Nexera`
- `M3`
- `LAURA COLE`
- `vertex`

Notes:

- `Nexera` should include a dot-grid cue
- `M3` should feel serif italic
- `LAURA COLE` should include an `LC` circular mark
- `vertex` should include subtle dot cues

These do not need real logo assets. Text-driven mock branding is acceptable.

## Implementation Shape

This should replace the current homepage on `main`.

Recommended file structure:

- `frontend/src/app/page.tsx`
- `frontend/src/app/globals.css`
- `frontend/src/components/landing/stellar-hero.tsx`
- `frontend/src/components/landing/stellar-navigation.tsx`
- `frontend/src/components/landing/stellar-tab-stage.tsx`
- `frontend/src/components/landing/stellar-logo-rail.tsx`
- `frontend/src/lib/landing/stellar-tabs.ts`
- `frontend/src/test/landing-page.test.tsx`

The tab stage should be a client component. The page assembly can remain simple and declarative.

## Testing Scope

Implementation should verify:

1. hero heading renders
2. CTA renders
3. all four tabs render
4. tab auto-cycle logic is wired safely
5. overlay content switches correctly by active tab
6. video element renders with expected attributes

## Success Criteria

The page is successful if:

1. it fully replaces the current homepage UI
2. it matches the provided Stellar-style structure closely
3. it feels visually crisp and premium on both mobile and desktop
4. the animated tab/video stage works cleanly
5. the page remains small, maintainable, and testable
