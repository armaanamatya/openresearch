# Rubric climb + leaderboard — screen captures (2026-05-23)

Visual evidence for the rubric-climb panel + leaderboard delivery
(`docs/superpowers/specs/2026-05-23-rubric-climb-leaderboard.md`).

Captured against `npm run dev` on port 3001. The lab captures use the existing
`?rlmFixture=1` URL (an instant-replayed fixture). The leaderboard captures
were re-shot with the backend running (`OPENRESEARCH_RUNS_ROOT=/tmp/reprolab_leaderboard_fixtures`)
against three seeded demo rows so the populated view is visible.

## Captures

| File | Viewport | Page | What to look for |
|---|---|---|---|
| `lab-climb-desktop.png` | 1440 × 900 | `/lab?rlmFixture=1` | Large rubric score, baseline→target bar, SVG line-chart sparkline, per-area chip row with status glyphs, climb annotation, exploration tree on right. |
| `lab-climb-mobile.png` | 390 × 844 | `/lab?rlmFixture=1` | Big "0.53" score, sparkline line, four area chips (✓ pass, ✓ pass, ◐ partial, ✗ fail), `baseline 0.22 → 0.53`, `+0.31 vs target 0.70`, italic *from candidate mixed-precision training* attribution tail. |
| `leaderboard-desktop.png` | 1440 × 900 | `/leaderboard` | Three rows ranked by score desc (0.71 / 0.58 / 0.42), all 10 columns visible, sort indicator on the score column, mode badge, paper link. |
| `leaderboard-mobile.png` | 390 × 844 | `/leaderboard` | Same three rows, columns scroll horizontally inside the bordered table-wrap; "Attention is all you need" wraps gracefully. |

## Notes

- The lab desktop view is dense by design — the rubric strip is band 2 of the
  4-band shell. The mobile capture shows the panel in isolation more clearly.
- The chip row uses `flex-wrap`, so the four area chips fall to a second line
  on the 390-wide mobile capture — intentional and consistent with the rest
  of the visual language.
- The leaderboard empty state is what a reviewer sees when they open the page
  with no completed runs on disk. With seeded runs (a docker-up integration
  test deferred to the next session per the plan), the table replaces the
  placeholder.
