# Verification — SDAR grant proposal & budget docs

**Verified:** 2026-06-02 · against on-disk repo artifacts + live Anthropic pricing page.
**Targets:** `findings/sdar_reproduction_grant_proposal.md`, `findings/sdar_reproduction_budget.md`.
**Verdict:** ✅ Every load-bearing empirical, pricing, and arithmetic claim checks out exactly. The one substantive issue found (the two paired docs disagreed on the Tier-2 "representative run" dollar figure: ~$75–145 vs ~$200–375) has been **resolved** by reconciling the budget §3–5 to its own §0. 3 genuinely minor items remain (staleness + illustrative figures), none affecting a dollar figure.

## Verified exactly (repo-grounded)
| Claim | Source | Result |
|---|---|---|
| §1 metered table — 6 PPO runs: rows 7/5/5/3/3/1, output tok, $ | `cost_ledger.jsonl` ×6 | **exact** (e.g. c9be: 7 rows / 51,565 / $1.94) |
| 24 sub-agent sessions, all `claude-sonnet-4-6` | ledgers | **exact** |
| Token-class split 3.1% out / 84.2% cache-read / 12.7% cache-create | ledgers | **exact** (of full total) |
| Per-session **min 47k / median 201k / mean 216k / max 520k** | ledgers | **all exact** — under the cache-create-excluded defn (in+cache_read+out) with conventional median (mean of 2 middle values) |
| Adam root: out 2,075 / cache-read 72,978 / cc+fresh 13,641 / ~89k / 19 iter | `best_runs/adam` | **exact** |
| Adam rubric 0.741 | `best_runs/adam/final_report.json` | **exact** (0.7413) |
| GPU catalog: 4090 $0.34 · A6000 $0.49 · L40S $0.86 · A100-40 $1.19 · A100-80 $1.89 · H100-80 $4.39 · H200 $7.99 (SECURE) | `gpu_catalog.py` | **all 7 exact** |
| Budget caps `OPENRESEARCH_MAX_RUN_GPU_USD`/`_PER_HOUR` default 10.0 | `cli.py` | **confirmed** |
| 11 attempts from the 2026-05-28 death-spiral | `prj_09047604.../attempts` | **exact** (11 dated 20260528) |

## Verified — arithmetic (recomputed)
- §4.3 per-run on verified rates: Sonnet 4.6 Hard **$4.76**, Light $0.71, Medium $1.78; Opus 4.8 Hard **$7.93** — all reproduce exactly. 10-tries-all-ranges: **$73 Sonnet / $121 Opus** ✓.
- §6 budget sum 2,500 + 5,000 + 30,000 + 7,500 = **$45,000** ✓.

## Verified — external pricing (live page, 2026-06-02)
`platform.claude.com/docs/.../pricing` confirms **verbatim**: Opus 4.8 `$5/$6.25/$10/$0.50/$25`; Sonnet 4.6 `$3/$3.75/$6/$0.30/$15`; Haiku 4.5 `$1/$1.25/$2/$0.10/$5`; "Opus 4.7+ new tokenizer … up to 35% more tokens." The proposal's "web-verified" claim is itself verified.

## Substantive issue — the two docs disagree on a headline dollar figure
**Tier-2 "single representative run" (identical row label: "smallest-two, A100-80, metered LLM"):**
- Proposal §6 (line 131): **~$70–140 + ~$5 LLM** (≈ $75–145).
- Budget §5 (line 214): **~$200–375**.

Same run, **2–3× apart.** Two compounding causes:
1. **Stale LLM term.** Budget §0 (added 2026-06-02) revised LLM/run *down* ~3× (Regime A → ~$4–5), and the proposal adopted it — but the budget's own §3 (Regime A "$20–40"), §4 (Tier-2 LLM "$30–60") and §5 summary were never reconciled to §0.
2. **GPU end mislabeled.** Budget §4 puts A100-80 GPU at **$68–136**, so an "A100-80 + metered LLM" run is ~$100–196 — the $200–375 is the **H100** end ($158–316) carried into a row labeled A100-80.

**✅ RESOLVED (2026-06-02).** Reconciled budget §3–5 to its own §0: §4 Tier-2 LLM line now ~$4–5 healthy / ~$15–25 retries with an A100-80-specific total (~$75–160); §5 representative run → **~$75–145** (matches proposal); §3 blockquote clarified that Regime A is the loose/program end, not a single-healthy-run figure. Both docs now agree.

## Minor items (do not affect any dollar figure)
1. **Attempt count now stale.** `prj_09047604` has **13** attempt dirs (11 from 05-28 + 2 added 05-31). "11 … 2026-05-28 death-spiral" is accurate; "11 attempt directories" is now stale. *(both docs)*
2. **Definitional wrinkle (budget §0.2).** The "per-session processed" headline (216k) *excludes* cache-creation, yet the same table's split row counts cache-create as 12.7% of the *full* total (248k) — two denominators in one table. The §4.3 dollar math correctly uses the full token counts incl. cache-create, so costs are unaffected. Honest fix: "~248k processed (216k excl. cache-creation)."
3. **Generated source sizes (budget §0.3).** Adam biggest `.py` = **61KB** (claimed 75KB); SDAR top-level `train.py` = 45KB, largest attempt = 81KB (claimed 58KB) — no file is 58KB. Illustrative only ("output is a thin slice"); qualitative point holds; likely stale after the 05-31 attempts.

## Notes (not errors)
- SDAR ~4M-token figure is explicitly **modeled/scaled** (no clean completed SDAR run); both docs disclose this (§0.7, §9c). Honest.
- Non-Claude rows (GPT-5/Kimi/DeepSeek) are flagged "indicative" — not independently verified, appropriately caveated.
- Version drift: budget §2 says "Opus 4.7" ($5/$25); proposal updated to "Opus 4.8" — both correct (4.5–4.8 all priced $5/$25 on the live page).
