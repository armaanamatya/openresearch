# Discord post — copy/paste this with the 2 PNGs attached

**Files to attach in Discord:**
- `docs/design/img/dag.png` — the build-order graph (the one image to attach if only one)
- `docs/design/img/gantt.png` — the timeline (optional second attachment)

---

## Message body (1,850 chars — fits Discord's 2000-char limit)

```
**OpenResearch — next-build proposal (need your 👍 / 👎 / comments)**

Two parallel tracks before the MS VP review. Graph attached ↓

**Track A · Context-Engineering Harness** (order: 1 → 3 → 2 → 4)
**1.** Token-budget allocator + provenance — every LLM call logs what got in / dropped / why
**3.** Prompt-caching audit — enforce a stable prefix, measure cache-hit %, $ saved
**2.** Lens-specific compaction — split papers into claims/methods/hyperparams/ablations lenses; each agent pulls only what it needs
**4.** Cross-run memory — embedding store of past primitive calls; warm-start new runs from similar prior reproductions

Why this order: each item's *measurements* unlock the next item's *design decisions*. No guessing budgets, lens sizes, or cache strategies.

**Track B · Demo / Wow Factor** (parallel)
**5.** Live "watch it think" — rubric score climbing in real-time during the self-improvement loop. Side pane in `/lab`.
**6.** Multi-model arena (`--mode arena`) — GPT-5 / Qwen3-Coder / Kimi K2.5 / Claude on the same paper in parallel. Model-agnostic story for MS.
**7.** Public `/bench` leaderboard — static page, papers × models × rubric × $ × wall-clock. Frames the project as benchmark-defining.

**Targets for the VP slide:**
• −40% input tokens per primitive
• ≥60% cache hit rate
• −50% $ per paper
• Rubric score flat or +0.05
• ≥10 papers on the public leaderboard at demo time

**Window:** Track A ~3 weeks, Track B ~2 weeks in parallel.

**What I need from you:**
👍 if the order + scope look right
🟡 if you want to discuss before we cut PRs
💬 reply with what's missing / what to drop

Full plan + diagrams: `docs/design/context-harness-plan.md` on branch `claude/great-tesla-xor5z` (PR #74)

Or vote with priorities: `docs/wow-factor-poll.html` (open in browser, paste your code in this thread)
```

---

## Even shorter version (under 1000 chars, if you want a quick poll)

```
**Next-build proposal — react with 1️⃣–7️⃣ for your top picks**

Track A · Harness (build order 1→3→2→4)
1️⃣ Token-budget allocator + provenance
3️⃣ Prompt-cache audit (cheap $ savings)
2️⃣ Lens-specific compaction
4️⃣ Cross-run memory (warm-start)

Track B · Demo (parallel)
5️⃣ Live rubric-score-climbing pane
6️⃣ Multi-model arena (GPT-5 vs Qwen vs Kimi vs Claude)
7️⃣ Public /bench leaderboard

Targets for MS VP slide: −40% input tokens, ≥60% cache hits, −50% $/paper, 10+ papers on /bench.

Window: ~3 weeks Track A, ~2 weeks Track B parallel.

Full plan: docs/design/context-harness-plan.md (PR #74)
React with your top 2-3 numbers 👇
```
