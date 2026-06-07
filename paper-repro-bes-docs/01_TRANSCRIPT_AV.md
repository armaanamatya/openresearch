# IMG_5104.MOV — Combined Audio + Visual Transcript

**Source:** `IMG_5104.MOV` (iPhone, portrait, 1080×1920, 30 fps, 5 min 53 s)
**Recorded:** 2026-06-05 · **Processed:** 2026-06-06
**What this is:** A coworker walking a whiteboard, explaining the team's current
architecture for **automated reproduction of ML research papers** and proposing a new
**BES (Bidirectional Evolutionary Search)** harness. The camera pans/zooms across one
large whiteboard while he points at regions.

**Method (all processing local — nothing uploaded):**
- Audio → `ffmpeg` (extract 16 kHz mono) → **faster-whisper large-v3** on GPU (language en, p=1.00, word timestamps).
- Visual → 1 fps frame extraction → dwell-frame detection (sharpest still frames where the camera paused) → full-resolution re-extraction → upscaled crops of dense regions.
- Raw artifacts in `out/` (`transcript.txt`, `transcript.json`, `frames/`, `hi/`, `crops/`).

**Speakers / names mentioned:** the presenter (narrating); **Abik** (did the BES research); **Marcus** (raised the SDAR paper). Audience = the people being asked for feedback.

---

## Part A — Spoken transcript (timestamped, grouped by topic)

> Verbatim wording, lightly grouped into paragraphs for readability. Segment-level timestamps are in `out/transcript.txt`.

**[00:01–00:09] Intro.** "Okay, so this is our current infrastructure. We were wondering about some new ideas that we had, and what y'all's thoughts were."

**[00:09–00:20] Orchestrator.** "Just to give an overview of what we currently have: we currently have the orchestrator, which is an **RLM**, which has like one context window and then **sub-agent variables**, which is built into RLM."

**[00:20–00:36] Comprehension.** "So the orchestrator agent first — we put in the paper, it'll **ingest** that; it's part of the **comprehension phase**. And then it'll go and **plan** it, try to **understand** it using **sub-agents**, which return sub-agent variables — which is part of RLM."

**[00:36–00:44] Baseline.** "From there it'll go **create a baseline**, which creates the **environment inside of Docker**, and then implements the baseline by **creating a training script**."

**[00:44–00:54] Run experiments.** "Both of these go when we start **running experiments**, which is right after it's created. And this is primarily where most of the time is being taken, because the agents are just writing a lot of code."

**[00:55–01:04] GPU.** "From there the experiments are run inside / using a **GPU**, either inside **RunPod** or a **server**, which is what we're using currently."

**[01:04–01:15] Rubric.** "As it goes to the run, it'll test with the **rubric** which it's being run against. This is from **PaperBench**. It has currently **six main criteria**."

**[01:16–01:40] Static vs. dynamic rubric.** "It is static and dynamic, in the sense that these criteria are static — they won't change — and the weighting for these is static. But inside the weighting (by weighting I mean a score), they have **leaf nodes** which determine the score, and each of those is **dynamic**. There's around **two to five per rubric category**, and those are **defined by the orchestrator**, so it'll change and update for the paper depending on the paper."

**[01:44–01:58] Pass → report.** "From there, I'm assuming that we have a positive **pass** (still in progress) and it meets the **overall score for the rubric, which is 0.6** — it'll create the **report**. Eventually we'll get the report so we don't have to worry about it; we want to work on the improvement, but that's to be determined."

**[01:58–02:24] Current issues.** "So to go into the main issues: obviously, **compute**. But more than anything right now we're kind of scared about **hallucination and context rot** — which RLM does serve to fix, but still a little bit of concern there. **Runs are constantly failing**, and there's a lot of **movement between these phases** — that's the main issue we're facing right now. And there's also **no testing on the training script**."

**[02:25–02:33] BES introduced.** "So what we're doing — we found a paper which is called **BES: Bidirectional Evolutionary Search**. It's meant for searching, but we're trying to **change the architecture to work for us**."

**[02:33–02:47] Goal tree / back pass.** "The way it works is it takes an **input** and then creates a **goal tree** using a **back pass**, which creates **sub-goals**. These are some examples for the **SDAR paper** of sub-goals that **Marcus** mentioned. It'll give a **scoring criteria**…"

**[02:47–03:07] Forward search / evolve.** "…then it'll go to **forward search**. It'll go and find like multiple different **solutions**, and then **select** the ones that are valid. And then it can also **evolve**, which is really interesting for us — in the sense of, let's say one has the first half correct and two has the second half correct, you can **combine and splice** those to create a new candidate."

**[03:07–03:23] Back pass loop.** "And then from there it'll **back pass**, in which it'll **update the candidate pool** and then **mark off these sub-goals**. And then if it's just stuck, it'll **refine the goal tree**. So it'll come here, update the candidate pool, then loop around; if it's stuck it'll go here; and if it's complete, it'll return that."

**[03:24–03:37] Apply to baseline.** "That's for searching, but we're looking at it as somewhat the infrastructure for us — primarily in the **baseline stage**, because, as mentioned, there's a lot of movement between here and there. And **Abik** has done a lot of research on this and he thinks this is really the way to go."

**[03:38–04:07] Expensive backend → new harness.** "He thinks currently it's computationally expensive in the backend — right here — because it's constantly running and failing, constantly going. But if we can create a **new harness which uses BES and TDD (test-driven development) to reduce the errors before they even happen** — it's going to be more expensive upfront, but in the end we won't be constantly running and dealing with these issues, and it should be good and cheaper in the long run."

**[04:07–04:18] Validation agent.** "And I also want to add a **validation agent** here, which would be **completely separate from the RLM orchestrator**, just to double-check this plan and ensure that it's good and there's no hallucination or anything."

**[04:19–04:38] BES inside the rubric.** "And then from there, we also had another idea of maybe using this **BES inside of the rubric** as the **leaf nodes** — creating sub-goals for these leaf goals, maybe using sub-agents — but there are some concerns with costs in that. But just overall incorporating BES in that."

**[04:38–05:02] Forward pass + pooling.** "And then we've only really talked about the **back pass** in the sense of creating sub-goals and using that, but **Abik also mentioned using forward pass and pooling inside of these runs** — like post-run failure. The only thing with that is we'd have to wait for a lot of failures to overall accumulate, and then **batch it off to the training script** and fix it from there and have it refined. The only thing is that it could be slower."

**[05:03–05:16] Forward pass for evolution.** "And then we want to incorporate the **forward pass for using the evolution** that we mentioned earlier, for potentially **innovation** and just reiterating the paper and fixing it there."

**[05:16–05:33] Wrap-up + concerns.** "I don't want to make it too long. So this is our overall plan — our overall new infrastructure idea. Our current concerns are **cost, speed, and effectiveness**: speed primarily in the forward pass; effectiveness we won't really know until we test; and cost, if we use a bunch of sub-agents."

**[05:33–05:52] Cost detail + the ask.** "That's our current main cost. We're using like cloud OAuth — like our **Claude Code account subscriptions** — but we're burning through the usage there, so I can imagine it's going to be only worse if we use more sub-agents. But that's our current infrastructure. We're just wondering if you **agree or disagree** with the addition, and **where we should go from here**."

---

## Part B — Whiteboard visual inventory (everything on the board)

The board has roughly six zones. Faint/erased red ink = annotations & "new idea" notes; black = primary diagrams.

### B1. Core pipeline (center) — "current infrastructure"
```
                         ┌─────────┐   red note: "All under Orch.
                         │  Orch   │    - uses RLM → 1 context window
                         └────┬────┘      w/ sub-agent vars"
                  ── Comprehension ──
   paper ───────────► ┌─────────┐
                      │ ingest  │
                      └────┬────┘
                       (plan)
                      ┌────▼───────┐
                      │ understand │──► ○ ○ ○ ○ ○   (sub-agents → sub-agent variables)
                      └────┬───────┘
                      ┌────▼──────────────────┐        ┌──────────┐
                      │ (baseline)            │ ·····► │  Docker  │
                      │  1. create env        │        └──────────┘
                      │  2. implement details │   red: "NEW IDEA (Baseline step)"
                      └────┬──────────────────┘
                           ▼
                      ┌─────────────┐   label: "Rubric"
                      │ Run         │◄──────────────── (loops with Run/Pass/Verify/Tests)
                      │ Experiment  │
                      └──┬───┬───┬──────────────┐
                  Run ◄──┘   │   │  Verify      │  Run Tests w/ run
                          ┌──▼─┐ │
                          │Pass│ │   red: "NEW idea (LAST)"
                          └──┬─┘ │   red: "forward pass + pooling (post-run failure)"
                             ▼
                      ┌─────────┐   blue: "meets overall score for rubric"
                      │ Report  │
                      └────┬────┘
                       implement TBD
              red: "Forward pass for Evolution + innovation here?"
```

### B2. Data / training / compute (center-right)
- **Data Loader** (oval) → **Training script** box: **S1 1hr, S2 1hr, S3 1hr** (three stages, ~1 hour each).
- Feeds / runs on **GPU**: **(RunPod)** and **(Server)**.
  - *Team clarification (post-recording):* the GPU backend is **configurable** — "RunPod" stands
    in for whatever compute is on hand: managed cloud (RunPod), a **local GPU cluster** (e.g.
    A5000s), or **Azure cloud GPUs**. The specific GPU choice doesn't matter to the architecture.
- **Docker** box near the baseline step.
- Red: **"ERROR — only from not meeting min threshold on rubric — COM…"**
- **"from PaperBench"**, arrow: **"Runs against this."**

### B3. Rubric (right)
```
Rubric   (from PaperBench · "Runs against this" · Static w/ paper-specific tailored guidelines)
┌───────────────────────────┐
│ Method to Code            │   • weights are STATIC
│ Data + Preprocessing      │   • leaf nodes determine the score
│ Experiment reproducibility│   • each leaf DYNAMIC (2–5 per rubric category),
│ Artifact completeness     │     defined by the orchestrator per-paper
│ Metric eval               │   • RED "NEW IDEA": make leaf nodes via
│ Results match             │     BES sub-goals (sub-agents create sub-goals)
└───────────────────────────┘
Pass threshold: overall rubric score ≥ 0.6
```

### B4. Issues (top-right)
- **issues:** Compute, [Sub-agent cost? — partly cut off]
- **Hallucination / context (rot)**
- **Runs failing** — "no / limited tests on training [script]"; current idea: "Harness sug[gestion]…"

### B5. BES — new infrastructure proposal (second half of board; a shark/orca doodle drawn around it)
```
Current LLMs              TDD +
 - Sparse Signal :(       Test Driven Development
 - Narrow Exploration

BES: Bidirectional Ev[olutionary] Search
   - "for Search, but change arch to work for us"

input
  │
  ▼
┌──────────────┐        ┌────────────────────────────┐
│  Goal Tree   │        │  back pass                 │
│ complete =   │◄───────│  - creates Sub-goals       │
│   return     │        │  SDAR (Marcus's paper) e.g.:│
│ else = loop  │        │     1. coding logic        │
│ gives        │        │     2. architecture        │
│ scoring      │        │  red "NEW IDEA":           │
│ criteria     │        │    sub-agents do sub-goals │
└──────┬───────┘        └────────────────────────────┘
       ▼
┌────────────────────────────┐
│  Forward Search            │
│   candidates [1] [2] [3]   │
│   Select  (valid ones)     │
│   Evolve → "New / Combine  │
│            + Evolve"        │   (splice partial solutions → new candidate)
└──────────┬─────────────────┘
           ▼
┌────────────────────────────┐      ┌──────────────────────┐
│  Back Pass                 │      │ is stuck →            │
│   update candidate pool    │─────►│   refine goal tree    │
│   mark off sub-goals       │      └──────────────────────┘
└────────────────────────────┘      complete → return
     (the whole loop is enclosed by a single feedback arrow)
```
- Left column: **"New infra: Thoughts?"** — "Current concerns: **Cost, speed, effectiveness**" · "Any Questions or Concerns?" · "Anything we should expand on?"
- Red notes (rationale, near the core pipeline): "Old infra: sub-agents accepted **one-shot training script**" · "**New Harness uses BES + TDD to reduce errors before they happen**" · "More expensive upfront; should reduce back-and-forth" · "use BES backwards-tree idea to **split into sub-goals**" · "After, uses **another external [validation] agent** to ensure plan is good" · "**in prog[ress] →**"

### B6. GPU cost comparison (top-left) & business notes
**Cost-to-run comparison** (the "640 / VRAM × $/hr" math on the board):

| GPU | H100 | RTX 6000* | B200 |
|---|---|---|---|
| VRAM (GB) | 80 | 96 | 180 |
| $ / hr | $3.00 | $1.90 | $5.50 |
| 640 / VRAM (≈ # units) | 8 | ~6 (6.6) | ~4 (3.5) |
| **Total run cost** | **$24.00** | **$11.40** | **$22.00** |

\* written "ATX6000"; given 96 GB + $1.90/hr this is most likely the **RTX 6000 (Blackwell, 96 GB)**. **RTX 6000 is the cheapest option** in their math. Nearby: **"Errors take hours."**

> *Team clarification (post-recording):* this table is **illustrative, not a decision** — the
> specific GPU "doesn't matter." Compute is a **configurable backend** (RunPod / local A5000-class
> cluster / Azure cloud GPUs, depending on what's available). Kept here only for completeness.

**Business / GTM notes (far right, peripheral to architecture):** a **GTM** list (1 post/day FB+email, 1 post/day IG, outreach + influencer); **"Partnership + Content"** with rough figures (≈25/mo dev cost, 10k SMM fee, 12k / 20k tiers); **"api costs"** (Apple correct + dev account compliance); a "pls give me ma[rcus?]" speech bubble from the shark doodle. *(Handwriting here is faint/partly occluded — figures approximate.)*

---

## Part C — Synchronized A/V walkthrough (what was said ↔ what was shown)

| Time | Spoken (topic) | On screen / pointed at |
|---|---|---|
| 00:01–00:09 | Intro — "current infrastructure + new ideas" | Wide shot, center pipeline + left cost table |
| 00:09–00:20 | Orchestrator = RLM (1 context window + sub-agent vars) | **Orch** box (top center) + red "uses RLM" note |
| 00:20–00:36 | Ingest → comprehension → plan → understand via sub-agents | **ingest / understand** boxes; the 5 sub-agent circles |
| 00:36–00:44 | Create baseline: env in Docker + training script | **(baseline) 1. create env 2. implement details**; **Docker** box |
| 00:44–00:54 | Run experiments — most time spent (writing code) | **Run Experiment** box |
| 00:55–01:04 | GPU: RunPod or server | **GPU (RunPod)/(Server)** boxes; **Data Loader / Training script S1–S3** |
| 01:04–01:40 | Rubric from PaperBench; 6 criteria; static weights + dynamic leaf nodes | **Rubric** box (Method-to-Code … Results match); red leaf-node notes |
| 01:44–01:58 | Pass ≥ 0.6 → Report; improvement TBD | **Pass → Report → implement TBD** |
| 01:58–02:24 | Issues: compute, hallucination/context rot, runs failing, no training-script tests | **issues** list (top-right); red "ERROR only from min threshold" |
| 02:25–02:33 | Found BES paper, repurposing it | Pans to BES zone; **"BES: Bidirectional Ev. Search"**, shark doodle |
| 02:33–03:23 | BES mechanics: goal tree ← back pass (subgoals); forward search (select/evolve); back pass (update pool, mark off; stuck→refine) | **Goal Tree / back pass / Forward Search / Back Pass** boxes; **SDAR** examples |
| 03:24–04:07 | Apply BES to baseline; new harness = BES + TDD; expensive upfront, cheaper long-run | Back to core pipeline; red "New Harness uses BES + TDD" notes |
| 04:07–04:18 | Add a separate validation agent | Red "external [validation] agent to ensure plan is good" |
| 04:19–04:38 | BES inside rubric as leaf nodes (cost concern) | Rubric box; red "BES subgoals" note |
| 04:38–05:16 | Forward pass + pooling (post-run-failure batch fixes); forward pass for evolution/innovation | Pipeline "forward pass + pooling" + "Forward pass for Evolution + innovation" notes |
| 05:16–05:52 | Wrap-up; concerns cost/speed/effectiveness; cloud-code-subscription cost burn; the ask | **"New infra: Thoughts? — Cost, speed, effectiveness; Any questions/concerns?"** |

---

## Uncertain readings (flagged for your review)
- **"ATX6000"** GPU label → almost certainly **RTX 6000 (96 GB)**; the VRAM row third value read as "80" but the on-board math uses 180, so B200 = 180 GB.
- **"Metric eval"** rubric row (written "metric curl/evl") and **"Experiment reproducibility"** (two stacked words) — wording approximate.
- **Issues** second bullet after "Compute" is cut off at frame edge.
- **GTM / Partnership** dollar figures are faint and partly occluded by the presenter — treat as indicative only.
- **"SDAR"** = a paper Marcus referenced (expansion not stated on the board or in audio).
- **"RLM"** = used as "one context window + sub-agent variables, mitigates context rot" — consistent with **Recursive Language Models**; the board does not spell out the acronym.
