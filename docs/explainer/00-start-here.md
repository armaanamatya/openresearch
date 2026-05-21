> **ReproLab Explainer · Index** — start here, then read **[01 — What ReproLab Is & Why ›](./01-overview.md)**

# 00 — Start Here

*The reading guide, mental model, and glossary for the ReproLab Explainer — a nine-part walkthrough of the codebase.*

## What this series is

The **ReproLab Explainer** is a nine-part series that teaches the ReproLab codebase from the ground up — what it does, why it is built the way it is, and how every subsystem works. It is written for someone who has never seen this repository before.

It is deliberately different from the docs already in the repo. `system_overview.md` and the files under `docs/agents/` are *reference* material — terse, and written for people who already have the context. This series is *pedagogical*: it builds intuition first, then goes deep, and assumes you know nothing about ReproLab specifically. Every non-obvious claim is anchored to a `path:line` you can open and check.

## Who this is for

A senior engineer joining or evaluating the project. You should be comfortable with Python, async I/O, and React — but you need **zero** prior knowledge of ReproLab. When you finish the series you will be able to navigate the codebase confidently, explain any subsystem to someone else, and know where a given change belongs.

## The 30-second mental model

ReproLab reproduces machine-learning papers automatically. A **14-stage pipeline** of **LLM agents** reads a paper, rebuilds its software environment, re-implements and runs the experiment, and scores the result against the paper's own claims. **Three verification gates** and an independent **audit chain** exist so the verdict can be trusted. Python drives the sequence deterministically; the LLM is called only for judgment; and every run is a directory on disk that can be resumed after a crash.

If you read only one more page, read **[01 — What ReproLab Is & Why](./01-overview.md)**.

## How to read this series

| # | Chapter | What you'll learn |
|---|---|---|
| **00** | Start Here *(this page)* | The reading guide, mental model, and glossary |
| **01** | [What ReproLab Is & Why](./01-overview.md) | The problem, one run end-to-end, the architecture in one picture |
| **02** | [The 14-Stage Pipeline](./02-the-pipeline.md) | The orchestrator state machine — the spine of the system |
| **03** | [Agents & the LLM Runtime](./03-agents-and-runtime.md) | How a stage becomes an LLM call; budgets, retries, the provider chain |
| **04** | [Verification, Scoring & Trust](./04-verification-and-trust.md) | The gates, the verifiers, the rubric, the Hermes audit chain |
| **05** | [Sandboxes & Environment Reconstruction](./05-sandboxes-and-environments.md) | Running untrusted code safely; rebuilding the environment |
| **06** | [Ingestion](./06-ingestion.md) | Turning a raw paper into structured, queryable knowledge |
| **07** | [State, Events & Persistence](./07-state-events-persistence.md) | Event sourcing, CQRS, and the knowledge layer |
| **08** | [Frontend & Operations](./08-frontend-and-ops.md) | The lab UI, the live bridge, configuration and deployment |

**Suggested paths**

- **The 20-minute tour** — read **01**, then **02**. That is the overview plus the spine; enough to hold a real conversation about the system.
- **The full path** — **01 → 08** in order. Each chapter builds on the previous one; this is the intended experience.
- **By need** — every chapter stands alone, with its own diagram, cross-references, and *Production Hardening* section. Jump straight to the subsystem you are about to touch.

Each chapter (01–08) closes with a **Production Hardening** section: concrete, file-anchored gaps for taking that subsystem from demo to production. Read end to end, those sections are a productionization roadmap.

## Glossary

The terms that recur across the series. Skim this now; refer back as needed.

| Term | Meaning |
|---|---|
| **project_id** | Stable ID for one reproduction, e.g. `prj_a1b2c3…`. Deterministically derived from the paper source, so the same paper always maps to the same ID. |
| **run** | One execution of the pipeline for a `project_id`. Its artifacts live in `runs/<project_id>/`. |
| **pipeline** | The 14-stage state machine (`orchestrator.py`) that drives a reproduction from `INGESTED` to `COMPLETE`. |
| **stage** | One of the 14 fixed states in `PipelineStage`. The system never has more than 14. |
| **agent** | One LLM invocation with a typed prompt, a tool allowlist, and schema-validated JSON output. Each stage runs one or more. |
| **runtime** | The provider-agnostic layer that executes an agent against Claude or OpenAI. |
| **gate** | A verification checkpoint (Gate 1 / 2 / 3) that decides whether a run may continue. |
| **verifier** | An agent that checks one aspect of a reproduction — method fidelity, metrics, artifacts, environment. |
| **rubric-verifier** | The agent that scores a reproduction against a PaperBench-style weighted rubric (0–1). |
| **PaperBench** | An external benchmark format for ML-paper reproduction; ReproLab can ingest its bundles and mirrors its rubric style. |
| **Hermes audit chain** | An independent oversight log recording whether each gate verdict was grounded in evidence. Stored in `runs/<id>/hermes/`. |
| **sandbox / backend** | The isolated environment where reconstructed code runs: `local_process`, `local_docker` (current default), or `runpod` (GPU, currently disabled). |
| **Track 3** | The rubric self-improvement loop — re-iterate improvements until the score target is met or a cap is hit. |
| **Track 4** | The environment build-and-repair loop — rebuild a broken `Dockerfile` from its build error, up to a cap. |
| **event store** | The append-only SQLite log of domain events; the authoritative record of everything that happened. |
| **CQRS** | Command/Query Responsibility Segregation — writes append events; reads come from projections and repositories. |
| **aggregate / projection** | DDD building blocks: an *aggregate* is a state machine folded from events; a *projection* is a read model rebuilt from them. |
| **SSE** | Server-Sent Events — the one-way stream that pushes live run updates from backend to UI. |
| **sdk vs offline mode** | `sdk` runs real LLM agents; `offline` is a deterministic, no-LLM path for CI and demos. |
| **fail-soft** | Design stance: on an unrecoverable sub-failure, finish the run with an honest partial verdict rather than crash. |

## A few notes before you start

- **Trust the code over the older docs.** Most files under `docs/agents/*.md` are `TODO` stubs; `system_overview.md` is accurate but terse. This series supersedes them as the primary explainer. `learn.md` and `CHANGELOG.md` are the project's append-only memory of bugs and changes — invaluable for the "why is it like this?" questions.
- **Citations are a hint, not a contract.** Every `path:line` reference in this series was accurate as of **2026-05-19**. File paths are stable; line numbers drift as code changes — open the file and look nearby.
- **The series was written by reading the code.** Each chapter was produced by analysing the actual source, with claims anchored so you can verify them. Where the code and the old docs disagreed, the code won — and the disagreements are noted.

Ready? → **[01 — What ReproLab Is & Why](./01-overview.md)**

---

**The ReproLab Explainer** — jump to any chapter:

▸ **00 · Start Here**  ·  [**01 · Overview**](./01-overview.md)  ·  [**02 · The Pipeline**](./02-the-pipeline.md)  ·  [**03 · Agents & Runtime**](./03-agents-and-runtime.md)  ·  [**04 · Verification & Trust**](./04-verification-and-trust.md)  ·  [**05 · Sandboxes**](./05-sandboxes-and-environments.md)  ·  [**06 · Ingestion**](./06-ingestion.md)  ·  [**07 · State & Events**](./07-state-events-persistence.md)  ·  [**08 · Frontend & Ops**](./08-frontend-and-ops.md)

[**01 · Overview**](./01-overview.md) ›
