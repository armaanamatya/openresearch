# HANDOFF — root-harness hardening (harden/root-harness)

Date: 2026-05-31 · Branch: `harden/root-harness` · Worktree: `/home/sww35/openresearch-harden`
Status: **design locked + committed; P0 DONE (ar5iv fallback, commit `5df6d19`); P1 next; P2–P6 not started.**

## Read these first (in order)
1. **`docs/superpowers/specs/2026-05-31-root-harness-hardening-design.md`** (committed `8ea804a`) — the authoritative plan. All decisions, phases P0–P6, per-phase tests, rollout/flag table, file:line anchors, HF contract (App A), honest rejection log (App B), SFT roadmap (App C), native-gap log (App D). **This is the source of truth — this HANDOFF only records live state + gotchas.**
2. Memory: `~/.claude/projects/-home-sww35-openresearch/memory/root-harness-hardening.md`.

## What's committed on this branch
- `8ea804a` docs(harden): the design doc.
- `9ba5dec` feat(openai-runtime): vLLM binding via OpenAIChatCompletionsModel (was uncommitted working-tree work; committed to give the worktree a clean base).
- `ec3fbc3` fix(ingest): paper-text override + arXiv fetch retry (the user's own work, committed before the worktree branched).
- Worktree is currently **clean** (P0's broken M2 attempt was reverted).

## Tooling gotchas (cost time last session — heed them)
- **Worktree has no own `.venv`.** Run tests with the main checkout's venv from the worktree cwd:
  `cd /home/sww35/openresearch-harden && /home/sww35/openresearch/.venv/bin/python -m pytest …`
  Confirmed it imports `backend` from the **worktree** (cwd wins). 
- **Bash cwd resets to `/home/sww35/openresearch` between calls** — always prefix each Bash with `cd /home/sww35/openresearch-harden &&`.
- **`Read` line numbers on `arxiv.py` were glitchy** (gaps, a possible tab in `_normalize_arxiv_id`). Match `Edit` `old_string`s on **unique content**, not line numbers, and **re-read the exact region** right before editing. Two arxiv edits silently failed last session due to old-string mismatch.

## P0 — paper fidelity ✅ DONE (commit `5df6d19`)
ar5iv fallback shipped. **Anchor correction (the handoff/design anchor was wrong — heed for future arxiv work):** the real source is `backend/services/ingestion/intake/fetchers/arxiv.py`, class **`ArxivFetcher`**, method **`_fetch_html`** — there is NO `parser/arxiv.py` and NO module-level `fetch_arxiv_html`. Constants are `_HTML_BASE_URL`/`_HTML_MIN_BYTES`/`_HTML_MAX_BYTES` (not `_ARXIV_HTML_URL`); validation is inline in the method (no `_looks_like_html`/`_has_article_marker` helpers).
- **What shipped:** added `_AR5IV_BASE_URL = "https://ar5iv.labs.arxiv.org/html"`; extracted `_try_fetch_html(html_url, arxiv_id) -> bytes|None` (fetch + the existing validation: status / 50 MB cap / is-html / `<article>`·`ltx_document` marker, never raises); `_fetch_html` loops `(_HTML_BASE_URL, _AR5IV_BASE_URL)` and writes the first valid body. Default-on, additive, no flag (design §10). Attributed to ml-intern `papers_tool.py` (Apache-2.0).
- **Tests:** `tests/test_ingestion_arxiv_fetcher_html.py` +2 — `test_html_falls_back_to_ar5iv_when_native_unavailable`, `test_html_prefers_native_arxiv_and_skips_ar5iv`. The fixtures route any url containing "html" to one response, so all 7 prior tests still pass; the 2 new ones dispatch native-vs-ar5iv on `"ar5iv" in url`. **182 passed / 0 regressions** in `pytest tests/ -k "ingest or parser or arxiv or remote_pdf or fetch or html or resolving or intake"` (handoff's "179" was approximate; 2 env skips: chromadb, tesseract).
- **M2 (arxiv `not_a_pdf` retry) — DROPPED (parked decision unchanged).** Conflicts with deliberate intent: `tests/test_issue12_intake_service.py::test_remote_fetch_non_pdf_emits_non_retryable_failure` + `::test_fetch_non_pdf_emits_non_retryable_failure` both assert non-PDF ⇒ `retryable=False`. Subsumed by ar5iv (design §8). Reopening requires changing those two tests + owner sign-off.

## P1–P6 — see the design doc §9. Quick map:
- **P1** Provider-runtime hardening: Gap A parity+hermetic (§3) **+ RuntimeGuard blacklist activation #7 (§4 — a real benchmark-integrity bug: `cli.py:1240` computes `blacklist_terms` then discards it; `to_runtime_spec` never sets `guard=`, so the paper's own repo is reachable from the authoring agent's networked Bash).** Files: `claude_runtime.py` (add `allowed_tools` via a shared MCP-merged `_tools_for_agent`, `setting_sources=[]`, `mcp_servers` always + `strict_mcp_config=True`, KEEP `bypassPermissions`), `registry.py` (`to_runtime_spec` `guard=` + fail-closed empty-tools guard), `base.py` (RuntimeGuard already exists at `base.py:35`), `context.py` (add `blocked_terms`), `invoke.py`, `cli.py:1240`. SDK `claude-agent-sdk==0.2.87` supports all needed kwargs (verified). Tests: cross-provider parity, permission_mode pin, hermetic, blacklist-blocks-paper-repo.
- **P2** Manifest (§5a): revive `services/runtime/artifacts.py` `commands.log`; enrich `experiment_runs.jsonl` at `_persist_experiment_result` (`primitives.py:3099`) with `experiment_run_id` (stop discarding `run_id` at `:3736`), `env_id`, `sandbox_backend`, structured command, `metrics_sha256`; back-link into `final_report`.
- **P3** Scoring truth (§5b/§5c): project `baseline_metrics` from artifact (kill model-injection at `report.py:695`); validated-citation clamp in `leaf_scorer.py` (**observe-first**, flip after one SDAR run).
- **P4** Budget/runtime (§6): `RuntimeBackend.exec` template-method centralizes enforcement (per-backend `cost_rate_usd_per_hour()`); upfront projected-cost gate (#5); sweeper ownership guard (#3) + preserve-in-flight (#4); watchdog unify (lift `_arm_watchdog` to RDR; delete dead `_ClusterWatchdog`); conformance test (all backends); +M1/M3/M4/#10.
- **P5** Loop & egress safety + guards: doom-loop detector (#1, closes BUG-LR-015 — sig MUST include stdout/result hash to not kill polling); chat secret-scrubber (#6); MCP default-off test; telemetry-stays-local test; boot validator (BUG-LR-014); leaderboard percentiles (#11).
- **P6** Capability (after P1's #7): GitHub reference-impl tool (#8, default-OFF, allowlist+guard-gated, needs `GITHUB_TOKEN`+`thefuzz`); run-complete notifications (#9, off-by-default).

## Rollout discipline (invariant 8): each phase ships its tests before the next is default-on. Behavioral changes default-on with `REPROLAB_*` hatch; citation-clamp observe-first. See design doc §10 for the full flag table.

## ml-intern reference
Cloned at `/tmp/ml-intern-src` (Apache-2.0). May not survive a reboot — re-clone with `git clone --depth 1 https://github.com/huggingface/ml-intern /tmp/ml-intern-src` if gone. Borrow **patterns only**, attribute each. It is an autonomous agent + HF-native toolset with **no built-in scoring** — a worker, not a harness; OpenResearch's `verify_against_rubric` stays the authority. Do NOT shell its headless auto-approval CLI (invariant 4) or enable its default trace-upload (invariant 5).

## Open decisions parked for the owner
- M2 (arxiv not_a_pdf retry): recommended DROP (see P0). Reconsider only with test-owner sign-off.
- `ApprovalService` is built but unwired from the RLM run path (design doc App D) — separate follow-up.
