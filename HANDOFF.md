# HANDOFF — root-harness hardening (harden/root-harness)

Date: 2026-05-31 · Branch: `harden/root-harness` · Worktree: `/home/sww35/openresearch-harden`
Status: **design locked + committed; P0 in progress (reverted to clean); P1–P6 not started.**

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

## P0 — paper fidelity (DO THIS FIRST; ~30 min)
Target files + the REAL test files (last session used wrong test filenames → silent no-ops):
- Source: `backend/services/ingestion/parser/arxiv.py` — the function is the **module-level `fetch_arxiv_html(arxiv_id)`**, NOT a class method `ArxivFetcher._fetch_html` (the design doc's anchor was slightly off). Validation helpers: `_looks_like_html`, `_has_article_marker`, `_HTML_MIN_BYTES`. Current single source: `_ARXIV_HTML_URL = "https://arxiv.org/html/{arxiv_id}"`.
- **TODO (ADOPT #2 — ar5iv fallback):** add `https://ar5iv.labs.arxiv.org/html/{arxiv_id}` as a 2nd source. Cleanest: extract a `_try_fetch_html(url) -> str|None` helper (fetch+validate), then loop `fetch_arxiv_html` over `(_ARXIV_HTML_URL, _AR5IV_HTML_URL)`, returning the first that passes. Attribute: `# Pattern adapted from huggingface/ml-intern papers_tool.py (Apache-2.0)`.
- **Test file (correct name):** `tests/test_ingestion_arxiv_fetcher_html.py`. Convention: `monkeypatch.setattr(arxiv_mod, "urlopen", fake)`, a `_FakeResponse`, `_html_body(...)`, `request.full_url`. Add: falls-back-to-ar5iv (native stub → ar5iv used), prefers-native (good native → ar5iv never queried, 1 call).
- **DROP M2 (arxiv `not_a_pdf` retry) — do NOT implement it.** It conflicts with deliberate existing intent: `tests/test_issue12_intake_service.py::test_remote_fetch_non_pdf_emits_non_retryable_failure` (line 345) and `::test_fetch_non_pdf_emits_non_retryable_failure` (line 257) both assert a non-PDF body is `retryable=False`. A non-PDF at a URL is a hard error (retrying the same URL yields the same non-PDF), and ar5iv already rescues freshly-posted papers via the HTML path. The ml-intern audit (design doc §8) flagged M2 as MARGINAL/"subsumed by ar5iv." If a future owner still wants it, it requires changing those two tests + sign-off — not a silent flip.
- Success check: new arxiv tests green + `pytest tests/ -k "ingest or parser or arxiv or remote_pdf or fetch or html or resolving or intake"` all green (no regressions; was 179 passing pre-M2).

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
