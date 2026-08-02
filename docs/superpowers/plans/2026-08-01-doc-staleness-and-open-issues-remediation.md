<!-- doc-meta: status=current; last-verified=2026-08-01 -->
# Doc Staleness & Open-Issues Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring every current-state doc into compliance with the two 2026 standing directives (⛔ OAuth forbidden 2026-08-01; GKE "NOT USED" never "parked" 2026-07-22), kill all ghost references to the deleted `docs/history/`, reconcile the tier-3/ablation doc contradictions, fix non-destructive repo hygiene, and land a permanent test guard so posture drift fails the suite.

**Architecture:** TDD on docs — Task 1 adds posture-guard tests to `tests/test_claude_md_fidelity.py` that FAIL against today's docs; Tasks 2–10 edit docs until they pass. Tasks 11–13 are hygiene + a canonical open-issues ledger. Nothing here touches `runs/`, `runs_logs/`, `gcp_logs/`, or `_archive/` (hard rule), and nothing is pushed or branch-pruned without operator sign-off (Task 12).

**Tech Stack:** Python/pytest (fidelity guard), `make docs-check`, plain markdown edits, git.

**Ground rules for the executor:**
- Git: descriptive present-tense commit headlines, **no Conventional-Commit prefixes, no Co-Authored-By/AI trailers**. Commit at task milestones. Do **not** push — pushing is operator-gated (Task 12).
- Never delete or move anything under `runs/`, `runs_logs/`, `gcp_logs/`, `_archive/`, `best_runs/` — only `.gitignore` entries are allowed for the first two.
- All pytest invocations on this machine need `OPENRESEARCH_MIN_DISK_GB=0` prefixed.
- Historical dossiers (`docs/periods/*.md`), `CHANGELOG.md` body entries, and `best_runs/**` narration are HISTORY — do not rewrite their OAuth/RunPod content; only current-state docs change.

---

### Task 1: Posture-guard tests (write failing tests first)

**Files:**
- Modify: `tests/test_claude_md_fidelity.py` (append at end)

- [ ] **Step 1: Append the three guard tests**

```python
# --- Cloud/auth posture guards (operator directives 2026-07-22 and 2026-08-01) ---

_POSTURE_DOCS = [
    "README.md",
    "CLAUDE.md",
    "ONBOARDING.md",
    "coworker.md",
    "docs/architecture.md",
    "docs/engineering-guide.md",
    "docs/operations.md",
    "docs/runbooks/2026-07-22-gcp-vm-e2e-run-procedure.md",
    "docs/runbooks/2026-08-01-remote-run-llm-auth.md",
    "docs/runbooks/2026-08-01-feature-ablation-gcp-runbook.md",
    "backend/agents/rlm/CLAUDE.md",
    "backend/services/runtime/CLAUDE.md",
    "infra/gcp/README.md",
    "infra/gcp/helm/README.md",
    "infra/azure/README.md",
    "docker/gke-cell-base/README.md",
]


def _posture_texts():
    for rel in _POSTURE_DOCS:
        path = _REPO / rel
        assert path.exists(), f"posture doc missing: {rel}"
        yield rel, path.read_text(encoding="utf-8")


def test_gke_posture_says_not_used_never_parked():
    """Directive 2026-07-22: every cloud-posture doc says GKE is NOT USED, never 'parked'."""
    for rel, text in _posture_texts():
        low = text.lower()
        if "gke" not in low:
            continue
        assert "not used" in low, f"{rel}: mentions GKE but never states it is NOT USED"
        assert "parked" not in low, f"{rel}: uses forbidden 'parked' wording for GKE"


def test_oauth_marked_forbidden_where_mentioned():
    """Directive 2026-08-01: any current doc that mentions OAuth must mark it forbidden."""
    import re

    marker = re.compile(r"(?i)(never use oauth|oauth is forbidden|never oauth|⛔[^\n]*oauth)")
    for rel, text in _posture_texts():
        if "oauth" not in text.lower():
            continue
        assert marker.search(text), f"{rel}: mentions OAuth without the forbidden marker"


def test_no_oauth_recommendation_phrases_survive():
    """Exact stale phrases that recommended OAuth as a usable path must be gone."""
    banned = [
        "Leave empty to use Claude CLI OAuth",
        "`claude login` for OAuth",
        "falls back to Claude CLI OAuth (free on subscription)",
        "Anthropic key/OAuth",
    ]
    for rel, text in _posture_texts():
        for phrase in banned:
            assert phrase not in text, f"{rel}: stale OAuth recommendation survives: {phrase!r}"
```

- [ ] **Step 2: Run the new tests to verify they FAIL against today's docs**

Run: `OPENRESEARCH_MIN_DISK_GB=0 .venv/bin/python -m pytest tests/test_claude_md_fidelity.py -q -k "posture or oauth"`
Expected: FAIL — `test_gke_posture_says_not_used_never_parked` (runbook says "parked"; `infra/gcp/README.md` never says "not used"), `test_oauth_marked_forbidden_where_mentioned` (README.md), `test_no_oauth_recommendation_phrases_survive` (README.md, gcp-vm runbook). If a listed file fails on `⛔`/marker matching for a reason not covered by Tasks 2–4, note it and adjust the fix tasks, not the test.

- [ ] **Step 3: Commit the failing guard (tests-first milestone)**

```bash
git add tests/test_claude_md_fidelity.py
git commit -m "Add cloud/auth posture guard tests for the GKE not-used and OAuth-forbidden directives"
```

---

### Task 2: Purge OAuth-as-usable-path wording from README.md

**Files:**
- Modify: `README.md:98-99, 110, 185, 199, 298-301, 306`

- [ ] **Step 1: Tech-stack rows (lines 98–99)** — replace:

```markdown
| Sub-agents | Claude Agent SDK (Sonnet) |
| Root models | GPT-5, Claude (API or OAuth), Qwen3-Coder, Azure OpenAI |
```

with:

```markdown
| Sub-agents | Claude Agent SDK (Sonnet by default; per-role override via `--models role=token,…`) |
| Root models | GPT-5, Claude (API key), Claude on Azure Foundry, Qwen3-Coder, Azure OpenAI |
```

- [ ] **Step 2: Prerequisites bullet (line 110)** — replace:

```markdown
- At least one LLM API key (`OPENAI_API_KEY` or `ANTHROPIC_API_KEY`), or `claude login` for OAuth
```

with:

```markdown
- At least one LLM API key: `OPENAI_API_KEY`, a **funded** `ANTHROPIC_API_KEY`, or Azure
  Foundry (`AZURE_FOUNDRY_*`). ⛔ **Never use OAuth** (`claude login` / `CLAUDE_CODE_OAUTH_TOKEN`
  / `--model claude-oauth`) — operator directive 2026-08-01; see
  `docs/runbooks/2026-08-01-remote-run-llm-auth.md`.
```

- [ ] **Step 3: Flags line (line 185)** — replace:

```markdown
**Flags:** `--mode {rlm,rdr,rlm-pure}`, `--provider {anthropic,openai}`, `--sandbox {auto,local,docker,azure,aws,gcp}`, `--model {gpt-5,claude,claude-oauth,qwen3-coder,azure}`, `--max-usd`, `--max-wall-clock`, `--vram-gb`
```

with:

```markdown
**Flags:** `--mode {rlm,rdr,rlm-pure}`, `--provider {anthropic,openai}`, `--sandbox {auto,local,docker,azure,aws,gcp}`, `--model {gpt-5,claude,sonnet-foundry,opus-foundry,qwen3-coder,azure}`, `--max-usd`, `--max-wall-clock`, `--vram-gb` (`claude-oauth` exists in code but is ⛔ forbidden — never use it)
```

- [ ] **Step 4: `ANTHROPIC_API_KEY` env-table row (line 199)** — replace:

```markdown
| `ANTHROPIC_API_KEY` | Optional | Sub-agents (Sonnet) and `--model claude`. **Leave empty to use Claude CLI OAuth** (`claude login`). A no-credit key does *not* fall back to OAuth — it hard-fails; see `CLAUDE.md` → "RLM auth". |
```

with:

```markdown
| `ANTHROPIC_API_KEY` | One auth path | Sub-agents (Sonnet) and `--model claude`. Must be a **funded** key — a no-credit key hard-fails with no fallback. ⛔ OAuth is forbidden (directive 2026-08-01); the sanctioned alternative surface is Azure Foundry (`sonnet-foundry`). See `CLAUDE.md` → "RLM auth". |
```

- [ ] **Step 5: LLM Auth Model section (lines 298–301)** — replace:

```markdown
1. **Root model** (RLM library) -- raw HTTP. Pick one: `--model gpt-5` (OpenAI), `--model claude` (Anthropic API key), `--model claude-oauth` (Claude CLI subscription), `--model azure` (Azure OpenAI).
2. **Sub-agents** (Claude Sonnet via `claude-agent-sdk`) -- uses `ANTHROPIC_API_KEY` if set and funded, otherwise falls back to Claude CLI OAuth (free on subscription).

For local development: use OpenAI for the root (~$1/run), OAuth for sub-agents ($0).
```

with:

```markdown
1. **Root model** (RLM library) -- raw HTTP. Pick one: `--model gpt-5` (OpenAI), `--model claude` (funded Anthropic API key), `--model sonnet-foundry` / `--model opus-foundry` (Anthropic on Azure Foundry), `--model azure` (Azure OpenAI).
2. **Sub-agents** (Claude Sonnet via `claude-agent-sdk`) -- a funded `ANTHROPIC_API_KEY` or the Foundry surface. ⛔ **Never OAuth** (operator directive 2026-08-01): no `--model claude-oauth`, no `CLAUDE_CODE_OAUTH_TOKEN`, no `claude login` — root or sub-agents, local or remote.

For local development: OpenAI root (~$1/run) + a funded/Foundry key for sub-agents. Full auth matrix: `docs/runbooks/2026-08-01-remote-run-llm-auth.md`.
```

- [ ] **Step 6: Current Limitations bullet (line 306)** — replace:

```markdown
- Cost ledger reports $0 for OAuth runs (SDK doesn't surface token counts).
```

with:

```markdown
- Cost ledger is blind to Foundry-routed LLM spend and idle GPU-node time — a `$0` there is not proof of $0; verify via `tokens_total.json`.
```

- [ ] **Step 7: Verify README no longer trips the guard**

Run: `OPENRESEARCH_MIN_DISK_GB=0 .venv/bin/python -m pytest tests/test_claude_md_fidelity.py -q -k "oauth"`
Expected: README.md assertions gone from the failure list (coworker.md/runbook failures remain until Tasks 3 and 10).

- [ ] **Step 8: Commit**

```bash
git add README.md
git commit -m "Purge OAuth-as-usable-path wording from README per the 2026-08-01 directive"
```

---

### Task 3: GCP-VM runbook — "parked" → "NOT USED", drop the OAuth remedy

**Files:**
- Modify: `docs/runbooks/2026-07-22-gcp-vm-e2e-run-procedure.md:12-16, 84, 255, 256`

- [ ] **Step 1: Capture the actual current guard message** (commit `bf3a937e` reworded it): run `grep -rn "ALLOW_GKE\|NOT USED" backend/services/runtime/ backend/agents/ | grep -i "raise\|error\|not used" | head -5` and read the raising line so the quote below matches the code verbatim. Adjust the quoted message in Step 2 if it differs.

- [ ] **Step 2: Header block (lines 12–16)** — replace:

```markdown
- **GKE is parked.** As of 2026-07-22, `--sandbox gcp` / `--sandbox gke` **fails loud**
  (`_backend_for_sandbox_mode` raises: *"GKE parked — use the campaign VM path"*). Set
  `OPENRESEARCH_ALLOW_GKE=1` to revive it, but only once the two blocked IAM grants are
  fixed (artifactregistry.reader + workloadIdentityUser). Design + rationale:
```

with:

```markdown
- **GKE is NOT USED.** As of 2026-07-22, `--sandbox gcp` / `--sandbox gke` **fails loud**
  (`_backend_for_sandbox_mode` raises — GKE is NOT USED; use the campaign VM path; wording
  updated in commit `bf3a937e`). The `OPENRESEARCH_ALLOW_GKE=1` escape hatch exists but is
  **inert and not a supported path** — the two missing IAM grants
  (artifactregistry.reader + workloadIdentityUser) were never applied. Design + rationale:
```

(keep the existing spec link line that follows unchanged)

- [ ] **Step 3: Line 84** — replace `hit the parked/fail-loud GKE path` with `hit the fail-loud GKE path (NOT USED)`.

- [ ] **Step 4: Troubleshooting row (line 255)** — replace:

```markdown
| `--sandbox gcp/gke` raises "GKE parked" | Intentional fail-loud (commit `86c00abe`) | Use the VM path; set `OPENRESEARCH_ALLOW_GKE=1` only after the IAM grants are fixed |
```

with:

```markdown
| `--sandbox gcp/gke` raises the GKE guard | Intentional fail-loud (commit `86c00abe`; message reworded to NOT USED in `bf3a937e`) | Use the VM path — GKE is NOT USED; the `OPENRESEARCH_ALLOW_GKE=1` hatch is not a supported path |
```

- [ ] **Step 5: Troubleshooting row (line 256)** — replace the remedy cell `Use a validated executor (Sonnet) via a working Anthropic key/OAuth` with `Use a validated executor (Sonnet) via a funded Anthropic key or Foundry (⛔ never OAuth)`.

- [ ] **Step 6: Verify + commit**

Run: `OPENRESEARCH_MIN_DISK_GB=0 .venv/bin/python -m pytest tests/test_claude_md_fidelity.py -q -k "posture or oauth"` — the gcp-vm runbook must no longer appear in failures.

```bash
git add docs/runbooks/2026-07-22-gcp-vm-e2e-run-procedure.md
git commit -m "Reword GCP-VM runbook GKE posture to NOT USED and drop the OAuth remedy"
```

---

### Task 4: GKE-not-used banners on infra/docker reference docs

**Files:**
- Modify: `infra/gcp/README.md:1-4`, `infra/gcp/helm/README.md:1`, `docker/gke-cell-base/README.md:1`

- [ ] **Step 1: `infra/gcp/README.md`** — insert directly under the `# ReproLab — GCP GKE GPU Backend: Terraform L1` title line:

```markdown
> # ⛔ GKE is NOT USED — operator directive (2026-07-22)
> The GKE backend fail-closes (`_backend_for_sandbox_mode` raises on `gcp`/`gke`; the
> `OPENRESEARCH_ALLOW_GKE=1` hatch is inert and not a supported path). These Terraform/Helm
> layers are kept in-tree **for reference only — do not provision from them.** The supported
> GCP path is the single-VM campaign route:
> [`docs/runbooks/2026-07-22-gcp-vm-e2e-run-procedure.md`](../../docs/runbooks/2026-07-22-gcp-vm-e2e-run-procedure.md).
```

- [ ] **Step 2: `infra/gcp/helm/README.md`** — insert the same blockquote under its title line (fix the relative link depth: `../../../docs/runbooks/...` if the file sits one level deeper — verify with `ls`).

- [ ] **Step 3: `docker/gke-cell-base/README.md`** — insert under its title line:

```markdown
> ⛔ **GKE is NOT USED** (operator directive 2026-07-22). This image is retained for the inert
> GKE cell path only — do not build/push for production. Supported paths: GCP single-VM,
> Azure/AKS, AWS/EKS.
```

- [ ] **Step 4: Verify + commit**

Run: `OPENRESEARCH_MIN_DISK_GB=0 .venv/bin/python -m pytest tests/test_claude_md_fidelity.py -q -k "posture"`
Expected: `test_gke_posture_says_not_used_never_parked` PASSES.

```bash
git add infra/gcp/README.md infra/gcp/helm/README.md docker/gke-cell-base/README.md
git commit -m "Add GKE-not-used banners to the inert GKE infra and docker reference docs"
```

---

### Task 5: backend/agents/rlm/CLAUDE.md — stale test ref, ghost spec ref, thinking-patch invariant

**Files:**
- Modify: `backend/agents/rlm/CLAUDE.md` (line 9, the MUSE-lite spec pointer, and the "Stability & correctness invariants" section)

- [ ] **Step 1: Fix the primitive-count sync pointer (line 9)** — replace `` keep this count and `tests/rlm/test_registry.py`'s `EXPECTED` set in sync `` with `` keep this count and `tests/test_claude_md_fidelity.py::test_custom_tools_count_matches_doc` in sync ``. (First confirm `tests/rlm/test_registry.py` really doesn't exist: `ls tests/rlm/test_registry.py` → expect "No such file".)

- [ ] **Step 2: Fix the MUSE-lite ghost spec pointer** — the line ends `Spec: `docs/history/specs/2026-05-30-...` design doc.` and `docs/history/` does not exist. Run `grep -n "lesson\|MUSE" docs/periods/2026-05.md | head -5` to find the dossier section, then replace the pointer with: `` Spec: consolidated into the 2026-05 dossier — `docs/periods/2026-05.md` (negative-lessons / MUSE-lite section). ``

- [ ] **Step 3: Document the thinking patch as a load-bearing import-time invariant.** First check it isn't already there: `grep -n "thinking" backend/agents/rlm/CLAUDE.md`. If absent, add this bullet to the "Stability & correctness invariants" section (alongside the safe-builtins and forced-iteration entries):

```markdown
- **Anthropic thinking-block patch** (`_anthropic_thinking_patch.py`, imported at the top of
  `run.py` next to the foundry/forced-iteration patches — keep it): (a) patches rlm
  `AnthropicClient.{completion,acompletion}` to extract concatenated text blocks, skipping
  thinking/tool-use blocks (crash was `content[0].text` on a ThinkingBlock); (b) patches the
  global anthropic SDK `Messages.create`/`AsyncMessages.create` to inject
  `thinking={"type": "disabled"}` for Foundry `claude-sonnet-5`/`claude-opus-4-8`, whose
  thinking default eats the `max_tokens` budget and truncates output. Byte-identical for all
  other models; idempotent. Tests: `tests/rlm/test_anthropic_thinking_patch.py`. Incident +
  full detail: `docs/runbooks/2026-08-01-remote-run-llm-auth.md`.
```

- [ ] **Step 4: Bump the doc-meta header** — `last-verified=2026-07-05` → `last-verified=2026-08-01`.

- [ ] **Step 5: Verify + commit**

Run: `OPENRESEARCH_MIN_DISK_GB=0 .venv/bin/python -m pytest tests/test_claude_md_fidelity.py -q` (all tests — citation resolution must stay green).

```bash
git add backend/agents/rlm/CLAUDE.md
git commit -m "Fix stale test and spec pointers in the RLM CLAUDE.md and document the thinking patch invariant"
```

---

### Task 6: tests/CLAUDE.md — describe the fidelity guard as it actually is

**Files:**
- Modify: `tests/CLAUDE.md` (the "As of today the guard reads only the root file…" paragraph)

- [ ] **Step 1:** Replace the paragraph beginning `As of today the guard reads only the root file` and ending `…rather than assuming it still checks root alone.` with:

```markdown
The guard reads the root file **plus every nested `CLAUDE.md`** (`_read_claude_docs()`
concatenates the set — landed 2026-07-05), so a documented env var, the primitive count, or a
`docs/*.md` citation may live in root OR in any nested file; the guard only needs to find it
somewhere in the set. If you add a doc-fidelity claim to a nested file, check the guard's
actual file list in `_read_claude_docs()` first.
```

- [ ] **Step 2:** Bump doc-meta `last-verified` to `2026-08-01`. Commit:

```bash
git add tests/CLAUDE.md
git commit -m "Describe the fidelity guard's nested-doc coverage as landed, not future work"
```

---

### Task 7: Kill the remaining `docs/history/` ghost references

**Files:**
- Modify: `infra/azure/STREAM-E-NOTES.md:4`, `infra/azure/helm/README.md:192`, `infra/azure/bicep/README.md:257`, `best_runs/adam/README.md:9`, `CHANGELOG.md` (header area)

`docs/history/` was consolidated into `docs/periods/<month>.md` dossiers during the 2026-07-22 cleanup. All four live-doc citations point at June specs → repoint to the 2026-06 dossier.

- [ ] **Step 1:** In each of the three `infra/azure/` files, replace the full `docs/history/specs/2026-06-…` path with: `` `docs/periods/2026-06.md` (consolidated dossier; the original `docs/history/` spec was pruned 2026-07-22) `` — keep the surrounding sentence intact.
- [ ] **Step 2:** In `best_runs/adam/README.md:9`, apply the same replacement pattern (spec was `2026-06-14-adam-score-optimization-design.md` → the 2026-06 dossier).
- [ ] **Step 3:** In `CHANGELOG.md`, add one line to the existing staleness note at the top (do NOT rewrite body entries):

```markdown
> Note (2026-08-01): `docs/history/*` paths cited in entries below were consolidated into
> `docs/periods/<month>.md` dossiers on 2026-07-22 and no longer exist on disk.
```

- [ ] **Step 4: Verify no live references remain**

Run: `grep -rn "docs/history" --include="*.md" . | grep -v node_modules | grep -v docs/periods | grep -v CHANGELOG.md`
Expected: no output (periods dossiers narrating history are exempt).

- [ ] **Step 5: Commit**

```bash
git add infra/azure/STREAM-E-NOTES.md infra/azure/helm/README.md infra/azure/bicep/README.md best_runs/adam/README.md CHANGELOG.md
git commit -m "Repoint surviving docs/history citations to the periods dossiers"
```

---

### Task 8: Reconcile the ablation plan with configs/ablation/arms.json

**Files:**
- Modify: `docs/superpowers/plans/2026-07-31-feature-ablation-campaign-plan.md:33, 35, 49, 114`

`arms.json` is what `scripts/merge_run_spec.py` actually consumes — its arm IDs and env names are canonical; the plan drifts from it.

- [ ] **Step 1: F1 row (line 33)** — replace the pseudo-notation toggle cell (`` `experiment_arm.bes.enabled=true`, candidates_per_cluster=2, select_metric=cluster_score ``) with the real env vars: `` `OPENRESEARCH_BES_ENABLED=1`, `OPENRESEARCH_BES_CANDIDATES_PER_CLUSTER=2`, `OPENRESEARCH_BES_SELECT_METRIC=cluster_score` (see `configs/ablation/arms.json` → `bes`) ``
- [ ] **Step 2: F2 row (line 35)** — replace `campaign anti-regression rail set — *confirm exact env*` with `` `OPENRESEARCH_CHAMPION_ARTIFACT=1`, `OPENRESEARCH_EVIDENCE_FINGERPRINT=1` (see `configs/ablation/arms.json` → `champion`) ``
- [ ] **Step 3: Remove the now-moot action item at line 49** (the "grep … to pin the exact env var for F2" note).
- [ ] **Step 4: Arm-name matrix (line 114)** — rename the `a_evid` row label to `champion` so it matches `arms.json` (verify with `python scripts/merge_run_spec.py --list`, which prints the canonical arm names).
- [ ] **Step 5: Verify**: `python scripts/merge_run_spec.py --list` output names all appear verbatim in the plan's matrix; `grep -n "a_evid" docs/superpowers/plans/2026-07-31-feature-ablation-campaign-plan.md` → no output.
- [ ] **Step 6: Commit**

```bash
git add docs/superpowers/plans/2026-07-31-feature-ablation-campaign-plan.md
git commit -m "Align ablation plan arm names and flag env vars with configs/ablation/arms.json"
```

---

### Task 9: Status banners on the executed tier-3 subproject plans

**Files:**
- Modify: `docs/superpowers/plans/2026-07-22-tier3-subproject-a-clean-gcp-vm-run.md:1` and `docs/superpowers/plans/2026-07-22-tier3-subproject-b-scheduler-apply.md:1`

The spec's banner says A+B are DONE (2026-07-23) while both plans present unchecked task lists — readers can't tell roadmap from history.

- [ ] **Step 1:** At the top of the sub-project **A** plan, insert:

```markdown
> **STATUS 2026-08-01 — EXECUTED (2026-07-22/23).** The checkboxes below are the as-authored
> roadmap and were not ticked during execution. Ground truth for what actually landed:
> `docs/progress/2026-07-22-tier3-adam-progress.md` and the status banner in
> `docs/superpowers/specs/2026-07-22-tier3-scheduler-adam-ab-design.md`. Note the disclosed
> deviation: the 5-field checkpoint/resume work (spec A-item-2) moved to sub-project B.
```

- [ ] **Step 2:** At the top of the sub-project **B** plan, insert the same banner minus the deviation sentence, plus: `Phase C (billed ADAM A/B on real GPU) remains ⛔ gated on operator GPU budget; Task 5 was deferred (see the progress log).` First reconcile against `docs/progress/2026-07-22-tier3-adam-progress.md` (§ around line 697: "Task 5 deferred; B not yet") — if the progress log contradicts "EXECUTED" for any B task, name the exception in the banner rather than overstating.
- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/plans/2026-07-22-tier3-subproject-a-clean-gcp-vm-run.md docs/superpowers/plans/2026-07-22-tier3-subproject-b-scheduler-apply.md
git commit -m "Mark executed tier-3 subproject plans with status banners pointing at ground truth"
```

---

### Task 10: Small-doc sweep — coworker.md, operations link, doc-meta refresh, flags.md regen

**Files:**
- Modify: `coworker.md`, `docs/operations.md:38`, `backend/services/runtime/CLAUDE.md:1`, `frontend/CLAUDE.md:1`, `configs/README.md:1`, `docs/reference/flags.md`

- [ ] **Step 1: coworker.md** — add under its title:

```markdown
> ⛔ **Never use OAuth** (operator directive 2026-08-01): no `--model claude-oauth`, no
> `CLAUDE_CODE_OAUTH_TOKEN`, no `claude login`. API keys only — Azure Foundry
> (`sonnet-foundry`) or a funded `ANTHROPIC_API_KEY`. Details:
> `docs/runbooks/2026-08-01-remote-run-llm-auth.md`.
```

- [ ] **Step 2: docs/operations.md:38** — normalize the runbook link to a correct relative path: `[2026-07-22-gcp-vm-e2e-run-procedure.md](runbooks/2026-07-22-gcp-vm-e2e-run-procedure.md)` (verify it resolves from `docs/`: `ls docs/runbooks/2026-07-22-gcp-vm-e2e-run-procedure.md`).
- [ ] **Step 3: doc-meta bumps** — in `backend/services/runtime/CLAUDE.md`, `frontend/CLAUDE.md`, `configs/README.md`: only after confirming the Task 1–9 sweep surfaced no other drift in that file, set `last-verified=2026-08-01`. (`backend/agents/rlm/CLAUDE.md` and `tests/CLAUDE.md` were bumped in Tasks 5–6.)
- [ ] **Step 4: flags.md** — it is generated; regenerate: `ls scripts/gen_flag_registry.py && .venv/bin/python scripts/gen_flag_registry.py` then `git diff --stat docs/reference/flags.md`. If the script path differs, find it with `grep -rn "flags.md" scripts/ Makefile` and use the real generator; do not hand-edit the file.
- [ ] **Step 5: Full verification + commit**

Run: `OPENRESEARCH_MIN_DISK_GB=0 .venv/bin/python -m pytest tests/test_claude_md_fidelity.py -q` → all pass (including the three Task 1 guards). Run `make docs-check` → OK.

```bash
git add coworker.md docs/operations.md backend/services/runtime/CLAUDE.md frontend/CLAUDE.md configs/README.md docs/reference/flags.md
git commit -m "Refresh doc-meta stamps, coworker auth caveat, operations link, and the generated flag registry"
```

---

### Task 11: Non-destructive repo hygiene

**Files:**
- Modify: `.gitignore`; `tests/rlm/test_max_turns_floor.py`, `tests/rlm/test_lifecycle_binding_parity.py`, `tests/rlm/test_figure_sidecars.py`, `tests/agents/rlm/test_run_experiment_cell_route.py`

- [ ] **Step 1: .gitignore** — append (this hides, never deletes — complies with the never-delete rule):

```gitignore
# local run backups / overnight monitor scratch (never committed, never deleted)
.demo_backups/
runs_logs/
```

Verify: `git status --porcelain | grep -c "demo_backups\|runs_logs"` → 0.

- [ ] **Step 2: Unused test imports** — `uvx ruff@0.15.16 check --select F401 --fix tests/rlm/test_max_turns_floor.py tests/rlm/test_lifecycle_binding_parity.py tests/rlm/test_figure_sidecars.py tests/agents/rlm/test_run_experiment_cell_route.py`, then run those four test files: `OPENRESEARCH_MIN_DISK_GB=0 .venv/bin/python -m pytest tests/rlm/test_max_turns_floor.py tests/rlm/test_lifecycle_binding_parity.py tests/rlm/test_figure_sidecars.py tests/agents/rlm/test_run_experiment_cell_route.py -q` → all pass.
- [ ] **Step 3: Prune the dead worktree record** (metadata only; the `/private/tmp` directory is already gone): `git worktree prune && git worktree list` → only the main checkout remains.
- [ ] **Step 4: Fast-forward local `main`** (it is 49 behind origin/main, 0 ahead — verify first: `git log origin/main..main --oneline | wc -l` must print 0, then `git fetch origin main:main`).
- [ ] **Step 5: Commit**

```bash
git add .gitignore tests/
git commit -m "Ignore local run-backup scratch dirs and drop unused test imports"
```

---

### Task 12: Operator-gated actions — DO NOT EXECUTE without explicit sign-off

- [ ] **Push the branch.** `fix/analysis-cleanups` holds **156 unpushed commits** (the whole consolidation + tier-3 + fixes line) on a single external volume — one disk failure loses it all. On sign-off: `git push -u origin fix/analysis-cleanups`.
- [ ] **Prune superseded branches** (unmerged SHAs — needs the operator's supersession judgment; tag before delete per the `backup/pruned/*` convention): `remove-runpod-railway-cleanup` (local + origin; 5 commits content-superseded by the trunk), local `chore/repo-consolidation` (its worktree is gone), and stale remotes `origin/gke-local-transport`, `origin/authoritative-scheduler-runtime`, `origin/scheduler-authority-runtime`, `origin/feat/grounded-self-improvement-harness-reliability`, `origin/feat/azure-bicep-canonical-aoai-hardening`, `origin/harden/resume-pause-reliability` (merged via PR #12). Recipe per branch: `git tag backup/pruned/<name> <sha> && git push origin backup/pruned/<name> && git branch -D <name> && git push origin --delete <name>`.
- [ ] **Tier-3 Phase C** (billed ADAM A/B on real GCP GPU) — gated on GPU budget; prerequisites are done per the progress log.
- [ ] **best_runs curation pass** — `docs/policies/artifacts.md:14` froze history "pending a separately reviewed curation pass"; schedule or explicitly drop it.

---

### Task 13: Canonical open-issues ledger

**Files:**
- Create: `docs/open-issues.md`
- Modify: `docs/README.md` (add one index line pointing at it)

- [ ] **Step 1: Create `docs/open-issues.md`:**

```markdown
<!-- doc-meta: status=current; last-verified=2026-08-01 -->
# Open issues — the honest ledger

One line per genuinely unresolved issue. When you fix one, delete its row in the same
commit. History/narrative lives in the periods dossiers, not here.

| Issue | Since | Status | Pointer |
|---|---|---|---|
| Tier-3 Phase C: billed ADAM A/B on real GPU never ran; scheduler authority default-ON needs ≥3 paired A/B runs + grader-σ gate + operator sign-off | 2026-07-22 | ⛔ gated on operator GPU budget | `docs/progress/2026-07-22-tier3-adam-progress.md` |
| Foundry cost blindness: `cost_ledger.jsonl` logs $0 for Foundry-routed LLM spend (no Foundry entries in `pricing.py`); real cost only in `tokens_total.json` | 2026-07 | open (documented caveat; candidate fix: Foundry price table in `pricing.py`) | root `CLAUDE.md` → "Cost visibility" |
| Ablation grid ops G2–G5 never fire (only G1 branching has ever fired) | 2026-07-31 | open — blocks full ablation campaign value | `docs/superpowers/plans/2026-07-31-feature-ablation-campaign-plan.md` §grid |
| `grok` is not a validated executor (emits no cell/commands manifest → evidence gate fails the run) | 2026-07-22 | open (documented limitation; validate only if grok should be a production executor) | gcp-vm runbook troubleshooting table |
| SDAR baseline: headline +7% never reproduced (best partial 0.363/0.600; Track B authors-trainer 0.456 verified) | 2026-06 | open — aspirational, canonical stress test | `best_runs/sdar/README.md` |
| Parser idempotence keys on the event store, not run-dir files — stale state needs a fresh `--project-id` | 2026-07-08 | open (by-design; documented workaround) | `backend/services/ingestion/` + gcp-vm runbook |
| No funded `ANTHROPIC_API_KEY`; `OPENAI_API_KEY` dead (401). Root-model selection prefers gpt-5 whenever `OPENAI_API_KEY` is present (`resolve_root_model`) — a dead key silently hijacks the root | 2026-06 | open (by-design selection; operative posture: Foundry keys) | `docs/runbooks/2026-08-01-remote-run-llm-auth.md` |
| `docs/policies/artifacts.md`: best_runs history frozen "pending a separately reviewed curation pass" that never happened | 2026-07-20 | open — schedule or drop | `docs/policies/artifacts.md` |

**Resolved recently (kept 30 days to kill stale memory, then delete):** 18 collection errors +
20 test failures — fixed in `954e3a8b` (suite now 10268 collected, 0 errors). k8s 409 retry
collision — fixed in `5c026301` (PR #12). Cutout FALSE-"failed" Tier-1 fixes — merged via the
consolidation trunk. GKE IAM grants + train-scope blockers — moot (GKE is NOT USED). OAuth
sub-agent SDK flakiness — moot (⛔ OAuth forbidden, 2026-08-01).
```

- [ ] **Step 2:** Add to `docs/README.md`'s index: `- [open-issues.md](open-issues.md) — the honest ledger of genuinely unresolved issues (delete rows as they close)`.
- [ ] **Step 3: Final full verification**

Run: `OPENRESEARCH_MIN_DISK_GB=0 .venv/bin/python -m pytest tests/test_claude_md_fidelity.py -q && make docs-check` → all green.

- [ ] **Step 4: Commit**

```bash
git add docs/open-issues.md docs/README.md
git commit -m "Add the canonical open-issues ledger and index it"
```

---

## Out of scope (deliberately)

- Rewriting historical docs (`docs/periods/*`, `CHANGELOG.md` body, `best_runs/**`, `gcp_logs/**`) — history stays as written; only banners/notes were added.
- Fixing the Foundry `pricing.py` cost gap, validating `grok`, ablation grid G2–G5, and Phase C — real engineering, tracked in `docs/open-issues.md`, each deserving its own spec.
- Ruff debt inside `backend/agents/rlm/skills/**` example scripts — cosmetic, example code.
- Deleting anything under `runs/`, `runs_logs/`, `gcp_logs/`, `_archive/` — hard rule, never.
