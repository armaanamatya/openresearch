<!-- doc-meta: status=current; last-verified=2026-07-06 -->
# Handoff — Merge the relevance-gated skill selection with the autonomous-upload UI branch

> **Date:** 2026-07-06 · **Status:** Current · self-contained action doc for landing **both** the
> skill-selection work (`reconcile/…`) **and** the autonomous-upload UI (`feat/autonomous-upload-ui`)
> onto `deepinvent/main`, in the right order, with the conflict surface pre-mapped.

## 0. TL;DR

The two workstreams do **not** share a base — `feat/autonomous-upload-ui` forked at `ddc82113`
(**before** the Release-1 skill library merged) and independently re-built a near-identical
Anthropic-Foundry provider layer. So a *direct* branch-to-branch merge is messy (the UI branch lacks
the entire Release-1 + reconcile superset). **Do not merge the two feature branches into each other.**

**Land them onto `deepinvent/main` in sequence instead:**

1. **PR-A — `reconcile/…` → `deepinvent/main`** (the 2 skill commits). **Verified conflict-free /
   fast-forwardable.** Skills + prereqs land clean.
2. **PR-B — bring `feat/autonomous-upload-ui` up to `deepinvent/main`, then PR it.** The conflicts are
   a **bounded 9-file cluster** in the Anthropic-Foundry layer + `run.py` + the run-spec + their tests
   — resolved by a simple rule: **take main's canonical Foundry/backbone, keep the UI branch's
   frontend + `spec_validator`.** My skill files (`skill_selection.py`, `leaf_scorer.py`,
   `primitives.py` hooks) **auto-merge clean** — they are not part of the conflict surface.

Everything default-OFF / byte-identical when off; author `lolout1 <appradhann@gmail.com>`; no
`Co-Authored-By` trailer; push only to `deepinvent` on request.

## 1. Branch topology (verified 2026-07-06)

| Ref | SHA | Note |
|---|---|---|
| `deepinvent/main` | `62ae4f73` | canonical mainline — has Anthropic-Foundry (`5daaff5b`) + Release-1 skills (`8f4944bf`) + lean CLAUDE.md (`14674f4e`) |
| `reconcile/grounded-self-improvement-on-main` | `cbcd75af` | **2 commits ahead of main** — the skill-selection work (below) |
| `feat/autonomous-upload-ui` | `846d4e06` | UI/UX branch (worktree `/home/abheekp/openresearch-autonomous-ui`), **pushed to `deepinvent/feat/autonomous-upload-ui`**, not merged |
| **merge-base**(reconcile, UI) | `ddc82113` | shared fork point — **BEFORE** `5daaff5b`/`8f4944bf` |

```
ddc82113 ──┬── 5daaff5b (Foundry) ── 8f4944bf (skills R1) ── 14674f4e ── [deepinvent/main 62ae4f73]
           │                                                                 ├── ed083363 (skill selection)
           │                                                                 └── cbcd75af (SDAR prereqs + run-spec skill flags)   ← reconcile HEAD
           │
           └── ee311db3 … 846d4e06   ← feat/autonomous-upload-ui  (its OWN parallel Foundry impl; NO Release-1, NO skill selection)
```

- `deepinvent/main` **is** an ancestor of `reconcile` (2 ahead) but **is NOT** an ancestor of the UI
  branch (`git merge-base --is-ancestor deepinvent/main feat/autonomous-upload-ui` → false).
- The canonical Foundry commit `5daaff5b` is **NOT** an ancestor of the UI branch — it carries its
  own parallel implementation (root of the add/add conflicts). The two impls are *close cousins*, not
  rewrites: `_anthropic_foundry_patch.py` diff(main..UI) = `+0/-9`, `foundry_anthropic.py` = `+12/-16`.

## 2. What each side brings

**`reconcile/…` (the 2 commits to land as PR-A):**
- `ed083363` — relevance-gated skill selection: new `backend/agents/rlm/skill_selection.py`
  (deterministic recall reuse + bounded LLM prune → `active_skills.json`), wired into
  `detect_environment` (the one trigger, no new primitive — count stays **19**), `consult_skill()`
  index focus, and the verifier via `leaf_scorer.score_reproduction(skill_context=…)`. Flag
  `OPENRESEARCH_SKILL_SELECT` (+ `_DETERMINISTIC`/`_CANDIDATES_MAX`/`_VERIFIER_BODIES`), default-OFF.
  26 new tests. Spec: `docs/superpowers/specs/2026-07-06-relevance-gated-skill-activation-design.md`.
- `cbcd75af` — SDAR execute-mode prereqs: `run.py` Foundry beta-header disable (+ the matching
  `test_coresidency_guard` update), `provisioner.py` symlink-preserve, and
  `configs/sdar_execute_run_spec.json` (REPO_COMMIT re-pin + `OPENRESEARCH_SKILLS`/`_SKILL_SELECT=1`).

**`feat/autonomous-upload-ui` (PR-B):** the upload→live-reproduction UI (neo-brutalist kit,
paper-landing/repo-confirm, spec-validation stepper, live agentic-reasoning session view, session
rail) — **56 clean-add `frontend/` files** — plus a backend `spec_validator` subsystem
(`spec_validator.py`, the `spec_validator` role, 4 SSE event builders, `run_pipeline_rlm` pre-loop
gate + report stamp, `GET /papers/{id}/repo`) and its **own** parallel Anthropic-Foundry layer.
`spec_validator.py` is **not on main** → it adds cleanly.

## 3. Conflict surface (verified via `git merge-tree`)

Merging the UI branch against `deepinvent/main` yields conflicts in exactly these **9 files** — all
in the Foundry / backbone layer, none in the skill or UI-frontend layers:

| File | Conflict kind | Resolution |
|---|---|---|
| `backend/agents/rlm/_anthropic_foundry_patch.py` | add/add | **Take main's** (canonical; UI's is a `-9`-line subset) |
| `backend/agents/runtime/foundry_anthropic.py` | add/add | **Take main's** (UI diverges only `+12/-16`; re-apply any genuine UI fix by hand) |
| `backend/agents/rlm/run.py` | content | **Layer UI's `spec_validator` pre-loop gate onto main's `run.py`** (main already carries the reconcile superset + the `cbcd75af` Foundry subprocess-env). The biggest manual merge — see §6. |
| `configs/sdar_execute_run_spec.json` | content | **Union the keys** — keep main's skill flags + REPO_COMMIT AND the UI branch's autonomous-profile additions (disjoint keys) |
| `tests/agents/rlm/test_anthropic_foundry_patch.py` | add/add | **Take main's** |
| `tests/agents/runtime/test_foundry_anthropic.py` | add/add | **Take main's** |
| `tests/rlm/test_coresidency_guard.py` | add/add | **Take main's** (has the `cbcd75af` subprocess-env assertion) |
| `tests/rlm/test_grader_transport_anthropic_foundry.py` | add/add | **Take main's** |
| `tests/services/context/workspace/tools/test_anthropic_messages_client_base_url.py` | add/add | **Take main's** |

**Auto-merges clean (no action):** `primitives.py` (my skill hooks vs UI's minor edits),
`grader_transport.py`, `report.py`, `role_models.py`. The skill-selection files
(`skill_selection.py`, `leaf_scorer.py`, `tests/rlm/test_skill_selection.py`) are **not** in the
surface at all.

**Root cause:** both branches built the Anthropic-Foundry foundation independently off `ddc82113`.
Resolving *toward main's canonical copy* collapses the whole cluster — the UI branch's Foundry code
is superseded by main's, so its parallel Foundry commits are effectively dropped, keeping only its
UI + `spec_validator` work.

## 4. Recommended strategy — main-first, two PRs

Why not a direct branch merge: the UI branch is ~295 commits behind main (forked pre-Release-1); a
direct merge drags every main-vs-UI delta (299 backend files differ, mostly main's work *missing*
from UI) into one conflicted diff. Rebasing/merging **onto main** instead means git only has to
reconcile the UI branch's *own* additions against the canonical base — the bounded 9-file cluster
above. This also matches the established cadence (PRs #4–7 landed reconcile onto `deepinvent/main`;
the UI branch is the next feature PR).

## 5. Step-by-step

### 5.1 PR-A — land the skill selection (conflict-free)
```bash
cd /home/abheekp/openresearch                      # reconcile worktree, HEAD cbcd75af
git status --short | grep -vE '^\?\? runs/'         # confirm the OUT-list (external-runs, app.py,
                                                    #   config.py, lab-sidebar, loose configs, logs)
                                                    #   is NOT staged — it stays out (see the
                                                    #   2026-07-05 reconcile handoff §3)
git push deepinvent reconcile/grounded-self-improvement-on-main
gh pr create --repo Deepinvent/scientific_article_generator \
  --base main --head reconcile/grounded-self-improvement-on-main \
  --title "Relevance-gated skill selection + SDAR execute-mode prereqs" \
  --body-file docs/superpowers/specs/2026-07-06-relevance-gated-skill-activation-design.md
```
Verified `git merge-tree --write-tree deepinvent/main cbcd75af` → **no conflicts**. Merge-commit
strategy (preserves the flag-by-flag trail the A/B gate references).

### 5.2 PR-B — update the UI branch onto the new main, then PR
Do this in the UI worktree so `reconcile` stays untouched:
```bash
cd /home/abheekp/openresearch-autonomous-ui        # feat/autonomous-upload-ui, HEAD 846d4e06
git fetch deepinvent
git merge deepinvent/main                           # brings Release-1 + skills + skill selection in
# → resolve the 9-file cluster per §3/§6, then:
git commit                                          # the merge commit (author lolout1, no trailer)
```
(Prefer `merge` over `rebase` here — a rebase would replay the UI branch's parallel-Foundry commits
one-by-one and re-conflict each; a single merge resolves the cluster once.)

### 5.3 Verify (both worktrees, before each PR)
```bash
.venv/bin/python -m pytest tests/ -q                # expect only the known host/env pre-existing
                                                    #   failures (AZURE_FOUNDRY .env creds, missing
                                                    #   tesseract, the parallel external-validator
                                                    #   env-leak) — zero NEW failures
.venv/bin/python -m pytest tests/rlm/test_skill_selection.py tests/rlm/test_registry.py \
    tests/test_claude_md_fidelity.py -q             # skills intact + primitive count == 19
uvx ruff@0.15.16 check backend/ tests/
(cd frontend && npm ci && npm run build && npx tsc --noEmit && npm test)   # UI branch only
```

### 5.4 Open PR-B
```bash
git push deepinvent feat/autonomous-upload-ui
gh pr create --repo Deepinvent/scientific_article_generator \
  --base main --head feat/autonomous-upload-ui \
  --title "Autonomous-upload UI + live-reproduction session view + spec_validator" \
  --body-file docs/superpowers/specs/2026-07-05-autonomous-upload-ui-and-live-reproduction-design.md
```

## 6. Conflict-resolution playbook (the `run.py` merge — the only real work)

`run.py` is the one file needing thought (the rest are "take main"). Both sides changed it:
- **main** already has: the reconcile superset + the `cbcd75af` Foundry-executor `subprocess_env`
  (`CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS`/`_ADVISOR_TOOL`) inside `_resolve_agent_runtime`.
- **UI** adds: the `spec_validator` **pre-loop gate** wired into `run_pipeline_rlm` (`+602/-21`
  lines), plus its own parallel Foundry-runtime resolution.

Resolution: **base the merged `run.py` on main's copy**, then **re-apply only the UI branch's
`spec_validator` additions** on top:
- Keep main's `_resolve_agent_runtime` (with the subprocess-env block) verbatim — discard the UI
  branch's parallel Foundry-runtime edits (superseded).
- Port the UI branch's `spec_validator` pre-loop gate + SSE emits + report stamp into
  `run_pipeline_rlm` at main's current structure. Cross-check it still calls the canonical
  `build_spec_validator_client` and reads the `spec_validator` role from the resolver.
- After resolving, grep for the tell-tales on both sides so nothing is dropped:
  `git grep -n "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS" run.py` (main's fix survived) **and**
  `git grep -n "spec_validator" backend/agents/rlm/run.py` (UI's gate survived).

For `configs/sdar_execute_run_spec.json`: the two edits touch **disjoint keys** — keep BOTH the
`OPENRESEARCH_SKILLS`/`_SKILL_SELECT`/`REPO_COMMIT` (main) and the autonomous-profile keys (UI).

## 7. Verification gates (must pass before PR-B merges)
- `test_claude_md_fidelity.py`: `PRIMITIVE_REGISTRY` == **19** and the doc says so (neither branch
  adds a primitive; the merge must not accidentally duplicate one).
- `test_skill_selection.py` (26) + `test_consult_skill.py` green — skills survive the merge.
- `test_coresidency_guard.py` green — main's subprocess-env assertion survived the `run.py` merge.
- `spec_validator` tests + the `frontend/` build/test — UI survives.
- Off-state sweep: with all `OPENRESEARCH_SKILL*` / `OPENRESEARCH_SPEC_VALIDATOR*` flags unset, the
  backend is byte-identical to today.

## 8. What stays OUT / risks
- **External-runs workstream stays OUT** of both PRs (uncommitted `app.py`/`config.py`/
  `lab-sidebar.tsx` + `backend/routes/external_runs.py` + `backend/services/external_monitor/` +
  `frontend/src/{app/api/external-runs,app/external-runs,components/lab/external,lib/external-runs}/`
  + `configs/external_runs.json.example`), per the 2026-07-05 reconcile handoff §3. Its own branch
  later.
- **Do not `git add -A`.** Stage explicit paths (both branches carry unrelated uncommitted trees +
  ~136 untracked `runs/` dirs).
- **Risk — silently dropping a UI Foundry fix.** The UI branch's `foundry_anthropic.py` diverges
  `+12/-16` from canonical; "take main" is correct for the *structure*, but eyeball the `+12` for any
  genuine bugfix worth porting before discarding.
- **Sandbox lock:** the autonomous UI profile is **gcp-not-gke** (per memory
  `project_autonomous_upload_ui`) — keep that in the merged run-spec/profile.

## 9. Author / contributor check (run before each push)
```bash
git log -3 --format='%h  %an <%ae>'         # every commit: lolout1 <appradhann@gmail.com>
git log deepinvent/main..HEAD --format='%B' | grep -i 'co-authored-by' && echo "TRAILER — remove" || echo "clean"
```
Fix a stray author with `git commit --amend --reset-author` (local config only — never
`-c user.email=…`).

## 10. Cross-references
- Skill-selection design + implementation notes: `docs/superpowers/specs/2026-07-06-relevance-gated-skill-activation-design.md` (§11).
- UI design + plan: `docs/superpowers/specs/2026-07-05-autonomous-upload-ui-and-live-reproduction-design.md`, `docs/superpowers/plans/2026-07-05-autonomous-upload-ui-implementation-plan.md`.
- The prior reconcile→main merge cadence + OUT-list: `docs/runbooks/2026-07-05-reconcile-to-main-merge-and-pr-handoff.md`.
- SDAR run readiness (post-merge, separate): the skills-ON A/B is staged on `reconcile` — VM
  `sdar-2model-a` (4×A100-80GB, us-central1-a), launch `scripts/sdar_phase1_foundry.sh`; OFF-arm
  reference `0.456` is the **authors' verl trainer** (Track B, `docs/audits/2026-07-04-sdar-gcp-runs-log-analysis.md`), not a harness run.
