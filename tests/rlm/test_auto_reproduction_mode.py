"""AUTO reproduction mode (OPENRESEARCH_REPRODUCTION_MODE=auto) — per-paper
resolution of execute-vs-from-scratch, and the honesty ledger that keeps the two
evidence qualities distinguishable.

WHY THIS MODE EXISTS. The locked product strategy is execute-mode-first: run the
authors' published code behind a verified metrics shim rather than have the LLM
re-implement the paper (from-scratch SDAR scored 0.0; the authors' trainer 0.456).
But a great many papers publish NO code. With the mode hard-pinned to ``execute``,
``assert_execute_mode_stamped`` RAISES for those papers, so a no-code paper
hard-fails the run. For a needle-in-a-haystack triage funnel that is the single
worst error available: the screen tier's expensive mistake is a FALSE NEGATIVE —
discarding a paper that would in fact have reproduced.

``auto`` resolves the mode PER-PAPER at repo-resolution time in ``_build_context``:
  * usable author repo cloned  -> ``execute`` (the high-evidence path)
  * no repo / clone failed     -> ``scratch`` (a REAL from-scratch attempt)

The fallback is legitimate but must never be silent — a from-scratch result and an
execute-mode result are NOT the same evidence quality, and a downstream
patent-triage consumer has to tell them apart. So the fallback is disclosed three
ways: an ``execute_mode_no_repo`` run_warning, ``repo_spec.json``'s resolved
``mode`` + ``fallback_from_execute``, and the report's ``reproduction`` block +
``degradations_taken[]`` ledger.

Crucially, ``assert_execute_mode_stamped`` STAYS fail-loud for an EXPLICIT
``execute`` request that silently downgraded — that is the real lie it guards, and
these tests pin that it does not regress.
"""

from __future__ import annotations

import json

import pytest

import backend.agents.rlm.run as run_mod
from backend.agents.rlm.report import _build_reproduction_block, _collect_degradations
from backend.services.ingestion.repo.manifest import RepoManifest
from backend.services.ingestion.repo.resolver import RepoResolver

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _fail_clone(spec, dest):
    """RepoProvisioner.clone double: the clone did not produce a usable repo."""
    return None


def _ok_clone(spec, dest):
    """RepoProvisioner.clone double: a usable repo landed on disk."""
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "train.py").write_text("print('authors code')\n", encoding="utf-8")
    return RepoManifest(
        path=str(dest),
        commit_sha="a" * 40,
        file_tree=["train.py"],
        key_files={"train.py": "print('authors code')"},
        size_mb=0.01,
    )


@pytest.fixture
def repo_env(monkeypatch):
    """Hermetic repo-first env. The suite is not env-hermetic yet (backend/config.py
    reads the dev's real .env), so every var these paths read is set/cleared here
    explicitly rather than trusted from the ambient environment."""
    monkeypatch.setenv("OPENRESEARCH_USE_AUTHOR_REPO", "1")
    monkeypatch.delenv("OPENRESEARCH_REPRODUCTION_MODE", raising=False)
    monkeypatch.delenv("OPENRESEARCH_REPO_LOCAL_PATH", raising=False)
    monkeypatch.delenv("OPENRESEARCH_REPO_COMMIT", raising=False)
    return monkeypatch


def _project(tmp_path):
    p = tmp_path / "prj"
    p.mkdir()
    return p


def _saved_spec(project_dir) -> dict:
    return json.loads((project_dir / "rlm_state" / "repo_spec.json").read_text())


def _warning_codes(events) -> list[str]:
    return [e.get("code") for e in events if e.get("event") == "run_warning"]


def _write_events(project_dir, events) -> None:
    """Persist emitted events to dashboard_events.jsonl — the channel
    _collect_degradations actually reads."""
    with (project_dir / "dashboard_events.jsonl").open("w", encoding="utf-8") as fh:
        for e in events:
            fh.write(json.dumps(e) + "\n")


# ---------------------------------------------------------------------------
# 1. the pure resolver: auto is resolved, never passed through
# ---------------------------------------------------------------------------


def test_resolver_auto_with_repo_resolves_to_execute():
    spec = RepoResolver.resolve("github:me/mine", [], set(), "auto")
    assert spec.url == "https://github.com/me/mine"
    # The RESOLVED mode, not the request: "auto" must never reach repo_spec.json,
    # because every downstream consumer compares mode == "execute".
    assert spec.mode == "execute"


def test_resolver_auto_without_repo_resolves_to_scratch():
    spec = RepoResolver.resolve(None, [], set(), "auto")
    assert spec.url is None
    assert spec.mode == "scratch"


@pytest.mark.parametrize(
    ("override", "expected"),
    [(None, "adapt"), ("", "adapt"), ("adapt", "adapt"),
     ("reference", "reference"), ("execute", "execute")],
)
def test_resolver_existing_modes_unchanged(override, expected):
    """Byte-identical: every pre-existing override value still stamps exactly what
    it stamped before `auto` was added."""
    assert RepoResolver.resolve("github:me/mine", [], set(), override).mode == expected


# ---------------------------------------------------------------------------
# 2. auto + resolvable repo -> execute, no warning
# ---------------------------------------------------------------------------


def test_auto_with_resolvable_repo_resolves_to_execute(tmp_path, repo_env):
    repo_env.setenv("OPENRESEARCH_REPRODUCTION_MODE", "auto")
    repo_env.setattr(
        "backend.services.ingestion.repo.provisioner.RepoProvisioner.clone", _ok_clone,
    )
    project_dir = _project(tmp_path)
    events: list[dict] = []

    repo_files, spec = run_mod._resolve_and_clone_repo(
        project_dir, "github:me/mine", set(), [], emit=events.append,
    )

    assert repo_files is not None          # the authors' code is in context
    assert spec.mode == "execute"
    saved = _saved_spec(project_dir)
    assert saved["mode"] == "execute"      # ground truth for every downstream reader
    assert saved["clone_succeeded"] is True
    assert saved["requested_mode"] == "auto"
    assert saved["fallback_from_execute"] is False
    # Nothing degraded — no fallback warning.
    assert "execute_mode_no_repo" not in _warning_codes(events)


def test_auto_execute_path_does_not_raise_through_build_context(tmp_path, repo_env):
    """The whole point of routing the assertion through _build_context."""
    repo_env.setenv("OPENRESEARCH_REPRODUCTION_MODE", "auto")
    repo_env.setattr(
        "backend.services.ingestion.repo.provisioner.RepoProvisioner.clone", _ok_clone,
    )
    project_dir = _project(tmp_path)

    ctx = run_mod._build_context(
        {"entries": []}, project_dir=project_dir, repo_url="github:me/mine",
    )

    assert ctx["repo_files"] is not None
    assert _saved_spec(project_dir)["mode"] == "execute"


# ---------------------------------------------------------------------------
# 3. auto + NO repo -> from-scratch, disclosed, never raises
# ---------------------------------------------------------------------------


def test_auto_without_repo_does_not_raise_and_falls_back(tmp_path, repo_env):
    """THE BUG THIS MODE FIXES: a no-code paper must still get a real attempt."""
    repo_env.setenv("OPENRESEARCH_REPRODUCTION_MODE", "auto")
    project_dir = _project(tmp_path)
    events: list[dict] = []

    # No repo_url and no discovered artifacts => the resolver finds nothing.
    ctx = run_mod._build_context(
        {"entries": []}, project_dir=project_dir, repo_url=None, emit=events.append,
    )

    assert ctx["repo_files"] is None       # from-scratch: no authors' code in context
    saved = _saved_spec(project_dir)
    assert saved["mode"] == "scratch"      # resolved HONESTLY, not "execute"
    assert saved["requested_mode"] == "auto"
    assert saved["fallback_from_execute"] is True
    assert "execute_mode_no_repo" in _warning_codes(events)


def test_auto_with_failed_clone_falls_back_to_scratch(tmp_path, repo_env):
    """A repo was found but would not clone => still no usable author code."""
    repo_env.setenv("OPENRESEARCH_REPRODUCTION_MODE", "auto")
    repo_env.setattr(
        "backend.services.ingestion.repo.provisioner.RepoProvisioner.clone", _fail_clone,
    )
    project_dir = _project(tmp_path)
    events: list[dict] = []

    run_mod._build_context(
        {"entries": []}, project_dir=project_dir, repo_url="github:me/mine",
        emit=events.append,
    )

    saved = _saved_spec(project_dir)
    assert saved["mode"] == "scratch"
    assert saved["fallback_from_execute"] is True
    assert saved["clone_succeeded"] is False
    codes = _warning_codes(events)
    assert "execute_mode_no_repo" in codes
    # The explicit-execute semantics ("will not silently reimplement") do NOT apply
    # to auto — auto's contract is precisely to reimplement, and to say so.
    assert "repo_execute_unavailable" not in codes


def test_auto_fallback_warning_names_the_evidence_gap(tmp_path, repo_env):
    """The warning must state that the authors' code did NOT run — that is the fact
    a triage consumer needs, not merely that 'something degraded'."""
    repo_env.setenv("OPENRESEARCH_REPRODUCTION_MODE", "auto")
    project_dir = _project(tmp_path)
    events: list[dict] = []

    run_mod._resolve_and_clone_repo(project_dir, None, set(), [], emit=events.append)

    msg = next(
        e["message"] for e in events
        if e.get("event") == "run_warning" and e.get("code") == "execute_mode_no_repo"
    )
    assert "from-scratch" in msg.lower()
    assert "no usable author repo" in msg.lower()


# ---------------------------------------------------------------------------
# 4. the fail-loud backstop is PRESERVED for an explicit `execute`
# ---------------------------------------------------------------------------


def test_explicit_execute_with_no_repo_still_raises(tmp_path, repo_env):
    """DO NOT REGRESS. An explicit `execute` that silently became scratch is a LIE
    about what ran; it must keep hard-failing. `auto` is the escape hatch, not a
    weakening of this backstop."""
    repo_env.setenv("OPENRESEARCH_REPRODUCTION_MODE", "execute")
    project_dir = _project(tmp_path)

    with pytest.raises(RuntimeError, match="execute"):
        run_mod._build_context(
            {"entries": []}, project_dir=project_dir, repo_url=None,
        )


def test_assert_execute_mode_stamped_still_raises_on_silent_downgrade():
    with pytest.raises(RuntimeError):
        run_mod.assert_execute_mode_stamped("execute", {"mode": "scratch"})
    with pytest.raises(RuntimeError):
        run_mod.assert_execute_mode_stamped("execute", None)


def test_assert_execute_mode_stamped_exempts_auto():
    """An `auto` run that honestly resolved to scratch is a DISCLOSED outcome, not a
    silent downgrade — the backstop must not fire on it."""
    run_mod.assert_execute_mode_stamped("auto", {"mode": "scratch"})
    run_mod.assert_execute_mode_stamped("auto", {"mode": "execute"})
    run_mod.assert_execute_mode_stamped("auto", None)


@pytest.mark.parametrize("mode", ["", "adapt", "reference"])
def test_assert_execute_mode_stamped_noop_for_other_modes(mode):
    run_mod.assert_execute_mode_stamped(mode, {"mode": "scratch"})


# ---------------------------------------------------------------------------
# 5. byte-identical off-state for the pre-existing modes
# ---------------------------------------------------------------------------


def test_adapt_no_repo_writes_no_disclosure_keys(tmp_path, repo_env):
    """Unset/adapt: repo_spec.json must not grow the auto-only disclosure keys."""
    project_dir = _project(tmp_path)

    run_mod._resolve_and_clone_repo(project_dir, None, set(), [], emit=None)

    saved = _saved_spec(project_dir)
    assert saved["mode"] == "scratch"
    assert "requested_mode" not in saved
    assert "fallback_from_execute" not in saved


def test_adapt_clone_failure_still_downgrades_with_legacy_warning(tmp_path, repo_env):
    repo_env.setattr(
        "backend.services.ingestion.repo.provisioner.RepoProvisioner.clone", _fail_clone,
    )
    project_dir = _project(tmp_path)
    events: list[dict] = []

    _, spec = run_mod._resolve_and_clone_repo(
        project_dir, "github:me/mine", set(), [], emit=events.append,
    )

    assert spec.mode == "scratch"
    codes = _warning_codes(events)
    assert "repo_clone_failed" in codes            # the pre-existing warning
    assert "execute_mode_no_repo" not in codes     # auto-only, must not leak


def test_explicit_execute_clone_failure_keeps_execute_and_legacy_warning(tmp_path, repo_env):
    """The existing execute-mode clone-failure contract is untouched by `auto`."""
    repo_env.setenv("OPENRESEARCH_REPRODUCTION_MODE", "execute")
    repo_env.setattr(
        "backend.services.ingestion.repo.provisioner.RepoProvisioner.clone", _fail_clone,
    )
    project_dir = _project(tmp_path)
    events: list[dict] = []

    _, spec = run_mod._resolve_and_clone_repo(
        project_dir, "github:me/mine", set(), [], emit=events.append,
    )

    assert spec.mode == "execute"                  # NOT downgraded
    codes = _warning_codes(events)
    assert "repo_execute_unavailable" in codes
    assert "execute_mode_no_repo" not in codes


# ---------------------------------------------------------------------------
# 6. report honesty — the two evidence qualities stay distinguishable
# ---------------------------------------------------------------------------


def test_report_reproduction_block_carries_resolved_execute_mode(tmp_path, repo_env):
    repo_env.setenv("OPENRESEARCH_REPRODUCTION_MODE", "auto")
    repo_env.setattr(
        "backend.services.ingestion.repo.provisioner.RepoProvisioner.clone", _ok_clone,
    )
    project_dir = _project(tmp_path)
    run_mod._resolve_and_clone_repo(project_dir, "github:me/mine", set(), [], emit=None)

    block = _build_reproduction_block(project_dir)

    assert block is not None
    assert block["mode"] == "execute"              # the RESOLVED mode
    assert block["requested_mode"] == "auto"
    assert block["repo_url"] == "https://github.com/me/mine"
    assert "fallback" not in block                 # nothing fell back


def test_report_reproduction_block_discloses_from_scratch_fallback(tmp_path, repo_env):
    """A triage consumer reading final_report.json must be able to see that the
    authors' code never ran — otherwise a from-scratch score is silently graded as
    if it were an execute-mode reproduction."""
    repo_env.setenv("OPENRESEARCH_REPRODUCTION_MODE", "auto")
    project_dir = _project(tmp_path)
    run_mod._resolve_and_clone_repo(project_dir, None, set(), [], emit=None)

    block = _build_reproduction_block(project_dir)

    assert block is not None, "a from-scratch fallback must still emit a reproduction block"
    assert block["mode"] == "scratch"             # NOT "execute"
    assert block["requested_mode"] == "auto"
    assert block["fallback"]["from"] == "execute"
    assert block["fallback"]["to"] == "scratch"
    assert "did NOT run" in block["fallback"]["evidence_note"]


def test_fallback_appears_in_the_degradations_ledger(tmp_path, repo_env):
    """A fallback IS a degradation — of evidence quality — so it belongs in the
    ledger the report already ships."""
    repo_env.setenv("OPENRESEARCH_REPRODUCTION_MODE", "auto")
    project_dir = _project(tmp_path)
    events: list[dict] = []
    run_mod._resolve_and_clone_repo(project_dir, None, set(), [], emit=events.append)
    _write_events(project_dir, events)

    ledger = _collect_degradations(project_dir)

    codes = [row["code"] for row in ledger]
    assert "execute_mode_no_repo" in codes
    row = next(r for r in ledger if r["code"] == "execute_mode_no_repo")
    assert row["count"] == 1
    assert row["last_message"]


def test_clean_execute_run_has_no_degradation_entry(tmp_path, repo_env):
    repo_env.setenv("OPENRESEARCH_REPRODUCTION_MODE", "auto")
    repo_env.setattr(
        "backend.services.ingestion.repo.provisioner.RepoProvisioner.clone", _ok_clone,
    )
    project_dir = _project(tmp_path)
    events: list[dict] = []
    run_mod._resolve_and_clone_repo(project_dir, "github:me/mine", set(), [], emit=events.append)
    _write_events(project_dir, events)

    codes = [row["code"] for row in _collect_degradations(project_dir)]
    assert "execute_mode_no_repo" not in codes


def test_adapt_run_with_no_repo_still_has_no_reproduction_block(tmp_path, repo_env):
    """Byte-identical: a non-auto repo-less run keeps emitting NO reproduction block."""
    project_dir = _project(tmp_path)
    run_mod._resolve_and_clone_repo(project_dir, None, set(), [], emit=None)

    assert _build_reproduction_block(project_dir) is None
