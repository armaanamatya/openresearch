"""EXECUTE mode (OPENRESEARCH_REPRODUCTION_MODE=execute) — the third repo-first
mode alongside adapt/reference. Unlike adapt (which seeds the repo into code/
and tells the implementer to fix it up) and reference (clone+read, reimplement
from scratch), execute seeds the repo into code/ *and* instructs the
implementer to run the authors' pipeline verbatim behind a thin harness shim.

This file pins the one seam #62 didn't already cover for reference mode: the
mode -> implementer-context mapping in `baseline_implementation.run_with_sdk`.
Seeding behavior and resolver normalization are covered in
tests/rlm/test_implement_baseline_repo.py and
tests/services/ingestion/repo/test_resolver.py respectively.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from backend.agents.baseline_implementation import run_with_sdk
from backend.agents.schemas import EnvironmentSpec, PaperClaimMap


def _capture_prompt(monkeypatch):
    """Patch collect_agent_text to capture the prompt argument; return the
    captured-list so the test can inspect what would have been sent."""
    captured: list[dict] = []

    async def _fake_collect(agent_name, prompt, **kwargs):
        captured.append({"agent": agent_name, "prompt": prompt})
        return ""

    # collect_agent_text is lazy-imported inside run_with_sdk, so patch at its
    # source module (backend.agents.runtime.invoke), not at the consumer.
    monkeypatch.setattr(
        "backend.agents.runtime.invoke.collect_agent_text",
        _fake_collect,
    )
    return captured


def _minimal_inputs(tmp_path: Path):
    """Build the minimum object set run_with_sdk needs."""
    runs_root = tmp_path / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    (runs_root / "prj_test").mkdir(parents=True, exist_ok=True)
    (runs_root / "prj_test" / "code").mkdir(parents=True, exist_ok=True)
    pcm = PaperClaimMap(core_contribution="test paper")
    env = EnvironmentSpec(dockerfile="FROM python:3.11", framework="pytorch")
    contract = None
    return runs_root, pcm, env, contract


def test_execute_mode_adds_execute_repo_note(tmp_path, monkeypatch):
    captured = _capture_prompt(monkeypatch)
    runs_root, pcm, env, contract = _minimal_inputs(tmp_path)
    asyncio.run(run_with_sdk(
        "prj_test", runs_root, pcm, env, contract,
        artifact_index={"mode": "execute", "repo_url": "https://github.com/x/y"},
    ))
    prompt = captured[0]["prompt"]
    assert "execute_repo_note" in prompt
    assert "EXECUTE mode" in prompt
    assert "Do NOT reimplement the method" in prompt
    # reference-mode's note must not leak into an execute-mode prompt
    assert "reference_repo_note" not in prompt


def test_reference_mode_still_adds_reference_repo_note_not_execute(tmp_path, monkeypatch):
    """Byte-identical-to-#62 regression: reference mode's existing behavior
    must be untouched by the new execute branch."""
    captured = _capture_prompt(monkeypatch)
    runs_root, pcm, env, contract = _minimal_inputs(tmp_path)
    asyncio.run(run_with_sdk(
        "prj_test", runs_root, pcm, env, contract,
        artifact_index={"mode": "reference", "repo_url": "https://github.com/x/y"},
    ))
    prompt = captured[0]["prompt"]
    assert "reference_repo_note" in prompt
    assert "execute_repo_note" not in prompt


def test_adapt_mode_adds_neither_note(tmp_path, monkeypatch):
    captured = _capture_prompt(monkeypatch)
    runs_root, pcm, env, contract = _minimal_inputs(tmp_path)
    asyncio.run(run_with_sdk(
        "prj_test", runs_root, pcm, env, contract,
        artifact_index={"mode": "adapt", "repo_url": "https://github.com/x/y"},
    ))
    prompt = captured[0]["prompt"]
    assert "reference_repo_note" not in prompt
    assert "execute_repo_note" not in prompt


def test_no_artifact_index_adds_neither_note(tmp_path, monkeypatch):
    """Flag-off / no-repo-run byte-identical baseline: an absent/empty
    artifact_index must not add either repo-mode note."""
    captured = _capture_prompt(monkeypatch)
    runs_root, pcm, env, contract = _minimal_inputs(tmp_path)
    asyncio.run(run_with_sdk("prj_test", runs_root, pcm, env, contract))
    prompt = captured[0]["prompt"]
    assert "reference_repo_note" not in prompt
    assert "execute_repo_note" not in prompt
