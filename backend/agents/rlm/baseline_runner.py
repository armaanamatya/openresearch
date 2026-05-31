"""Isolated-subprocess runner for the baseline code-writing SDK call.

The claude-agent-sdk's ``_process_query_inner`` async generator can hit
``RuntimeError: aclose(): asynchronous generator is already running`` when
``asyncio.run`` tears down its loop — and once that happens **in-process** it
poisons the whole reproduction process, so every subsequent ``implement_baseline``
retry fails too (the 2026-05-30 SDAR run died 8/8 iterations this way, never
writing a single ``train.py``).

This module is the child entrypoint run via ``multiprocessing`` (spawn): it runs
``run_with_sdk`` in a *fresh* interpreter + event loop, so an ``aclose()`` race
crashes only the child (clean non-zero exit) and the parent stays pristine — the
next attempt spawns a brand-new child. The child writes the generated code to
disk exactly as the in-process path does; it returns only the three fields the
parent reads (``commands_to_run`` / ``diff_summary`` / ``assumptions_applied``)
via the result queue, and signals SDK-stream liveness by touching a heartbeat
file the parent polls (the parent's stall watchdog reads its mtime).

Kept deliberately lean so ``spawn`` re-imports little in the child.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def run_baseline_in_child(
    result_q: Any,
    heartbeat_path: str,
    project_id: str,
    runs_root: str,
    pcm: Any,
    env: Any,
    contract: Any,
    artifact_index: Any,
    kwargs: dict,
) -> None:
    """Child process body: run ``run_with_sdk`` and put a small result dict on the queue.

    Never raises out of the process — ANY failure (including the SDK aclose race
    or an import error) is reported on the queue as ``{"ok": False, ...}`` so the
    parent can classify it as a repairable failure and retry with a fresh child.
    """
    import asyncio

    hb = Path(heartbeat_path)

    def _on_event() -> None:
        try:
            hb.touch()
        except Exception:  # noqa: BLE001 — liveness is best-effort
            pass

    try:
        from backend.agents.baseline_implementation import run_with_sdk

        res = asyncio.run(
            run_with_sdk(
                project_id,
                Path(runs_root),
                pcm,
                env,
                contract,
                artifact_index,
                runtime=None,  # rebuilt from env in this fresh process
                on_event=_on_event,
                **kwargs,
            )
        )
        result_q.put({
            "ok": True,
            "commands_to_run": list(getattr(res, "commands_to_run", []) or []),
            "diff_summary": getattr(res, "diff_summary", "") or "",
            "assumptions_applied": list(getattr(res, "assumptions_applied", []) or []),
        })
    except BaseException as exc:  # noqa: BLE001 — report everything, incl. the aclose race
        import traceback

        try:
            result_q.put({
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[:4000],
            })
        except Exception:  # noqa: BLE001 — queue may be broken if the loop is wrecked
            pass
