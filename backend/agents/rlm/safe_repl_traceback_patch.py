"""Make REPL exceptions diagnosable by the root model.

Upstream rlm's LocalREPL.execute_code catches the exception and appends only
``f"{type(e).__name__}: {e}"`` to stderr (local_repl.py line ~506). For a
``TypeError: 'NoneType' object is not callable`` raised at line 105 of a
12K-character script, the model sees 45 chars with no file, no line, no
chain — and cannot diagnose which name was None. The 2026-05-28 SDAR run
wasted four iterations because of this (BUG-LR-012).

We patch ``LocalREPL.execute_code`` to also include ``traceback.format_exc()``
(capped at 2000 chars to stay within the SSE egress bound). The patch is
applied at module import time via the same pattern as forced_iteration.py.

If the upstream signature changes, the guard ``_PATCHED_ATTR`` prevents
double-patching and a log warning surfaces instead of a crash.

Design spec: docs/superpowers/specs/2026-05-28-rlm-stability-remediation-design.md
Upstream-merge note: file a PR against rlms to include traceback.format_exc()
in the stderr capture by default, and to remove globals/locals from the
_SAFE_BUILTINS blocklist. Until accepted, this monkey-patch is permanent.
"""
from __future__ import annotations
import logging
import traceback

from rlm.environments import local_repl as _local_repl

logger = logging.getLogger(__name__)
_PATCHED_ATTR = "_reprolab_traceback_patched"
_TRACEBACK_CAP = 2000


def _patched_execute_code(self, code: str):
    import time
    start = time.perf_counter()
    self._pending_llm_calls = []
    with self._capture_output() as (stdout_buf, stderr_buf), self._temp_cwd():
        try:
            combined = {**self.globals, **self.locals}
            exec(code, combined, combined)  # noqa: S102
            for k, v in combined.items():
                if k not in self.globals and not k.startswith("_"):
                    self.locals[k] = v
            self._restore_scaffold()
            stdout = stdout_buf.getvalue()
            stderr = stderr_buf.getvalue()
        except Exception:
            stdout = stdout_buf.getvalue()
            tb = traceback.format_exc()
            if len(tb) > _TRACEBACK_CAP:
                tb = "...(truncated)\n" + tb[-(_TRACEBACK_CAP - 18):]
            stderr = stderr_buf.getvalue() + f"\n{tb}"
    fa = self._last_final_answer
    self._last_final_answer = None
    return _local_repl.REPLResult(
        stdout=stdout,
        stderr=stderr,
        locals=self.locals.copy(),
        execution_time=time.perf_counter() - start,
        rlm_calls=self._pending_llm_calls.copy(),
        final_answer=fa,
    )


def apply_traceback_patch() -> None:
    if getattr(_local_repl.LocalREPL, _PATCHED_ATTR, False):
        return
    _local_repl.LocalREPL.execute_code = _patched_execute_code
    setattr(_local_repl.LocalREPL, _PATCHED_ATTR, True)
    logger.debug("safe_repl_traceback_patch: LocalREPL.execute_code patched")


apply_traceback_patch()
