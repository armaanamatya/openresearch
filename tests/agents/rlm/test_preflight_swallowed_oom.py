"""Tests for the swallowed-backward-OOM pre-flight check.

Covers ``preflight_ast._check_swallowed_backward_oom`` (BES Phase 2 Component C,
spec ``docs/superpowers/specs/2026-06-07-bes-integration/phase-2-preflight-retry-reduction.md``
§4) and its wiring into the public ``scan_code_dir`` API.

The anti-pattern: a training step wraps ``loss.backward()`` / ``optimizer.step()``
in a ``try`` whose handler catches the OOM (``RuntimeError`` /
``torch.cuda.OutOfMemoryError`` / bare ``Exception``) and SKIPS the step
(``continue`` / ``pass`` / ``break``) WITHOUT re-raising or shrinking a batch/scale
variable. The script swallows the OOM and exits 0 — today caught only POSTflight by
a log scan. This static check surfaces it before sandbox dispatch.

FP-tight: the legitimate catch-OOM-shrink-and-retry pattern (handler re-raises OR
mutates a batch/scale variable) MUST NOT flag. The check is always ``soft`` (never
short-circuits the run) per the spec.
"""
from __future__ import annotations

import ast
from pathlib import Path

from backend.agents.rlm.preflight_ast import (
    PreflightViolation,
    _check_swallowed_backward_oom,
    scan_code_dir,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(src: str) -> list[PreflightViolation]:
    """Parse ``src`` and run the swallowed-OOM check, returning its violations."""
    tree = ast.parse(src)
    out: list[PreflightViolation] = []
    _check_swallowed_backward_oom(tree, Path("train_cell.py"), out)
    return out


def _swallow_hits(vs: list[PreflightViolation]) -> list[PreflightViolation]:
    """The swallowed-OOM check is the only one emitting this soft detail shape."""
    return [v for v in vs if "swallowed-OOM" in v.detail or "swallows the OOM" in v.detail]


def _write(code_dir: Path, name: str, body: str) -> Path:
    p = code_dir / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Case 1: try: loss.backward() except RuntimeError: continue  → FLAGS (soft)
# ---------------------------------------------------------------------------


def test_swallow_and_skip_flags_soft() -> None:
    out = _run("""\
def train_step(loss, optimizer):
    try:
        loss.backward()
        optimizer.step()
    except RuntimeError:
        continue
""")
    assert len(out) == 1, f"Expected exactly one swallow violation, got: {out}"
    v = out[0]
    assert v.severity == "soft", f"Component C is soft-only, got {v.severity!r}"
    assert v.file == "train_cell.py"
    assert v.line > 0
    assert "RuntimeError" in v.suggested_fix
    assert "swallow" in v.detail.lower() or "silent_oom" in v.detail


def test_bare_except_continue_flags() -> None:
    """A bare ``except:`` around backward that just continues is the broad-catch
    swallow — flagged soft."""
    out = _run("""\
def train_step(loss):
    try:
        loss.backward()
    except:
        continue
""")
    hits = _swallow_hits(out)
    assert len(hits) == 1 and hits[0].severity == "soft", out


def test_oom_error_pass_flags() -> None:
    """``except torch.cuda.OutOfMemoryError: pass`` (attribute type) → flagged."""
    out = _run("""\
def train_step(loss):
    try:
        loss.backward()
    except torch.cuda.OutOfMemoryError:
        pass
""")
    hits = _swallow_hits(out)
    assert len(hits) == 1 and hits[0].severity == "soft", out


# ---------------------------------------------------------------------------
# Case 2: try: loss.backward() except RuntimeError: batch_size //= 2; continue
#         → does NOT flag (shrink-and-retry, the FP guard)
# ---------------------------------------------------------------------------


def test_shrink_and_retry_does_not_flag() -> None:
    out = _run("""\
def train_step(loss, batch_size):
    try:
        loss.backward()
    except RuntimeError:
        batch_size //= 2
        continue
""")
    assert out == [], f"Shrink-and-retry must NOT flag (FP guard): {out}"


def test_scale_attr_shrink_does_not_flag() -> None:
    """Shrinking a ``self.batch_scale`` attribute (AnnAssign/Assign on an
    Attribute target) is a real retry — must not flag."""
    out = _run("""\
def train_step(self, loss):
    try:
        loss.backward()
    except OutOfMemoryError:
        self.batch_scale = self.batch_scale * 0.5
        return
""")
    assert out == [], f"Attribute batch/scale shrink must NOT flag: {out}"


def test_grad_accum_shrink_does_not_flag() -> None:
    out = _run("""\
def train_step(loss):
    try:
        loss.backward()
    except Exception:
        grad_accum_steps = grad_accum_steps * 2
        continue
""")
    assert out == [], f"grad_accum shrink must NOT flag: {out}"


# ---------------------------------------------------------------------------
# Case 3: try: loss.backward() except: raise  → does NOT flag (re-raise)
# ---------------------------------------------------------------------------


def test_reraise_does_not_flag() -> None:
    out = _run("""\
def train_step(loss):
    try:
        loss.backward()
    except:
        raise
""")
    assert out == [], f"Re-raising handler must NOT flag: {out}"


def test_reraise_runtime_error_does_not_flag() -> None:
    out = _run("""\
def train_step(loss):
    try:
        loss.backward()
    except RuntimeError as exc:
        log("oom")
        raise exc
""")
    assert out == [], f"Re-raise (``raise exc``) must NOT flag: {out}"


# ---------------------------------------------------------------------------
# FP guards: non-OOM except type, and no backward/step in the body.
# ---------------------------------------------------------------------------


def test_non_oom_except_type_does_not_flag() -> None:
    """``except ValueError:`` is not an OOM-shaped catch — do not flag even if it
    skips, since this cannot be masking a CUDA OOM."""
    out = _run("""\
def train_step(loss):
    try:
        loss.backward()
    except ValueError:
        continue
""")
    assert out == [], f"Non-OOM except type must NOT flag: {out}"


def test_no_backward_or_step_in_body_does_not_flag() -> None:
    """A try with no ``.backward()``/``.step()`` in the body is irrelevant."""
    out = _run("""\
def load(path):
    try:
        data = read(path)
    except RuntimeError:
        continue
""")
    assert out == [], f"No backward/step in body → must NOT flag: {out}"


# ---------------------------------------------------------------------------
# Case 4: scan_code_dir end-to-end returns the soft violation and NEVER raises.
# ---------------------------------------------------------------------------


def test_scan_code_dir_surfaces_swallow_violation_end_to_end(tmp_path: Path) -> None:
    _write(tmp_path, "train_cell.py", """\
def train_step(loss, optimizer):
    try:
        loss.backward()
        optimizer.step()
    except RuntimeError:
        continue
""")
    violations = scan_code_dir(tmp_path)
    hits = _swallow_hits(violations)
    assert len(hits) == 1, f"Expected one swallow violation through public API: {violations}"
    assert hits[0].severity == "soft"
    assert hits[0].file == "train_cell.py"


def test_scan_code_dir_never_raises_on_swallow_dir(tmp_path: Path) -> None:
    """Fail-soft contract: scan_code_dir must never raise even with the swallow
    case present alongside an unparseable sibling."""
    _write(tmp_path, "train_cell.py", """\
def step(loss):
    try:
        loss.backward()
    except RuntimeError:
        pass
""")
    _write(tmp_path, "broken.py", "def oops(:\n    pass\n")
    # Must not raise; returns a list.
    violations = scan_code_dir(tmp_path)
    assert isinstance(violations, list)
    assert len(_swallow_hits(violations)) == 1, violations


def test_shrink_and_retry_clean_through_public_api(tmp_path: Path) -> None:
    """The FP guard holds through the public API: a shrink-and-retry file yields
    no swallow violation."""
    _write(tmp_path, "train_cell.py", """\
def step(loss, batch_size):
    try:
        loss.backward()
    except RuntimeError:
        batch_size //= 2
        continue
""")
    assert _swallow_hits(scan_code_dir(tmp_path)) == []
