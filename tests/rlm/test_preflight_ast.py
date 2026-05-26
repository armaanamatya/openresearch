"""Tests for backend.agents.rlm.preflight_ast — scoped AST pre-flight (PR-γ.1).

Covers the four BLOCKING shapes and three NON-BLOCKING shapes the spec requires.
All fixtures use synthetic .py files written to tmp_path.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from backend.agents.rlm.preflight_ast import PreflightViolation, scan_code_dir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(code_dir: Path, name: str, body: str) -> Path:
    p = code_dir / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    return p


def _hard(vs: list[PreflightViolation]) -> list[PreflightViolation]:
    return [v for v in vs if v.severity == "hard"]


# ---------------------------------------------------------------------------
# BLOCKING: same-file, same-class missing attribute access
# ---------------------------------------------------------------------------


def test_missing_attr_on_local_class_is_blocked(tmp_path: Path) -> None:
    """WakeSleepVAE.reparameterize called but not defined on the class → BLOCKS."""
    _write(tmp_path, "train.py", """\
class WakeSleepVAE:
    def encode(self, x):
        return x

model = WakeSleepVAE()
z, mu, log_var = model.reparameterize(model.encode(x))
""")
    violations = scan_code_dir(tmp_path)
    hard = _hard(violations)
    assert len(hard) >= 1
    # The violation should point at the offending attribute.
    attrs = [v.missing_attr for v in hard]
    assert "reparameterize" in attrs, f"Expected 'reparameterize' in {attrs}"
    # Class name should be surfaced.
    classes = [v.class_name for v in hard if v.missing_attr == "reparameterize"]
    assert any(c == "WakeSleepVAE" for c in classes), (
        f"Expected class_name=WakeSleepVAE, got {classes}"
    )


def test_defined_attr_does_not_block(tmp_path: Path) -> None:
    """Same class with reparameterize defined → no violation."""
    _write(tmp_path, "train.py", """\
class WakeSleepVAE:
    def encode(self, x):
        return x

    def reparameterize(self, z, mu, log_var):
        return z

model = WakeSleepVAE()
z = model.reparameterize(model.encode(x), mu, log_var)
""")
    violations = scan_code_dir(tmp_path)
    attr_hits = [v for v in violations if v.missing_attr == "reparameterize"]
    assert not attr_hits, f"Should not flag defined method, got: {attr_hits}"


# ---------------------------------------------------------------------------
# NON-BLOCKING: dynamic setattr BEFORE the call suppresses the violation
# ---------------------------------------------------------------------------


def test_setattr_before_call_does_not_block(tmp_path: Path) -> None:
    """setattr(model, 'reparameterize', ...) present → DOES NOT BLOCK."""
    _write(tmp_path, "train.py", """\
class WakeSleepVAE:
    def encode(self, x):
        return x

model = WakeSleepVAE()
setattr(model, "reparameterize", lambda z, mu, lv: z)
z = model.reparameterize(model.encode(x), mu, log_var)
""")
    violations = scan_code_dir(tmp_path)
    attr_hits = [v for v in violations if v.missing_attr == "reparameterize"]
    assert not attr_hits, (
        f"setattr should suppress the violation, got: {attr_hits}"
    )


# ---------------------------------------------------------------------------
# BLOCKING: import-from of a locally-defined symbol that doesn't exist
# ---------------------------------------------------------------------------


def test_local_import_nonexistent_symbol_blocks(tmp_path: Path) -> None:
    """from .models import Bar when models.py has only class Foo → BLOCKS."""
    _write(tmp_path, "models.py", """\
class Foo:
    pass
""")
    _write(tmp_path, "train.py", """\
from models import Bar  # Bar doesn't exist in models.py

model = Bar()
""")
    violations = scan_code_dir(tmp_path)
    hard = _hard(violations)
    import_hits = [v for v in hard if v.missing_attr == "Bar"]
    assert import_hits, (
        f"Expected violation for missing 'Bar' in models.py, got violations: {violations}"
    )


def test_local_import_existing_symbol_does_not_block(tmp_path: Path) -> None:
    """from models import Foo where Foo is defined → no violation."""
    _write(tmp_path, "models.py", """\
class Foo:
    pass
""")
    _write(tmp_path, "train.py", """\
from models import Foo

model = Foo()
""")
    violations = scan_code_dir(tmp_path)
    import_hits = [v for v in violations if v.missing_attr == "Foo"]
    assert not import_hits, f"Defined symbol should not block, got: {import_hits}"


# ---------------------------------------------------------------------------
# NON-BLOCKING: cross-file attribute access on externally imported class
# ---------------------------------------------------------------------------


def test_cross_file_external_class_does_not_block(tmp_path: Path) -> None:
    """Attribute access on a class imported from an external library → DOES NOT BLOCK."""
    _write(tmp_path, "train.py", """\
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

model = MyModel()
# .parameters() comes from nn.Module which is external — should NOT flag.
optimizer = None
for p in model.parameters():
    pass
""")
    violations = scan_code_dir(tmp_path)
    # MyModel is locally defined but parameters() comes from nn.Module (external parent).
    # The checker should NOT flag cross-file inherited attributes.
    param_hits = [v for v in violations if v.missing_attr == "parameters"]
    assert not param_hits, f"External-inherited attr should not block, got: {param_hits}"


# ---------------------------------------------------------------------------
# BLOCKING: pure syntax error in a .py file
# ---------------------------------------------------------------------------


def test_syntax_error_in_file_blocks(tmp_path: Path) -> None:
    """A .py file with a SyntaxError → hard violation with the line number."""
    _write(tmp_path, "train.py", """\
class Foo:
    def broken(self
        pass  # missing closing paren
""")
    violations = scan_code_dir(tmp_path)
    hard = _hard(violations)
    syntax_hits = [v for v in hard if "SyntaxError" in v.detail or v.missing_attr is None]
    assert syntax_hits, f"SyntaxError should produce a hard violation, got: {hard}"
    # Line number should be non-zero.
    assert any(v.line > 0 for v in syntax_hits), "Violation should carry a line number"


def test_syntax_error_violation_has_correct_file(tmp_path: Path) -> None:
    """The violation's file attribute points to the file with the SyntaxError."""
    _write(tmp_path, "models.py", """\
def ok():
    pass
""")
    _write(tmp_path, "train.py", """\
def broken(:
    pass
""")
    violations = scan_code_dir(tmp_path)
    hard = _hard(violations)
    assert hard, "Should have at least one hard violation"
    # The violation file should be train.py (models.py is fine).
    files = {v.file for v in hard}
    assert "train.py" in files, f"Expected train.py in {files}"


# ---------------------------------------------------------------------------
# Regression: empty directory / non-existent directory
# ---------------------------------------------------------------------------


def test_empty_dir_returns_empty(tmp_path: Path) -> None:
    """Empty directory → empty violations list (no crash)."""
    assert scan_code_dir(tmp_path) == []


def test_nonexistent_dir_returns_empty() -> None:
    """Non-existent directory → empty violations list (no crash)."""
    assert scan_code_dir(Path("/tmp/__no_such_dir_preflight_ast_test__")) == []


# ---------------------------------------------------------------------------
# Integration: hook into validate_code_pre_flight
# ---------------------------------------------------------------------------


def test_hook_into_validate_code_pre_flight(tmp_path: Path) -> None:
    """Ensure scan_code_dir violations are bridged into validate_code_pre_flight."""
    from backend.agents.rlm.pre_flight_validator import validate_code_pre_flight

    _write(tmp_path, "train.py", """\
class SimpleVAE:
    def encode(self, x):
        return x

model = SimpleVAE()
output = model.decode(model.encode(x))  # decode not defined
""")
    # Call with no paper_targets and no base_image — only γ.1 violations fire.
    result = validate_code_pre_flight(tmp_path, None, base_image=None)
    hard = [v for v in result if v.severity == "hard"]
    decode_hits = [v for v in hard if "decode" in v.detail]
    assert decode_hits, (
        f"Expected 'decode' violation to be surfaced by validate_code_pre_flight, "
        f"got: {result}"
    )
