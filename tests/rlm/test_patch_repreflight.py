"""Dark-switches Task 1 — patch-mode must re-validate.

`implement_baseline` patch-mode applies a diff and writes train.py back. A patch can
apply cleanly yet not yield clean code. `_patched_code_still_violates` re-runs the AST
preflight so a patch that left confident violations falls through to a full rewrite
(cheaper than burning a GPU to rediscover the bug). Fail-soft on scan error.
"""
import backend.agents.rlm.primitives as prim


def test_still_violates_true_when_scan_nonempty(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "backend.agents.rlm.preflight_ast.scan_code_dir", lambda d: ["v1"]
    )
    assert prim._patched_code_still_violates(str(tmp_path)) is True


def test_clean_when_scan_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "backend.agents.rlm.preflight_ast.scan_code_dir", lambda d: []
    )
    assert prim._patched_code_still_violates(str(tmp_path)) is False


def test_scan_error_is_failsoft(monkeypatch, tmp_path):
    def _boom(d):
        raise RuntimeError("scan blew up")
    monkeypatch.setattr("backend.agents.rlm.preflight_ast.scan_code_dir", _boom)
    # fail-soft: a scan error must NOT block the patch path (returns False)
    assert prim._patched_code_still_violates(str(tmp_path)) is False
