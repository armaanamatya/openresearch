"""Central structured-logging config — opt-in, no default behavior change.

Provides a JSON formatter (queryable logs), env-driven level control (less noise
/ better levels via OPENRESEARCH_LOG_LEVEL), a project-id correlation filter, and
an idempotent configure_logging(). Hermetic — the configure_logging test
snapshots/restores the root logger so it never pollutes other tests.
"""
from __future__ import annotations

import json
import logging

from backend.logging_config import (
    JsonFormatter,
    ProjectIdFilter,
    configure_logging,
    resolve_level,
)


def _record(msg="hi", level=logging.INFO, **extra):
    rec = logging.LogRecord("t", level, __file__, 1, msg, (), None)
    for k, v in extra.items():
        setattr(rec, k, v)
    return rec


# --------------------------------------------------------------------------- #
# level resolution
# --------------------------------------------------------------------------- #
def test_resolve_level_by_name():
    assert resolve_level("DEBUG") == logging.DEBUG
    assert resolve_level("info") == logging.INFO
    assert resolve_level("Warning") == logging.WARNING


def test_resolve_level_by_number():
    assert resolve_level("10") == 10


def test_resolve_level_defaults_to_info_on_junk():
    assert resolve_level("") == logging.INFO
    assert resolve_level("not-a-level") == logging.INFO
    assert resolve_level(None) == logging.INFO


# --------------------------------------------------------------------------- #
# JSON formatter (queryable)
# --------------------------------------------------------------------------- #
def test_json_formatter_emits_valid_json_with_core_fields():
    out = JsonFormatter().format(_record("hello", level=logging.WARNING))
    obj = json.loads(out)
    assert obj["level"] == "WARNING"
    assert obj["message"] == "hello"
    assert obj["logger"] == "t"
    assert "ts" in obj


def test_json_formatter_includes_extra_fields():
    out = JsonFormatter().format(_record("m", project_id="prj_abc", gate="grader_integrity"))
    obj = json.loads(out)
    assert obj["project_id"] == "prj_abc"
    assert obj["gate"] == "grader_integrity"


def test_json_formatter_includes_exception():
    try:
        raise ValueError("boom")
    except ValueError:
        import sys
        rec = logging.LogRecord("t", logging.ERROR, __file__, 1, "err", (), sys.exc_info())
    obj = json.loads(JsonFormatter().format(rec))
    assert "ValueError" in obj["exception"]
    assert "boom" in obj["exception"]


# --------------------------------------------------------------------------- #
# project-id correlation filter
# --------------------------------------------------------------------------- #
def test_project_id_filter_injects_default():
    f = ProjectIdFilter("prj_xyz")
    rec = _record()
    assert f.filter(rec) is True
    assert rec.project_id == "prj_xyz"


def test_project_id_filter_does_not_clobber_existing():
    f = ProjectIdFilter("prj_default")
    rec = _record(project_id="prj_explicit")
    f.filter(rec)
    assert rec.project_id == "prj_explicit"


# --------------------------------------------------------------------------- #
# configure_logging — idempotent, no handler duplication
# --------------------------------------------------------------------------- #
def test_configure_logging_is_idempotent():
    root = logging.getLogger()
    saved = root.handlers[:]
    saved_level = root.level
    try:
        root.handlers = []
        configure_logging(level="DEBUG", fmt="json")
        n1 = len(root.handlers)
        configure_logging(level="DEBUG", fmt="json")
        n2 = len(root.handlers)
        assert n1 == 1
        assert n2 == 1  # second call does not add a duplicate handler
        assert root.level == logging.DEBUG
    finally:
        root.handlers = saved
        root.setLevel(saved_level)
