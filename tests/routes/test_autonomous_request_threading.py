"""Task 2: `autonomous` + `run_spec` request fields + run_spec threading.

`autonomous` (default False) only needs to reach `_start_python_run` — a later
task (T3, `apply_autonomous_profile_override`) gives it behavior. `run_spec`
must reach the child subprocess's `cmd_reproduce` Namespace via the `common`
dict `_python_script` embeds, so `getattr(args, "run_spec", None)` sees it.

Test-strategy note (overrides the original brief's `_build_common_for_test`
snippet, which assumed a `FileLiveRunService` method that doesn't exist —
`common`/`_python_script` are module-level): the `_python_script`-threading
tests below call the REAL function and parse the embedded config back out of
the generated script text with a test-file-local regex helper. No production
test-seam, no subprocess spawn — mirrors the existing
`test_python_script_threads_repo_url` precedent in
`tests/test_repo_url_inputs.py`, but recovers the actual dict (not just a
substring match) so the OFF-state assertion can check the resolved value.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from starlette.testclient import TestClient

from backend.app import StartArxivRunRequest, create_app
from backend.services.events.live_runs import (
    LiveRunState,
    StartRunRequest,
    _python_script,
)


# --------------------------------------------------------------------------- #
# StartRunRequest / StartArxivRunRequest defaults + declarations
# --------------------------------------------------------------------------- #


def test_request_has_autonomous_and_run_spec_defaults():
    r = StartRunRequest()
    assert r.autonomous is False and r.run_spec is None


def test_start_run_request_accepts_autonomous_and_run_spec():
    r = StartRunRequest(autonomous=True, run_spec="configs/autonomous_reproduction_run_spec.json")
    assert r.autonomous is True
    assert r.run_spec == "configs/autonomous_reproduction_run_spec.json"


def test_start_arxiv_request_declares_autonomous_default_false():
    # Mirrors test_advanced_field_forwarding.py::test_arxiv_request_declares_advanced_fields —
    # a hermetic pydantic-level check, no HTTP round trip needed since both
    # StartArxivRunRequest.autonomous and StartRunRequest.autonomous are the
    # same concrete `bool` type (no None-coercion risk on this leg).
    r = StartArxivRunRequest(url="https://arxiv.org/abs/2605.15155")
    assert r.autonomous is False


def test_start_arxiv_request_accepts_autonomous_true():
    r = StartArxivRunRequest(url="https://arxiv.org/abs/2605.15155", autonomous=True)
    assert r.autonomous is True


# --------------------------------------------------------------------------- #
# _python_script: run_spec threads through the embedded `common` config AND
# the uploaded-paper Namespace that cmd_reproduce reads args.run_spec from.
# --------------------------------------------------------------------------- #


def _extract_common(script: str) -> dict[str, Any]:
    """Recover the `common` dict `_python_script` double-JSON-embeds.

    The generated script contains a line shaped like
    ``config = json.loads("<json-encoded-json-encoded-common-dict>")`` (the
    embed site does ``json.loads({json.dumps(json.dumps(common))})``). One
    json.loads unwraps the outer string layer back to the inner JSON text;
    the second parses that text into the actual dict.
    """
    m = re.search(r"config = json\.loads\((.+)\)", script)
    assert m, "could not find the embedded `config = json.loads(...)` line in the generated script"
    return json.loads(json.loads(m.group(1)))


def test_python_script_threads_run_spec_into_common_and_namespace(tmp_path: Path):
    req = StartRunRequest(run_spec="configs/autonomous_reproduction_run_spec.json")
    script = _python_script(req, project_id="p1", runs_root=tmp_path, uploaded_paper=None)

    common = _extract_common(script)
    assert common["run_spec"] == "configs/autonomous_reproduction_run_spec.json"

    # The uploaded-paper branch builds `Namespace(**{**_REPRODUCE_DEFAULTS, ...})`
    # for the direct `cmd_reproduce` call — both branches of the runtime `if` are
    # always present in the generated source (the `if` only executes at subprocess
    # run time), so this static text check is valid even with uploaded_paper=None.
    assert '"run_spec": config["run_spec"]' in script


def test_python_script_off_state_run_spec_resolves_to_none(tmp_path: Path):
    """Hard OFF-state invariant: an unset run_spec resolves to None, so
    cmd_reproduce's `getattr(args, "run_spec", None)` sees None and never
    calls `_load_run_spec` — byte-identical to today. Assert the RESOLVED
    value, not literal script text (the embedded JSON legitimately gains a
    `"run_spec": null` member that wasn't there before)."""
    script = _python_script(StartRunRequest(), project_id="p1", runs_root=tmp_path, uploaded_paper=None)
    common = _extract_common(script)
    assert common["run_spec"] is None


# --------------------------------------------------------------------------- #
# HTTP-level regression guard for /runs/upload. `_optional_form_bool` returns
# `bool | None`; `StartRunRequest.autonomous` is a concrete `bool` (not
# Optional) per the brief. Passing a bare None into a strict-bool pydantic
# field raises a ValidationError, so the form-read call MUST coerce
# (`bool(_optional_form_bool(...))`) or every existing /runs/upload caller
# that omits the brand-new `autonomous` field would start 422ing today.
# --------------------------------------------------------------------------- #


class _FakeUploadService:
    def __init__(self) -> None:
        self.started: StartRunRequest | None = None
        self.state = LiveRunState(
            projectId="prj_auto",
            outputDir="runs/prj_auto",
            runMode="rlm",
            llmProvider="anthropic",
            status="queued",
            payload=None,
            log="",
        )

    async def start_uploaded_run(self, request: StartRunRequest, *, file_name: str, content: bytes):
        self.started = request
        self.state.sourceKind = "uploaded_pdf"
        self.state.sourceLabel = file_name
        return self.state


def test_upload_route_omitting_autonomous_defaults_false_not_422():
    service = _FakeUploadService()
    client = TestClient(create_app(run_service=service))

    response = client.post(
        "/runs/upload",
        data={"mode": "rlm", "provider": "anthropic"},
        files={"paper": ("paper.pdf", b"%PDF-demo", "application/pdf")},
    )

    assert response.status_code == 202, response.text
    assert service.started is not None
    assert service.started.autonomous is False


def test_upload_route_forwards_autonomous_true():
    service = _FakeUploadService()
    client = TestClient(create_app(run_service=service))

    response = client.post(
        "/runs/upload",
        data={"mode": "rlm", "provider": "anthropic", "autonomous": "true"},
        files={"paper": ("paper.pdf", b"%PDF-demo", "application/pdf")},
    )

    assert response.status_code == 202, response.text
    assert service.started is not None
    assert service.started.autonomous is True
