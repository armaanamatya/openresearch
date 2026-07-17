"""POST /paper/estimate — pre-run budget estimation.

Spec: docs/superpowers/specs/2026-05-25-budget-estimation-design.md §HTTP API
Invariant 7: this handler never spawns a subprocess.
Invariant 10: on failure, return a 200 with error_message so the UI can
  surface "Skip estimate and start anyway" without blocking the user.

Security (2026-07-13 fix): this route is gated by the same X-Demo-Secret
mechanism as the other run-start routes in backend/app.py
(``_enforce_demo_gate``) -- it triggers an LLM call and a GPU-resolution pass
and, for ``source_kind="pdf_path"``, reads a caller-supplied path off disk.
On a gated deployment it must not be reachable unauthenticated, exactly like
POST /runs and POST /runs/upload. A ``pdf_path`` source is containment-checked
against the runs directory (``resolve_allowed_pdf_path`` in
backend/services/pricing/estimator.py) BEFORE any file is touched; a path
outside the runs directory is rejected with a clean 400 -- that is a security
decision, not an "estimator failure," so it is never folded into invariant
10's 200-with-error-string fallback.
"""

from __future__ import annotations

import hmac
import logging
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.config import get_settings
from backend.services.pricing.estimator import (
    PdfPathNotAllowedError,
    estimate_paper_budget,
    resolve_allowed_pdf_path,
)

logger = logging.getLogger(__name__)

router = APIRouter()


class EstimateRequest(BaseModel):
    source_kind: Literal["arxiv_id", "arxiv_url", "pdf_path"] = "arxiv_id"
    source: str
    recipe_mode: Literal["strict", "compressed", "both"] = "both"


def _enforce_demo_gate(provided_secret: str | None, configured_secret: str) -> None:
    """Require a matching X-Demo-Secret header.

    Mirrors ``backend.app._enforce_demo_gate`` (also duplicated locally in
    backend/routes/messages.py) rather than importing across route modules.
    When ``configured_secret`` is empty the gate is disabled (local dev).
    """
    if not configured_secret:
        return
    if not provided_secret or not hmac.compare_digest(provided_secret, configured_secret):
        raise HTTPException(status_code=401, detail="A valid demo access secret is required.")


def _runs_root() -> Path:
    settings = get_settings()
    return Path(settings.runs_root) if settings.runs_root else Path("runs")


@router.post("/paper/estimate")
async def estimate_budget(
    request: Request,
    x_demo_secret: str | None = Header(default=None),
) -> JSONResponse:
    """Return a PaperBudgetEstimate for a paper before a run starts.

    Accepts either:
    - JSON body: {"source_kind", "source", "recipe_mode"}
    - Multipart form: "paper" file field (same shape as /runs/upload) +
      optional "recipe_mode" field

    On estimator failure, returns 200 with {"error": "..."} so the UI can
    surface "Skip estimate and start anyway" (invariant 10). A rejected
    request (missing/invalid demo secret, or a pdf_path outside the runs
    directory) is NOT an estimator failure and returns a 4xx instead -- see
    the module docstring.
    """
    _enforce_demo_gate(x_demo_secret, get_settings().demo_secret)

    content_type = request.headers.get("content-type", "")

    if "multipart/form-data" in content_type:
        form = await request.form()
        paper_file = form.get("paper")
        if paper_file is None or not hasattr(paper_file, "read"):
            raise HTTPException(status_code=400, detail="No 'paper' file field in multipart upload.")
        pdf_bytes = await paper_file.read()
        recipe_mode_raw = str(form.get("recipe_mode", "both"))
        if recipe_mode_raw not in ("strict", "compressed", "both"):
            recipe_mode_raw = "both"

        # Server-generated scratch path *inside* runs_root, named from a
        # fresh uuid4 (never derived from client input) -- mirrors the
        # .lab_uploads convention in backend/services/events/live_runs.py.
        # Writing under runs_root (rather than the OS temp dir) means this
        # path satisfies the same pdf_path containment check as any other
        # source, with no bypass flag needed.
        uploads_dir = _runs_root() / ".estimate_uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = uploads_dir / f"{uuid4().hex}.pdf"
        tmp_path.write_bytes(pdf_bytes)
        try:
            return await _run_estimate(
                source_kind="pdf_path",
                source=str(tmp_path),
                recipe_mode=recipe_mode_raw,
            )
        finally:
            tmp_path.unlink(missing_ok=True)
    else:
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Request body must be JSON or multipart.")

        try:
            req = EstimateRequest(**body)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        source = req.source
        if req.source_kind == "pdf_path":
            # Validate/authorize BEFORE _run_estimate's try/except-200 block
            # (invariant 10 covers genuine estimator failures only): a
            # pdf_path outside the runs directory is rejected here with a
            # clean 400 and the file is never opened.
            try:
                source = str(resolve_allowed_pdf_path(req.source, (_runs_root(),)))
            except PdfPathNotAllowedError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        return await _run_estimate(
            source_kind=req.source_kind,
            source=source,
            recipe_mode=req.recipe_mode,
        )


async def _run_estimate(
    source_kind: str,
    source: str,
    recipe_mode: str,
) -> JSONResponse:
    """Inner runner — separated so multipart and JSON share the same path."""
    runs_root = _runs_root()

    try:
        result = await estimate_paper_budget(
            source,
            source_kind=source_kind,
            recipe_mode=recipe_mode,
            runs_root=runs_root,
        )
        return JSONResponse(content=result)
    except PdfPathNotAllowedError as exc:
        # Defense in depth: estimate_paper_budget() re-validates containment
        # internally (backend/services/pricing/estimator.py). The JSON-body
        # branch above already rejects before ever reaching here; this only
        # fires if that pre-check were somehow bypassed, and it must still
        # be a 400 -- never folded into the 200-with-error-string fallback
        # below.
        logger.warning("estimate: rejected pdf_path outside allowed root: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 — invariant 10: never block the user
        logger.warning(
            "estimate: failed for source_kind=%s source=%r: %s",
            source_kind,
            source,
            exc,
        )
        return JSONResponse(
            status_code=200,
            content={
                "error": str(exc),
                "error_type": type(exc).__name__,
                "fallback_available": True,
            },
        )
