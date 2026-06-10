"""POST /paper/estimate — pre-run budget estimation.

Spec: docs/superpowers/specs/2026-05-25-budget-estimation-design.md §HTTP API
Invariant 7: this handler never spawns a subprocess.
Invariant 10: on failure, return a 200 with error_message so the UI can
  surface "Skip estimate and start anyway" without blocking the user.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.config import get_settings
from backend.services.pricing.estimator import estimate_paper_budget

logger = logging.getLogger(__name__)

router = APIRouter()


class EstimateRequest(BaseModel):
    source_kind: Literal["arxiv_id", "arxiv_url", "pdf_path"] = "arxiv_id"
    source: str
    recipe_mode: Literal["strict", "compressed", "both"] = "both"


@router.post("/paper/estimate")
async def estimate_budget(request: Request) -> JSONResponse:
    """Return a PaperBudgetEstimate for a paper before a run starts.

    Accepts either:
    - JSON body: {"source_kind", "source", "recipe_mode"}
    - Multipart form: "paper" file field (same shape as /runs/upload) +
      optional "recipe_mode" field

    On estimator failure, returns 200 with {"error": "..."} so the UI can
    surface "Skip estimate and start anyway" (invariant 10).
    """
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

        import tempfile
        import os
        suffix = ".pdf"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name
        try:
            return await _run_estimate(
                source_kind="pdf_path",
                source=tmp_path,
                recipe_mode=recipe_mode_raw,
            )
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    else:
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Request body must be JSON or multipart.")

        try:
            req = EstimateRequest(**body)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        return await _run_estimate(
            source_kind=req.source_kind,
            source=req.source,
            recipe_mode=req.recipe_mode,
        )


async def _run_estimate(
    source_kind: str,
    source: str,
    recipe_mode: str,
) -> JSONResponse:
    """Inner runner — separated so multipart and JSON share the same path."""
    settings = get_settings()
    runs_root = Path(settings.runs_root) if settings.runs_root else Path("runs")

    try:
        result = await estimate_paper_budget(
            source,
            source_kind=source_kind,
            recipe_mode=recipe_mode,
            runs_root=runs_root,
        )
        return JSONResponse(content=result)
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
