"""PaperSource discriminated union — what the user submits."""

from __future__ import annotations

import re
from typing import Annotated, Literal
from urllib.parse import unquote, urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator


_ARXIV_URL_RE = re.compile(
    r"(?:^|/)arxiv\.org/(?:abs|pdf)/(?P<id>\d{4}\.\d{4,5}(?:v\d+)?)(?:\.pdf)?$",
    re.IGNORECASE,
)
_ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}(?:v\d+)?$", re.IGNORECASE)
_DOI_URL_PREFIXES = ("https://doi.org/", "http://doi.org/", "doi:")


class PdfPath(BaseModel):
    """A local file path pointing to a PDF.

    `path` may be absolute or relative; the intake service resolves it
    to an absolute path before hashing for project_id derivation, so the
    same file under different relative paths produces the same project_id.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["pdf_path"] = "pdf_path"
    path: str


class ArxivId(BaseModel):
    """An arXiv paper identifier or arxiv.org abs/pdf URL."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["arxiv"] = "arxiv"
    arxiv_id: str

    @field_validator("arxiv_id")
    @classmethod
    def normalize_arxiv_id(cls, value: str) -> str:
        candidate = value.strip()
        match = _ARXIV_URL_RE.search(candidate)
        if match:
            candidate = match.group("id")
        if candidate.lower().startswith("arxiv:"):
            candidate = candidate.split(":", 1)[1]
        if not _ARXIV_ID_RE.match(candidate):
            raise ValueError(f"Invalid arXiv identifier: {value!r}")
        return candidate


class DoiRef(BaseModel):
    """A DOI identifier or doi.org URL."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["doi"] = "doi"
    doi: str

    @field_validator("doi")
    @classmethod
    def normalize_doi(cls, value: str) -> str:
        candidate = value.strip()
        lower = candidate.lower()
        for prefix in _DOI_URL_PREFIXES:
            if lower.startswith(prefix):
                candidate = candidate[len(prefix):]
                break
        parsed = urlparse(candidate)
        if parsed.scheme and parsed.netloc:
            candidate = parsed.path.lstrip("/")
        candidate = unquote(candidate).strip()
        if not candidate.lower().startswith("10.") or "/" not in candidate:
            raise ValueError(f"Invalid DOI: {value!r}")
        return candidate


# Discriminated union — Pydantic picks the right class on `kind`.
PaperSource = Annotated[PdfPath | ArxivId | DoiRef, Field(discriminator="kind")]


__all__ = ["ArxivId", "DoiRef", "PaperSource", "PdfPath"]
