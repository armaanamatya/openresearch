"""Regex-based first-pass artifact discovery.

This adapter intentionally favors high-precision URL extraction over
guessing ambiguous dataset names from prose. It captures repository,
dataset, and issue/discussion metadata when papers cite concrete URLs.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from urllib.parse import urlparse

from backend.services.ingestion.discovery.model import (
    DiscoveredArtifact,
    DiscoveredArtifactKind,
    artifact_ref_id_for,
)


_URL_RE = re.compile(r"https?://[^\s\]\)\},;\"']+")
_GITHUB_RE = re.compile(
    r"^github\.com/(?P<owner>[^/\s]+)/(?P<repo>[^/\s]+)"
    r"(?:/(?P<section>issues|pull|discussions)/(?P<number>\d+))?",
    re.IGNORECASE,
)
_HF_DATASET_RE = re.compile(
    r"^huggingface\.co/datasets/(?P<owner>[^/\s]+)/(?P<name>[^/\s]+)",
    re.IGNORECASE,
)
_KAGGLE_DATASET_RE = re.compile(
    r"^www\.kaggle\.com/datasets/(?P<owner>[^/\s]+)/(?P<name>[^/\s]+)",
    re.IGNORECASE,
)


class RegexArtifactDiscoveryAdapter:
    name = "regex_url_discovery"

    def discover(self, *, project_id: str, text: str) -> list[DiscoveredArtifact]:
        found: "OrderedDict[tuple[DiscoveredArtifactKind, str], DiscoveredArtifact]" = (
            OrderedDict()
        )
        for match in _URL_RE.finditer(text):
            raw_url = match.group(0).rstrip(".")
            parsed = urlparse(raw_url)
            host_path = f"{parsed.netloc}{parsed.path}".rstrip("/")
            artifact = self._artifact_for_url(
                project_id=project_id,
                url=raw_url,
                host_path=host_path,
                evidence_quote=_evidence_window(text, match.start(), match.end()),
            )
            if artifact is not None:
                found.setdefault((artifact.kind, artifact.locator), artifact)
        return list(found.values())

    def _artifact_for_url(
        self,
        *,
        project_id: str,
        url: str,
        host_path: str,
        evidence_quote: str,
    ) -> DiscoveredArtifact | None:
        github = _GITHUB_RE.match(host_path)
        if github:
            owner = github.group("owner")
            repo = github.group("repo").removesuffix(".git")
            section = github.group("section")
            number = github.group("number")
            if section == "issues":
                kind = DiscoveredArtifactKind.issue
                locator = f"github:{owner}/{repo}#issue-{number}"
            elif section == "discussions":
                kind = DiscoveredArtifactKind.discussion
                locator = f"github:{owner}/{repo}#discussion-{number}"
            else:
                kind = DiscoveredArtifactKind.repository
                locator = f"github:{owner}/{repo}"
            return _make_artifact(project_id, kind, locator, url, evidence_quote)

        hf_dataset = _HF_DATASET_RE.match(host_path)
        if hf_dataset:
            locator = f"huggingface:{hf_dataset.group('owner')}/{hf_dataset.group('name')}"
            return _make_artifact(
                project_id,
                DiscoveredArtifactKind.dataset,
                locator,
                url,
                evidence_quote,
            )

        kaggle_dataset = _KAGGLE_DATASET_RE.match(host_path)
        if kaggle_dataset:
            locator = f"kaggle:{kaggle_dataset.group('owner')}/{kaggle_dataset.group('name')}"
            return _make_artifact(
                project_id,
                DiscoveredArtifactKind.dataset,
                locator,
                url,
                evidence_quote,
            )

        return None


def _make_artifact(
    project_id: str,
    kind: DiscoveredArtifactKind,
    locator: str,
    url: str,
    evidence_quote: str,
) -> DiscoveredArtifact:
    return DiscoveredArtifact(
        id=artifact_ref_id_for(project_id=project_id, kind=kind, locator=locator),
        project_id=project_id,
        kind=kind,
        locator=locator,
        url=url,
        title=locator,
        metadata={"source": "url"},
        evidence_quote=evidence_quote,
        confidence=0.9,
    )


def _evidence_window(text: str, start: int, end: int, radius: int = 160) -> str:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    return " ".join(text[left:right].split())


__all__ = ["RegexArtifactDiscoveryAdapter"]
