#!/usr/bin/env python3
"""Documentation freshness & consistency checker.

Enforces the policy in ``docs/policies/documentation.md``. Pure stdlib; run from
anywhere inside the repo:

    python scripts/docs_freshness_check.py        # or: make docs-check

Exit 1 (FAIL) on:
  * a tracked PDF outside the approved fixture locations
  * a current-state doc (listed in current-docs.txt) missing its freshness marker
  * a broken relative link in any current-state doc or the README
  * a working-note file reappearing at the repo root (masquerading as current)
  * a README inline reference to a missing repo file/script

Exit 0 with WARN on:
  * a current-state doc whose last-verified date is older than STALE_DAYS

The check is manifest-driven on purpose: only docs in current-docs.txt are held
to a freshness date, so the ~50 intentionally-dated runbooks/specs are never
flagged as noise.
"""
from __future__ import annotations

import datetime
import fnmatch
import re
import subprocess
import sys
from pathlib import Path

STALE_DAYS = 90

# Tracked PDFs are input fixtures (papers don't go stale). They are allowed only
# in these locations; a PDF anywhere else is probably an accidental commit.
APPROVED_PDF_GLOBS = [
    "demo_paper.pdf",
    "paperbench*.pdf",
    "best_runs/**/*.pdf",
    "docs/**/*.pdf",
    "third_party/**/*.pdf",
]

# These were development working-notes; they live in docs/archive/ now. If any
# reappears at the repo ROOT it would masquerade as a current top-level doc.
FORBIDDEN_ROOT_BASENAMES = {
    "progress.md",
    "runlog.md",
    "learn.md",
    "uiprogress.md",
    "frontend_integration.md",
}

MARKER_RE = re.compile(r"last-verified=(\d{4}-\d{2}-\d{2})")
MD_LINK_RE = re.compile(r"\[(?:[^\]]*)\]\(([^)]+)\)")
# Inline-code repo paths *with a file extension* in the README, e.g.
# `scripts/foo.py`, `backend/app.py`, `best_runs/x/final_report.md`. Bare
# directory references are not checked (too prose-ambiguous); see policy §5.
README_PATH_RE = re.compile(
    r"`((?:backend|frontend|scripts|docs|best_runs|runs|third_party|data|tools)"
    r"/[\w./-]+\.\w+)`"
)


def repo_root() -> Path:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=Path(__file__).resolve().parent,
            text=True,
        ).strip()
        return Path(out)
    except Exception:
        return Path(__file__).resolve().parent.parent


def tracked_files(root: Path) -> list[str]:
    out = subprocess.check_output(["git", "ls-files"], cwd=root, text=True)
    return [line for line in out.splitlines() if line]


def load_manifest(root: Path) -> list[str]:
    manifest = root / "docs/policies/current-docs.txt"
    paths: list[str] = []
    for line in manifest.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            paths.append(line)
    return paths


def check_pdfs(root: Path, tracked: list[str], fails: list[str]) -> None:
    for f in tracked:
        if not f.lower().endswith(".pdf"):
            continue
        if not any(fnmatch.fnmatch(f, g) for g in APPROVED_PDF_GLOBS):
            fails.append(
                f"PDF outside approved fixture locations: {f} "
                f"(allowed: {', '.join(APPROVED_PDF_GLOBS)})"
            )


def check_root_working_notes(root: Path, fails: list[str]) -> None:
    for name in sorted(FORBIDDEN_ROOT_BASENAMES):
        if (root / name).exists():
            fails.append(
                f"Working-note '{name}' is back at repo root — it must live in "
                f"docs/archive/ with an ARCHIVED banner, not masquerade as a "
                f"current top-level doc."
            )


def check_markers(
    root: Path, manifest: list[str], fails: list[str], warns: list[str]
) -> None:
    today = datetime.date.today()
    for rel in manifest:
        p = root / rel
        if not p.exists():
            fails.append(f"current-docs.txt lists a missing file: {rel}")
            continue
        head = "\n".join(p.read_text(errors="replace").splitlines()[:15])
        m = MARKER_RE.search(head)
        if not m:
            fails.append(
                f"Current-state doc missing freshness marker "
                f"(`<!-- doc-meta: status=current; last-verified=YYYY-MM-DD -->` "
                f"in first 15 lines): {rel}"
            )
            continue
        try:
            verified = datetime.date.fromisoformat(m.group(1))
        except ValueError:
            fails.append(f"Unparseable last-verified date in {rel}: {m.group(1)}")
            continue
        age = (today - verified).days
        if age > STALE_DAYS:
            warns.append(
                f"{rel}: last verified {verified} ({age} days ago) — re-verify "
                f"against the code and bump the marker."
            )


def _check_links_in(root: Path, rel: str, fails: list[str]) -> None:
    doc = root / rel
    if not doc.exists():
        return
    base = doc.parent
    for target in MD_LINK_RE.findall(doc.read_text(errors="replace")):
        target = target.strip()
        # strip optional link title:  (path "title")
        if " " in target:
            target = target.split(" ", 1)[0]
        target = target.strip("<>")
        if (
            not target
            or target.startswith(("http://", "https://", "mailto:", "#"))
        ):
            continue
        path_part = target.split("#", 1)[0]
        if not path_part:
            continue
        resolved = (base / path_part).resolve()
        if not resolved.exists():
            fails.append(f"Broken link in {rel}: '{target}' -> {path_part}")


def check_links(root: Path, manifest: list[str], fails: list[str]) -> None:
    for rel in manifest:
        _check_links_in(root, rel, fails)


def check_readme_inline_paths(root: Path, fails: list[str]) -> None:
    readme = root / "README.md"
    if not readme.exists():
        return
    for ref in set(README_PATH_RE.findall(readme.read_text(errors="replace"))):
        if any(tok in ref for tok in ("<", ">", "*", "_id", "{")):
            continue  # placeholder, not a literal path
        if not (root / ref).exists():
            fails.append(f"README references a missing path: `{ref}`")


def main() -> int:
    root = repo_root()
    tracked = tracked_files(root)
    manifest = load_manifest(root)

    fails: list[str] = []
    warns: list[str] = []

    check_pdfs(root, tracked, fails)
    check_root_working_notes(root, fails)
    check_markers(root, manifest, fails, warns)
    check_links(root, manifest, fails)
    check_readme_inline_paths(root, fails)

    print(f"docs-check: {len(manifest)} current-state docs, "
          f"{sum(1 for f in tracked if f.lower().endswith('.pdf'))} tracked PDFs.")
    for w in warns:
        print(f"  WARN  {w}")
    for f in fails:
        print(f"  FAIL  {f}")

    if fails:
        print(f"\ndocs-check FAILED: {len(fails)} problem(s). "
              f"See docs/policies/documentation.md.")
        return 1
    print("docs-check OK" + (f" ({len(warns)} warning(s))" if warns else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
