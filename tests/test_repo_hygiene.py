"""Repo-hygiene invariants (D2/D3): no large binary archives in git, and the
tracked runs/ artifacts stay within the .gitignore whitelist (small high-value
per-run files only — no train logs, dashboards, code/, reports/, or attempts/)."""
import subprocess
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]

# depth-1 per-run files the whitelist intentionally tracks. Mirror of the
# .gitignore `!/runs/*/<name>` re-include rules — keep the two lists in sync.
_RUNS_WHITELIST = {
    "final_report.json",
    "final_report.md",
    "demo_status.json",
    "batch_child.log",
    "experiment_runs.jsonl",
    "cost_ledger.jsonl",
    "tokens_total.json",
}


def _tracked(pathspec: str) -> list[str]:
    out = subprocess.check_output(
        ["git", "ls-files", pathspec], cwd=_REPO
    ).decode()
    return [ln for ln in out.splitlines() if ln]


def test_no_tracked_zip_archives():
    zips = _tracked("*.zip")
    assert zips == [], f"binary .zip archives must not be tracked: {zips}"


def test_tracked_run_artifacts_within_whitelist():
    offenders = []
    for f in _tracked("runs/"):
        parts = f.split("/")
        # allowed shape: runs/<project_dir>/<whitelisted-basename>
        if not (len(parts) == 3 and parts[2] in _RUNS_WHITELIST):
            offenders.append(f)
    assert offenders == [], (
        "tracked runs/ artifacts outside the whitelist (nested or heavy logs): "
        f"{offenders}"
    )


def test_no_tracked_but_gitignored_files():
    """A file that is both tracked and gitignored is a contradiction: git
    silently stops reporting local changes to it, and it usually means an
    ignore rule was added without `git rm --cached` (how 6.5MB under
    paper-repro-bes-docs/ lingered after commit 3203c17 ignored the dir).
    Either track it (remove the ignore rule) or untrack it — never both."""
    out = subprocess.check_output(
        ["git", "ls-files", "-i", "-c", "--exclude-standard"], cwd=_REPO
    ).decode()
    offenders = [ln for ln in out.splitlines() if ln]
    assert offenders == [], (
        f"files tracked despite matching .gitignore (git rm --cached them): {offenders}"
    )
