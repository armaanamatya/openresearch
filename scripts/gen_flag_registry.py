#!/usr/bin/env python3
"""Generate docs/reference/flags.md — the authoritative inventory of every
``OPENRESEARCH_*`` feature flag the backend reads.

The flag surface is large (300+ vars) and mostly read ad-hoc via
``os.environ.get(...)`` rather than through ``backend/config.py`` Settings, so
there is no single place to answer "what is this flag, what's its default, is it
documented, is it managed by config?". This script reconstructs that inventory
from the source of truth (the code) so the registry never drifts: re-run it and
diff.

Usage:
    .venv/bin/python scripts/gen_flag_registry.py          # write docs/reference/flags.md
    .venv/bin/python scripts/gen_flag_registry.py --check   # exit 1 if the file is stale

Pure stdlib. No behavior change anywhere; read-only over the tree.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BACKEND = REPO / "backend"
CONFIG = BACKEND / "config.py"
OUT = REPO / "docs" / "reference" / "flags.md"

FLAG_RE = re.compile(r"OPENRESEARCH_[A-Z0-9_]+")
# os.environ.get("FLAG", <default>) | os.getenv("FLAG", <default>) — capture default literal if present.
READ_RE = re.compile(
    r"""os\.(?:environ\.get|getenv)\(\s*["'](OPENRESEARCH_[A-Z0-9_]+)["']\s*(?:,\s*(?P<default>[^)]+?))?\s*\)"""
)
# Settings field:  ^    name: type = default   (class-body indent, lowercase field)
FIELD_RE = re.compile(r"^    ([a-z][a-z0-9_]*)\s*:\s*[^\s=]+\s*=\s*(.+?)\s*(?:#.*)?$")


def iter_py() -> list[Path]:
    return sorted(
        p for p in BACKEND.rglob("*.py") if "__pycache__" not in p.parts
    )


def claude_documentation_text() -> str:
    """Read the root plus owned nested CLAUDE docs.

    Flag ownership is deliberately split by subsystem, so a registry that only
    reads the lean root document falsely marks documented sandbox/RLM flags as
    undocumented.  Keep the same exclusions as the doc-fidelity guard.
    """
    skip = {"runs", "node_modules", ".venv", ".git", "openscience-ref", "site-packages"}
    texts: list[str] = []
    for path in sorted(REPO.glob("**/CLAUDE.md")):
        rel = path.relative_to(REPO)
        if any(part in skip for part in rel.parts):
            continue
        texts.append(path.read_text(errors="replace"))
    return "\n".join(texts)


def config_managed_flags() -> dict[str, str]:
    """Map OPENRESEARCH_<FIELD> -> default literal for every Settings field."""
    managed: dict[str, str] = {}
    in_settings = False
    for line in CONFIG.read_text().splitlines():
        if re.match(r"class \w+\(BaseSettings\)", line):
            in_settings = True
            continue
        if in_settings and line and not line[0].isspace() and line.startswith("def "):
            break
        m = FIELD_RE.match(line)
        if m:
            name, default = m.group(1), m.group(2).strip()
            if default.startswith("Field("):
                default = "Field(...)"
            managed["OPENRESEARCH_" + name.upper()] = default
    return managed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="exit 1 if flags.md is stale")
    args = ap.parse_args()

    managed = config_managed_flags()
    claude_txt = claude_documentation_text()

    all_flags: set[str] = set()
    read_sites: dict[str, int] = defaultdict(int)
    adhoc_default: dict[str, str] = {}
    adhoc_files: dict[str, set[str]] = defaultdict(set)

    for path in iter_py():
        text = path.read_text(errors="replace")
        for f in FLAG_RE.findall(text):
            all_flags.add(f)
        rel = str(path.relative_to(REPO))
        for m in READ_RE.finditer(text):
            flag = m.group(1)
            read_sites[flag] += 1
            adhoc_files[flag].add(rel)
            if flag not in adhoc_default and m.group("default"):
                adhoc_default[flag] = m.group("default").strip()

    all_flags |= set(managed)

    families: dict[str, list[str]] = defaultdict(list)
    for f in all_flags:
        parts = f.split("_")
        fam = "_".join(parts[:2]) if len(parts) > 2 else f
        families[fam].append(f)

    n_managed = sum(1 for f in all_flags if f in managed)
    n_adhoc = sum(1 for f in all_flags if f not in managed)
    n_documented = sum(1 for f in all_flags if f in claude_txt)

    lines: list[str] = []
    lines.append("# OPENRESEARCH_* Feature-Flag Registry")
    lines.append("")
    lines.append(
        "> **Generated — do not hand-edit.** Regenerate with "
        "`.venv/bin/python scripts/gen_flag_registry.py`. "
        "Source of truth is the code, not this file."
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Total distinct flags:** {len(all_flags)}")
    lines.append(f"- **Managed by `config.py` Settings (typed, default known):** {n_managed}")
    lines.append(f"- **Ad-hoc `os.environ` reads (no central default):** {n_adhoc}")
    lines.append(f"- **Mentioned in a `CLAUDE.md`:** {n_documented} ({100 * n_documented // max(len(all_flags), 1)}%)")
    lines.append("")
    lines.append("Legend: **cfg** = typed in `config.py`; **doc** = appears in a `CLAUDE.md`; "
                 "**sites** = ad-hoc read count; **default** = literal default at first read site (best effort).")
    lines.append("")

    # Actionable: a flag that is BOTH typed in config.py AND still read directly
    # from os.environ is a "config field bypassed" case — the typed default is
    # dead because call sites re-read the env. These are the safe-ish
    # consolidation candidates (verify per-flag: many are read at call-time on
    # purpose for test monkeypatching / per-run toggling).
    bypassed = sorted(f for f in all_flags if f in managed and read_sites.get(f, 0) > 0)
    lines.append("## Config fields bypassed by ad-hoc reads")
    lines.append("")
    lines.append(
        f"{len(bypassed)} flags are typed in `config.py` **and** still read directly via "
        "`os.environ`, so the typed default is dead at those call sites. Review before "
        "consolidating — some are call-time reads on purpose (test monkeypatch / per-run toggle)."
    )
    lines.append("")
    if bypassed:
        lines.append("| Flag | cfg default | ad-hoc reads |")
        lines.append("|---|---|:--:|")
        for f in bypassed:
            d = managed.get(f, "").replace("|", "\\|")[:40]
            lines.append(f"| `{f}` | `{d}` | {read_sites[f]} |")
        lines.append("")

    for fam in sorted(families):
        flags = sorted(families[fam])
        lines.append(f"### `{fam}_*`")
        lines.append("")
        lines.append("| Flag | cfg | doc | sites | default |")
        lines.append("|---|:--:|:--:|:--:|---|")
        for f in flags:
            cfg = "✅" if f in managed else ""
            doc = "✅" if f in claude_txt else ""
            sites = read_sites.get(f, 0)
            default = managed.get(f) or adhoc_default.get(f) or ""
            default = default.replace("|", "\\|")[:40]
            lines.append(f"| `{f}` | {cfg} | {doc} | {sites or ''} | `{default}` |" if default
                         else f"| `{f}` | {cfg} | {doc} | {sites or ''} | |")
        lines.append("")

    content = "\n".join(lines) + "\n"

    if args.check:
        existing = OUT.read_text() if OUT.exists() else ""
        if existing.strip() != content.strip():
            print("flags.md is STALE — run: .venv/bin/python scripts/gen_flag_registry.py", file=sys.stderr)
            return 1
        print("flags.md is up to date.")
        return 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(content)
    print(f"Wrote {OUT.relative_to(REPO)}: {len(all_flags)} flags "
          f"({n_managed} cfg-managed, {n_adhoc} ad-hoc, {n_documented} documented).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
