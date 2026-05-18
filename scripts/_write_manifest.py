"""Write a manifest.json describing every file under a launch's log dir.

Invoked by scripts/dev.sh on cleanup. Walks the directory, hashes each file
with sha256, counts lines for text-shaped files, and emits a single JSON doc
to <log_dir>/manifest.json.

Stdlib only. Best-effort: if a file disappears mid-walk (e.g. uvicorn writes
to it as we read), we skip it and keep going. Never raises to the caller —
manifest is a convenience, not load-bearing.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

TEXT_SUFFIXES = {".log", ".jsonl", ".json", ".md", ".txt", ".yml", ".yaml"}
SKIP_NAMES = {"manifest.json"}
CHUNK = 1 << 16


def _hash_and_lines(path: Path, want_lines: bool) -> tuple[str, int | None, int]:
    h = hashlib.sha256()
    size = 0
    lines = 0 if want_lines else None
    with path.open("rb") as f:
        while True:
            chunk = f.read(CHUNK)
            if not chunk:
                break
            h.update(chunk)
            size += len(chunk)
            if want_lines:
                lines += chunk.count(b"\n")
    return h.hexdigest(), lines, size


def build_manifest(root: Path) -> dict:
    entries = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.name in SKIP_NAMES:
            continue
        try:
            want_lines = path.suffix.lower() in TEXT_SUFFIXES
            sha, lines, size = _hash_and_lines(path, want_lines)
        except (OSError, FileNotFoundError):
            continue
        rel = path.relative_to(root).as_posix()
        entry = {"path": rel, "size": size, "sha256": sha}
        if lines is not None:
            entry["lines"] = lines
        entries.append(entry)
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "root": str(root),
        "file_count": len(entries),
        "total_bytes": sum(e["size"] for e in entries),
        "files": entries,
    }


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: _write_manifest.py <log_dir>", file=sys.stderr)
        return 2
    root = Path(argv[1]).resolve()
    if not root.is_dir():
        print(f"_write_manifest.py: not a directory: {root}", file=sys.stderr)
        return 1
    manifest = build_manifest(root)
    out = root / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"manifest -> {out} ({manifest['file_count']} files, {manifest['total_bytes']} bytes)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except Exception as exc:
        print(f"_write_manifest.py: best-effort failure: {exc}", file=sys.stderr)
        sys.exit(0)
