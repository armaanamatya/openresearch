"""UTF-8-safe JSON I/O helpers.

Cross-platform encoding hardening — see docs/design/cross-platform-encoding-fix.md.

On Windows, `Path.write_text(...)` / `Path.read_text()` default to cp1252,
which crashes on `→`, `—`, Greek letters, smart quotes, etc. — all of which
routinely appear in audit JSON, claim maps, environment specs, and final
reports. macOS (UTF-8 since Catalina) and most Linux distros run UTF-8 by
default, so the bug ships green there and explodes on Windows.

Use these helpers for every JSON file the backend writes or reads.

    from backend.utils.io import write_json, read_json
    write_json(path, payload)
    payload = read_json(path)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(path: Path, payload: Any, *, indent: int = 2) -> None:
    """Atomically-friendly JSON write with UTF-8 and ensure_ascii=False.

    `ensure_ascii=False` keeps the on-disk JSON readable when it contains
    non-Latin-1 text — UTF-8 handles those bytes natively.
    """
    path.write_text(
        json.dumps(payload, indent=indent, ensure_ascii=False),
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    """Read a JSON file written with utf-8 encoding."""
    return json.loads(path.read_text(encoding="utf-8"))
