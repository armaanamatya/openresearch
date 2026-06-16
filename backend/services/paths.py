"""Cross-platform path normalization for user-provided file paths.

The lab UI / CLI accept paths that may originate from a different host OS
(e.g. a Windows path pasted into a WSL backend). This module transparently
converts those paths to the host's filesystem convention.

Key rule: only transform values that LOOK like file paths. Strings that
look like arXiv IDs, URLs, or DOIs are returned unchanged — callers
upstream of this module distinguish those.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess


# Windows absolute path: drive letter + colon + slash or backslash.
_WIN_ABS_RE = re.compile(r"^([A-Za-z]):[\\/]")

# WSL UNC path from Windows. The distro name is captured but discarded —
# we assume we're inside the distro the path is rooted at. This is fine
# for the common single-distro case; multi-distro users would need to
# adjust manually.
_WSL_UNC_RE = re.compile(
    r"^\\\\wsl(?:\$|\.localhost)\\[^\\]+\\(.*)$",
    re.IGNORECASE,
)


def normalize_path_input(value: str) -> str:
    """Normalize a user-provided path across Windows / WSL / macOS.

    Behaviour:
      * Strips surrounding quotes (cmd / Explorer paste convention).
      * Expands ``~`` to the user's home on POSIX hosts.
      * Converts Windows absolute paths to WSL mount paths on POSIX hosts:
        ``C:\\X\\Y`` -> ``/mnt/c/X/Y``. Uses ``wslpath -u`` when available
        for the most accurate conversion; falls back to pure-Python parsing
        when the binary is absent (Linux without WSL utilities, macOS).
      * Converts ``\\\\wsl$\\<distro>\\<path>`` UNC paths to ``/<path>`` on
        POSIX hosts.
      * Returns the input unchanged if it doesn't match any path pattern
        (e.g. arXiv IDs, URLs, DOIs).

    Args:
        value: A user-provided string.

    Returns:
        The normalized string. Identity-preserving for non-path inputs and
        for paths already canonical on the host platform.
    """
    if not isinstance(value, str) or not value:
        return value
    s = value.strip()

    # Strip surrounding double or single quotes.
    if len(s) >= 2 and (
        (s.startswith('"') and s.endswith('"'))
        or (s.startswith("'") and s.endswith("'"))
    ):
        s = s[1:-1].strip()

    posix_host = os.name == "posix"

    # Case 1: WSL UNC path from Windows on a POSIX host.
    m = _WSL_UNC_RE.match(s)
    if m and posix_host:
        inner = m.group(1).replace("\\", "/")
        if not inner.startswith("/"):
            inner = "/" + inner
        return inner

    # Case 2: Windows absolute path on a POSIX host.
    m = _WIN_ABS_RE.match(s)
    if m and posix_host:
        # Prefer wslpath when available (handles edge cases natively).
        if shutil.which("wslpath"):
            try:
                result = subprocess.run(
                    ["wslpath", "-u", s],
                    capture_output=True, text=True, timeout=5, check=False,
                )
                if result.returncode == 0 and result.stdout.strip():
                    return result.stdout.strip()
            except (subprocess.SubprocessError, OSError):
                pass  # Fall through to pure-Python conversion.
        # Pure-Python fallback. Drive letter is lowercased to match WSL convention.
        drive = m.group(1).lower()
        rest = s[m.end():].replace("\\", "/")
        if not rest.startswith("/"):
            rest = "/" + rest
        return f"/mnt/{drive}{rest}"

    # Case 3: tilde-home on POSIX hosts.
    if posix_host and s.startswith("~"):
        return os.path.expanduser(s)

    # Case 4: already canonical for the host platform — return unchanged.
    return s


__all__ = ["normalize_path_input"]
