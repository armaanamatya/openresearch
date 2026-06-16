#!/usr/bin/env bash
# load_env_file [ENV_FILE] [PYTHON_BIN] — export KEY=VALUE pairs from a .env
# file into the current shell WITHOUT source-ing it (values are data, not
# code). The parse is delegated to python-dotenv — the exact parser
# pydantic-settings uses — so the process env this exports can never diverge
# from what the backend would read from the file itself: inline comments,
# quoting, `export ` prefixes, spaces around '=', CRLF endings, multi-line
# quoted values, and duplicate-key last-wins all behave identically
# (audit 2026-06-09: a hand-rolled bash parser kept inline `# comments`
# inside values; the corrupted export OUTRANKED pydantic's correct file parse
# and crashed Settings() on Literal fields — a container restart loop
# triggered by .env.example's own suggested header line).
#
# Vars already present in the environment (compose `environment:`,
# `docker run -e`) keep winning over .env — normal compose precedence.
#
# Usage:  . /load_env.sh && load_env_file /app/.env /opt/venv/bin/python
load_env_file() {
    local env_file="${1:-/app/.env}"
    local python_bin="${2:-/opt/venv/bin/python}"
    [[ -f "${env_file}" ]] || return 0
    if ! command -v "${python_bin}" >/dev/null 2>&1; then
        echo "[load_env] WARNING: ${python_bin} not found; NOT loading ${env_file}" >&2
        return 0
    fi
    local _kv _key _value
    # NUL-delimited so values may contain anything but NUL (incl. newlines).
    while IFS= read -r -d '' _kv; do
        _key="${_kv%%=*}"
        _value="${_kv#*=}"
        if [[ -z "${!_key+x}" ]]; then
            export "${_key}=${_value}"
        fi
    done < <("${python_bin}" - "${env_file}" <<'PYEOF'
import re
import sys

from dotenv import dotenv_values

for key, value in dotenv_values(sys.argv[1]).items():
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key or ""):
        # Not a valid shell identifier — exporting it would crash the shell.
        continue
    sys.stdout.write(f"{key}={'' if value is None else value}\x00")
PYEOF
)
    unset _kv _key _value
}
