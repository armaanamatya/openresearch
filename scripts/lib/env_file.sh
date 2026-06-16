#!/usr/bin/env bash
# env_value_from_file KEY FILE — print KEY's value from a dotenv-format file.
#
# Faithful to python-dotenv's grammar for the constructs .env.example uses.
# This matters because the caller (start.sh) exports the result as process
# env, which pydantic-settings ranks ABOVE its own env_file parse — so any
# divergence here silently shadows the correct value, and a Literal-typed
# field (e.g. OPENRESEARCH_DEFAULT_SANDBOX) turns the divergence into a hard
# ValidationError at boot (audit 2026-06-09: the .env.example header's own
# suggested line `OPENRESEARCH_DEFAULT_SANDBOX=local   # no Docker...` used
# to export the comment as part of the value).
#
# Grammar covered (pinned to dotenv_values in
# tests/scripts/test_env_file_parsers.py):
#   - optional `export ` prefix; optional spaces around `=`
#   - LAST occurrence of a duplicated key wins
#   - unquoted values: inline ` # comment` stripped (the `#` must follow
#     whitespace), trailing whitespace trimmed
#   - single/double-quoted values keep embedded `#`; a comment may follow
#     the closing quote
#   - CRLF tolerated (trailing \r stripped per line)
# Known limitation: multi-line quoted values are not supported (not used in
# .env.example; use the python-dotenv-delegating docker/load_env.sh where a
# venv is guaranteed).
# Compatible with macOS bash 3.2. Exit 1 when the key is absent.
env_value_from_file() {
    local key="$1"
    local file="$2"
    [[ -f "${file}" ]] || return 1
    local line value found=0
    while IFS= read -r line || [[ -n "$line" ]]; do
        line="${line%$'\r'}"
        [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
        if [[ "$line" =~ ^[[:space:]]*(export[[:space:]]+)?${key}[[:space:]]*=[[:space:]]*(.*)$ ]]; then
            value="${BASH_REMATCH[2]}"
            if [[ "$value" =~ ^\"([^\"]*)\"[[:space:]]*(#.*)?$ ]]; then
                value="${BASH_REMATCH[1]}"
            elif [[ "$value" =~ ^\'([^\']*)\'[[:space:]]*(#.*)?$ ]]; then
                value="${BASH_REMATCH[1]}"
            else
                # dotenv semantics: a comment starts at whitespace+'#'
                # (KEY=local# keeps the '#'); then trim trailing whitespace.
                value="${value%%[[:space:]]#*}"
                value="${value%"${value##*[![:space:]]}"}"
            fi
            found=1
        fi
    done < "${file}"
    [[ "${found}" == "1" ]] || return 1
    printf "%s" "${value}"
}
