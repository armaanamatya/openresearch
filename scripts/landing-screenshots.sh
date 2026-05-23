#!/usr/bin/env bash
set -euo pipefail

URL="${1:-http://localhost:3001}"
NAME="${2:-page}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$ROOT/frontend/tmp-screens"

mkdir -p "$OUT_DIR"

(
  cd "$ROOT/frontend"
  npx playwright screenshot "$URL" "$OUT_DIR/${NAME}-full.png" --viewport-size=1440,900 --full-page
  npx playwright screenshot "$URL" "$OUT_DIR/${NAME}-mobile.png" --viewport-size=390,844 --full-page
)

printf 'OK -> frontend/tmp-screens/%s-*.png\n' "$NAME"
