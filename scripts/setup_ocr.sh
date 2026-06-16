#!/usr/bin/env bash
# setup_ocr.sh — provision Tesseract language data for the OCR paper-parser fallback.
#
# The OCR fallback (backend/services/ingestion/parser/ocr_parser.py) is the last
# stage of the ingestion cascade (HTML > PDF > OCR). It needs the Tesseract `eng`
# language data, which is absent in some environments.
#
# Known environment wrinkle: when the only `tesseract` on PATH is a confined
# *snap*, it (a) ships no language data and (b) runs in a private mount namespace
# so it cannot read the host /tmp where pytesseract stages page images. The snap
# CAN read $HOME (via the `home` interface), so this script installs the language
# data under $HOME and prints the two env vars the snap needs.
#
# The OCR fallback fails soft when unprovisioned; tests/test_ingestion_ocr_parser.py
# skips with an accurate reason. Run this script, export the printed vars, and the
# OCR fallback (and its test) become functional.
set -euo pipefail

TESSDATA_DIR="${HOME}/.local/share/tessdata"
ENG="${TESSDATA_DIR}/eng.traineddata"
# tessdata_fast: ~4 MB, fast and adequate for paper-text OCR.
URL="https://github.com/tesseract-ocr/tessdata_fast/raw/main/eng.traineddata"

mkdir -p "${TESSDATA_DIR}"
if [ -s "${ENG}" ]; then
    echo "eng.traineddata already present: ${ENG}"
else
    echo "Downloading eng.traineddata -> ${ENG}"
    curl -fSL -o "${ENG}" "${URL}"
fi
ls -la "${ENG}"

cat <<NOTE

OCR language data provisioned. Export these so the parser (and its test) run:

    export TESSDATA_PREFIX="${TESSDATA_DIR}"

If 'tesseract' on PATH is a confined snap — check with
'readlink -f \$(which tesseract)'; a /snap/... path means yes — it also cannot
read the host /tmp, so point the temp dir at a \$HOME location:

    mkdir -p "\${HOME}/.cache/openresearch_ocr_tmp"
    export TMPDIR="\${HOME}/.cache/openresearch_ocr_tmp"

A non-snap 'apt install tesseract-ocr tesseract-ocr-eng' avoids both wrinkles.
NOTE
