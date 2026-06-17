#!/usr/bin/env bash
# azure_build_cell_image.sh — Build and push the reprolab cell base image to ACR
#
# Usage:
#   scripts/azure_build_cell_image.sh <acr-name-or-login-server> [tag=<git-short-sha>]
#
# Examples:
#   scripts/azure_build_cell_image.sh sciartacr
#   scripts/azure_build_cell_image.sh sciartacr.azurecr.io
#   scripts/azure_build_cell_image.sh sciartacr abc1234
#
# On success, prints the OPENRESEARCH_AZURE_BASE_IMAGE= line to copy into .env.
# NEVER use :latest — always use a pinned tag (git SHA or explicit version).

set -euo pipefail

# ─── Args ─────────────────────────────────────────────────────────────────────
ACR_INPUT="${1:-}"
TAG="${2:-}"

if [[ -z "$ACR_INPUT" ]]; then
  echo "ERROR: ACR name or login server is required." >&2
  echo "Usage: $0 <acr-name-or-login-server> [tag=<git-short-sha>]" >&2
  exit 1
fi

# ─── Dependency check ─────────────────────────────────────────────────────────
if ! command -v az &>/dev/null; then
  echo "ERROR: 'az' (Azure CLI) is not installed or not on PATH." >&2
  echo "Install: https://learn.microsoft.com/en-us/cli/azure/install-azure-cli" >&2
  exit 1
fi

# ─── Auth check ───────────────────────────────────────────────────────────────
if ! az account show &>/dev/null; then
  echo "ERROR: Not logged in to Azure CLI. Run: az login" >&2
  exit 1
fi

# ─── Resolve ACR name vs full login server ────────────────────────────────────
# Accept either "sciartacr" (name) or "sciartacr.azurecr.io" (login server).
if [[ "$ACR_INPUT" == *.azurecr.io ]]; then
  ACR_LOGIN_SERVER="$ACR_INPUT"
  ACR_NAME="${ACR_INPUT%.azurecr.io}"
else
  ACR_NAME="$ACR_INPUT"
  ACR_LOGIN_SERVER="${ACR_INPUT}.azurecr.io"
fi

# ─── Resolve tag (default: git short SHA) ─────────────────────────────────────
if [[ -z "$TAG" ]]; then
  if ! command -v git &>/dev/null; then
    echo "ERROR: 'git' not found and no tag argument supplied." >&2
    echo "Usage: $0 <acr-name-or-login-server> <tag>" >&2
    exit 1
  fi
  TAG=$(git rev-parse --short HEAD 2>/dev/null) || {
    echo "ERROR: Not in a git repo and no tag argument supplied." >&2
    echo "Usage: $0 <acr-name-or-login-server> <tag>" >&2
    exit 1
  }
fi

IMAGE_REF="${ACR_LOGIN_SERVER}/reprolab-cell:${TAG}"

# ─── Verify Dockerfile exists ────────────────────────────────────────────────
DOCKERFILE="docker/aks-cell-base/Dockerfile"
CONTEXT_DIR="docker/aks-cell-base/"

if [[ ! -f "$DOCKERFILE" ]]; then
  echo "ERROR: Dockerfile not found at ${DOCKERFILE}" >&2
  echo "  Run from the repo root, or check that docker/aks-cell-base/ exists." >&2
  exit 1
fi

# ─── Build and push ───────────────────────────────────────────────────────────
echo "Building and pushing cell image via ACR Build Tasks..."
echo "  Registry:   ${ACR_LOGIN_SERVER}"
echo "  Image:      reprolab-cell:${TAG}"
echo "  Dockerfile: ${DOCKERFILE}"
echo ""

az acr build \
  --registry "$ACR_NAME" \
  --image "reprolab-cell:${TAG}" \
  --file "$DOCKERFILE" \
  "$CONTEXT_DIR"

echo ""
echo "Build complete. Add the following to your .env:"
echo ""
echo "  OPENRESEARCH_AZURE_BASE_IMAGE=${IMAGE_REF}"
echo ""
echo "IMPORTANT: Never use :latest — always pin a specific tag."
echo "  The tag '${TAG}' is derived from the current git commit."
echo "  Re-run this script after each Dockerfile change to get a new pinned tag."
