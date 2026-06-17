#!/usr/bin/env bash
# build_orchestrator_image.sh — Build and push the ReproLab orchestrator
# (control-plane) image to Google Artifact Registry.
#
# This is the image the GKE/AKS orchestrator Deployment + CronJob run
# (infra/gcp/helm orchestrator.image).  It is built from docker/orchestrator/
# Dockerfile (Python backend + Node + the pinned `claude` CLI; NO CUDA).
#
# Usage:
#   scripts/build_orchestrator_image.sh <ar-host-and-repo> [tag]
#
# Where <ar-host-and-repo> is the Artifact Registry host + project + repo, i.e.
#   REGION-docker.pkg.dev/PROJECT/REPO
# (the same shape as `terraform -chdir=infra/gcp output -raw artifact_registry_url`).
# It is NOT hardcoded here — pass it in, or set ORCHESTRATOR_AR_REPO in the env.
#
# Examples:
#   scripts/build_orchestrator_image.sh us-central1-docker.pkg.dev/my-proj/reprolab
#   scripts/build_orchestrator_image.sh us-central1-docker.pkg.dev/my-proj/reprolab sha-abc1234
#   ORCHESTRATOR_AR_REPO=us-central1-docker.pkg.dev/my-proj/reprolab \
#       scripts/build_orchestrator_image.sh
#
# On success, prints the orchestrator.image= line to copy into the helm install.
# NEVER use :latest — always a pinned tag (git short SHA by default).

set -euo pipefail

# ─── Args ─────────────────────────────────────────────────────────────────────
AR_REPO="${1:-${ORCHESTRATOR_AR_REPO:-}}"
TAG="${2:-${ORCHESTRATOR_IMAGE_TAG:-}}"

# The repo's own name within the AR host (the image basename).
IMAGE_NAME="reprolab-orchestrator"

if [[ -z "$AR_REPO" ]]; then
  echo "ERROR: Artifact Registry host+repo is required." >&2
  echo "Usage: $0 <REGION-docker.pkg.dev/PROJECT/REPO> [tag]" >&2
  echo "   or: ORCHESTRATOR_AR_REPO=<...> $0 [tag]" >&2
  echo "Get it from: terraform -chdir=infra/gcp output -raw artifact_registry_url" >&2
  exit 1
fi

# Strip a trailing slash if the caller passed one.
AR_REPO="${AR_REPO%/}"

# ─── Resolve to the repo root (so the build context + Dockerfile resolve) ─────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DOCKERFILE="${REPO_ROOT}/docker/orchestrator/Dockerfile"

if [[ ! -f "$DOCKERFILE" ]]; then
  echo "ERROR: Dockerfile not found at ${DOCKERFILE}" >&2
  exit 1
fi

# ─── Dependency check ─────────────────────────────────────────────────────────
if ! command -v docker &>/dev/null; then
  echo "ERROR: 'docker' is not installed or not on PATH." >&2
  exit 1
fi

# ─── Resolve tag (default: git short SHA) ─────────────────────────────────────
if [[ -z "$TAG" ]]; then
  if ! command -v git &>/dev/null; then
    echo "ERROR: 'git' not found and no tag argument supplied." >&2
    echo "Usage: $0 <REGION-docker.pkg.dev/PROJECT/REPO> <tag>" >&2
    exit 1
  fi
  TAG="sha-$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null)" || {
    echo "ERROR: Not in a git repo and no tag argument supplied." >&2
    echo "Usage: $0 <REGION-docker.pkg.dev/PROJECT/REPO> <tag>" >&2
    exit 1
  }
fi

if [[ "$TAG" == "latest" ]]; then
  echo "ERROR: refusing to push :latest — pass a pinned tag (git SHA or version)." >&2
  exit 1
fi

IMAGE_REF="${AR_REPO}/${IMAGE_NAME}:${TAG}"

# ─── Configure docker auth for the AR host (best-effort, one-time) ────────────
# REGION-docker.pkg.dev → the host docker must authenticate against.
AR_HOST="${AR_REPO%%/*}"
if command -v gcloud &>/dev/null; then
  echo "Configuring docker auth for ${AR_HOST} (gcloud auth configure-docker)..."
  gcloud auth configure-docker "$AR_HOST" --quiet || \
    echo "WARN: 'gcloud auth configure-docker ${AR_HOST}' failed; assuming docker is already authed." >&2
else
  echo "WARN: 'gcloud' not found; assuming docker is already authenticated to ${AR_HOST}." >&2
fi

# ─── Build and push ───────────────────────────────────────────────────────────
echo "Building orchestrator control-plane image..."
echo "  Registry:   ${AR_HOST}"
echo "  Image:      ${IMAGE_NAME}:${TAG}"
echo "  Dockerfile: ${DOCKERFILE}"
echo "  Context:    ${REPO_ROOT}"
echo ""

docker build \
  -f "$DOCKERFILE" \
  -t "$IMAGE_REF" \
  "$REPO_ROOT"

echo ""
echo "Pushing ${IMAGE_REF}..."
docker push "$IMAGE_REF"

echo ""
echo "Build + push complete. Use the following in your helm install/upgrade:"
echo ""
echo "  --set orchestrator.image=${IMAGE_REF}"
echo ""
echo "IMPORTANT: Never use :latest — the tag '${TAG}' is pinned."
echo "  Re-run this script after each change to docker/orchestrator/Dockerfile"
echo "  or backend/ to get a new pinned tag."
