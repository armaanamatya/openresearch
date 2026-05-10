#!/usr/bin/env bash
# Runpod preflight + (optional) end-to-end smoke for the openresearch pipeline.
#
# This script intentionally hits the same REST endpoint that
# backend/services/runtime/runpod_backend.py uses, with the same bearer-token
# auth scheme, so a green run here means `--sandbox runpod` will authenticate
# successfully when you launch a pipeline.
#
# Usage:
#   scripts/runpod_check.sh                # preflight only (read-only, free)
#   scripts/runpod_check.sh --start-pod    # also boots a tiny pod, runs
#                                          # nvidia-smi over SSH, then destroys
#                                          # it (COSTS MONEY — minutes-scale)
#
# Exit codes:
#   0  everything green
#   1  bad usage / unexpected error
#   2  required env var missing
#   3  RunPod API auth failed
#   4  configured GPU type not currently offered
#   5  SSH key missing / wrong permissions / mismatched pair
#   6  --start-pod end-to-end smoke failed

set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve repo root (script lives in <repo>/scripts/) and load .env
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${REPO_ROOT}/.env"

START_POD=0
for arg in "$@"; do
    case "$arg" in
        --start-pod) START_POD=1 ;;
        -h|--help)
            sed -n '2,18p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg" >&2
            exit 1
            ;;
    esac
done

if [[ ! -f "${ENV_FILE}" ]]; then
    echo "FAIL  .env not found at ${ENV_FILE}" >&2
    exit 2
fi

# Load .env without exporting its values to the user's interactive shell — only
# this script process inherits them. `set -a` would export everything; we use
# explicit per-line export to keep blast radius small.
while IFS= read -r line || [[ -n "$line" ]]; do
    # Skip blank lines and comments.
    [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
    # Only export simple KEY=VALUE pairs, ignoring anything that looks shell-y.
    if [[ "$line" =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then
        key="${BASH_REMATCH[1]}"
        value="${BASH_REMATCH[2]}"
        # Strip surrounding single or double quotes.
        if [[ "$value" =~ ^\"(.*)\"$ ]]; then value="${BASH_REMATCH[1]}"; fi
        if [[ "$value" =~ ^\'(.*)\'$ ]]; then value="${BASH_REMATCH[1]}"; fi
        export "${key}=${value}"
    fi
done < "${ENV_FILE}"

# ---------------------------------------------------------------------------
# Pretty printers
# ---------------------------------------------------------------------------
if [[ -t 1 ]]; then
    GREEN=$'\033[0;32m'; RED=$'\033[0;31m'; YELLOW=$'\033[0;33m'; DIM=$'\033[2m'; RESET=$'\033[0m'
else
    GREEN=""; RED=""; YELLOW=""; DIM=""; RESET=""
fi
ok()    { printf "  ${GREEN}OK${RESET}    %s\n" "$*"; }
warn()  { printf "  ${YELLOW}WARN${RESET}  %s\n" "$*"; }
fail()  { printf "  ${RED}FAIL${RESET}  %s\n" "$*" >&2; }
step()  { printf "\n${DIM}== %s ==${RESET}\n" "$*"; }

normalize_ssh_key_path() {
    local raw="$1"
    # Convert Windows-style absolute paths when this script runs in WSL/Linux.
    # C:\Users\name\.ssh\id_ed25519 -> /mnt/c/Users/name/.ssh/id_ed25519
    if [[ "$raw" =~ ^([A-Za-z]):\\ ]]; then
        local drive="${BASH_REMATCH[1],,}"
        local tail="${raw:2}"
        tail="${tail//\\//}"
        printf "/mnt/%s/%s" "$drive" "${tail#/}"
        return 0
    fi
    printf "%s" "$raw"
}

mask() {
    # Show first 4 + last 4 chars of a secret, with the middle masked.
    local s="$1"
    local n="${#s}"
    if (( n <= 8 )); then
        printf "****"
    else
        printf "%s...%s" "${s:0:4}" "${s: -4}"
    fi
}

require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        fail "Required command not found: $1"
        exit 1
    fi
}

require_cmd curl
require_cmd python3

# Tiny JSON helpers backed by python3 stdlib so we don't depend on jq.
# Usage:  json_get '<path expr>' < file        # e.g. json_get 'len(d)'
#         json_eval '<expr>' < file            # raw python expression on `d`
json_eval() {
    local expr="$1"
    python3 -c "
import json, sys
d = json.load(sys.stdin)
val = ${expr}
if isinstance(val, (dict, list)):
    print(json.dumps(val))
elif val is None:
    print('')
else:
    print(val)
"
}

# ---------------------------------------------------------------------------
# 1. Required env vars
# ---------------------------------------------------------------------------
step "1. Environment"

# Apply config.py defaults so the user can leave them unset in .env.
: "${REPROLAB_RUNPOD_API_BASE_URL:=https://rest.runpod.io/v1}"
: "${REPROLAB_RUNPOD_IMAGE:=runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04}"
: "${REPROLAB_RUNPOD_GPU_TYPE:=NVIDIA GeForce RTX 4090}"
: "${REPROLAB_RUNPOD_GPU_COUNT:=1}"
: "${REPROLAB_RUNPOD_CLOUD_TYPE:=SECURE}"
: "${REPROLAB_RUNPOD_CONTAINER_DISK_GB:=50}"
: "${REPROLAB_RUNPOD_VOLUME_GB:=20}"
: "${REPROLAB_RUNPOD_VOLUME_MOUNT_PATH:=/workspace}"
: "${REPROLAB_RUNPOD_NETWORK_VOLUME_ID:=}"
: "${REPROLAB_RUNPOD_DATA_CENTER_IDS:=}"
: "${REPROLAB_RUNPOD_SSH_USER:=root}"
: "${REPROLAB_RUNPOD_BOOT_TIMEOUT_SECONDS:=900}"
: "${REPROLAB_RUNPOD_DELETE_ON_DESTROY:=true}"

# Fall back to RUNPOD_API_KEY if REPROLAB_RUNPOD_API_KEY is unset
# (mirrors RunpodBackend.__init__).
API_KEY="${REPROLAB_RUNPOD_API_KEY:-${RUNPOD_API_KEY:-}}"

if [[ -z "${API_KEY}" ]]; then
    fail "REPROLAB_RUNPOD_API_KEY (or RUNPOD_API_KEY) is empty"
    exit 2
fi
ok "API key set ($(mask "${API_KEY}"))"

if [[ -z "${REPROLAB_RUNPOD_SSH_KEY_PATH:-}" ]]; then
    fail "REPROLAB_RUNPOD_SSH_KEY_PATH is empty"
    exit 2
fi
ok "SSH key path set (${REPROLAB_RUNPOD_SSH_KEY_PATH})"

if [[ -z "${REPROLAB_RUNPOD_SSH_PUBLIC_KEY:-}" ]]; then
    SSH_KEY_CANDIDATE="$(eval echo "${REPROLAB_RUNPOD_SSH_KEY_PATH}")"  # expand ~
    if [[ -f "${SSH_KEY_CANDIDATE}" ]] && command -v ssh-keygen >/dev/null 2>&1; then
        DERIVED_PUBLIC_KEY="$(ssh-keygen -y -f "${SSH_KEY_CANDIDATE}" 2>/dev/null || true)"
        if [[ -n "${DERIVED_PUBLIC_KEY}" ]]; then
            REPROLAB_RUNPOD_SSH_PUBLIC_KEY="${DERIVED_PUBLIC_KEY}"
            export REPROLAB_RUNPOD_SSH_PUBLIC_KEY
            ok "SSH public key derived from private key"
        fi
    fi
fi
if [[ -z "${REPROLAB_RUNPOD_SSH_PUBLIC_KEY:-}" ]]; then
    fail "REPROLAB_RUNPOD_SSH_PUBLIC_KEY is empty (and could not be derived from private key)"
    exit 2
fi
ok "SSH public key available"

ok "GPU type:   ${REPROLAB_RUNPOD_GPU_TYPE}"
ok "GPU count:  ${REPROLAB_RUNPOD_GPU_COUNT}"
ok "Cloud type: ${REPROLAB_RUNPOD_CLOUD_TYPE}"
ok "Image:      ${REPROLAB_RUNPOD_IMAGE}"
ok "API base:   ${REPROLAB_RUNPOD_API_BASE_URL}"

# ---------------------------------------------------------------------------
# 2. RunPod API auth (GET /pods)
# ---------------------------------------------------------------------------
step "2. RunPod API authentication"

AUTH_URL="${REPROLAB_RUNPOD_API_BASE_URL%/}/pods"
AUTH_BODY="$(mktemp)"
AUTH_CODE="$(curl -sS -o "${AUTH_BODY}" -w "%{http_code}" \
    -H "Authorization: Bearer ${API_KEY}" \
    -H "Accept: application/json" \
    "${AUTH_URL}" || true)"

if [[ "${AUTH_CODE}" != "200" ]]; then
    fail "GET ${AUTH_URL} returned HTTP ${AUTH_CODE}"
    echo "       body:" >&2
    sed 's/^/         /' "${AUTH_BODY}" >&2 || true
    rm -f "${AUTH_BODY}"
    exit 3
fi

POD_COUNT="$(json_eval 'len(d) if isinstance(d, list) else (len(d.get("data") or d.get("pods") or []))' < "${AUTH_BODY}" 2>/dev/null || echo "?")"
ok "Auth OK — account currently has ${POD_COUNT} pod(s)"
rm -f "${AUTH_BODY}"

# ---------------------------------------------------------------------------
# 3. GPU availability
# ---------------------------------------------------------------------------
# RunPod's REST v1 API does not expose a GPU type listing endpoint
# (that lives on the legacy GraphQL API at https://api.runpod.io/graphql).
# The configured GPU id is validated by the API at pod creation time, so the
# only way to definitively confirm availability is `--start-pod`.
step "3. GPU offering check"
warn "REST v1 doesn't expose GPU listings — '${REPROLAB_RUNPOD_GPU_TYPE}' will be"
warn "validated at pod creation. Use --start-pod for an end-to-end smoke test."

# ---------------------------------------------------------------------------
# 4. SSH key sanity
# ---------------------------------------------------------------------------
step "4. SSH key"

SSH_KEY="$(eval echo "${REPROLAB_RUNPOD_SSH_KEY_PATH}")"  # expand ~
SSH_KEY="$(normalize_ssh_key_path "${SSH_KEY}")"

if [[ ! -f "${SSH_KEY}" ]]; then
    fail "Private key not found: ${SSH_KEY}"
    exit 5
fi
ok "Private key exists: ${SSH_KEY}"

# Permissions check (Linux/macOS only — Windows over WSL inherits Linux mode).
if command -v stat >/dev/null 2>&1; then
    PERM="$(stat -c '%a' "${SSH_KEY}" 2>/dev/null || stat -f '%A' "${SSH_KEY}" 2>/dev/null || echo "")"
    if [[ -n "${PERM}" && "${PERM}" != "600" && "${PERM}" != "400" ]]; then
        warn "Private key permissions are ${PERM} (recommend 600 or 400). OpenSSH may refuse to use it."
        echo "       Fix with: chmod 600 ${SSH_KEY}" >&2
    elif [[ -n "${PERM}" ]]; then
        ok "Private key permissions: ${PERM}"
    fi
fi

PUB_FROM_PRIV="$(ssh-keygen -y -f "${SSH_KEY}" 2>/dev/null || true)"
if [[ -z "${PUB_FROM_PRIV}" ]]; then
    warn "Could not derive public key from private key (ssh-keygen failed). Skipping pair check."
else
    # Compare just the key payload (field 2), not the comment, since the .env
    # comment "appradhann@gmail.com" won't appear in ssh-keygen -y output.
    PUB_PAYLOAD_ENV="$(echo "${REPROLAB_RUNPOD_SSH_PUBLIC_KEY}" | awk '{print $2}')"
    PUB_PAYLOAD_FILE="$(echo "${PUB_FROM_PRIV}" | awk '{print $2}')"
    if [[ -n "${PUB_PAYLOAD_ENV}" && "${PUB_PAYLOAD_ENV}" == "${PUB_PAYLOAD_FILE}" ]]; then
        ok "REPROLAB_RUNPOD_SSH_PUBLIC_KEY matches the private key"
    else
        fail "REPROLAB_RUNPOD_SSH_PUBLIC_KEY does NOT match ${SSH_KEY}"
        echo "       The pod will boot with the .env public key but your" >&2
        echo "       private key won't be able to log in." >&2
        echo "       Fix with:" >&2
        echo "         REPROLAB_RUNPOD_SSH_PUBLIC_KEY=\"$(ssh-keygen -y -f "${SSH_KEY}")\"" >&2
        exit 5
    fi
fi

# ---------------------------------------------------------------------------
# 5. Optional: end-to-end pod smoke test
# ---------------------------------------------------------------------------
if [[ "${START_POD}" -ne 1 ]]; then
    echo
    printf "${GREEN}All preflight checks passed.${RESET}\n"
    printf "Run with ${YELLOW}--start-pod${RESET} to also boot a real pod and run nvidia-smi over SSH (costs money).\n"
    exit 0
fi

step "5. End-to-end pod smoke test (this WILL spend money)"

require_cmd ssh
require_cmd ssh-keygen

POD_NAME="reprolab-smoke-$(date +%s)"
echo "Creating pod ${POD_NAME} with ${REPROLAB_RUNPOD_GPU_TYPE}..."

CREATE_PAYLOAD="$(
    POD_NAME="${POD_NAME}" \
    CLOUD="${REPROLAB_RUNPOD_CLOUD_TYPE}" \
    IMAGE="${REPROLAB_RUNPOD_IMAGE}" \
    GPU="${REPROLAB_RUNPOD_GPU_TYPE}" \
    GPU_COUNT="${REPROLAB_RUNPOD_GPU_COUNT}" \
    DISK="${REPROLAB_RUNPOD_CONTAINER_DISK_GB}" \
    VOL="${REPROLAB_RUNPOD_VOLUME_GB}" \
    MOUNT="${REPROLAB_RUNPOD_VOLUME_MOUNT_PATH}" \
    PUBKEY="${REPROLAB_RUNPOD_SSH_PUBLIC_KEY}" \
    DCS="${REPROLAB_RUNPOD_DATA_CENTER_IDS}" \
    NV="${REPROLAB_RUNPOD_NETWORK_VOLUME_ID}" \
    python3 -c "
import json, os
payload = {
    'name': os.environ['POD_NAME'],
    'cloudType': os.environ['CLOUD'],
    'computeType': 'GPU',
    'imageName': os.environ['IMAGE'],
    'gpuTypeIds': [os.environ['GPU']],
    'gpuCount': int(os.environ['GPU_COUNT']),
    'containerDiskInGb': int(os.environ['DISK']),
    'volumeInGb': int(os.environ['VOL']),
    'volumeMountPath': os.environ['MOUNT'],
    'ports': ['22/tcp'],
    'supportPublicIp': True,
    'env': {
        'PUBLIC_KEY': os.environ['PUBKEY'],
        'SSH_PUBLIC_KEY': os.environ['PUBKEY'],
    },
}
dcs = [s.strip() for s in os.environ.get('DCS','').split(',') if s.strip()]
if dcs:
    payload['dataCenterIds'] = dcs
nv = os.environ.get('NV','').strip()
if nv:
    payload['networkVolumeId'] = nv
print(json.dumps(payload))
")"

POD_RESP="$(curl -sS -X POST \
    -H "Authorization: Bearer ${API_KEY}" \
    -H "Content-Type: application/json" \
    -d "${CREATE_PAYLOAD}" \
    "${REPROLAB_RUNPOD_API_BASE_URL%/}/pods")"

POD_ID="$(echo "${POD_RESP}" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    sys.exit(0)
print(d.get('id') or '')
")"
if [[ -z "${POD_ID}" ]]; then
    fail "Pod creation failed"
    echo "${POD_RESP}" | python3 -m json.tool >&2 2>/dev/null || echo "${POD_RESP}" >&2
    exit 6
fi
ok "Pod created: ${POD_ID}"

cleanup_pod() {
    echo "Destroying pod ${POD_ID}..."
    curl -sS -X DELETE \
        -H "Authorization: Bearer ${API_KEY}" \
        "${REPROLAB_RUNPOD_API_BASE_URL%/}/pods/${POD_ID}" >/dev/null || true
    ok "Pod ${POD_ID} delete request sent"
}
trap cleanup_pod EXIT

DEADLINE=$(( $(date +%s) + REPROLAB_RUNPOD_BOOT_TIMEOUT_SECONDS ))
PUBLIC_IP=""
SSH_PORT=""
echo "Waiting up to ${REPROLAB_RUNPOD_BOOT_TIMEOUT_SECONDS}s for pod to expose SSH..."
while (( $(date +%s) < DEADLINE )); do
    POD_INFO="$(curl -sS \
        -H "Authorization: Bearer ${API_KEY}" \
        "${REPROLAB_RUNPOD_API_BASE_URL%/}/pods/${POD_ID}")"
    read -r PUBLIC_IP SSH_PORT < <(echo "${POD_INFO}" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    print('')
    sys.exit(0)
ip = d.get('publicIp') or d.get('publicIP') or ''
pm = d.get('portMappings') or {}
port = pm.get('22') or pm.get('22/tcp') or ''
print(f'{ip} {port}')
")
    if [[ -n "${PUBLIC_IP}" && -n "${SSH_PORT}" ]]; then
        # Probe TCP reachability before attempting SSH.
        if (echo > "/dev/tcp/${PUBLIC_IP}/${SSH_PORT}") 2>/dev/null; then
            break
        fi
    fi
    sleep 5
done

if [[ -z "${PUBLIC_IP}" || -z "${SSH_PORT}" ]]; then
    fail "Pod never exposed SSH within ${REPROLAB_RUNPOD_BOOT_TIMEOUT_SECONDS}s"
    exit 6
fi
ok "Pod reachable at ${REPROLAB_RUNPOD_SSH_USER}@${PUBLIC_IP}:${SSH_PORT}"

SSH_OPTS=(
    -i "${SSH_KEY}"
    -p "${SSH_PORT}"
    -o StrictHostKeyChecking=no
    -o UserKnownHostsFile=/dev/null
    -o ConnectTimeout=15
    -o LogLevel=ERROR
)

# Retry SSH a few times — sshd often comes up a beat after the port is open.
SSH_OK=0
for attempt in 1 2 3 4 5; do
    if ssh "${SSH_OPTS[@]}" "${REPROLAB_RUNPOD_SSH_USER}@${PUBLIC_IP}" \
        'nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader' \
        2>/tmp/runpod_ssh_err; then
        SSH_OK=1; break
    fi
    sleep 5
done

if [[ "${SSH_OK}" -ne 1 ]]; then
    fail "SSH / nvidia-smi failed"
    sed 's/^/       /' /tmp/runpod_ssh_err >&2 || true
    exit 6
fi

ok "End-to-end smoke passed"
echo
printf "${GREEN}Runpod is fully wired up. You can launch the pipeline with:${RESET}\n"
printf "  python -m backend.cli reproduce <paper.pdf> --sandbox runpod\n"
