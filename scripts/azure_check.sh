#!/usr/bin/env bash
# Azure compute sandbox preflight + (optional) end-to-end smoke for the
# openresearch pipeline.  Mirrors scripts/runpod_check.sh exit-code discipline.
#
# This script verifies the same auth surface that
# backend/services/runtime/azure_backend.py uses (DefaultAzureCredential),
# so a green run here means `--sandbox azure` will authenticate when you
# launch a pipeline.
#
# Usage:
#   scripts/azure_check.sh                 # preflight only (read-only, free)
#   scripts/azure_check.sh --start-vm      # also boots a tiny CPU SKU, runs
#                                          # `echo hello` over SSH, then destroys
#                                          # it (COSTS A FEW CENTS — minutes-scale)
#
# Exit codes:
#   0  everything green
#   1  bad usage / unexpected error
#   2  required env var missing
#   3  Azure auth failed (DefaultAzureCredential.get_token raised)
#   4  configured REPROLAB_AZURE_VM_SIZE not available in REPROLAB_AZURE_REGION
#   5  SSH key missing / wrong permissions / mismatched pair
#   6  --start-vm end-to-end smoke failed

set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve repo root (script lives in <repo>/scripts/) and load .env
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${REPO_ROOT}/.env"

START_VM=0
for arg in "$@"; do
    case "$arg" in
        --start-vm) START_VM=1 ;;
        -h|--help)
            sed -n '2,22p' "$0"
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

# Load .env (same conservative parser as runpod_check.sh).
while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
    if [[ "$line" =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then
        key="${BASH_REMATCH[1]}"
        value="${BASH_REMATCH[2]}"
        if [[ "$value" =~ ^\"(.*)\"$ ]]; then value="${BASH_REMATCH[1]}"; fi
        if [[ "$value" =~ ^\'(.*)\'$ ]]; then value="${BASH_REMATCH[1]}"; fi
        if [[ -z "${!key+x}" ]]; then
            export "${key}=${value}"
        fi
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

# ---------------------------------------------------------------------------
# Step 1 — required env vars
# ---------------------------------------------------------------------------
step "1) Required env vars"

SUB_ID="${REPROLAB_AZURE_SUBSCRIPTION_ID:-${AZURE_SUBSCRIPTION_ID:-}}"
if [[ -z "${SUB_ID}" ]]; then
    fail "REPROLAB_AZURE_SUBSCRIPTION_ID is empty (also tried AZURE_SUBSCRIPTION_ID)"
    exit 2
fi
ok "Subscription id: ${SUB_ID:0:8}…"

REGION="${REPROLAB_AZURE_REGION:-eastus}"
VM_SIZE="${REPROLAB_AZURE_VM_SIZE:-Standard_NC6s_v3}"
SSH_KEY_PATH="${REPROLAB_AZURE_SSH_KEY_PATH:-${HOME}/.ssh/id_ed25519}"
SSH_USER="${REPROLAB_AZURE_SSH_USER:-azureuser}"
ok "Region:  ${REGION}"
ok "VM size: ${VM_SIZE}"

# ---------------------------------------------------------------------------
# Step 2 — Python + Azure SDK importable
# ---------------------------------------------------------------------------
step "2) Python + Azure SDK"

PY="${REPROLAB_PYTHON_BIN:-python3}"
if ! command -v "${PY}" >/dev/null 2>&1; then
    PY="python"
fi
if ! command -v "${PY}" >/dev/null 2>&1; then
    fail "Neither python3 nor python is on PATH"
    exit 1
fi
ok "Python: $(${PY} --version 2>&1)"

if ! ${PY} -c "from azure.identity import DefaultAzureCredential; from azure.mgmt.compute import ComputeManagementClient" 2>/dev/null; then
    fail "Azure SDK not importable — run: pip install -r backend/requirements.txt"
    exit 1
fi
ok "Azure SDK importable"

# ---------------------------------------------------------------------------
# Step 3 — auth (DefaultAzureCredential)
# ---------------------------------------------------------------------------
step "3) Azure auth"

if ! ${PY} - <<PY_EOF
import sys
from azure.identity import DefaultAzureCredential
try:
    cred = DefaultAzureCredential()
    token = cred.get_token("https://management.azure.com/.default")
    print(f"  token expires in: {token.expires_on - __import__('time').time():.0f}s")
except Exception as exc:
    print(f"  auth failure: {exc}", file=sys.stderr)
    sys.exit(3)
PY_EOF
then
    fail "DefaultAzureCredential could not obtain a token.  Run 'az login' or set AZURE_CLIENT_ID/_SECRET/_TENANT_ID."
    exit 3
fi
ok "DefaultAzureCredential resolved"

# ---------------------------------------------------------------------------
# Step 4 — SKU availability in the configured region
# ---------------------------------------------------------------------------
step "4) SKU '${VM_SIZE}' availability in '${REGION}'"

if ! ${PY} - <<PY_EOF
import os, sys
from azure.identity import DefaultAzureCredential
from azure.mgmt.compute import ComputeManagementClient
sub = os.environ.get("REPROLAB_AZURE_SUBSCRIPTION_ID") or os.environ.get("AZURE_SUBSCRIPTION_ID")
region = os.environ.get("REPROLAB_AZURE_REGION", "eastus")
vm_size = os.environ.get("REPROLAB_AZURE_VM_SIZE", "Standard_NC6s_v3")
cred = DefaultAzureCredential()
client = ComputeManagementClient(cred, sub)
hit = False
restricted = False
for sku in client.resource_skus.list():
    if sku.name != vm_size:
        continue
    locations = [l.lower() for l in (sku.locations or [])]
    if region.lower() not in locations:
        continue
    hit = True
    for r in (sku.restrictions or []):
        if str(getattr(r, "reason_code", "")) in {"NotAvailableForSubscription", "QuotaId"}:
            restricted = True
            print(f"  RESTRICTION: {r.reason_code}", file=sys.stderr)
    break
if not hit:
    print(f"  '{vm_size}' is not offered in '{region}'", file=sys.stderr)
    sys.exit(4)
if restricted:
    sys.exit(4)
PY_EOF
then
    fail "VM size ${VM_SIZE} is not available in ${REGION} (or requires a quota increase)"
    exit 4
fi
ok "SKU ${VM_SIZE} available in ${REGION}"

# ---------------------------------------------------------------------------
# Step 5 — SSH key
# ---------------------------------------------------------------------------
step "5) SSH key"

if [[ ! -f "${SSH_KEY_PATH}" ]]; then
    fail "SSH private key not found: ${SSH_KEY_PATH}"
    fail "  Generate with: ssh-keygen -t ed25519 -f ${SSH_KEY_PATH}"
    exit 5
fi
key_perms="$(stat -c '%a' "${SSH_KEY_PATH}" 2>/dev/null || stat -f '%Lp' "${SSH_KEY_PATH}" 2>/dev/null || echo '???')"
if [[ "${key_perms}" != "600" && "${key_perms}" != "400" ]]; then
    warn "SSH private key permissions are ${key_perms} (expected 600 or 400)."
    warn "  Fix with: chmod 600 ${SSH_KEY_PATH}"
fi
ok "SSH private key: ${SSH_KEY_PATH} (perm ${key_perms})"

if [[ -n "${REPROLAB_AZURE_SSH_PUBLIC_KEY:-}" ]]; then
    ok "REPROLAB_AZURE_SSH_PUBLIC_KEY is set — will be used verbatim."
elif [[ -f "${SSH_KEY_PATH}.pub" ]]; then
    ok "Public key sibling found: ${SSH_KEY_PATH}.pub"
else
    if ssh-keygen -y -f "${SSH_KEY_PATH}" >/dev/null 2>&1; then
        ok "Public key derivable via ssh-keygen -y"
    else
        fail "No public key — set REPROLAB_AZURE_SSH_PUBLIC_KEY or create ${SSH_KEY_PATH}.pub"
        exit 5
    fi
fi

# ---------------------------------------------------------------------------
# Step 6 — (optional) end-to-end smoke
# ---------------------------------------------------------------------------
if [[ "${START_VM}" -eq 1 ]]; then
    step "6) End-to-end smoke (Standard_B1s, ~\$0.01/hr — actual cost <\$0.01)"
    warn "This provisions a real Azure VM and deletes it.  Press Ctrl-C within 5s to cancel."
    sleep 5

    if ! ${PY} - <<'PY_EOF'
import asyncio, os, sys, tempfile, uuid
from pathlib import Path
from backend.services.runtime.azure_backend import AzureBackend
from backend.services.runtime.interface import SandboxConfig

async def main():
    tmp = Path(tempfile.mkdtemp(prefix="azure-smoke-"))
    (tmp / "hello.txt").write_text("hi\n")
    backend = AzureBackend(
        subscription_id=os.environ["REPROLAB_AZURE_SUBSCRIPTION_ID"],
        region=os.environ.get("REPROLAB_AZURE_REGION", "eastus"),
        vm_size="Standard_B1s",  # CPU-only, cheapest available
        image="Canonical:0001-com-ubuntu-server-jammy:22_04-lts:latest",
        ssh_key_path=os.environ.get("REPROLAB_AZURE_SSH_KEY_PATH") or None,
        ssh_public_key=os.environ.get("REPROLAB_AZURE_SSH_PUBLIC_KEY", ""),
        # B-series VMs do NOT support Premium_LRS; StandardSSD_LRS works
        # on every SKU and is the AzureBackend default.
        os_disk_tier="StandardSSD_LRS",
        max_boot_seconds=300,
    )
    config = SandboxConfig(
        project_id="smoke",
        run_id=uuid.uuid4().hex[:8],
        project_root=tmp,
    )
    sandbox = await backend.create_sandbox(config)
    try:
        result = await backend.exec(sandbox, "echo hello-from-azure", timeout=30)
        print(f"  exec exit={result.exit_code} stdout={result.stdout.strip()!r}")
        if result.exit_code != 0:
            sys.exit(6)
    finally:
        await backend.destroy(sandbox)

asyncio.run(main())
PY_EOF
    then
        fail "End-to-end smoke failed"
        exit 6
    fi
    ok "End-to-end smoke succeeded — VM destroyed."
fi

step "All checks passed"
ok "Azure sandbox is ready.  Launch with: --sandbox azure"
