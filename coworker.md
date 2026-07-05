# Coworker Access — Foundry Models + GCP

Two systems, two auth models:

- **Foundry models** → work from the `.env` **alone**. Paste the key, done.
- **GCP** → needs the `.env` **plus one key file** (`gcp-key.json`). GCP auth is an identity, not a bearer key, so it can't live in `.env` by itself.

---

## 1. Foundry models — works from `.env` alone

Add these to your `.env` (I'll send you the key separately):

```dotenv
AZURE_FOUNDRY_ENDPOINT=https://appradhann-4738-resource.services.ai.azure.com/openai/v1/
AZURE_FOUNDRY_DEPLOYMENT=grok-4.3
AZURE_FOUNDRY_API_KEY=<key I send you>
```

The **same key** serves both surfaces:

| Surface | Base URL | Deployments |
|---|---|---|
| OpenAI-compatible | `.../openai/v1/` | `grok-4.3`, `gpt-chat-latest` |
| Anthropic (Messages) | `.../anthropic/v1/messages` | `claude-opus-4-8`, `claude-sonnet-5` |

Anthropic surface auth: headers `x-api-key: <key>` + `anthropic-version: 2023-06-01`.

**Verify:**

```bash
# OpenAI-compatible
curl -s https://appradhann-4738-resource.services.ai.azure.com/openai/v1/chat/completions \
  -H "Authorization: Bearer $AZURE_FOUNDRY_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"grok-4.3","messages":[{"role":"user","content":"ping"}]}'

# Anthropic (Opus 4.8 / Sonnet 5)
curl -s https://appradhann-4738-resource.services.ai.azure.com/anthropic/v1/messages \
  -H "x-api-key: $AZURE_FOUNDRY_API_KEY" -H "anthropic-version: 2023-06-01" \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-sonnet-5","max_tokens":16,"messages":[{"role":"user","content":"ping"}]}'
```

No Azure login required.

---

## 2. GCP — project `deepinvent-ext-ut`

Reference values (non-secret):

| Field | Value |
|---|---|
| Project ID | `deepinvent-ext-ut` |
| Region / zone | `us-central1` / `us-central1-c` |
| GPU quota | A100-40 — 8 on-demand, 16 preemptible |

### Owner: one-time prep (run once, then send `gcp-key.json`)

```bash
gcloud iam service-accounts create coworker-access \
  --project deepinvent-ext-ut --display-name "Coworker access"

SA=coworker-access@deepinvent-ext-ut.iam.gserviceaccount.com
for R in roles/container.developer roles/artifactregistry.reader \
         roles/storage.objectAdmin roles/iam.serviceAccountUser roles/compute.viewer; do
  gcloud projects add-iam-policy-binding deepinvent-ext-ut \
    --member="serviceAccount:$SA" --role="$R"
done

gcloud iam service-accounts keys create gcp-key.json --iam-account "$SA"
# → send gcp-key.json to the coworker (password manager, not Slack)
```

### Coworker: activate it

Save the file (e.g. `~/.gcp/gcp-key.json`) and point `.env` at it:

```dotenv
GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/gcp-key.json
```

Then activate for `gcloud` + `kubectl`:

```bash
gcloud auth activate-service-account --key-file="$GOOGLE_APPLICATION_CREDENTIALS"
gcloud config set project deepinvent-ext-ut

# only when a GKE cluster is running (see caveat):
gcloud container clusters list --project deepinvent-ext-ut          # get the name
gcloud container clusters get-credentials <name> --region us-central1 --project deepinvent-ext-ut
```

The `GOOGLE_APPLICATION_CREDENTIALS` line covers the Python SDK path automatically.

> **Caveat:** the GKE cluster is spun up/down to save cost. If `clusters list` is empty, none is running — the owner brings it up via `infra/gcp/envs/deepinvent` before GPU runs.

---

## What gets sent

| File | Contains | Handling |
|---|---|---|
| `.env` | Foundry key + `GOOGLE_APPLICATION_CREDENTIALS` path | secret — password manager |
| `gcp-key.json` | GCP service-account key | secret — password manager |

Keep both out of git and chat. Revoke GCP access anytime with:

```bash
gcloud iam service-accounts delete coworker-access@deepinvent-ext-ut.iam.gserviceaccount.com
```
