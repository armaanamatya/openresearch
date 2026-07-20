# RunPod scheduler-runtime validation — 2026-07-20

## Result

The bounded RunPod data-plane smoke completed successfully after two real
integration defects were fixed. The final run was
`prj_runpod_scheduler_smoke_20260720d`. It used the normal `runpod` sandbox, a
one-GPU `NVIDIA GeForce RTX 4090` pod, and no LLM calls. Its deterministic
evidence recorded at completion:

- `demo_status.json`: `completed` / `reproduced`.
- `experiment_runs.jsonl`: `success=true`, `exit_code=0`,
  `metrics_sha256=7acb43c03d0164d4ca2cfea2ab1e24011b73d2f3c5232676a40110a8383dc9bc`.
- `metrics.json`: `{ "sanity_ok": 1.0, "gpu_visible": 1.0 }`.
- `exec.log`: both the import preflight and `nvidia-smi` succeeded; the visible
  device was an RTX 4090.
- `runpodctl pod list --all`: `[]` immediately after completion, proving the
  owned-pod teardown path ran.

The four local smoke directories (including two failed diagnostic attempts)
were moved to macOS Trash after validation, so they cannot leave a dirty
worktree or be accidentally committed. The immutable metric SHA, logs, exact
failure chronology, and cost evidence needed to audit the result are retained
in this runbook.

The command was deliberately bounded: `--max-usd 0.25`,
`--max-wall-clock 480`, `--max-pod-seconds 240`,
`--max-run-gpu-usd 0.50`, `--max-gpu-usd-per-hour 1.50`, and one GPU. The
account balance changed from `$20.8820184766` before this validation sequence
to `$20.8614162424` after it, an observed upper bound of `$0.0206022342` for
the sequence. RunPod's per-pod billing endpoint had not yet posted the current
hour, so this balance delta—not an empty local cost ledger—is the cost evidence.

## Exact command

No API key is printed or stored in this repository. The explicit opt-in reads
the credential already held by `runpodctl`:

```bash
PYTHONPATH="$PWD" \
OPENRESEARCH_RUNPOD_CLOUD_TYPE=SECURE \
OPENRESEARCH_RUNPOD_USE_CLI_CREDENTIALS=1 \
.venv/bin/python -m backend.cli reproduce papers/allcnn.pdf \
  --source-kind pdf_path --sandbox runpod --sanity \
  --project-id prj_runpod_scheduler_smoke_20260720d \
  --gpu-mode prefer --command-timeout 120 \
  --max-usd 0.25 --max-wall-clock 480 --max-pod-seconds 240 \
  --max-run-gpu-usd 0.50 --max-gpu-usd-per-hour 1.50 \
  --force-single-gpu --vram-gb 16
```

## Failures found and repaired

1. **CLI/application credential mismatch.** `runpodctl doctor` was authenticated
   but the application only accepted environment/config API keys, so the first
   attempt failed before pod creation. `RunpodBackend` now accepts
   `~/.runpod/config.toml` only behind
   `OPENRESEARCH_RUNPOD_USE_CLI_CREDENTIALS=1`. OFF is environment-only and
   byte-identical; explicit API keys retain precedence. Hermetic OFF/ON tests
   cover this bridge.
2. **Remote preflight host-path leak.** The always-on import smoke embedded the
   host-relative `runs/.../code` path inside its shell command. A remote pod
   therefore failed at `cd` before executing `sanity.py`. Both preflight and
   optional execution smoke now have a `uses_sandbox_workdir` form; RunPod uses
   the backend-guaranteed uploaded work directory instead.
3. **Sanity GPU intent was dropped.** `--gpu-mode prefer` did not reach
   `RunContext`, leaving the record as `auto`, and the old smoke did not prove a
   GPU was visible. The CLI now threads the requested mode and, for RunPod with
   GPU mode other than `off`, fails closed unless `nvidia-smi` returns a device.
   The final evidence records `gpu_visible=1.0` and `gpu_mode=prefer`.

## Scheduler validation boundary

The RunPod run validates the billed remote execution path: credential
resolution, pod creation, remote workdir selection, upload, preflight,
deterministic metrics, artifact sync, ledger/status writes, GPU visibility,
and teardown. It is intentionally not a campaign, so
`asha_shadow_report.py` correctly reports no `campaign/attempts.jsonl`.

The receipt-gated scheduler tree itself is exercised hermetically by
`test_scheduler_authority_controller.py`, `test_scheduler_runtime.py`,
`test_scheduler_evidence.py`, `test_scheduler_lineage_sink.py`, and the ASHA
adapter/core tests. It proves fan-out, atomic launch claims, controller-attested
receipts, deterministic promote/freeze/literal-`training_diverged` kill, frozen
pool revival, separate fidelity/width meters, cost/A100 caps, and lineage
reconciliation. The ordinary serial campaign remains deliberately audit-only:
it has no automatic producer for controller-attested paper-step/checkpoint
receipts, so `OPENRESEARCH_SCHEDULER_AUTHORITATIVE` stays default-OFF and does
not emit `applied:true`.

## Cloud posture

This does not remove the documented GCP Workload Identity blocker or validate
the EKS backend. Those are separate cloud control-plane prerequisites. The
RunPod validation is the current evidence-valid GPU path and does not weaken
the deterministic evidence gate.
