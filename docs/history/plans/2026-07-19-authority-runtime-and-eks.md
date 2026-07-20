# Authoritative scheduler runtime and EKS execution

## Goal

Replace the pre-authority audit-only seam with a real, evidence-gated branch
runtime. Add AWS as an EKS/S3 Kubernetes backend with the same cap, artifact,
and cancellation semantics as GKE. Neither change may make a default flag live
or treat an LLM grade, a wall-clock value, or a local cost ledger as authority.

## TDD sequence

1. Define and test a stdlib-only `scheduler_evidence` contract: immutable,
   paper-pinned step ladders and branch-rung receipts. Missing/tampered metric,
   checkpoint, optimizer/LR/RNG/data-order state, canonical evidence bundle, or
   provenance invalidates a receipt. `final_report.score` is never read.
2. Extend the cell harness to mint these receipts only after durable rung
   completion; test an uninterrupted and resumed checkpoint chain have the same
   validated state. Keep the existing cell resume predicate unchanged when the
   scheduler flag is off.
3. Add a tree-only durable branch projection (queue, active lease, frozen pool,
   and literal true-kill records) to campaign state/ledger. Test serial-off
   rows/states remain byte-identical and every queue mutation is write-ahead.
4. Append factual branch-tree events through the SQLite event store only after
   their corresponding durable projection mutation. Test event order, F10 dedup,
   concurrency retry, and absence of events for rejected receipts.
5. Authoritatively map verified branch observations only after base policy
   terminal evaluation. Test all four base terminal kinds preserve all five
   decision keys; promote queues a resume, freeze verifies then pools, and only
   literal `training_diverged` can true-kill.
6. Loosen the paired A/B gate only after producers exist. Require complete paired
   controls, full terminal evidence equality, branch/attempt/checkpoint linkage,
   and provider-attested cost records. Add a single full passing fixture and
   adversarial forged-artifact rejections.
7. Implement EKS+S3 by adapting the generic Kubernetes backend, not by adding a
   second scheduler. Test lazy imports, STS/kubeconfig/S3 preflight, prefix
   isolation, pinned image, IRSA, selectors, caps, timeout/cancellation, and
   AWS cell-matrix dispatch without network access.
8. Only after the hermetic suite is green, execute read-only AWS/GCP preflight.
   Launch separately approved, tightly capped shadow runs first; monitor each
   run and reconcile Job/Pod/node/provider artifacts before any A/B conclusion.

## Non-negotiable runtime contracts

- Fidelity is `to_step` in the paper-pinned ladder; GPU dollars set only width.
- A receipt is a harness-verified input, not agent-authored JSON trusted on
  presence alone.
- Underperformance freezes to a verified checkpoint. Deletion requires the
  literal receipt cause `training_diverged`.
- Event rows describe completed durable transitions, never an advisory proposal.
- EKS workers use IRSA + prefix-scoped S3 and pinned ECR images; never static AWS
  credentials in a pod. Provider billing exports are reconciled after execution,
  not inferred from `cost_ledger.jsonl`.
