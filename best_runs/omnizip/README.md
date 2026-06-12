# OmniZip (2511.14582) — best so far: 0.664 reproduced (beat its own floor)

*OmniZip: Audio-Guided Token Compression for Omni-modal LLM Inference* — a
TRAINING-FREE inference paper: audio-guided token pruning accelerates
Qwen2.5-Omni audio-video prefill (~2.5-3.4x claimed) while holding accuracy.
A different reproduction shape from the training papers: the work is paired
measurement (baseline vs compressed, identical subsets/prompts/decoding),
not gradient descent — the rubric's heavy leaves are algorithmic fidelity
(windowed audio-retention scoring, inverse video-pruning allocation,
audio-anchor preservation, the interleaved merge) and measurement honesty
(same bounded subset, warmup excluded, per-cell wall-clock).

Best: **0.6638 reproduced** (2026-06-12, one 8.4h iteration, meets_target
TRUE against the floored 0.6563 target — the second run in the campaign to
beat its own best ancestor, after All-CNN v4). Ran with the full stack:
floored target, pinned rubric, seeded best attempt, prior-attempt evidence,
leaf-triage repair plans. Residual weak leaves are judgment-class
(algorithmic fidelity), not evidence gaps. The paper hint (`backend/agents/prompts/paper_hints.py`,
"2511.14582") carries the algorithm invariants and the paired-measurement
protocol.

Files: `final_report.{json,md}`, leaf-by-leaf `rubric_evaluation.json`,
`rubric_tree.json`.
