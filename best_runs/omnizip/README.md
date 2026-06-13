# OmniZip (2511.14582) — best so far: 0.692 reproduced (3rd straight floor-beat)

*OmniZip: Audio-Guided Token Compression for Omni-modal LLM Inference* — a
TRAINING-FREE inference paper: audio-guided token pruning accelerates
Qwen2.5-Omni audio-video prefill (~2.5-3.4x claimed) while holding accuracy.
A different reproduction shape from the training papers: the work is paired
measurement (baseline vs compressed, identical subsets/prompts/decoding),
not gradient descent — the rubric's heavy leaves are algorithmic fidelity
(windowed audio-retention scoring, inverse video-pruning allocation,
audio-anchor preservation, the interleaved merge) and measurement honesty
(same bounded subset, warmup excluded, per-cell wall-clock).

Best: **0.6919 reproduced** (2026-06-13, meets_target TRUE against the floored 0.6638 target — OmniZip's THIRD consecutive record, each run starting from the prior's best via the seeded-ancestor + champion + floored-target rails). Full stack: floored target, pinned rubric, seeded best attempt, prior-attempt evidence, leaf-triage repair plans. Residual weak leaves are judgment-class
(algorithmic fidelity), not evidence gaps. The paper hint (`backend/agents/prompts/paper_hints.py`,
"2511.14582") carries the algorithm invariants and the paired-measurement
protocol.

Files: `final_report.{json,md}`, leaf-by-leaf `rubric_evaluation.json`,
`rubric_tree.json`.
