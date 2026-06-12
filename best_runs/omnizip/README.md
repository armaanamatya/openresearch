# OmniZip (2511.14582) — best so far: 0.656 partial

*OmniZip: Audio-Guided Token Compression for Omni-modal LLM Inference* — a
TRAINING-FREE inference paper: audio-guided token pruning accelerates
Qwen2.5-Omni audio-video prefill (~2.5-3.4x claimed) while holding accuracy.
A different reproduction shape from the training papers: the work is paired
measurement (baseline vs compressed, identical subsets/prompts/decoding),
not gradient descent — the rubric's heavy leaves are algorithmic fidelity
(windowed audio-retention scoring, inverse video-pruning allocation,
audio-anchor preservation, the interleaved merge) and measurement honesty
(same bounded subset, warmup excluded, per-cell wall-clock).

Best attempt: **0.6563 partial** (2026-06-11, 9 iterations, ~6.8h). The
campaign for this paper is young — a follow-up run with the full current
stack (floored target 0.6563, pinned rubric, seeded best attempt,
prior-attempt evidence, leaf-triage repair plans) was in flight when this
package was cut; the paper hint (`backend/agents/prompts/paper_hints.py`,
"2511.14582") carries the algorithm invariants and the paired-measurement
protocol.

Files: `final_report.{json,md}`, leaf-by-leaf `rubric_evaluation.json`,
`rubric_tree.json`.
