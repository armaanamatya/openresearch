# Final Benchmark Report

**Project:** `ui_rlm_openai_1779662820532`  
**Benchmark:** PaperBench-style final benchmark  
**Task:** `reprolab-demo/ppo-cartpole-v1`  
**Verdict:** `reproduced_with_caveats`

The hardcoded demo ships with a deterministic PaperBench-style comparison so the UI has a realistic final benchmark surface before a long live run finishes.

## Source Artifact

| Field | Value |
| --- | --- |
| PDF | `reprolab-demo-paper.pdf` |
| Stored in generated code root | `paper.pdf` |
| Pages | 12 |
| Size | 2923532 bytes |
| SHA256 | `e78feadadbdbb0b601b3c2bcc81404722cd431a489b307545f9b7bea1e8c4f5b` |

## Final Metric Comparison

| Metric | Paper target | Reproduced value | Delta |
| --- | ---: | ---: | ---: |
| mean_reward | 475.0 | 492.3 | +17.3 |

## PaperBench-Style Rubric

| Area | Score | Evidence |
| --- | ---: | --- |
| Paper understanding | 0.96 | `paper_claim_map.json` |
| Environment reconstruction | 0.92 | `Dockerfile` |
| Baseline implementation | 0.91 | `train.py` |
| Execution artifacts | 0.88 | `metrics.json`, `commands.log`, `provenance.json` |
| Comparison quality | 0.90 | `final_benchmark_report.md` |

## Generated Codebase Root

The generated code root is designed to be inspectable without the dashboard:

```text
code/
  paper.pdf
  README.md
  Dockerfile
  train.py
  config.json
  commands.log
  paperbench_comparison.json
  final_benchmark_report.md
  logs/paperbench_eval.log
```
