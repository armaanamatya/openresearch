# PPO Baseline — CartPole-v1 Reproduction

This directory is the generated code root for `ui_rlm_openai_1779657640555`.

## Algorithm

Clipped-objective PPO (Schulman et al., 2017) targeting a mean episode reward
of ≥ 475.0 over 100 evaluation episodes on CartPole-v1.

| Hyperparameter | Value | Assumption |
|---|---|---|
| Learning rate | 3e-4 | Paper |
| Adam ε | 1e-5 | A001 |
| Weight init | Orthogonal | A002 |
| LR schedule | Linear decay | A003 |
| Advantage norm | Per-minibatch | A004 |
| Value loss clip | ✓ | A005 |
| max_grad_norm | 0.5 | A006 |
| GAE λ | 0.95 | A007 |
| Entropy coef | 0.01 | A008 |
| Minibatch size | 64 | Paper |
| Total timesteps | 500,000 | Paper |
| Eval episodes | 100 | Paper |

## How to Run

### Docker (recommended)

```bash
docker build -t ppo-cartpole .
docker run --rm -v $(pwd)/outputs:/artifacts \
    -e OUTPUT_DIR=/artifacts \
    ppo-cartpole
```

### Local

```bash
pip install -r requirements.txt
python train.py            # writes metrics.json to $OUTPUT_DIR (or ./outputs)
```

The script auto-detects GPU availability. On CPU-only machines it caps training
at 200,000 timesteps (still uses the real CartPole-v1 environment — no surrogates).

## Outputs

| File | Description |
|---|---|
| `$OUTPUT_DIR/metrics.json` | Flat JSON with `reward` (mean 100-ep eval) |
| `$OUTPUT_DIR/model.pth` | Saved actor-critic weights |

### metrics.json shape

```json
{
  "reward": 495.3,
  "mean_eval_reward_100ep": 495.3,
  "total_timesteps_run": 500000,
  "num_eval_episodes": 100,
  "training_tail_mean_reward": 498.1,
  "wall_time_seconds": 142.7,
  "target_reward": 475.0,
  "target_met": true
}
```

## Review Artifacts

- `paperbench_comparison.json` - structured benchmark comparison
- `final_benchmark_report.md` - human-readable benchmark report
- `logs/paperbench_eval.log` - PaperBench-style evaluator log
- `reprolab_manifest.json` - source and artifact manifest
