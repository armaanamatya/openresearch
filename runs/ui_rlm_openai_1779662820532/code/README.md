# PPO on CartPole-v1 — Reproduction

Implements **Proximal Policy Optimization (PPO)** on the `CartPole-v1` environment.
Target: mean episode reward ≥ **475.0** over 100 evaluation episodes.

## Algorithm

- **Actor-Critic** shared-trunk MLP with orthogonal initialisation
- **GAE** (Generalized Advantage Estimation, λ=0.95)
- **Clipped surrogate objective** (ε=0.2) — stochastic gradient ascent on the PPO objective
- Entropy bonus (coef=0.01) for exploration
- Value function loss coefficient = 0.5
- Gradient clipping (max_norm=0.5)

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Total timesteps | 500 000 (GPU) / 200 000 (CPU) |
| Rollout steps (N_steps) | 2048 |
| Mini-batch size | 64 |
| PPO epochs per rollout | 10 |
| Learning rate (Adam) | 3e-4 |
| Discount γ | 0.99 |
| GAE λ | 0.95 |
| Clip ε | 0.2 |
| Entropy coef | 0.01 |
| Value coef | 0.5 |
| Eval episodes | 100 |
| Random seeds | 3 (0, 1, 2) |

## How to run

    # Local (CPU)
    pip install -r requirements.txt
    python train.py

    # Specify seeds manually
    python train.py --seeds 0 1 2

    # Override timesteps
    python train.py --total-timesteps 500000

    # Evaluation episodes
    python train.py --eval-episodes 100

## Docker

    docker build -t ppo-cartpole .
    docker run --rm -e OUTPUT_DIR=/artifacts -v $(pwd)/out:/artifacts ppo-cartpole

## Outputs

All outputs are written to $OUTPUT_DIR (defaults to the code root for local runs):

- metrics.json        — Flat JSON: mean_reward, std_reward, target_met, …
- run_detail.json     — Per-seed reward histories
- train.log           — Full training log
- checkpoints/        — Model weights per seed

## Assumptions applied

- ENV001: PyTorch 2.2.0
- ENV002: Python 3.11
- ENV003: CPU-only (no GPU required)
