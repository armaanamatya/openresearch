# PPO CartPole-v1 Reproduction

Faithful reproduction of **Proximal Policy Optimization Algorithms**
Schulman et al., 2017 — https://arxiv.org/abs/1707.06347

---

## What this implements

- **PPO-Clip** variant (Equation 7 from the paper)
- **Generalized Advantage Estimation (GAE)** with lambda=0.95
- Shared **Actor-Critic** network: two-layer MLP (64 hidden units, tanh), with
  separate policy and value heads
- Multiple epochs of **minibatch SGD** per rollout (10 epochs, mini-batch 64)
- **Entropy bonus**, value loss coefficient, gradient norm clipping
- **CartPole-v1** environment from Gymnasium
- Target: **mean episode reward >= 475.0** after **500,000 timesteps**

---

## How to run

### Local (Python >=3.11)

```bash
pip install -r requirements.txt
python train.py --config config.json
```

Results are written to `metrics.json` in `$OUTPUT_DIR` (defaults to the script
directory when the env var is unset).

### Docker

```bash
docker build -t ppo-cartpole .
docker run --rm -e OUTPUT_DIR=/out -v $(pwd)/out:/out ppo-cartpole
```

---

## Hyperparameters

All hyperparameters live in `config.json`:

| Parameter         | Value   | Source          |
|-------------------|---------|-----------------|
| total_timesteps   | 500,000 | Paper           |
| learning_rate     | 3e-4    | Paper (Adam)    |
| batch_size        | 64      | Paper           |
| n_steps           | 2048    | PPO default     |
| n_epochs          | 10      | PPO default     |
| gamma             | 0.99    | PPO default     |
| gae_lambda        | 0.95    | GAE paper       |
| clip_coef         | 0.2     | PPO default     |
| ent_coef          | 0.0     | Not needed for CartPole |
| vf_coef           | 0.5     | PPO default     |
| max_grad_norm     | 0.5     | PPO default     |
| hidden_size       | 64      | PPO reference impl |
| eval_episodes     | 100     | Paper           |

---

## Output files

| File                              | Description                                      |
|-----------------------------------|--------------------------------------------------|
| `$OUTPUT_DIR/metrics.json`        | Final metrics (mean reward, target met, timing)  |
| `$OUTPUT_DIR/resolved_config.json`| Merged config actually used at runtime           |
| `$OUTPUT_DIR/best_model.pt`       | Model checkpoint at best eval reward             |
| `$OUTPUT_DIR/final_model.pt`      | Model checkpoint after all 500K timesteps        |

---

## Expected results

```json
{
  "mean_reward_100ep": 500.0,
  "best_eval_reward": 500.0,
  "total_timesteps": 500000,
  "target_reward": 475.0,
  "target_met": true
}
```

CartPole-v1 is capped at 500 steps per episode; PPO consistently solves it
(reward >= 475) within 200K-300K timesteps with these hyperparameters.

---

## Assumptions applied

| ID     | Detail                          | Chosen value    |
|--------|---------------------------------|-----------------|
| ENV001 | PyTorch version (not stated)    | 2.2.0           |
| ENV002 | Python version (not stated)     | 3.11            |
| ENV003 | GPU requirement                 | CPU-only        |
