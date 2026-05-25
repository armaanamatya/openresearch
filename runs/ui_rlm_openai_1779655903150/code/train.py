import os
import torch
import gymnasium as gym
from torch import nn
from torch.optim import Adam
from torch.distributions import Categorical

# Check for GPU
HAS_GPU = torch.cuda.is_available()
device = 'cuda' if HAS_GPU else 'cpu'

env_name = "CartPole-v1"

def train():
    env = gym.make(env_name)
    obs_space = env.observation_space.shape[0]
    action_space = env.action_space.n

    policy = nn.Sequential(
        nn.Linear(obs_space, 128),
        nn.ReLU(),
        nn.Linear(128, action_space),
        nn.Softmax(dim=-1)
    ).to(device)

    optimizer = Adam(policy.parameters(), lr=3e-4)
    num_episodes = 500  # reduced for testing
    episode_rewards = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            obs = torch.tensor(obs, dtype=torch.float32, device=device)
            action_prob = policy(obs)
            action_dist = Categorical(action_prob)
            action = action_dist.sample()

            obs, reward, done, truncated, info = env.step(action.item())
            total_reward += reward

            loss = -action_dist.log_prob(action) * reward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        episode_rewards.append(total_reward)

    average_reward = sum(episode_rewards[-100:]) / min(len(episode_rewards), 100)
    print(f"Average reward over last 100 episodes: {average_reward}")

    # Write metrics
    with open(os.path.join(os.environ["OUTPUT_DIR"], "metrics.json"), "w") as f:
        f.write(json.dumps({'average_reward': average_reward}))

if __name__ == "__main__":
    train()
