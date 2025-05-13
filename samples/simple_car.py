import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions
from torch import optim

## DOES NOT SEEM TO WORK, POLICY UPDATES PLATEAU ##

# init cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Cuda: ", True if torch.cuda.is_available() else False)

# init env
env = gym.make("CarRacing-v3", render_mode='human', continuous=False)

# info
print("Observation space: ", env.observation_space)
## Output: (96,96,3) | Range: [0, 255] | dtype: unit8
print("\n-\n")
print("Action space: ", env.action_space)
### Output: Discrete(5) | 0-4: do nothing, steer left, steer right, gas, brake

# nn seed
SEED = 1111
env.reset(seed=SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# cnn policy network: inputs observation, outputs action
class PolicyNetwork(nn.Module):
    def __init__(self, output_dim, dropout):
        super().__init__()
        # Changed input channels from 3 to 1 for grayscale images
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8)
        self.pool1 = nn.MaxPool2d(kernel_size=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=8)
        self.pool2 = nn.MaxPool2d(kernel_size=4)
        self.fc1 = nn.Linear(576, 64)
        self.fc2 = nn.Linear(64, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))  # (B, 16, D/2, H/2, W/2)
        x = self.pool2(F.relu(self.conv2(x)))  # (B, 32, D/4, H/4, W/4)
        x = x.reshape(x.size(0), -1)  # Flatten (using reshape and not flatten since we want the batch dim, or just use flatten and then unsqueeze)
        # print(x.shape)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# sample test
# sample_input = torch.randn(1, 3, 96, 96)
# model = PolicyNetwork(output_dim=5, dropout=0.2)
# out = model(sample_input)

# print(out.shape)

# Stepwise results
def calculate_step(rewards, discount_factor):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + R * discount_factor
        returns.insert(0, R)
    returns = torch.tensor(returns, device=device)
    normalized_results = (returns - returns.mean())/returns.std()
    return normalized_results

# Forward pass for each step
def forward_pass(env, policy, discount_factor, render_flag):
    log_prob_actions = []
    rewards = []
    done = False
    episode_returns = 0
    policy.train()
    observation, info = env.reset(seed = SEED)
    while not done:
        # Convert RGB to grayscale and normalize
        rgb_observation = torch.FloatTensor(observation/255.0).unsqueeze(0)
        # Convert RGB to grayscale using weighted average (standard conversion weights)
        grayscale_observation = 0.299 * rgb_observation[:, :, :, 0] + 0.587 * rgb_observation[:, :, :, 1] + 0.114 * rgb_observation[:, :, :, 2]
        # Add channel dimension back and move to proper shape [B, C, H, W]
        norm_observation = grayscale_observation.unsqueeze(1).to(device)
        
        action_prediction = policy(norm_observation)
        action_prob = F.softmax(action_prediction, dim=-1)
        dist = distributions.Categorical(action_prob)
        action = dist.sample()
        log_prob_action = dist.log_prob(action)
        log_prob_actions.append(log_prob_action.unsqueeze(0))
        observation, reward, terminated, truncated, _ = env.step(action.item())
        if render_flag:
            env.render()
        done = terminated or truncated
        rewards.append(reward)
        episode_returns += reward
    log_prob_actions = torch.cat(log_prob_actions)
    stepwise_returns = calculate_step(rewards, discount_factor)
    return episode_returns, stepwise_returns, log_prob_actions

# Train
def teach(config: dict):  
    # Hyperparams from the config
    epochs = config["epochs"]
    discount_factor = config["discount_factor"]
    n_trials = config["n_trials"]
    reward_threshold = config["reward_threshold"]
    print_interval = config["print_interval"]
    output_dim = env.action_space.n  # Fixed: use action space size instead of np_random
    dropout = config["dropout"]
    LR = config["learning_rate"]
    render_flag = config["render"]
    
    episode_returns = []
    policy = PolicyNetwork(output_dim, dropout).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=LR)

    total_episodes = 0
    while total_episodes < epochs:
        batch_loss = 0
        batch_return = 0
        # Acculumate returns over n_trials
        for _ in range(n_trials):
            episode_return, stepwise_returns, log_prob_actions = forward_pass(env, policy, discount_factor, render_flag)
            batch_return += episode_return
            batch_loss += -(stepwise_returns * log_prob_actions).sum()
            episode_returns.append(episode_return)
            total_episodes += 1
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        
        mean_episode_return = np.mean(episode_returns[-print_interval:])
        if total_episodes % print_interval < n_trials:
            print(f'| Episode: {total_episodes:3} | Mean Rewards: {mean_episode_return:5.1f} |')
        if mean_episode_return >= reward_threshold:
            print(f'Reached reward threshold in {total_episodes} episodes')
            break

if __name__=='__main__':
    # Hyperparams
    config = {
        "epochs": 500,
        "discount_factor": 0.99,
        "n_trials": 5,          
        "reward_threshold": 900,
        "print_interval": 10,
        "dropout": 0.2,
        "learning_rate": 0.01,
        "render": True       
    }
    
    teach(config)
