import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions
from torch import optim

# Check if CUDA is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

env = gym.make("CartPole-v1", render_mode='human')

# Observation space info
print("Observation space: ", env.observation_space)
observation, info = env.reset()
print("Sample observation: ", observation)

# Action space info
print("Action space: ", env.action_space)
# The output is Discrete(2) -> agent can perform two actions, either move the cart to the left or to the right

# Sample agent with gym

SEED = 1111
env.reset(seed=SEED)

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# Setting up the policies -> ann
# Function of the policy network - map observed state to actions
# Input - observation, Output - right action

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.layer1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.layer2(x)
        return x

# Out goal - maximize the total return
# Return at each step - cumulative sum of rewards obtained until that point
# Future rewards are to be adjusted using a discount factor

# Calculate stepwise results
def calculate_stepwise_returns(rewards, discount_factor):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + R * discount_factor
        returns.insert(0, R)

    returns = torch.tensor(returns, device=device)
    normalized_results = (returns-returns.mean())/returns.std()
    return normalized_results

# Forward pass
def forward_pass(env, policy, discount_factor):
    log_prob_actions = []
    rewards = []
    done = False
    episode_return = 0
    policy.train()
    observation, info = env.reset()
    while not done:
        observation = torch.FloatTensor(observation).unsqueeze(0).to(device)
        action_pred = policy(observation)
        action_prob = F.softmax(action_pred, dim=-1)
        dist = distributions.Categorical(action_prob)
        action = dist.sample()
        log_prob_action = dist.log_prob(action)
        observation, reward, terminated, truncated, info = env.step(action.item())
        env.render()
        done = terminated or truncated
        log_prob_actions.append(log_prob_action)
        rewards.append(reward)
        episode_return += reward
    log_prob_actions = torch.cat(log_prob_actions)
    stepwise_returns = calculate_stepwise_returns(rewards, discount_factor)
    return episode_return, stepwise_returns, log_prob_actions 

def calculate_loss(stepwise_returns, log_prob_actions):
    loss = -(stepwise_returns * log_prob_actions).sum()
    return loss

def update_policy(stepwise_returns, log_prob_actions, optimizer):
    stepwise_returns = stepwise_returns.detach()
    loss = calculate_loss(stepwise_returns, log_prob_actions)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

# Train the policy network
def main():
    # Hyperparams
    MAX_EPOCHS = 500
    DISCOUNT_FACTOR = 0.99
    N_TRIALS = 25
    REWARD_THRESHOLD = 475
    PRINT_INTERVAL = 10
    INPUT_DIM = env.observation_space.shape[0]
    HIDDEN_DIM = 128
    OUTPUT_DIM = env.action_space.n
    DROPOUT = 0.5
    episode_returns = []
    policy = PolicyNetwork(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, DROPOUT).to(device)
    LR = 0.01
    optimizer = optim.Adam(policy.parameters(), lr=LR)
    
    for episode in range(1, MAX_EPOCHS+1):
        episode_return, stepwise_returns, log_prob_actions = forward_pass(env, policy, DISCOUNT_FACTOR)
        _ = update_policy(stepwise_returns, log_prob_actions, optimizer)
        episode_returns.append(episode_return)
        mean_episode_return = np.mean(episode_returns[-N_TRIALS:])
        if episode % PRINT_INTERVAL == 0:
            print(f'| Episode: {episode:3} | Mean Rewards: {mean_episode_return:5.1f} |')
        if mean_episode_return >= REWARD_THRESHOLD:
            print(f'Reached reward threshold in {episode} episodes')
            break

if __name__=='__main__':
    main()
