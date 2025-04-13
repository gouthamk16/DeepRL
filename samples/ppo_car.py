import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions
from torch import optim
import cv2  # Added for image preprocessing

# Check if CUDA is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Change environment from CartPole to CarRacing
env = gym.make("CarRacing-v3", render_mode='human', continuous=False)

# Observation space info
print("Observation space: ", env.observation_space)
observation, info = env.reset()
print("Sample observation shape: ", observation.shape)

# Action space info
print("Action space: ", env.action_space)
# CarRacing-v3 has a Discrete(5) action space when continuous=False
# The actions are: [do nothing, left, right, gas, brake]

# Sample agent with gym
SEED = 1111
env.reset(seed=SEED)

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# Preprocessing function for image observations
def preprocess_observation(observation):
    # Convert RGB to grayscale
    gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
    # Resize to 84x84
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    # Normalize values to [0, 1]
    normalized = resized / 255.0
    return normalized

# Setting up the networks for PPO - Policy Network and Value Network
# PPO uses an actor-critic architecture

class CNNPolicy(nn.Module):
    def __init__(self, output_dim, dropout=0.1):
        super().__init__()
        # Convolutional layers for image processing
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)  # Input: 1 channel (grayscale)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate the flattened size after convolutions
        # For 84x84 input, after convolutions: 64 x 7 x 7 = 3136
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x should be grayscale image with shape [batch, 1, 84, 84]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten the convolutional features
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def get_action_and_log_prob(self, x):
        action_pred = self(x)
        action_prob = F.softmax(action_pred, dim=-1)
        dist = distributions.Categorical(action_prob)
        action = dist.sample()
        log_prob_action = dist.log_prob(action)
        return action, log_prob_action, dist.entropy()
    
    def get_log_prob(self, x, action):
        action_pred = self(x)
        action_prob = F.softmax(action_pred, dim=-1)
        dist = distributions.Categorical(action_prob)
        log_prob_action = dist.log_prob(action)
        return log_prob_action, dist.entropy()

class CNNValue(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        # Convolutional layers for image processing - same architecture as policy
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Value head
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x should be grayscale image with shape [batch, 1, 84, 84]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten the convolutional features
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
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
        # print(observation, info)
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

# Function to collect trajectories/experience
def collect_trajectories(env, policy, value_net, num_steps):
    observations = []
    actions = []
    rewards = []
    dones = []
    log_probs = []
    values = []
    
    observation, _ = env.reset()
    done = False
    episode_reward = 0
    episode_rewards = []
    
    for _ in range(num_steps):
        # Preprocess the observation (convert to grayscale and resize)
        processed_observation = preprocess_observation(observation)
        
        # Add channel dimension and convert to tensor
        obs_tensor = torch.FloatTensor(processed_observation).unsqueeze(0).unsqueeze(0).to(device)
        
        # Get action, log probability, and value
        with torch.no_grad():
            action, log_prob, _ = policy.get_action_and_log_prob(obs_tensor)
            value = value_net(obs_tensor)
        
        next_observation, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        
        # Store the experience (store the raw observation for later preprocessing)
        observations.append(observation)
        actions.append(action.item())
        rewards.append(reward)
        dones.append(done)
        log_probs.append(log_prob.item())
        values.append(value.item())
        
        episode_reward += reward
        
        if done:
            observation, _ = env.reset()
            episode_rewards.append(episode_reward)
            episode_reward = 0
        else:
            observation = next_observation
    
    # Convert lists to numpy arrays for rewards, dones, log_probs, and values
    # Keep observations as a list of raw images
    rewards = np.array(rewards)
    dones = np.array(dones)
    log_probs = np.array(log_probs)
    values = np.array(values)
    
    return observations, actions, rewards, dones, log_probs, values, episode_rewards

# Calculate advantages using Generalized Advantage Estimation (GAE)
def compute_gae(rewards, values, dones, next_value, gamma, gae_lambda):
    advantages = np.zeros_like(rewards)
    last_gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_non_terminal = 1.0 - dones[t]
            next_values = next_value
        else:
            next_non_terminal = 1.0 - dones[t]
            next_values = values[t + 1]
        
        delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
        advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
    
    returns = advantages + values
    
    return returns, advantages

# PPO update function
def ppo_update(policy, value_net, policy_optimizer, value_optimizer, observations, actions, log_probs, returns, advantages, clip_ratio, value_coef, entropy_coef, k_epochs=4):
    # Process all observations in the batch
    processed_observations = []
    for obs in observations:
        processed_obs = preprocess_observation(obs)
        processed_observations.append(processed_obs)
    
    # Convert to tensor with proper dimensions [batch, channel, height, width]
    processed_observations = np.array(processed_observations)
    observations_tensor = torch.FloatTensor(processed_observations).unsqueeze(1).to(device)  # Add channel dimension
    
    actions = torch.LongTensor(actions).to(device)
    old_log_probs = torch.FloatTensor(log_probs).to(device)
    returns = torch.FloatTensor(returns).to(device)
    advantages = torch.FloatTensor(advantages).to(device)
    
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # Optimize policy for multiple epochs
    for _ in range(k_epochs):
        # Get current policy log probabilities and entropy
        new_log_probs, entropy = policy.get_log_prob(observations_tensor, actions)
        
        # Compute ratio (π_θ / π_θ_old)
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # Compute surrogate losses
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
        
        # PPO policy loss
        policy_loss = -torch.min(surr1, surr2).mean() - entropy_coef * entropy.mean()
        
        # Update policy
        policy_optimizer.zero_grad()
        policy_loss.backward()
        # Optional: gradient clipping
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        policy_optimizer.step()
        
        # Update value function
        value_pred = value_net(observations_tensor).squeeze()
        value_loss = F.mse_loss(value_pred, returns) * value_coef
        
        value_optimizer.zero_grad()
        value_loss.backward()
        # Optional: gradient clipping
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), 0.5)
        value_optimizer.step()
    
    return policy_loss.item(), value_loss.item()

# Train the policy network using PPO
def main():
    # Hyperparameters for PPO - adjusted for CarRacing
    MAX_EPOCHS = 1000  # Increased for more complex environment
    GAMMA = 0.99  # Discount factor
    GAE_LAMBDA = 0.95  # GAE lambda parameter
    CLIP_RATIO = 0.2  # PPO clip parameter
    STEPS_PER_EPOCH = 2048  # Increased steps per update for complex environment
    K_EPOCHS = 10  # More optimization epochs for complex task
    LR_POLICY = 0.0001  # Reduced learning rate for stability
    LR_VALUE = 0.0003  # Reduced learning rate for stability
    VALUE_COEF = 0.5  # Value loss coefficient
    ENTROPY_COEF = 0.02  # Increased entropy coefficient for more exploration
    REWARD_THRESHOLD = 900  # CarRacing is considered solved at 900 score
    N_TRIALS = 10
    PRINT_INTERVAL = 5
    
    # Network parameters - already defined by CNNPolicy and CNNValue
    OUTPUT_DIM = env.action_space.n  # 5 for CarRacing-v3 with continuous=False
    DROPOUT = 0.2  # Increased dropout for better generalization
    
    # Initialize networks and optimizers
    policy = CNNPolicy(OUTPUT_DIM, DROPOUT).to(device)
    value_net = CNNValue(DROPOUT).to(device)
    
    policy_optimizer = optim.Adam(policy.parameters(), lr=LR_POLICY)
    value_optimizer = optim.Adam(value_net.parameters(), lr=LR_VALUE)
    
    # Tracking metrics
    all_episode_rewards = []
    policy_losses = []
    value_losses = []
    
    # Main training loop
    for epoch in range(1, MAX_EPOCHS + 1):
        # Collect trajectories
        observations, actions, rewards, dones, log_probs, values, episode_rewards = collect_trajectories(
            env, policy, value_net, STEPS_PER_EPOCH)
        
        # Record episode rewards
        if episode_rewards:
            all_episode_rewards.extend(episode_rewards)
        
        # Compute GAE
        # Get the value of the last observation for bootstrapping
        if len(observations) > 0:
            # Preprocess the last observation
            processed_last_obs = preprocess_observation(observations[-1])
            processed_last_obs = torch.FloatTensor(processed_last_obs).unsqueeze(0).unsqueeze(0).to(device)
            
            with torch.no_grad():
                last_value = value_net(processed_last_obs).cpu().numpy()[0, 0]
        else:
            last_value = 0
        
        returns, advantages = compute_gae(rewards, values, dones, last_value, GAMMA, GAE_LAMBDA)
        
        # Update policy and value network with PPO
        policy_loss, value_loss = ppo_update(
            policy, value_net, policy_optimizer, value_optimizer,
            observations, actions, log_probs, returns, advantages,
            CLIP_RATIO, VALUE_COEF, ENTROPY_COEF, K_EPOCHS)
        
        policy_losses.append(policy_loss)
        value_losses.append(value_loss)
        
        # Logging
        if epoch % PRINT_INTERVAL == 0 or epoch == 1:
            # Calculate mean reward over last N_TRIALS episodes
            if len(all_episode_rewards) > 0:
                recent_rewards = all_episode_rewards[-N_TRIALS:] if len(all_episode_rewards) >= N_TRIALS else all_episode_rewards
                mean_reward = np.mean(recent_rewards)
                
                print(f'| Epoch: {epoch:3d} | Mean Reward: {mean_reward:7.1f} | Policy Loss: {np.mean(policy_losses):.4f} | Value Loss: {np.mean(value_losses):.4f} |')
                policy_losses = []
                value_losses = []
                
                # Check if we've solved the environment
                if mean_reward >= REWARD_THRESHOLD:
                    print(f'Solved CarRacing-v3 in {epoch} epochs with average reward {mean_reward:.2f}!')
                    # Save the trained model
                    torch.save(policy.state_dict(), 'carracing_policy.pth')
                    torch.save(value_net.state_dict(), 'carracing_value.pth')
                    print("Models saved.")
                    break
    
    # Close the environment
    env.close()

if __name__=='__main__':
    main()
