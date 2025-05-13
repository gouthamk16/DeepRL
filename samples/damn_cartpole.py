import argparse
import gymnasium as gym
import numpy as np
from itertools import count
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt

## Snatched the code from the pytroch repo, made it better.

## Before reading the code, a small brainwash: 0.7 works for shaping rewards, don't know why but it works.

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', # Tried many discount factor, 0.99 is the best, even 0.999 breaks the entire system
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

env = gym.make('CartPole-v1', render_mode="human" if args.render else None)
env.reset(seed=args.seed)
torch.manual_seed(args.seed)

## ORIGINAL NETWORK IMPLEMENTED BY PYTORCH 
# class Policy(nn.Module):
#     def _init_(self):
#         super(Policy, self)._init_()
#         self.affine1 = nn.Linear(4, 128)
#         self.dropout = nn.Dropout(p=0.6)
#         self.affine2 = nn.Linear(128, 2)

#         self.saved_log_probs = []
#         self.rewards = []

#     def forward(self, x):
#         x = self.affine1(x)
#         x = self.dropout(x)
#         x = F.relu(x)
#         action_scores = self.affine2(x)
#         return F.softmax(action_scores, dim=1)

## Slightly improved version -> converges faster
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, 64)
        self.affine3 = nn.Linear(64, 2)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = F.relu(x)
        x = self.affine2(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine3(x)
        return F.softmax(action_scores, dim=1)


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-2) 
eps = np.finfo(np.float32).eps.item()

# Lists to store reward data for plotting
episode_rewards = []
running_rewards = []


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


def finish_episode():
    R = 0
    policy_loss = []
    returns = deque()
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R # Discount future rewards
        returns.appendleft(R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        # policy_loss.append(-log_prob * R * len(returns) * 0.7) # Scaling the loss by the length of the episode. Longer episode - more weight (stupid)
        policy_loss.append(-log_prob * R)
    # print(policy_loss[:5])
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def plot_rewards():
    plt.figure(figsize=(12, 6))
    
    # Plot episode rewards
    plt.subplot(2, 1, 1)
    plt.plot(episode_rewards, 'g-')
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    # Plot running rewards
    plt.subplot(2, 1, 2)
    plt.plot(running_rewards, 'r-')
    plt.title('Running Average Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.tight_layout()
    plt.savefig('cartpole_rewards.png')
    # plt.show()


def main():
    running_reward = 10
    for i_episode in count(1):
        state, _ = env.reset()
        ep_reward = 0
        
        for t in range(1, 10000):  # Don't infinite loop while learning
            action = select_action(state)
            state, reward, done, _, _ = env.step(action)
            if args.render:
                env.render()
            # policy.rewards.append(reward)
            # Finding a way to provide a dynamic varable instead of 0.7
            # shaping_weight = max(0.7 * (1-i_episode/1000), 0.01) # Does not seem to work. Takes very long to converge.
            shaping_weight = 0.7
            policy.rewards.append(reward + t * shaping_weight) # makes the later actions more valuable. - Amplifies the gradients for actions in the later states - but still doesnot solve the credit assignment problem, violates the policy invariance theorem..
            ep_reward += reward
            
            if done:
                break
        
        short_term_importance = 0.05
        running_reward = short_term_importance * ep_reward + (1 - short_term_importance) * running_reward # Exponential moving average for the running reward
        episode_rewards.append(ep_reward)
        running_rewards.append(running_reward)
        
        finish_episode()
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))
        if running_reward > (env.spec.reward_threshold):
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break
        # if i_episode % 1000 == 0:
        #     break

    # Plot rewards after training is complete
    plot_rewards()


if __name__ == '__main__':
    main()