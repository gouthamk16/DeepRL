## Trying rl with decision transformer on the lunar lander environment

# If this is a simple robot, we can use the robots position and velocity to move the robot to a particular position
# Creatying a gym environment for the robot. Actions are move forward, move backward, turn left, turn right, stop.
# The closer the robot is to the target pposition, the higher the reward. Terminate the episode when the robot reaches the target or after a specific timeout.

from typing import Optional
import numpy as np
import gymnasium as gym

class LunarLander:
    def __init__(self):
        self.env = gym.make()
        self.num_workers = 2
        self.running_reward = 0
    def get_env(env_id):

