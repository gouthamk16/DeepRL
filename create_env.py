# Create a rl based environment for robotic applications

# Certain stuff to consider:
# 1. Observation space, action space, reward function etc etc..
# 2. The robot itself (function that takes in an action, outputs the observation and reward)

# Action space:
# </> Move left
# </> Move right

# Observation space:
# </> Coordinates of the arm
# </> Velocity of the arm


from typing import List, Optional

class robotEnv():
    def __init__(self):
        self.action_space = 2
        self.observation_space = [[], []]
        self.reward_list = []
        self.actions = []
        self.observations = []


if __name__=='__main__':
    env = robotEnv()
    print(env.action_space)
    print(env.observation_space)
