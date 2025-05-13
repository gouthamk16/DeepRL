## DeepRL

Trying out RL algorithms on gym environments using neural nets. Researching new stuff.

Useless, barely working samples for policy optimization using neural nets in [samples](samples/). 

### Fix:
1. Proximal Policy Optimization (PPO) on the CarRacing-v3 env.
2. Fix the policy updates (reward fluctuates and makes no sense, might be the vanilla network), try KL divergence instead of the plain loss function, try removing the log probability from the loss calculation.