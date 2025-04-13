## DeepRL

Trying out RL algorithms on gym environments using neural nets. Researching new stuff.

Useless, barely working samples for policy optimization using neural nets in [samples](samples/)

### Fix:
1. Create a sample environment (kind of replicating [gymnasium](https://github.com/Farama-Foundation/Gymnasium))
2. Proximal Policy Optimization (PPO) on the CarRacing-v3 env.
3. Fix the policy updates (reward fluctuates and makes no sense, might be the vanilla network), try KL divergence instead of the plain loss function, try removing the log probability from the loss calculation.