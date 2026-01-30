# SAC / TD3 / DDPG

Actor-critic pour actions continues (off-policy).

## Idée clé

**DDPG** : DQN pour continuous actions. Actor μ(s) outputs action, Critic Q(s,a) evaluates.  
**TD3** : DDPG + tricks (twin critics, delayed updates, target noise).  
**SAC** : Maximum entropy RL. Robuste, state-of-the-art continuous control.

**SAC** : `max E[Σ r_t + α H(π)]` where H = entropy → encourage exploration.

## Exemples concrets

```python
# TD3 example (simplified)
import torch
import torch.nn as nn

actor = nn.Sequential(nn.Linear(state_dim, 256), nn.ReLU(), nn.Linear(256, action_dim), nn.Tanh())
critic1 = nn.Sequential(nn.Linear(state_dim + action_dim, 256), nn.ReLU(), nn.Linear(256, 1))
critic2 = nn.Sequential(nn.Linear(state_dim + action_dim, 256), nn.ReLU(), nn.Linear(256, 1))

# ... target networks

for step in range(max_steps):
    action = actor(state) + noise
    next_state, reward, done, _ = env.step(action)
    
    # Update critic with twin Q
    target_action = actor_target(next_state) + noise
    target_Q1 = critic1_target(torch.cat([next_state, target_action], dim=1))
    target_Q2 = critic2_target(torch.cat([next_state, target_action], dim=1))
    target_Q = reward + gamma * torch.min(target_Q1, target_Q2)
    
    Q1 = critic1(torch.cat([state, action], dim=1))
    Q2 = critic2(torch.cat([state, action], dim=1))
    critic_loss = nn.MSELoss()(Q1, target_Q) + nn.MSELoss()(Q2, target_Q)
    
    # Update actor (delayed)
    if step % policy_freq == 0:
        actor_loss = -critic1(torch.cat([state, actor(state)], dim=1)).mean()
```

## Quand l'utiliser

- ✅ **Continuous control** : Robotics, locomotion
- ✅ **Off-policy** : Sample efficient
- ✅ **High-dimensional** : Complex tasks

## Forces

✅ **Continuous actions**  
✅ **Sample efficient** : Off-policy  
✅ **Stable** : TD3/SAC tricks

## Limites

❌ **Complex** : Many hyperparams  
❌ **Discrete actions** → DQN better

## Variantes / liens

**DDPG** : Base (2016)  
**TD3** : DDPG + stabilization (2018)  
**SAC** : Maximum entropy (2018, SOTA)

## Références

- **DDPG** : Lillicrap et al., 2015
- **TD3** : Fujimoto et al., 2018
- **SAC** : Haarnoja et al., 2018
