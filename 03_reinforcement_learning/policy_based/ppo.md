# PPO

Policy gradient stable (clipped objective), très util​isé.

## Idée clé

**PPO (Proximal Policy Optimization)** : Policy gradient **stable** via **clipped objective**. State-of-the-art pour continuous control.

**Clipped Surrogate Objective** :
```
r_t(θ) = π_θ(a|s) / π_θ_old(a|s)  # Probability ratio

L^CLIP(θ) = E[min(r_t·Â, clip(r_t, 1-ε, 1+ε)·Â)]

→ Empêche trop gros policy updates
```

**Advantage** : Â = R - V(s) (how good is action vs average)

## Exemples concrets

```python
import torch
import torch.nn as nn
from torch.distributions import Categorical

class PPO:
    def __init__(self, state_dim, action_dim):
        self.policy = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        
        self.value = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()),
            lr=3e-4
        )
    
    def act(self, state):
        probs = self.policy(torch.FloatTensor(state))
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)
    
    def update(self, states, actions, old_log_probs, returns, advantages, epsilon=0.2):
        for _ in range(10):  # Multiple epochs
            # Current policy
            probs = self.policy(states)
            dist = Categorical(probs)
            log_probs = dist.log_prob(actions)
            
            # Ratio
            ratio = torch.exp(log_probs - old_log_probs)
            
            # Clipped objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-epsilon, 1+epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            values = self.value(states).squeeze()
            value_loss = nn.functional.mse_loss(values, returns)
            
            # Total loss
            loss = policy_loss + 0.5 * value_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

# Training loop
ppo = PPO(state_dim, action_dim)

for episode in range(1000):
    states, actions, rewards, log_probs = [], [], [], []
    
    state = env.reset()
    for t in range(max_steps):
        action, log_prob = ppo.act(state)
        next_state, reward, done, _ = env.step(action)
        
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        log_probs.append(log_prob)
       
        if done:
            break
        state = next_state
    
    # Compute returns and advantages
    returns = compute_returns(rewards, gamma)
    advantages = compute_advantages(states, rewards, ppo.value)
    
    # Update
    ppo.update(
        torch.FloatTensor(states),
        torch.LongTensor(actions),
        torch.stack(log_probs),
        torch.FloatTensor(returns),
        torch.FloatTensor(advantages)
    )
```

## Quand l'utiliser

- ✅ **Continuous control** : Robotique, locomotion
- ✅ **High-dimensional** : Complex tasks
- ✅ **Stable training needed**

**Quand NE PAS utiliser** :
- ❌ Sample efficiency critical → Off-policy (SAC)
- ❌ Discrete only → DQN peut suffire

## Forces

✅ **Stable** : Clipped objective  
✅ **State-of-the-art** : Robotics, games  
✅ **Simple** : Peu d'hyperparams  
✅ **Versatile** : Discrete + continuous

## Limites

❌ **Sample inefficiency** : On-policy  
❌ **Compute** : Multiple epochs per batch  
❌ **Tuning** : Epsilon, epochs, etc.

## Variantes / liens

**PPO-Clip** : Version standard  
**PPO-Penalty** : KL penalty au lieu de clip  
**TRPO** : Précurseur (plus complexe)

## Références

- **PPO** : Schulman et al., 2017 - "Proximal Policy Optimization"
- **OpenAI Baselines** : [github.com/openai/baselines](https://github.com/openai/baselines)
