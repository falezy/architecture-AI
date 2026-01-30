# REINFORCE

Policy gradient simple basé sur retours.

## Idée clé

**REINFORCE** : Monte Carlo policy gradient. `∇J(θ) = E[∇log π(a|s) × G_t]` où G_t = return depuis t.

## Exemples concrets

```python
import torch
import torch.nn as nn

policy = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, action_dim), nn.Softmax(dim=-1))
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

for episode in range(1000):
    states, actions, rewards = [], [], []
    state = env.reset()
    
    while not done:
        probs = policy(torch.FloatTensor(state))
        action = torch.multinomial(probs, 1).item()
        next_state, reward, done, _ = env.step(action)
        
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        state = next_state
    
    # Compute returns
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    
    returns = torch.FloatTensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-9)  # Normalize
    
    # Policy gradient
    loss = 0
    for s, a, G in zip(states, actions, returns):
        probs = policy(torch.FloatTensor(s))
        loss += -torch.log(probs[a]) * G
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Quand l'utiliser

- ✅ **Simple baseline** : Start simple
- ✅ **Continuous actions** : Works directly

## Forces

✅ **Simple** : Easy to implement  
✅ **Unbiased** : Monte Carlo

## Limites

❌ **High variance** : No baseline  
❌ **Sample inefficient** : Monte Carlo

## Variantes / liens

**With baseline** : Reduce variance → Actor-Critic  
**PPO** : Modern stable version

## Références

- **Williams** : "REINFORCE", 1992
