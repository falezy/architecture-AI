# A2C / A3C

Actor-critic synchrones/asynchrones.

## Idée clé

**A2C/A3C** : **Actor** (policy) + **Critic** (value). A3C = asynchronous parallel workers, A2C = synchronous.

**Actor-Critic** :
```
Actor π(a|s): Choose action
Critic V(s): Evaluate state value
Advantage A(s,a) = Q(s,a) - V(s) = r + γV(s') - V(s)

Actor loss: -log π(a|s) × A(s,a)
Critic loss: (V(s) - target)²
```

## Exemples concrets

```python
import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU())
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)
    
    def forward(self, x):
        shared = self.shared(x)
        return self.actor(shared), self.critic(shared)

model = ActorCritic(state_dim, action_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

state = env.reset()
log_probs, values, rewards = [], [], []

for t in range(max_steps):
    state_t = torch.FloatTensor(state)
    logits, value = model(state_t)
    
    probs = torch.softmax(logits, dim=-1)
    action = torch.multinomial(probs, 1).item()
    
    next_state, reward, done, _ = env.step(action)
    
    log_probs.append(torch.log(probs[action]))
    values.append(value)
    rewards.append(reward)
    
    if done:
        break
    state = next_state

# Compute returns
returns = compute_returns(rewards, gamma=0.99)
returns_t = torch.FloatTensor(returns)
values_t = torch.cat(values)
log_probs_t = torch.stack(log_probs)

# Advantage
advantages = returns_t - values_t.detach()

# Losses
actor_loss = -(log_probs_t * advantages).mean()
critic_loss = ((returns_t - values_t) ** 2).mean()
loss = actor_loss + 0.5 * critic_loss

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

## Quand l'utiliser

- ✅ **Continuous/Discrete actions**  
- ✅ **Lower variance** vs REINFORCE  
- ✅ **Parallel training** (A3C)

**Quand NE PAS utiliser** : ❌ Need stability → PPO

## Forces

✅ **Lower variance** : Critic baseline  
✅ **Fast** : Parallel workers (A3C)  
✅ **Versatile** : Works well

## Limites

❌ **Less stable que PPO**  
❌ **Sample inefficient** : On-policy

## Variantes / liens

**A3C** : Asynchronous parallel  
**A2C** : Synchronous (plus utilisé)  
**PPO** : Plus stable version

## Références

- **A3C** : Mnih et al., 2016
- **OpenAI Baselines** : A2C implementation
