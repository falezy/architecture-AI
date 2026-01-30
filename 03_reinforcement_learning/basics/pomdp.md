# POMDP

RL sous observabilité partielle (croyance sur l'état).

## Idée clé

**POMDP** : MDP where agent doesn't fully observe state. `(S, A, P, R, Ω, O)` où Ω = observations, O(o|s,a) = observation model.

**Belief state** : `b(s) = P(s|history)` = probability distribution over states.

**Solution** : Policy π(a|b) over belief states. Or use RNN/LSTM to remember history.

## Exemples concrets

```python
# RNN-based POMDP solution
import torch
import torch.nn as nn

class RecurrentPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(obs_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, obs_sequence, hidden=None):
        lstm_out, hidden = self.lstm(obs_sequence, hidden)
        action_logits = self.fc(lstm_out[:, -1, :])
        return action_logits, hidden

model = RecurrentPolicy(obs_dim, action_dim, hidden_dim=128)
hidden = None

for episode in range(1000):
    obs_history = []
    state = env.reset()
    
    for t in range(max_steps):
        obs = get_observation(state)
        obs_history.append(obs)
        
        obs_seq = torch.FloatTensor(obs_history).unsqueeze(0)
        action_logits, hidden = model(obs_seq, hidden)
        
        action = torch.softmax(action_logits, dim=-1).argmax().item()
        next_state, reward, done, _ = env.step(action)
        
        if done:
            break
        state = next_state
```

## Quand l'utiliser

- ✅ **Partial observability** : Sensors limited
- ✅ **Memory needed** : History matters
- ✅ **Real-world** : Most environments

**Quand NE PAS utiliser** : ❌ Full observability → MDP

## Forces

✅ **Realistic** : Real-world partial obs  
✅ **Memory** : RNN/LSTM solution  
✅ **Generalizes MDP**

## Limites

❌ **Complex** : Belief state tracking  
❌ **Computationally expensive**  
❌ **Hard to solve optimally**

## Variantes / liens

**MDP** : Full observability  
**Dec-POMDP** : Multi-agent partial obs  
**RNN-based** : Practical solution

## Références

- **Kaelbling et al** : "Planning and Acting in Partially Observable Domains", 1998
