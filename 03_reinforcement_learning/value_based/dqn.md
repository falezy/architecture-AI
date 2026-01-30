# DQN

Q-learning avec réseau neuronal (Deep Q-Network).

## Idée clé

**DQN** : Q-learning avec neural network au lieu de table. Breakthrough RL (Atari 2015).

**Innovations** :
```
1. Experience Replay: Store (s,a,r,s') in buffer, sample random batches
2. Target Network: Separate network for stability
3. CNN: Pour images (Atari)

Loss = E[(r + γ max_a' Q_target(s',a') - Q(s,a))²]
```

## Exemples concrets

```python
import torch
import torch.nn as nn
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, x):
        return self.net(x)

# Replay buffer
replay_buffer = deque(maxlen=10000)

# Networks
q_network = DQN(state_dim, action_dim)
target_network = DQN(state_dim, action_dim)
target_network.load_state_dict(q_network.state_dict())

optimizer = torch.optim.Adam(q_network.parameters(), lr=1e-3)

for episode in range(1000):
    state = env.reset()
    for t in range(max_steps):
        # Epsilon-greedy
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = q_network(torch.FloatTensor(state)).argmax().item()
        
        next_state, reward, done, _ = env.step(action)
        replay_buffer.append((state, action, reward, next_state, done))
        
        # Training
        if len(replay_buffer) > batch_size:
            batch = random.sample(replay_buffer, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            
            states_t = torch.FloatTensor(states)
            actions_t = torch.LongTensor(actions)
            rewards_t = torch.FloatTensor(rewards)
            next_states_t = torch.FloatTensor(next_states)
            dones_t = torch.FloatTensor(dones)
            
            # Q-values
            q_values = q_network(states_t).gather(1, actions_t.unsqueeze(1))
            
            # Target Q-values
            with torch.no_grad():
                next_q_values = target_network(next_states_t).max(1)[0]
                targets = rewards_t + gamma * next_q_values * (1 - dones_t)
            
            # Loss
            loss = nn.functional.mse_loss(q_values.squeeze(), targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Update target network
        if t % target_update_freq == 0:
            target_network.load_state_dict(q_network.state_dict())
        
        if done:
            break
        state = next_state
```

## Quand l'utiliser

- ✅ **High-dimensional states** : Images, etc.
- ✅ **Discrete actions** : Atari games
- ✅ **Off-policy** : Sample efficiency

**Quand NE PAS utiliser** :
- ❌ Continuous actions → DDPG, SAC
- ❌ Besoin on-policy → PPO

## Forces

✅ **Scalable** : Neural approximation  
✅ **Atari success** : Surhumain sur beaucoup de jeux  
✅ **Off-policy** : Replay buffer efficient

## Limites

❌ **Overestimation bias** : Max operator  
❌ **Instable** : Sans tricks (replay, target net)  
❌ **Discrete actions only**

## Variantes / liens

**Double DQN** : Réduit overestimation  
**Dueling DQN** : Separate V(s) et A(s,a)  
**Rainbow DQN** : Combine améliorations  
**Prioritized Experience Replay** : Sample important transitions

## Références

- **DQN** : Mnih et al., 2015 - "Human-level control through deep RL"
- **Double DQN** : van Hasselt et al., 2015
- **Dueling DQN** : Wang et al., 2016
