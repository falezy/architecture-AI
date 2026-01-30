# Model-based RL

Apprend un modèle du monde pour planifier/imaginer.

## Idée clé

**Model-based RL** : Apprend dynamics model `s' = f(s,a)` puis utilise pour planning/imagination. Sample efficient vs model-free.

**Dyna** : Combine real experience + simulated experience from learned model.

## Exemples concrets

```python
import torch
import torch.nn as nn

# World model
class WorldModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, state_dim)  # Predict next state
        )
    
    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))

world_model = WorldModel(state_dim, action_dim)
policy = Policy(state_dim, action_dim)

# Dyna-style training
for episode in range(1000):
    # Real experience
    state = env.reset()
    for t in range(max_steps):
        action = policy(state)
        next_state, reward, done, _ = env.step(action)
        
        # Train world model
        pred_next = world_model(state, action)
        model_loss = nn.MSELoss()(pred_next, next_state)
        model_loss.backward()
        
        # Train policy on real experience
        policy.update(state, action, reward, next_state)
        
        # Simulated experience (planning)
        for _ in range(k_planning_steps):
            sim_state = sample_from_buffer()
            sim_action = policy(sim_state)
            sim_next_state = world_model(sim_state, sim_action)
            sim_reward = estimate_reward(sim_state, sim_action)
            
            # Train policy on simulated experience
            policy.update(sim_state, sim_action, sim_reward, sim_next_state)
        
        if done:
            break
        state = next_state
```

## Quand l'utiliser

- ✅ **Sample efficiency critical** : Expensive / dangerous real interactions
- ✅ **Transferable dynamics** : Similar tasks
- ✅ **Planning horizon** : Multi-step lookahead

**Quand NE PAS utiliser** : ❌ Complex dynamics → Model errors → Model-free safer

## Forces

✅ **Sample efficient** : Learn from imagination  
✅ **Transferable** : Model reusable  
✅ **Planning** : Multi-step reasoning

## Limites

❌ **Model errors** : Compound over time  
❌ **Complex** : Model + policy learning  
❌ **Stochastic env** : Hard to model

## Variantes / liens

**Dyna** : Real + simulated experience  
**PETS** : Probabilistic ensemble  
**World Models** : VAE + RNN + controller  
**MuZero** : Learned model for planning (AlphaZero)

## Références

- **Dyna** : Sutton, 1990
- **PETS** : Chua et al., 2018
- **MuZero** : Schrittwieser et al., 2020
