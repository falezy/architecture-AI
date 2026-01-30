# Q-learning

Apprentissage de la valeur d'action (tabulaire).

## Idée clé

**Q-learning** : Apprend **Q(s,a)** = valeur d'action via Bellman equation. Off-policy, model-free.

**Q-table update** :
```
Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]

α = learning rate
γ = discount factor
```

**Epsilon-greedy exploration** :
```python
if random() < ε:
    action = random_action()  # Explore
else:
    action = argmax_a Q(s,a)   # Exploit
```

## Exemples concrets

```python
import numpy as np

# GridWorld environment
Q = np.zeros((n_states, n_actions))
α, γ, ε = 0.1, 0.99, 0.1

for episode in range(1000):
    s = env.reset()
    done = False
    
    while not done:
        # Epsilon-greedy
        if np.random.random() < ε:
            a = np.random.randint(n_actions)
        else:
            a = np.argmax(Q[s])
        
        s_next, r, done, _ = env.step(a)
        
        # Q-update (off-policy)
        Q[s,a] += α * (r + γ * np.max(Q[s_next]) - Q[s,a])
        
        s = s_next
```

## Quand l'utiliser

- ✅ **Petits états/actions** : Tabular
- ✅ **Learning from scratch**  
- ✅ **Off-policy** : Learn from old experience

**Quand NE PAS utiliser** :
- ❌ Large state spaces → DQN
- ❌ Continuous actions → Actor-Critic

## Forces

✅ **Simple** : Easy to implement  
✅ **Off-policy** : Sample efficient  
✅ **Convergence garantie** : Sous conditions  
✅ **Foundation** : Base pour DQN

## Limites

❌ **Not scalable** : Tabular only  
❌ **Slow** : Must visit all states  
❌ **No generalization** : Each state independent

## Variantes / liens

**DQN** : Q-learning avec neural network  
**SARSA** : On-policy variant (learns actual policy)  
**Double Q-learning** : Reduce overestimation

## Références

- **Watkins & Dayan** : Q-Learning, 1992
- **Sutton & Barto** : "Reinforcement Learning: An Introduction", 2018
