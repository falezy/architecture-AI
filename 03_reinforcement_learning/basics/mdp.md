# MDP

Cadre standard RL : états, actions, transitions, récompenses.

## Idée clé

**MDP (Markov Decision Process)** : Framework mathematique pour RL. `(S, A, P, R, γ)`.

```
S: États
A: Actions  
P(s'|s,a): Transition probabilities
R(s,a,s'): Rewards
γ: Discount factor

Markov property: P(s'|s,a) indépendant du passé
```

**Bellman Equation** : `V(s) = max_a [R(s,a) + γ Σ P(s'|s,a)V(s')]`

**Goal** : Find optimal policy π*(s) maximizing expected return `E[Σ γ^t r_t]`

## Exemples concrets

```python
import numpy as np

# Simple GridWorld MDP
states = [(i,j) for i in range(4) for j in range(4)]
actions = ['up', 'down', 'left', 'right']
gamma = 0.9

# Value iteration
V = {s: 0 for s in states}
for _ in range(100):
    V_new = {}
    for s in states:
        values = []
        for a in actions:
            s_next = transition(s, a)
            r = reward(s, a, s_next)
            values.append(r + gamma * V[s_next])
        V_new[s] = max(values)
    V = V_new

# Extract policy
policy = {s: actions[np.argmax([R(s,a) + gamma*V[next(s,a)] for a in actions])] for s in states}
```

## Quand l'utiliser

- ✅ **RL framework** : Foundation  
- ✅ **Fully observable** : Know state
- ✅ **Planning** : Model-based

**Quand NE PAS utiliser** : ❌ Partial observability → POMDP

## Forces

✅ **Mathematical framework** : Rigorous  
✅ **Bellman equations** : Dynamic programming

## Limites

❌ **Full observability** : Restrictive  
❌ **Known model** : Often unrealistic

## Variantes / liens

**POMDP** : Partial observability  
**Semi-MDP** : Variable timesteps

## Références

- **Bellman** : Dynamic Programming, 1957
- **Sutton & Barto** : RL book, 2018
