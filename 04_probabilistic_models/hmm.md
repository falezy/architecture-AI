# HMM

Modèle de Markov caché pour séquences (probabiliste).

## Idée clé

**HMM (Hidden Markov Model)** : **Hidden states** génèrent **observations**. Transitions entre états cachés.

```
Hidden: S₁ → S₂ → S₃ (Markov chain)
Observed: O₁   O₂   O₃ (emissions)

P(O|S), P(S'|S) = learned parameters
```

**3 problèmes** :
1. **Evaluation** : P(observations) → Forward algorithm
2. **Decoding** : Best state sequence → Viterbi
3. **Learning** : Parameters → Baum-Welch (EM)

## Exemples concrets

```python
from hmmlearn import hmm
import numpy as np

# HMM gaussien
model = hmm.GaussianHMM(n_components=3, covariance_type="full")

# Training
X = np.array([[0.5], [1.2], [0.8], [2.1], [1.9], [0.3]])
model.fit(X)

# Viterbi (meilleure séquence d'états)
log_prob, states = model.decode(X)
print("Hidden states:", states)

# Prédiction
new_obs = np.array([[1.5]])
log_prob = model.score(new_obs)

# Generate samples
samples, states = model.sample(n_samples=10)
```

## Quand l'utiliser

- ✅ **Speech recognition** : Phonèmes → audio
- ✅ **NLP** : POS tagging
- ✅ **Time series** : Régimes cachés
- ✅ **Bioinformatics** : Gene finding

**Quand NE PAS utiliser** : ❌ Long dependencies → RNN, Transformer

## Forces

✅ **Interprétable** : États explicites  
✅ **Probabiliste** : Uncertainty  
✅ **Efficient algorithms** : Forward, Viterbi

## Limites

❌ **Markov assumption** : Limited memory  
❌ **Manual states** : Need to choose K  
❌ **Linear sequences only**

## Variantes / liens

**Gaussian HMM** : Continuous observations  
**Discrete HMM** : Discrete emissions  
**CRF** : Discriminative alternative

## Références

- **Rabiner** : "A Tutorial on HMM", 1989
- **hmmlearn** : [github.com/hmmlearn](https://github.com/hmmlearn/hmmlearn)
