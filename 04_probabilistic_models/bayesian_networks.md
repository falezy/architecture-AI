# Bayesian Networks

Graphes probabilistes pour inférence causale/probabiliste.

## Idée clé

**Bayesian Network** : **DAG** (Directed Acyclic Graph) encode conditional dependencies. `P(X₁,...,Xₙ) = ∏ P(Xᵢ|Parents(Xᵢ))`

## Exemples concrets

```python
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# Define structure
model = BayesianNetwork([('Rain', 'Sprinkler'), ('Rain', 'Grass'), ('Sprinkler', 'Grass')])

# Learn parameters from data
model.fit(data, estimator=MaximumLikelihoodEstimator)

# Inference
infer = VariableElimination(model)
prob = infer.query(['Grass'], evidence={'Rain': 1})
print(prob)
```

## Quand l'utiliser

- ✅ **Causal reasoning**  
- ✅ **Uncertain knowledge**  
- ✅ **Medical diagnosis**

## Forces

✅ **Interpretable** : DAG structure  
✅ **Probabilistic** : Uncertainty

## Limites

❌ **Scalability** : Exact inference NP-hard  
❌ **Structure learning** : Difficult

## Variantes / liens

**Markov Networks** : Undirected

## Références

- **Pearl** : "Probabilistic Reasoning", 1988
