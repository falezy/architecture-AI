# Gaussian Processes

Modèle non-paramétrique avec incertitude (régression/classif).

## Idée clé

**GP** : Distribution sur **functions**. `f ~ GP(m(x), k(x,x'))`. Kernel k encode similarity.

**GP Regression** : Posterior `f|data ~ N(μ*, Σ*)` donne mean + uncertainty.

## Exemples concrets

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# Kernel
kernel = RBF(length_scale=1.0)

# GP model
gp = GaussianProcessRegressor(kernel=kernel, alpha=0.1)
gp.fit(X_train, y_train)

# Predict with uncertainty
y_pred, sigma = gp.predict(X_test, return_std=True)

# Plot
plt.plot(X_test, y_pred, label='Mean')
plt.fill_between(X_test.ravel(), y_pred - 2*sigma, y_pred + 2*sigma, alpha=0.3, label='±2σ')
```

## Quand l'utiliser

- ✅ **Small data** : Sample efficient  
- ✅ **Uncertainty critical** : Medical, safety
- ✅ **Bayesian optimization**

**Quand NE PAS utiliser** : ❌ Big data → O(n³)

## Forces

✅ **Uncertainty quantification**  
✅ **Non-parametric** : Flexible  
✅ **Bayesian**

## Limites

❌ **Scalability** : O(n³)  
❌ **Kernel choice** : Manual

## Variantes / liens

**Sparse GP** : Scalable approximations  
**Deep GP** : Stacked GPs

## Références

- **Rasmussen & Williams** : GP for ML, 2006
- **GPyTorch** : [gpytorch.ai](https://gpytorch.ai/)
