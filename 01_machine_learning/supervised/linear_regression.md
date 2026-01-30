# Linear Regression

Régression linéaire pour prédire une variable continue.

## Idée clé

La régression linéaire modélise la relation entre une **variable cible** (y) et une ou plusieurs **variables explicatives** (X) par une fonction linéaire. L'objectif est de trouver les coefficients qui minimisent l'erreur de prédiction.

**Formule générale** :
```
y = β₀ + β₁·x₁ + β₂·x₂ + ... + βₙ·xₙ + ε
```
- `y` : variable à prédire (ex: prix d'une maison)
- `x₁, x₂, ..., xₙ` : variables explicatives (ex: surface, nombre de chambres)
- `β₀` : intercept (ordonnée à l'origine)
- `β₁, β₂, ..., βₙ` : coefficients (pentes)
- `ε` : erreur (bruit)

**Objectif** : Minimiser la **Mean Squared Error (MSE)** :
```
MSE = (1/n) Σ(yᵢ - ŷᵢ)²
```
où `ŷᵢ = β₀ + β₁·x₁ + ... + βₙ·xₙ`

**Méthode de résolution** :
- **Équation normale** : `β = (XᵀX)⁻¹Xᵀy` (solution analytique)
- **Gradient Descent** : Optimisation itérative (pour grandes données)

### Équation normale vs Descente de gradient : Quand utiliser quoi ?

| Critère | Équation normale | Descente de gradient |
|---------|------------------|---------------------|
| **Formule** | `β = (XᵀX)⁻¹Xᵀy` | Itérations : `θ = θ - α·∇J(θ)` |
| **Complexité** | O(n³) | O(knd) k=itérations |
| **Petites données** (n < 10,000) | ✅ **Recommandé** : rapide, solution exacte | Possible mais inutile |
| **Grandes données** (n > 100,000) | ❌ Trop lent | ✅ **Recommandé** : efficace |
| **Nombreuses features** (d > 10,000) | ❌ Impossible (inversion matricielle) | ✅ Fonctionne bien |
| **Hyperparamètres** | Aucun | Learning rate α, nb itérations |
| **Solution** | Exacte | Approximation |
| **Utilisation** | `LinearRegression()` | `SGDRegressor()` |

#### ✅ Utilisez l'équation normale (solution directe)

**Quand** : Régression linéaire classique avec peu de données et features

```python
from sklearn.linear_model import LinearRegression

# ✅ Par défaut, utilise l'équation normale
model = LinearRegression()
model.fit(X, y)  # Solution exacte en une étape
```

**Avantages** :
- Pas d'hyperparamètres à tuner
- Solution exacte (pas d'approximation)
- Très rapide pour petits datasets

#### 🔄 Utilisez la descente de gradient

**Quand** :
1. **Grandes données** (n > 100,000)
2. **Nombreuses features** (d > 10,000)
3. **Logistic Regression** (pas de solution analytique)
4. **Régularisation Lasso** (pas de solution fermée)
5. **Online learning** (données en flux)

```python
from sklearn.linear_model import SGDRegressor

# Stochastic Gradient Descent
model = SGDRegressor(
    max_iter=1000,
    learning_rate='adaptive',  # Ajuste α automatiquement
    early_stopping=True
)
model.fit(X, y)
```

**Exemple avec descente de gradient manuelle** :
```python
import numpy as np

# Données
X = np.array([1, 2, 3, 4, 5])
y = 2 * X + 1 + np.random.randn(5) * 0.5

# Initialisation
theta_0, theta_1 = 0, 0  # Coefficients
alpha = 0.01  # Learning rate
n_iterations = 100

for i in range(n_iterations):
    # Prédictions
    y_pred = theta_0 + theta_1 * X
    
    # Gradients (dérivées partielles de MSE)
    gradient_0 = (2/len(X)) * np.sum(y_pred - y)
    gradient_1 = (2/len(X)) * np.sum((y_pred - y) * X)
    
    # Mise à jour
    theta_0 -= alpha * gradient_0
    theta_1 -= alpha * gradient_1

print(f"Résultat: y = {theta_0:.2f} + {theta_1:.2f}·x")
```

#### 📊 Variantes de gradient descent

| Type | Données/itération | Quand l'utiliser |
|------|------------------|------------------|
| **Batch GD** | Toutes | Petits datasets (stable mais lent) |
| **Stochastic GD (SGD)** | 1 exemple | Très grandes données (rapide, bruyant) |
| **Mini-batch GD** | 32-256 exemples | ✅ **Best practice** (compromis) |

#### 🎯 En pratique avec scikit-learn

**Scikit-learn choisit automatiquement la meilleure méthode** :

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression

# Équation normale (solution directe)
LinearRegression()        # ✅ Équation normale
Ridge(alpha=1.0)         # ✅ Équation normale (solution analytique existe)

# Gradient descent (obligatoire ou recommandé)
Lasso(alpha=0.1)         # ✅ Coordinate Descent (variante de GD)
LogisticRegression()     # ✅ L-BFGS (toujours, pas de solution fermée)
SGDRegressor()           # ✅ Stochastic GD explicite (grandes données)
```

#### 💡 Règle pratique

```python
# Petites/moyennes données (n < 10,000) → Équation normale
if n_samples < 10000 and n_features < 1000:
    model = LinearRegression()  # Rapide, exact

# Grandes données ou nombreuses features → Gradient descent  
else:
    model = SGDRegressor()  # Scalable
```

## Exemples concrets

### 1. Régression linéaire simple : Prédire le prix d'une maison

**Scénario** : Vous avez des données sur 100 maisons avec leur surface (m²) et leur prix (€). Vous voulez prédire le prix d'une nouvelle maison.

**Données d'exemple** :
```
Surface (m²)  →  Prix (€)
50            →  150,000
80            →  240,000
120           →  360,000
```

**Code Python avec scikit-learn** :
```python
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 1. Données d'entraînement
X_train = np.array([50, 80, 100, 120, 150]).reshape(-1, 1)  # Surface (m²)
y_train = np.array([150000, 240000, 300000, 360000, 450000])  # Prix (€)

# 2. Créer et entraîner le modèle
model = LinearRegression()
model.fit(X_train, y_train)

# 3. Afficher les coefficients
print(f"Intercept (β₀): {model.intercept_:.2f} €")
print(f"Coefficient (β₁): {model.coef_[0]:.2f} €/m²")
# Résultat : y = 0 + 3000·x  (approximativement)

# 4. Prédire le prix d'une maison de 90 m²
nouvelle_surface = np.array([[90]])
prix_predit = model.predict(nouvelle_surface)
print(f"Prix prédit pour 90 m²: {prix_predit[0]:,.0f} €")
# Résultat : ~270,000 €

# 5. Évaluer le modèle
from sklearn.metrics import r2_score, mean_squared_error
y_pred = model.predict(X_train)
print(f"R² Score: {r2_score(y_train, y_pred):.3f}")  # Proche de 1 = bon
print(f"RMSE: {np.sqrt(mean_squared_error(y_train, y_pred)):,.0f} €")
```

**Visualisation** :
```python
plt.scatter(X_train, y_train, color='blue', label='Données réelles')
plt.plot(X_train, model.predict(X_train), color='red', label='Régression')
plt.xlabel('Surface (m²)')
plt.ylabel('Prix (€)')
plt.legend()
plt.show()
```

---

### 2. Régression linéaire multiple : Prédire un salaire

**Scénario** : Prédire le salaire d'un employé en fonction de son **expérience** (années) et son **niveau d'éducation** (1=Bachelor, 2=Master, 3=PhD).

**Données d'exemple** :
```
Expérience | Éducation | Salaire (k€)
2          | 1         | 35
5          | 2         | 50
10         | 3         | 75
```

**Code Python** :
```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 1. Données
X = np.array([
    [2, 1],   # 2 ans, Bachelor
    [5, 2],   # 5 ans, Master
    [10, 3],  # 10 ans, PhD
    [3, 1],   # 3 ans, Bachelor
    [7, 2],   # 7 ans, Master
    [15, 3],  # 15 ans, PhD
])
y = np.array([35, 50, 75, 38, 55, 90])  # Salaire en k€

# 2. Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. Entraîner le modèle
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Coefficients
print(f"Intercept: {model.intercept_:.2f} k€")
print(f"Coef Expérience: {model.coef_[0]:.2f} k€/an")
print(f"Coef Éducation: {model.coef_[1]:.2f} k€/niveau")
# Résultat : Salaire = 20 + 4·expérience + 8·éducation (approx.)

# 5. Prédire pour un employé (8 ans, Master)
nouveau_profil = np.array([[8, 2]])
salaire_predit = model.predict(nouveau_profil)
print(f"Salaire prédit: {salaire_predit[0]:.1f} k€")
# Résultat : ~56 k€

# 6. Évaluation sur test set
from sklearn.metrics import r2_score, mean_absolute_error
y_pred_test = model.predict(X_test)
print(f"R² (test): {r2_score(y_test, y_pred_test):.3f}")
print(f"MAE (test): {mean_absolute_error(y_test, y_pred_test):.2f} k€")
```

---

### 3. Implementation from scratch (NumPy)

**Comprendre les mathématiques** :
```python
import numpy as np

# Données simples
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 5, 4, 5])

# Ajouter une colonne de 1 pour l'intercept
X_b = np.c_[np.ones((len(X), 1)), X]  # [1, x]

# Équation normale: β = (X^T X)^-1 X^T y
beta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
print(f"β₀ (intercept): {beta[0]:.3f}")
print(f"β₁ (pente): {beta[1]:.3f}")

# Prédiction
X_new = np.array([[0], [6]])
X_new_b = np.c_[np.ones((len(X_new), 1)), X_new]
y_pred = X_new_b @ beta
print(f"Prédictions: {y_pred}")
```

## Quand l'utiliser

- ✅ **Relation linéaire** : La relation entre X et y est approximativement linéaire
- ✅ **Interprétabilité** : Besoin de comprendre l'impact de chaque variable (coefficients)
- ✅ **Prédiction continue** : Variable cible numérique (prix, température, salaire)
- ✅ **Baseline** : Modèle de référence simple avant d'essayer des modèles complexes
- ✅ **Peu de données** : Fonctionne bien avec peu d'exemples (contrairement aux DNN)

**Cas d'usage typiques** :
- 🏠 **Immobilier** : Prix en fonction de surface, localisation, nombre de chambres
- 💰 **Finance** : Prédire le chiffre d'affaires en fonction du budget marketing
- 📈 **Économie** : Relation entre PIB et chômage
- 🌡️ **Sciences** : Température en fonction de l'altitude
- 📊 **A/B Testing** : Impact d'une variable sur une métrique

## Forces

✅ **Simplicité** : Facile à comprendre et à implémenter  
✅ **Rapide** : Entraînement très rapide (solution analytique)  
✅ **Interprétable** : Les coefficients indiquent l'importance de chaque variable  
✅ **Peu de données** : Fonctionne avec peu d'exemples d'entraînement  
✅ **Pas d'hyperparamètres** : Aucun tuning nécessaire (version de base)  
✅ **Inférence instantanée** : Prédiction = simple multiplication matricielle  
✅ **Robuste au surapprentissage** : Avec régularisation (Ridge, Lasso)

## Limites

❌ **Hypothèse de linéarité** : Ne capture pas les relations non-linéaires (x²)  
❌ **Sensible aux outliers** : Les valeurs extrêmes biaisent les coefficients  
❌ **Multicolinéarité** : Problème si les variables X sont fortement corrélées  
❌ **Homoscédasticité requise** : Variance de l'erreur doit être constante  
❌ **Dimensionnalité** : Problème si p >> n (plus de variables que d'exemples)  
❌ **Features engineering** : Nécessite parfois des transformations manuelles (log, polynômes)  
❌ **Prédictions bornées** : Peut prédire des valeurs impossibles (prix négatif)

**Exemple de limitation** :
```python
# Relation non-linéaire : y = x²
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = X.flatten() ** 2  # [1, 4, 9, 16, 25]

# Régression linéaire échoue (R² faible)
model = LinearRegression().fit(X, y)
print(f"R² Score: {model.score(X, y):.3f}")  # ~0.8 (pas terrible)

# Solution : Polynomial Features
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)  # [1, x, x²]
model_poly = LinearRegression().fit(X_poly, y)
print(f"R² Score (poly): {model_poly.score(X_poly, y):.3f}")  # 1.0 (parfait)
```

## Variantes / liens

### Extensions de la régression linéaire

**1. Régularisation** : Pénaliser les coefficients pour éviter le surapprentissage
- **Ridge (L2)** : `Loss = MSE + α·Σβ²` → coefficients plus petits
  ```python
  from sklearn.linear_model import Ridge
  model = Ridge(alpha=1.0)  # α contrôle la régularisation
  ```
- **Lasso (L1)** : `Loss = MSE + α·Σ|β|` → sélection de features (β=0)
  ```python
  from sklearn.linear_model import Lasso
  model = Lasso(alpha=0.1)  # Certains β deviennent exactement 0
  ```
- **Elastic Net** : Combinaison de Ridge + Lasso
  ```python
  from sklearn.linear_model import ElasticNet
  model = ElasticNet(alpha=0.1, l1_ratio=0.5)
  ```

**2. Régression polynomiale** : Capturer les relations non-linéaires
```python
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=3)  # x, x², x³
X_poly = poly.fit_transform(X)
model = LinearRegression().fit(X_poly, y)
```

**3. Régression robuste** : Résistant aux outliers
```python
from sklearn.linear_model import HuberRegressor, RANSACRegressor
model = HuberRegressor()  # Moins sensible aux outliers
```

### Relations avec d'autres modèles

- **Logistic Regression** : Version classification (sigmoïde au lieu de linéaire)
- **SVR (Support Vector Regression)** : Régression avec marge (kernel pour non-linéarité)
- **Decision Trees** : Capture automatiquement les non-linéarités
- **Random Forest / Gradient Boosting** : Ensemble de trees (plus précis mais moins interprétable)
- **Neural Networks** : Généralisation avec fonctions d'activation non-linéaires
- **GAM (Generalized Additive Models)** : Somme de fonctions non-linéaires

### Prétraitement associé

- **Standardisation** : Mettre les features à la même échelle
  ```python
  from sklearn.preprocessing import StandardScaler
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)
  ```
- **Encodage catégoriel** : One-hot encoding pour variables catégorielles
  ```python
  from sklearn.preprocessing import OneHotEncoder
  encoder = OneHotEncoder()
  X_encoded = encoder.fit_transform(X_cat)
  ```

## Références

### Documentation et tutoriels
- **Scikit-learn** : [Linear Models](https://scikit-learn.org/stable/modules/linear_model.html)
- **StatQuest** : [Linear Regression Explained](https://www.youtube.com/watch?v=nk2CQITm_eo) (YouTube)
- **Andrew Ng** : [ML Course - Linear Regression](https://www.coursera.org/learn/machine-learning)

### Livres
- **"An Introduction to Statistical Learning"** (James et al., 2021) - Chapitre 3
- **"The Elements of Statistical Learning"** (Hastie et al., 2009) - Chapitre 3
- **"Pattern Recognition and Machine Learning"** (Bishop, 2006) - Chapitre 3

### Outils Python
```python
# Scikit-learn (le plus populaire)
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# Statsmodels (plus de statistiques détaillées)
import statsmodels.api as sm
model = sm.OLS(y, X).fit()
print(model.summary())  # P-values, intervalles de confiance, etc.

# NumPy (implémentation manuelle)
beta = np.linalg.lstsq(X, y, rcond=None)[0]
```

### Tests statistiques associés
- **Test de normalité des résidus** : Shapiro-Wilk
- **Test d'hétéroscédasticité** : Breusch-Pagan
- **Test de multicolinéarité** : VIF (Variance Inflation Factor)
- **Test de significativité** : P-values des coefficients
