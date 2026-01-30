# SVM

Classification/régression via marges maximales, kernels possibles.

## Idée clé

**SVM (Support Vector Machine)** est un algorithme de classification qui trouve la **frontière de décision optimale** en maximisant la **marge** entre les classes. La marge est la distance entre la frontière et les points les plus proches de chaque classe (appelés **support vectors**).

**Principe** :
1. Trouver l'hyperplan qui sépare les classes avec la **plus grande marge possible**
2. Les points les plus proches de la frontière sont les **support vectors**
3. Seuls les support vectors influencent la position de la frontière
4. Utiliser des **kernels** pour gérer les données non-linéairement séparables

**Formule (cas linéaire)** :
```
f(x) = w·x + b
Prédiction: sign(f(x)) = { +1 si f(x) ≥ 0
                          { -1 sinon
```
- `w` : vecteur de poids (normal à l'hyperplan)
- `b` : biais (intercept)
- Marge = `2/||w||`

**Objectif d'optimisation** :
```
Maximiser: marge = 2/||w||
Équivalent à minimiser: ||w||² / 2
Sous contrainte: yᵢ(w·xᵢ + b) ≥ 1  pour tout i
```

**Visualisation (2D)** :
```
        Classe +1
          •  •  •
         ╱       ╲
   ─────●─────────●───── Hyperplan
       ╱ Support   ╲
      •  Vectors    •
    Classe -1
    
    ← Marge →
```

**Marge douce (Soft Margin)** :
- Paramètre `C` : contrôle le compromis entre marge large et erreurs
- `C` grand : Marge étroite, peu d'erreurs (peut overfitter)
- `C` petit : Marge large, tolère plus d'erreurs (plus régularisé)

**Kernel Trick** :
- Permet de transformer des données non-linéairement séparables
- Projette les données dans un espace de dimension supérieure
- Kernels populaires : linéaire, polynomial, RBF (Gaussian), sigmoïd

## Exemples concrets

### 1. Classification linéaire : Données séparables

**Scénario** : Classifier deux classes linéairement séparables.

**Code Python avec SVM linéaire** :
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 1. Générer données linéairement séparables
X, y = make_blobs(
    n_samples=100, 
    centers=2, 
    n_features=2,
    center_box=(-5, 5),
    random_state=42
)

# 2. Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. SVM linéaire
model = SVC(
    kernel='linear',    # Kernel linéaire
    C=1.0,              # Paramètre de régularisation
    random_state=42
)
model.fit(X_train, y_train)

# 4. Prédictions
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print(f"\nNombre de support vectors: {len(model.support_vectors_)}")
print(f"Indices des support vectors: {model.support_}")

# 5. Visualisation
def plot_svm_decision_boundary(model, X, y, title):
    # Créer grille
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.02),
        np.arange(y_min, y_max, 0.02)
    )
    
    # Prédire sur grille
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Tracer
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
    
    # Tracer les support vectors
    plt.scatter(
        model.support_vectors_[:, 0],
        model.support_vectors_[:, 1],
        s=200, 
        linewidth=2,
        facecolors='none', 
        edgecolors='green',
        label='Support Vectors'
    )
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend()
    plt.colorbar()
    plt.show()

plot_svm_decision_boundary(model, X, y, 'SVM Linéaire - Frontière de décision')

# 6. Équation de l'hyperplan
w = model.coef_[0]
b = model.intercept_[0]
print(f"\nÉquation de l'hyperplan:")
print(f"  {w[0]:.3f}·x₁ + {w[1]:.3f}·x₂ + {b:.3f} = 0")
print(f"  Marge: {2 / np.linalg.norm(w):.3f}")
```

---

### 2. Classification non-linéaire : Kernel RBF

**Scénario** : Données non-linéairement séparables (cercles concentriques).

**Code Python avec kernel RBF (Gaussian)** :
```python
from sklearn.datasets import make_circles

# 1. Données circulaires (non-linéaires)
X, y = make_circles(
    n_samples=200, 
    factor=0.5,      # Ratio entre cercles
    noise=0.1,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 2. Essayer SVM linéaire (va échouer)
svm_linear = SVC(kernel='linear')
svm_linear.fit(X_train, y_train)
print(f"SVM Linéaire accuracy: {svm_linear.score(X_test, y_test):.2%}")

# 3. SVM avec kernel RBF (Gaussian)
svm_rbf = SVC(
    kernel='rbf',    # Radial Basis Function
    C=1.0,           # Régularisation
    gamma='scale',   # Influence de chaque exemple (défaut: 1/(n_features * X.var()))
    random_state=42
)
svm_rbf.fit(X_train, y_train)
print(f"SVM RBF accuracy: {svm_rbf.score(X_test, y_test):.2%}")

# 4. Visualisation
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# SVM linéaire
ax = axes[0]
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
Z = svm_linear.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
ax.set_title('SVM Linéaire (échec)')

# SVM RBF
ax = axes[1]
Z = svm_rbf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
ax.scatter(
    svm_rbf.support_vectors_[:, 0],
    svm_rbf.support_vectors_[:, 1],
    s=200, linewidth=2, facecolors='none', edgecolors='green',
    label='Support Vectors'
)
ax.set_title(f'SVM RBF (accuracy: {svm_rbf.score(X_test, y_test):.0%})')
ax.legend()

plt.tight_layout()
plt.show()

print(f"\nNombre de support vectors (RBF): {len(svm_rbf.support_vectors_)}")
```

---

### 3. Comparaison des kernels

**Code pour comparer linéaire, polynomial, RBF** :
```python
from sklearn.datasets import make_moons

# Données en forme de lunes (non-linéaires)
X, y = make_moons(n_samples=200, noise=0.15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Tester différents kernels
kernels = {
    'Linéaire': SVC(kernel='linear', C=1),
    'Polynomial (deg=3)': SVC(kernel='poly', degree=3, C=1),
    'RBF': SVC(kernel='rbf', C=1, gamma='scale'),
    'Sigmoïd': SVC(kernel='sigmoid', C=1)
}

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for idx, (name, model) in enumerate(kernels.items()):
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    
    # Visualiser
    ax = axes[idx]
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
    ax.scatter(
        model.support_vectors_[:, 0],
        model.support_vectors_[:, 1],
        s=100, linewidth=1.5, facecolors='none', edgecolors='green'
    )
    ax.set_title(f'{name}\nAccuracy: {accuracy:.0%}, SV: {len(model.support_vectors_)}')

plt.tight_layout()
plt.show()
```

---

### 4. Tuning de C et gamma (kernel RBF)

**Code pour comprendre l'impact de C et gamma** :
```python
# Impact de C (régularisation)
C_values = [0.1, 1, 10, 100]
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

X, y = make_circles(n_samples=200, factor=0.5, noise=0.1, random_state=42)

for idx, C in enumerate(C_values):
    model = SVC(kernel='rbf', C=C, gamma='scale')
    model.fit(X, y)
    
    ax = axes[idx]
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k', s=30)
    ax.set_title(f'C={C}\nSV: {len(model.support_vectors_)}')

plt.tight_layout()
plt.show()

# Impact de gamma
gamma_values = [0.1, 1, 10, 100]
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for idx, gamma in enumerate(gamma_values):
    model = SVC(kernel='rbf', C=1, gamma=gamma)
    model.fit(X, y)
    
    ax = axes[idx]
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k', s=30)
    ax.set_title(f'gamma={gamma}\nSV: {len(model.support_vectors_)}')

plt.tight_layout()
plt.show()
```

**Interprétation** :
- **C petit** : Marge large, tolère erreurs → underfitting
- **C grand** : Marge étroite, peu d'erreurs → overfitting
- **gamma petit** : Influence large (frontière lisse)
- **gamma grand** : Influence locale (frontière complexe) → overfitting

---

### 5. SVM pour régression (SVR)

**Code pour Support Vector Regression** :
```python
from sklearn.svm import SVR
import numpy as np

# 1. Données de régression avec bruit
np.random.seed(42)
X = np.sort(5 * np.random.rand(100, 1), axis=0)
y = np.sin(X).ravel() + np.random.randn(100) * 0.1

# 2. Comparer différents kernels
svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr_lin = SVR(kernel='linear', C=100, epsilon=0.1)
svr_poly = SVR(kernel='poly', C=100, degree=3, epsilon=0.1)

# 3. Entraîner
models = [svr_rbf, svr_lin, svr_poly]
names = ['RBF', 'Linear', 'Polynomial']

X_test = np.linspace(0, 5, 300)[:, np.newaxis]

plt.figure(figsize=(12, 6))
plt.scatter(X, y, color='darkorange', label='Données')

for model, name, color in zip(models, names, ['navy', 'red', 'green']):
    model.fit(X, y)
    y_pred = model.predict(X_test)
    plt.plot(X_test, y_pred, color=color, linewidth=2, label=f'SVR {name}')

plt.xlabel('X')
plt.ylabel('y')
plt.title('Support Vector Regression - Comparaison kernels')
plt.legend()
plt.show()
```

---

### 6. Grid Search pour tuning optimal

**Code pour trouver les meilleurs hyperparamètres** :
```python
from sklearn.model_selection import GridSearchCV

# Données
X, y = make_circles(n_samples=200, factor=0.5, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Grille de paramètres
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    'kernel': ['rbf', 'poly', 'sigmoid']
}

# Grid Search
svm = SVC()
grid_search = GridSearchCV(
    svm, 
    param_grid, 
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"Meilleurs paramètres: {grid_search.best_params_}")
print(f"Meilleur score CV: {grid_search.best_score_:.3f}")
print(f"Score test: {grid_search.score(X_test, y_test):.3f}")

# Entraîner avec meilleurs paramètres
best_model = grid_search.best_estimator_
print(f"\nNombre de support vectors: {len(best_model.support_vectors_)}")
```

## Quand l'utiliser

- ✅ **Données moyenne/haute dimensionnalité** : Fonctionne bien avec beaucoup de features
- ✅ **Frontières complexes** : Kernels permettent des décisions non-linéaires
- ✅ **Petits/moyens datasets** : Moins de 10,000 exemples (scaling en O(n²) ou O(n³))
- ✅ **Robustesse aux outliers** : Seuls les support vectors comptent
- ✅ **Marges claires** : Classes bien séparées
- ✅ **Classification binaire** : Excellente performance (extension multi-classe possible)

**Cas d'usage typiques** :
- 📝 **Text classification** : Catégorisation de documents, spam detection
- 🧬 **Bioinformatique** : Classification de protéines, gènes
- 🖼️ **Vision** : Reconnaissance de visages (avec features HOG/SIFT)
- 💊 **Médecine** : Diagnostic (nombreuses features, peu d'exemples)
- 💰 **Finance** : Scoring de crédit

**Quand NE PAS utiliser** :
- ❌ Très grandes données (>100,000) → trop lent → Logistic Regression, Random Forest
- ❌ Beaucoup de bruit → Random Forest plus robuste
- ❌ Besoin de probabilités calibrées → predict_proba de SVM peu fiable (utiliser Platt scaling)
- ❌ Interprétabilité critique → Decision Tree, Linear Regression
- ❌ Images/texte brut → Deep Learning (CNN, Transformers)

## Forces

✅ **Frontières complexes** : Kernels permettent séparations non-linéaires  
✅ **Robuste en haute dimension** : Fonctionne bien avec d >> n  
✅ **Memory efficient** : Utilise seulement les support vectors  
✅ **Versatile** : Nombreux kernels (linéaire, RBF, polynomial, custom)  
✅ **Régularisation intégrée** : Paramètre C contrôle overfitting  
✅ **Base théorique solide** : Optimisation convexe bien définie

**Exemple de robustesse en haute dimension** :
```python
from sklearn.datasets import make_classification

# 200 features, 100 exemples (d >> n)
X, y = make_classification(n_samples=100, n_features=200, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# SVM fonctionne bien
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
print(f"SVM accuracy (200 features): {svm.score(X_test, y_test):.2%}")

# Random Forest moins bon
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
print(f"Random Forest accuracy: {rf.score(X_test, y_test):.2%}")
```

## Limites

❌ **Lent sur grandes données** : O(n²) à O(n³) en complexité  
❌ **Choix du kernel difficile** : Nécessite expérimentation  
❌ **Tuning hyperparamètres** : C, gamma, kernel à optimiser  
❌ **Pas de probabilités** : predict_proba peu fiable (Platt scaling requis)  
❌ **Sensible à l'échelle** : Nécessite normalisation des features  
❌ **Pas d'interprétabilité** : Difficile d'expliquer (sauf kernel linéaire)  
❌ **Multi-classe** : Extension OvO ou OvR (pas natif)

**Temps d'entraînement** :
```python
import time
from sklearn.datasets import make_classification

# Comparer scaling avec taille des données
for n in [100, 1000, 5000]:
    X, y = make_classification(n_samples=n, n_features=20, random_state=42)
    
    start = time.time()
    svm = SVC(kernel='rbf')
    svm.fit(X, y)
    elapsed = time.time() - start
    
    print(f"n={n:5d}: {elapsed:.2f}s")

# Output typique:
# n=  100: 0.01s
# n= 1000: 0.15s
# n= 5000: 3.50s  (croissance rapide!)
```

**Normalisation obligatoire** :
```python
from sklearn.preprocessing import StandardScaler

# Sans normalisation
X, y = make_classification(n_samples=200, n_features=2, random_state=42)
X[:, 0] *= 1000  # Feature 1 entre 0-1000
X[:, 1] *= 0.01  # Feature 2 entre 0-0.01

svm_no_scale = SVC(kernel='rbf')
svm_no_scale.fit(X, y)
print(f"Sans normalisation: {svm_no_scale.score(X, y):.2%}")

# Avec normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
svm_scaled = SVC(kernel='rbf')
svm_scaled.fit(X_scaled, y)
print(f"Avec normalisation: {svm_scaled.score(X_scaled, y):.2%}")
```

## Variantes / liens

### Types de SVM

**1. SVC (Support Vector Classification)** :
```python
from sklearn.svm import SVC

# Classification binaire/multi-classe
svc = SVC(
    C=1.0,              # Régularisation (plus grand = moins de régularisation)
    kernel='rbf',       # 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
    degree=3,           # Degré du polynomial (si kernel='poly')
    gamma='scale',      # Coefficient du kernel ('scale', 'auto', float)
    class_weight=None,  # 'balanced' pour classes déséquilibrées
    probability=False,  # Activer predict_proba (mais plus lent)
    random_state=42
)
```

**2. SVR (Support Vector Regression)** :
```python
from sklearn.svm import SVR

# Régression
svr = SVR(
    kernel='rbf',
    C=1.0,
    epsilon=0.1,    # Tube epsilon (tolérance d'erreur)
    gamma='scale'
)
```

**3. LinearSVC** (plus rapide pour kernel linéaire) :
```python
from sklearn.svm import LinearSVC

# Optimisé pour kernel linéaire (beaucoup plus rapide)
linear_svc = LinearSVC(
    C=1.0,
    max_iter=1000,
    dual=True  # dual=False si n_samples > n_features
)
```

**4. NuSVC** (alternative à C) :
```python
from sklearn.svm import NuSVC

# Utilise nu au lieu de C (interprétation différente)
nu_svc = NuSVC(
    nu=0.5,  # Borne supérieure sur fraction d'erreurs (0 < nu ≤ 1)
    kernel='rbf'
)
```

### Kernels disponibles

**Formules des kernels** :

1. **Linéaire** : `K(x, x') = x · x'`
   ```python
   SVC(kernel='linear')
   ```

2. **Polynomial** : `K(x, x') = (gamma·x·x' + coef0)^degree`
   ```python
   SVC(kernel='poly', degree=3, gamma='scale', coef0=0)
   ```

3. **RBF (Gaussian)** : `K(x, x') = exp(-gamma·||x - x'||²)`
   ```python
   SVC(kernel='rbf', gamma='scale')
   ```

4. **Sigmoïd** : `K(x, x') = tanh(gamma·x·x' + coef0)`
   ```python
   SVC(kernel='sigmoid', gamma='scale', coef0=0)
   ```

5. **Custom kernel** :
   ```python
   def my_kernel(X, Y):
       # Implémenter votre propre kernel
       return np.dot(X, Y.T)
   
   SVC(kernel=my_kernel)
   ```

### Hyperparamètres clés

**C (régularisation)** :
- `C` grand (ex: 100) → Marge étroite, peu d'erreurs → risque overfitting
- `C` petit (ex: 0.1) → Marge large, tolère erreurs → risque underfitting
- Par défaut: `C=1.0`

**gamma (kernel RBF)** :
- `gamma` grand → Influence locale étroite → risque overfitting
- `gamma` petit → Influence large → frontière lisse
- `'scale'` : `1 / (n_features * X.var())` (recommandé)
- `'auto'` : `1 / n_features`

**degree (kernel polynomial)** :
- `degree=2` : Frontière quadratique
- `degree=3` : Frontière cubique (défaut)
- Plus le degré augmente, plus la frontière est complexe

### Relations avec d'autres modèles

- **Logistic Regression** : Similaire au SVM linéaire mais avec loss différent
- **Perceptron** : Ancêtre de SVM (pas de marge maximale)
- **Neural Networks** : Kernel RBF ≈ couche cachée RBF
- **Kernel Methods** : Kernel PCA, Kernel Ridge Regression utilisent même principe
- **AdaBoost** : Autre approche pour frontières complexes

### Preprocessing recommandé

**Pipeline complet** :
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Pipeline: normalisation + SVM
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Indispensable pour SVM
    ('svm', SVC(kernel='rbf', C=1, gamma='scale'))
])

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

### Probabilités avec Platt Scaling

```python
# Activer predict_proba (mais plus lent)
svm_proba = SVC(kernel='rbf', probability=True)
svm_proba.fit(X_train, y_train)

# Probabilités calibrées
probas = svm_proba.predict_proba(X_test)
print(probas[:5])  # Probabilités pour chaque classe
```

## Références

### Documentation et tutoriels
- **Scikit-learn** : [SVM Guide](https://scikit-learn.org/stable/modules/svm.html)
- **StatQuest** : [SVM Explained](https://www.youtube.com/watch?v=efR1C6CvhmE) (YouTube)
- **Andrew Ng** : [ML Course - SVM](https://www.coursera.org/learn/machine-learning) (Coursera)

### Livres
- **"An Introduction to Statistical Learning"** (James et al., 2021) - Chapitre 9
- **"The Elements of Statistical Learning"** (Hastie et al., 2009) - Chapitre 12
- **"Pattern Recognition and Machine Learning"** (Bishop, 2006) - Chapitre 7

### Papers fondamentaux
- **SVM original** : Cortes & Vapnik, 1995 - "Support-Vector Networks"
- **Kernel Trick** : Boser, Guyon & Vapnik, 1992 - "Training Algorithm for Optimal Margin Classifiers"
- **SMO Algorithm** : Platt, 1998 - "Sequential Minimal Optimization"

### Théorie

**Kernel Trick** :
```
Au lieu de calculer φ(x) (projection haute dimension),
on calcule K(x, x') = φ(x) · φ(x') directement

Exemple RBF:
- Projection explicite: dimension infinie !
- Kernel: simple fonction exp(-||x-x'||²)
```

**Dualité** :
```
Problème primal: minimiser ||w||² / 2
Problème dual: maximiser Σαᵢ - ½ΣΣαᵢαⱼyᵢyⱼK(xᵢ,xⱼ)

→ Résolution du dual (plus efficace avec kernels)
```

### Comparaison de performance

**Benchmark (MNIST digits, 10 classes)** :
```
Algorithme          Accuracy    Temps
Logistic Regression   92%       5s
Decision Tree         87%       3s
Random Forest         96%       45s
SVM (linear)          94%       120s
SVM (RBF)             98%       350s
Neural Network        99%       180s

→ SVM RBF: Excellente accuracy mais lent
```

### Tuning rapide (règles empiriques)

**Recommandations** :
1. **Toujours normaliser** : `StandardScaler`
2. **Commencer avec RBF** : kernel='rbf', C=1, gamma='scale'
3. **Si trop lent** : Utiliser `LinearSVC` ou sous-échantillonner
4. **Grid Search** : Tester C=[0.1, 1, 10, 100], gamma=[0.001, 0.01, 0.1, 1]
5. **Classes déséquilibrées** : `class_weight='balanced'`

**Exemple tuning rapide** :
```python
# Étape 1: Tester kernel linéaire (rapide)
svm_lin = SVC(kernel='linear', C=1)
svm_lin.fit(X_train, y_train)
score_lin = svm_lin.score(X_test, y_test)

# Étape 2: Si linéaire insuffisant, tester RBF
if score_lin < 0.85:
    svm_rbf = SVC(kernel='rbf', C=1, gamma='scale')
    svm_rbf.fit(X_train, y_train)
    score_rbf = svm_rbf.score(X_test, y_test)
    
    # Étape 3: Si RBF meilleur, tuner C et gamma
    if score_rbf > score_lin:
        # Grid search sur C et gamma
        pass
```
