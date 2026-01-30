# PCA

Réduction de dimension linéaire par variance maximale.

## Idée clé

**PCA (Principal Component Analysis)** est une technique de **réduction de dimension linéaire** qui projette les données sur un sous-espace de plus faible dimension en **maximisant la variance**. Elle transforme les features corrélées en un ensemble de **composantes principales** orthogonales (non-corrélées).

**Principe** :
1. Centrer les données (moyenne = 0)
2. Calculer la matrice de covariance
3. Trouver les vecteurs propres (eigenvectors) et valeurs propres (eigenvalues)
4. Trier les composantes par variance (eigenvalue décroissant)
5. Projeter les données sur les k premiers vecteurs propres

**Mathématiques** :
```
Input: X (n × d)  → n exemples, d features

1. Centrer: X_centered = X - mean(X)

2. Covariance: C = (1/n) X_centered^T X_centered  (d × d)

3. Eigen-décomposition: C v = λ v
   - v : eigenvector (composante principale)
   - λ : eigenvalue (variance expliquée)

4. Projection: Z = X_centered · V_k  (n × k)
   - V_k : matrice des k premiers eigenvectors
   - Z : données projetées

5. Reconstruction: X_approx = Z · V_k^T + mean(X)
```

**Variance expliquée** :
```
Variance totale = Σ λᵢ
                  i=1..d

Variance expliquée par k composantes = Σ λᵢ / Σ λⱼ
                                        i=1..k  j=1..d

Objectif: Garder k tel que variance ≥ 95%
```

**Visualisation** :
```
3D → 2D par PCA:

     z                    PC2
     │  • •            ↑
     │ •  •           │  • •
     │•  •    →       │ •  •
     └───── y         └────→ PC1
    ╱  •  •              • •
   x

Original: 3 features    Réduit: 2 composantes
Corrélées               Orthogonales
```

**Différence avec autres méthodes** :
| Aspect | PCA | t-SNE | UMAP | Autoencoders |
|--------|-----|-------|------|--------------|
| **Type** | Linéaire | Non-linéaire | Non-linéaire | Non-linéaire |
| **Vitesse** | Très rapide | Lent | Rapide | Moyen |
| **Préserve** | Variance globale | Structure locale | Local + global | Appris |
| **Déterministe** | Oui | Non | Non | Non |
| **Nouveau point** | Facile (projection) | Difficile | Moyen | Facile |

## Exemples concrets

### 1. PCA de base : Réduction de dimension 3D → 2D

**Scénario** : Réduire des données 3D en 2D pour visualisation.

**Code Python avec PCA** :
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs

# 1. Générer données 3D
X, y = make_blobs(n_samples=300, n_features=3, centers=3, random_state=42)

# 2. PCA: 3D → 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 3. Variance expliquée
explained_variance = pca.explained_variance_ratio_
print(f"Variance expliquée par PC1: {explained_variance[0]:.2%}")
print(f"Variance expliquée par PC2: {explained_variance[1]:.2%}")
print(f"Variance totale expliquée: {explained_variance.sum():.2%}")

# 4. Composantes principales (eigenvectors)
components = pca.components_
print(f"\nComposante principale 1: {components[0]}")
print(f"Composante principale 2: {components[1]}")

# 5. Visualisation
fig = plt.figure(figsize=(15, 5))

# Données originales 3D
ax = fig.add_subplot(131, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='viridis', s=50)
ax.set_title('Données originales (3D)')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')

# Projection 2D
ax = fig.add_subplot(132)
ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', s=50)
ax.set_title(f'PCA 2D ({explained_variance.sum():.1%} variance)')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.grid(True, alpha=0.3)

# Scree plot (variance par composante)
ax = fig.add_subplot(133)
ax.bar(range(1, 3), explained_variance, color='steelblue')
ax.set_xlabel('Composante principale')
ax.set_ylabel('Variance expliquée')
ax.set_title('Scree Plot')
ax.set_xticks([1, 2])

plt.tight_layout()
plt.show()
```

---

### 2. Choisir le nombre de composantes : Scree plot

**Code pour déterminer k optimal** :
```python
from sklearn.datasets import load_digits

# Dataset MNIST digits (64 features)
digits = load_digits()
X = digits.data
y = digits.target

print(f"Données originales: {X.shape}")

# PCA avec toutes les composantes
pca_full = PCA()
pca_full.fit(X)

# Variance expliquée cumulée
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

# Trouver nombre de composantes pour 95% variance
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
n_components_99 = np.argmax(cumulative_variance >= 0.99) + 1

print(f"Composantes pour 95% variance: {n_components_95}")
print(f"Composantes pour 99% variance: {n_components_99}")

# Visualisation
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scree plot
ax = axes[0]
ax.plot(range(1, len(pca_full.explained_variance_ratio_) + 1), 
        pca_full.explained_variance_ratio_, 'bo-', linewidth=2)
ax.set_xlabel('Composante principale')
ax.set_ylabel('Variance expliquée')
ax.set_title('Scree Plot')
ax.grid(True, alpha=0.3)
ax.axvline(x=n_components_95, color='r', linestyle='--', label=f'95% variance (k={n_components_95})')
ax.legend()

# Variance cumulée
ax = axes[1]
ax.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'go-', linewidth=2)
ax.axhline(y=0.95, color='r', linestyle='--', label='95% seuil')
ax.axhline(y=0.99, color='orange', linestyle='--', label='99% seuil')
ax.set_xlabel('Nombre de composantes')
ax.set_ylabel('Variance cumulée')
ax.set_title('Variance Expliquée Cumulée')
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()

# Réduction avec k optimal
pca_reduced = PCA(n_components=n_components_95)
X_reduced = pca_reduced.fit_transform(X)
print(f"\nDonnées réduites: {X_reduced.shape}")
print(f"Ratio de compression: {X.shape[1] / X_reduced.shape[1]:.1f}x")
```

---

### 3. Visualisation des composantes principales

**Code pour comprendre ce que PCA capture** :
```python
from sklearn.datasets import load_digits

# Charger digits
digits = load_digits()
X = digits.data  # 1797 images 8x8 = 64 pixels
y = digits.target

# PCA
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X)

# Visualiser les 10 premières composantes principales
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.ravel()

for i in range(10):
    # Reshape composante en image 8x8
    component = pca.components_[i].reshape(8, 8)
    
    ax = axes[i]
    ax.imshow(component, cmap='RdBu', interpolation='nearest')
    ax.set_title(f'PC{i+1} ({pca.explained_variance_ratio_[i]:.1%})')
    ax.axis('off')

plt.suptitle('Composantes Principales (Eigenvectors)', fontsize=14)
plt.tight_layout()
plt.show()

# Visualiser projection 2D
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='tab10', s=30, alpha=0.7)
plt.colorbar(scatter, label='Digit')
plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%})')
plt.title('Digits projetés sur PC1-PC2')
plt.grid(True, alpha=0.3)
plt.show()
```

---

### 4. Reconstruction : Compression avec perte

**Code pour voir la perte d'information** :
```python
from sklearn.datasets import load_digits

# Charger un digit
digits = load_digits()
original = digits.data[0].reshape(8, 8)

# Tester différents nombres de composantes
n_components_list = [1, 2, 5, 10, 20, 40, 64]
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.ravel()

# Original
axes[0].imshow(original, cmap='gray')
axes[0].set_title('Original (64 features)')
axes[0].axis('off')

for idx, n_components in enumerate(n_components_list, start=1):
    # PCA
    pca = PCA(n_components=n_components)
    pca.fit(digits.data)
    
    # Projection
    compressed = pca.transform(digits.data[0:1])
    
    # Reconstruction
    reconstructed = pca.inverse_transform(compressed)
    reconstructed_img = reconstructed[0].reshape(8, 8)
    
    # Erreur de reconstruction
    mse = np.mean((original - reconstructed_img) ** 2)
    variance = pca.explained_variance_ratio_.sum()
    
    ax = axes[idx]
    ax.imshow(reconstructed_img, cmap='gray')
    ax.set_title(f'k={n_components} ({variance:.1%} var)\nMSE={mse:.3f}')
    ax.axis('off')

plt.tight_layout()
plt.show()
```

---

### 5. PCA pour preprocessing : Avant classification

**Code montrant amélioration avec PCA** :
```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import time

# Données
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.3, random_state=42
)

# Sans PCA
start = time.time()
model_no_pca = LogisticRegression(max_iter=1000)
model_no_pca.fit(X_train, y_train)
time_no_pca = time.time() - start
score_no_pca = model_no_pca.score(X_test, y_test)

# Avec PCA
start = time.time()
pipeline_pca = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=20)),  # 64 → 20
    ('classifier', LogisticRegression(max_iter=1000))
])
pipeline_pca.fit(X_train, y_train)
time_pca = time.time() - start
score_pca = pipeline_pca.score(X_test, y_test)

# Comparaison
print("Sans PCA:")
print(f"  Accuracy: {score_no_pca:.2%}")
print(f"  Temps: {time_no_pca:.3f}s")
print(f"  Features: 64")

print("\nAvec PCA (k=20):")
print(f"  Accuracy: {score_pca:.2%}")
print(f"  Temps: {time_pca:.3f}s")
print(f"  Features: 20")
print(f"  Speedup: {time_no_pca / time_pca:.1f}x")

# Variance expliquée
pca = pipeline_pca.named_steps['pca']
print(f"\nVariance conservée: {pca.explained_variance_ratio_.sum():.1%}")
```

---

### 6. Biplot : Features + Observations

**Code pour visualiser features et données simultanément** :
```python
from sklearn.datasets import load_iris

# Charger Iris
iris = load_iris()
X = iris.data
y = iris.target
features = iris.feature_names

# PCA 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Biplot
fig, ax = plt.subplots(figsize=(10, 8))

# Scatter des observations
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', s=50, alpha=0.6)

# Arrows des features (loadings)
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

for i, feature in enumerate(features):
    ax.arrow(0, 0, loadings[i, 0], loadings[i, 1],
             head_width=0.1, head_length=0.1, fc='red', ec='red', linewidth=2)
    ax.text(loadings[i, 0] * 1.15, loadings[i, 1] * 1.15, feature,
            fontsize=12, ha='center', va='center', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax.set_title('Biplot PCA - Iris Dataset')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=0, color='k', linewidth=0.5)

plt.colorbar(scatter, label='Species')
plt.tight_layout()
plt.show()
```

## Quand l'utiliser

- ✅ **Haute dimensionnalité** : Réduire features corrélées (d > 50)
- ✅ **Visualisation** : Projeter en 2D/3D pour explorer
- ✅ **Preprocessing** : Avant classification/régression (accélère)
- ✅ **Compression** : Stockage efficient avec reconstruction
- ✅ **Multicollinéarité** : Éliminer corrélations entre features
- ✅ **Bruit** : Filtrer composantes de faible variance

**Cas d'usage typiques** :
- 🖼️ **Vision** : Réduction de dimensions d'images (eigenfaces)
- 🧬 **Génomique** : Analyse de données d'expression de gènes
- 📊 **Finance** : Réduction de facteurs de risque
- 🔊 **Audio** : Compression de signaux
- 📝 **NLP** : Réduction après TF-IDF ou embeddings

**Quand NE PAS utiliser** :
- ❌ Besoins non-linéaires → t-SNE, UMAP, Autoencoders
- ❌ Interprétabilité critique → Features originaux
- ❌ Features déjà orthogonales → Pas de gain
- ❌ Très peu de features (d < 10) → Pas nécessaire
- ❌ Information dans variance faible → Kernel PCA, LDA

## Forces

✅ **Très rapide** : O(min(n·d², d³)) complexity  
✅ **Déterministe** : Toujours même résultat  
✅ **Théorie solide** : Mathématiquement bien défini  
✅ **Interprétable** : Composantes = combinaisons linéaires  
✅ **Nouveau point** : Projection facile (`transform`)  
✅ **Reversible** : Reconstruction via `inverse_transform`

**Exemple de vitesse** :
```python
import time

# 10,000 points, 1000 features
X_large = np.random.randn(10000, 1000)

# PCA très rapide
start = time.time()
pca = PCA(n_components=50)
X_reduced = pca.fit_transform(X_large)
print(f"PCA: {time.time() - start:.2f}s")

# t-SNE beaucoup plus lent
from sklearn.manifold import TSNE
start = time.time()
# tsne = TSNE(n_components=2)
# X_tsne = tsne.fit_transform(X_large)  # Prendrait plusieurs minutes!
print(f"t-SNE: trop lent pour 10k points (> 5 min)")
```

## Limites

❌ **Linéaire uniquement** : Assume relations linéaires  
❌ **Variance ≠ information** : Info peut être dans faible variance  
❌ **Sensible à l'échelle** : Nécessite normalisation  
❌ **Perd interprétabilité** : Composantes = mélanges de features  
❌ **Outliers** : Biaisent les composantes  
❌ **Pas de labels** : N'utilise pas y (unsupervised)

**Problème de linéarité** :
```python
# Données circulaires (non-linéaires)
theta = np.linspace(0, 2*np.pi, 200)
X_circle = np.column_stack([np.cos(theta), np.sin(theta)])

# PCA échoue à capturer structure
pca = PCA(n_components=1)
X_pca = pca.fit_transform(X_circle)

# Kernel PCA réussit
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components=1, kernel='rbf', gamma=10)
X_kpca = kpca.fit_transform(X_circle)

# Visualisation
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].scatter(X_circle[:, 0], X_circle[:, 1], c=theta, cmap='viridis')
axes[0].set_title('Original (cercle)')

axes[1].scatter(X_pca[:, 0], np.zeros_like(X_pca), c=theta, cmap='viridis')
axes[1].set_title('PCA linéaire (perd structure)')

axes[2].scatter(X_kpca[:, 0], np.zeros_like(X_kpca), c=theta, cmap='viridis')
axes[2].set_title('Kernel PCA (préserve ordre)')

plt.show()
```

**Normalisation obligatoire** :
```python
# Données avec échelles différentes
X = np.random.randn(100, 2)
X[:, 0] *= 1000  # Feature 1: 0-1000
X[:, 1] *= 1     # Feature 2: 0-1

# Sans normalisation: PC1 dominé par feature 1
pca_no_scale = PCA(n_components=2)
pca_no_scale.fit(X)
print(f"Sans normalisation, PC1: {pca_no_scale.components_[0]}")

# Avec normalisation
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca_scaled = PCA(n_components=2)
pca_scaled.fit(X_scaled)
print(f"Avec normalisation, PC1: {pca_scaled.components_[0]}")
```

## Variantes / liens

### Hyperparamètres clés

```python
PCA(
    n_components=None,      # Nombre de composantes (int, float 0-1, 'mle', None)
    whiten=False,           # Normaliser composantes à variance 1
    svd_solver='auto',      # 'auto', 'full', 'arpack', 'randomized'
    random_state=None       # Pour svd_solver='randomized'
)
```

**Recommandations** :
- **n_components** :
  - `int` : Nombre exact (ex: 50)
  - `float` : Variance à conserver (ex: 0.95 pour 95%)
  - `'mle'` : Estimation automatique (Minka's MLE)
  - `None` : Toutes les composantes
- **whiten** : `True` si features doivent avoir même variance
- **svd_solver** : 
  - `'auto'` : Choisit automatiquement (défaut)
  - `'randomized'` : Plus rapide pour grandes matrices

### Méthodes importantes

```python
# Fit + transform
X_reduced = pca.fit_transform(X)

# Transform nouveau point
X_new_reduced = pca.transform(X_new)

# Reconstruction (inverse)
X_reconstructed = pca.inverse_transform(X_reduced)

# Variance expliquée
variance_ratio = pca.explained_variance_ratio_  # Par composante
total_variance = variance_ratio.sum()

# Composantes principales (eigenvectors)
components = pca.components_  # Shape: (n_components, n_features)

# Eigenvalues
eigenvalues = pca.explained_variance_
```

### Variantes de PCA

**1. Kernel PCA** (non-linéaire) :
```python
from sklearn.decomposition import KernelPCA

# Pour données non-linéairement séparables
kpca = KernelPCA(
    n_components=2,
    kernel='rbf',     # 'linear', 'poly', 'rbf', 'sigmoid'
    gamma=10
)
X_kpca = kpca.fit_transform(X)
```

**2. Incremental PCA** (pour grandes données) :
```python
from sklearn.decomposition import IncrementalPCA

# Traitement par batches (économe en mémoire)
ipca = IncrementalPCA(n_components=50, batch_size=1000)
for batch in batches:
    ipca.partial_fit(batch)

X_reduced = ipca.transform(X)
```

**3. Sparse PCA** (composantes sparses) :
```python
from sklearn.decomposition import SparsePCA

# Composantes avec peu de coefficients non-nuls (interprétabilité)
spca = SparsePCA(n_components=10, alpha=0.1)
X_spca = spca.fit_transform(X)
```

**4. Randomized PCA** :
```python
# Déjà intégré dans PCA(svd_solver='randomized')
# Plus rapide pour n_components << min(n_samples, n_features)
pca = PCA(n_components=50, svd_solver='randomized')
```

### Relations avec d'autres modèles

- **SVD (Singular Value Decomposition)** : PCA utilise SVD en interne
- **LDA (Linear Discriminant Analysis)** : Supervised (utilise labels y)
- **t-SNE** : Non-linéaire, pour visualisation uniquement
- **UMAP** : Non-linéaire, plus rapide que t-SNE
- **Autoencoders** : Non-linéaire deep learning
- **Factor Analysis** : Modèle génératif probabiliste

### Preprocessing recommandé

**Pipeline standard** :
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Pipeline: normalisation + PCA
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # OBLIGATOIRE pour PCA!
    ('pca', PCA(n_components=0.95))  # Garder 95% variance
])

X_reduced = pipeline.fit_transform(X)
```

**Avec classification** :
```python
from sklearn.linear_model import LogisticRegression

# Pipeline complet
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=50)),
    ('classifier', LogisticRegression())
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
```

## Références

### Documentation et tutoriels
- **Scikit-learn** : [PCA](https://scikit-learn.org/stable/modules/decomposition.html#pca)
- **StatQuest** : [PCA Explained](https://www.youtube.com/watch?v=FgakZw6K1QQ) (YouTube)
- **3Blue1Brown** : [Eigenvectors and Eigenvalues](https://www.youtube.com/watch?v=PFDu9oVAE-g)

### Livres
- **"Pattern Recognition and Machine Learning"** (Bishop, 2006) - Chapitre 12
- **"The Elements of Statistical Learning"** (Hastie et al., 2009) - Chapitre 14.5
- **"Hands-On Machine Learning"** (Géron, 2019) - Chapitre 8

### Papers fondamentaux
- **PCA** : Pearson, 1901 - "On Lines and Planes of Closest Fit to Systems of Points in Space"
- **Kernel PCA** : Schölkopf et al., 1998 - "Nonlinear Component Analysis as a Kernel Eigenvalue Problem"
- **Incremental PCA** : Ross et al., 2008 - "Incremental Learning for Robust Visual Tracking"

### Théorie

**Complexité** :
```
Temps: O(min(n·d², d³))
  - SVD complète: O(n·d²) si n > d
  - Randomized SVD: O(n·d·k) plus rapide

Espace: O(n·d + k·d)
  - k = nombre de composantes
```

**Mathématiques (SVD)** :
```
X_centered = U Σ V^T

Où:
- U (n × k): scores (données projetées / √n)
- Σ (k × k): valeurs singulières (√eigenvalues)
- V (d × k): loadings (composantes principales)

X_reduced = U Σ = X_centered V
```

**Benchmark de performance** :
```
Dataset: 10,000 points, 1000 features, k=50

Méthode             Temps    Mémoire
PCA (full SVD)      2.5s     80 MB
PCA (randomized)    0.8s     80 MB
Incremental PCA     1.2s     20 MB
Kernel PCA (RBF)    45s      800 MB
t-SNE               >300s    200 MB

→ PCA randomized: Optimal vitesse/précision
```

### Tuning rapide (règles empiriques)

**Workflow recommandé** :
```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Normaliser (OBLIGATOIRE)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. PCA avec toutes composantes pour voir variance
pca_full = PCA()
pca_full.fit(X_scaled)

# 3. Plot variance cumulée
cumsum = np.cumsum(pca_full.explained_variance_ratio_)
plt.plot(cumsum)
plt.axhline(y=0.95, color='r', linestyle='--')
plt.xlabel('Composantes')
plt.ylabel('Variance cumulée')
plt.show()

# 4. Choisir k pour 95% variance
k = np.argmax(cumsum >= 0.95) + 1
print(f"k optimal (95%): {k}")

# 5. PCA final
pca = PCA(n_components=k)
X_reduced = pca.fit_transform(X_scaled)
```

**Choix de k** :
```python
# Méthode 1: Variance seuil (ex: 95%)
pca = PCA(n_components=0.95)

# Méthode 2: Nombre fixe (ex: visualisation 2D)
pca = PCA(n_components=2)

# Méthode 3: Scree plot (chercher "coude")
# Où la variance commence à décroître lentement

# Méthode 4: MLE automatique
pca = PCA(n_components='mle')
```

**Optimisation mémoire** :
```python
# Si données trop grandes pour mémoire → Incremental PCA
from sklearn.decomposition import IncrementalPCA

ipca = IncrementalPCA(n_components=50, batch_size=1000)

# Traiter par batches
for i in range(0, len(X), 1000):
    batch = X[i:i+1000]
    ipca.partial_fit(batch)

X_reduced = ipca.transform(X)
```
