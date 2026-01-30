# Gaussian Mixture Models (GMM)

Clustering probabiliste via mélanges gaussiens (EM).

## Idée clé

**GMM (Gaussian Mixture Model)** est un modèle de **clustering probabiliste** qui suppose que les données proviennent d'un **mélange de plusieurs distributions gaussiennes**. Contrairement à K-Means qui assigne chaque point à un seul cluster (hard assignment), GMM calcule des **probabilités d'appartenance** pour chaque cluster (soft assignment).

**Principe** :
1. Les données sont générées par K composantes gaussiennes
2. Chaque composante a sa propre **moyenne** (μₖ), **covariance** (Σₖ), et **poids** (πₖ)
3. Pour chaque point, calculer la probabilité d'appartenir à chaque cluster
4. Utiliser l'algorithme **EM (Expectation-Maximization)** pour optimiser les paramètres

**Formule de densité** :
```
p(x) = Σ πₖ · N(x | μₖ, Σₖ)
      k=1..K

Où:
- πₖ : poids de la composante k (Σπₖ = 1)
- N(x | μₖ, Σₖ) : distribution gaussienne
- K : nombre de composantes
```

**Probabilité d'appartenance (soft assignment)** :
```
p(k | x) = πₖ · N(x | μₖ, Σₖ) / p(x)

→ Probabilité que x appartienne au cluster k
```

**Algorithme EM** :
```
Initialisation: μₖ, Σₖ, πₖ aléatoires

Répéter jusqu'à convergence:
  E-step: Calculer p(k|x) pour chaque point x
  M-step: Mettre à jour μₖ, Σₖ, πₖ avec maximum de vraisemblance
```

**Différence avec K-Means** :
| Aspect | K-Means | GMM |
|--------|---------|-----|
| **Assignment** | Hard (1 cluster) | Soft (probabilités) |
| **Forme clusters** | Sphériques | Ellipsoïdales (covariances) |
| **Output** | Labels | Probabilités |
| **Algorithme** | Lloyd | EM |
| **Robustesse** | Sensible aux outliers | Plus robuste |
| **Vitesse** | Rapide | Plus lent |

## Exemples concrets

### 1. Clustering probabiliste : Données 2D

**Scénario** : Identifier des clusters ellipsoïdaux dans des données 2D.

**Code Python avec GMM** :
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

# 1. Générer données avec 3 clusters
X, y_true = make_blobs(
    n_samples=300, 
    centers=3,
    cluster_std=[1.0, 1.5, 0.5],  # Variances différentes
    random_state=42
)

# 2. Entraîner GMM
gmm = GaussianMixture(
    n_components=3,        # Nombre de composantes gaussiennes
    covariance_type='full', # Type de matrice de covariance
    max_iter=100,
    random_state=42
)
gmm.fit(X)

# 3. Prédire les clusters
y_pred = gmm.predict(X)

# 4. Probabilités d'appartenance (soft assignment)
probas = gmm.predict_proba(X)
print("Probabilités pour le premier point:")
print(f"  Cluster 0: {probas[0, 0]:.2%}")
print(f"  Cluster 1: {probas[0, 1]:.2%}")
print(f"  Cluster 2: {probas[0, 2]:.2%}")

# 5. Paramètres des composantes
print("\nParamètres des composantes gaussiennes:")
for i in range(3):
    print(f"\nComposante {i}:")
    print(f"  Moyenne: {gmm.means_[i]}")
    print(f"  Poids: {gmm.weights_[i]:.3f}")
    print(f"  Covariance shape: {gmm.covariances_[i].shape}")

# 6. Score (log-vraisemblance moyenne)
score = gmm.score(X)
print(f"\nLog-likelihood moyenne: {score:.3f}")

# 7. Visualisation
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Données originales
ax = axes[0]
ax.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', s=50, alpha=0.6)
ax.set_title('Données originales')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')

# Prédictions GMM avec ellipses
ax = axes[1]
ax.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', s=50, alpha=0.6)

# Tracer les ellipses (2 std)
from matplotlib.patches import Ellipse

def draw_ellipse(position, covariance, ax, **kwargs):
    # Décomposition en valeurs propres
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2 * 2 * np.sqrt(eigenvalues)  # 2 std
    
    ellipse = Ellipse(position, width, height, angle=angle, **kwargs)
    ax.add_patch(ellipse)

for i in range(3):
    draw_ellipse(
        gmm.means_[i], 
        gmm.covariances_[i], 
        ax, 
        alpha=0.2, 
        edgecolor='red', 
        linewidth=2,
        facecolor='none'
    )
    # Marquer les centres
    ax.plot(gmm.means_[i, 0], gmm.means_[i, 1], 'rx', markersize=15, markeredgewidth=3)

ax.set_title('GMM Clustering (avec ellipses de covariance)')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')

plt.tight_layout()
plt.show()
```

---

### 2. Comparaison K-Means vs GMM

**Code pour montrer la différence** :
```python
from sklearn.cluster import KMeans

# Données avec clusters ellipsoïdaux (non-sphériques)
np.random.seed(42)
X_ellipse = np.dot(
    np.random.randn(200, 2),
    [[2, 0], [0, 0.5]]  # Matrice d'étirement
) + [5, 5]

X_circle = np.random.randn(200, 2) + [0, 0]
X = np.vstack([X_ellipse, X_circle])

# K-Means (assume clusters sphériques)
kmeans = KMeans(n_clusters=2, random_state=42)
y_kmeans = kmeans.fit_predict(X)

# GMM (peut modéliser ellipses)
gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
y_gmm = gmm.fit_predict(X)

# Visualisation
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# K-Means
ax = axes[0]
ax.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis', s=30, alpha=0.6)
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
           c='red', s=200, marker='X', edgecolors='black', linewidth=2, label='Centres')
ax.set_title('K-Means (clusters sphériques)')
ax.legend()

# GMM
ax = axes[1]
ax.scatter(X[:, 0], X[:, 1], c=y_gmm, cmap='viridis', s=30, alpha=0.6)
for i in range(2):
    draw_ellipse(gmm.means_[i], gmm.covariances_[i], ax, 
                 alpha=0.2, edgecolor='red', linewidth=2, facecolor='none')
    ax.plot(gmm.means_[i, 0], gmm.means_[i, 1], 'rx', markersize=15, markeredgewidth=3)
ax.set_title('GMM (clusters ellipsoïdaux)')

plt.tight_layout()
plt.show()
```

---

### 3. Density Estimation : Générer de nouvelles données

**Scénario** : Utiliser GMM pour modéliser la densité et générer des échantillons synthétiques.

**Code Python** :
```python
# 1. Entraîner GMM sur données existantes
X, _ = make_blobs(n_samples=300, centers=3, random_state=42)

gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.fit(X)

# 2. Générer de nouveaux échantillons
n_samples = 500
X_new, y_new = gmm.sample(n_samples)

# 3. Calculer densité de probabilité
x_min, x_max = X[:, 0].min() - 2, X[:, 0].max() + 2
y_min, y_max = X[:, 1].min() - 2, X[:, 1].max() + 2
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 100),
    np.linspace(y_min, y_max, 100)
)
Z = np.exp(gmm.score_samples(np.c_[xx.ravel(), yy.ravel()]))
Z = Z.reshape(xx.shape)

# 4. Visualisation
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Données originales + densité
ax = axes[0]
ax.contourf(xx, yy, Z, levels=20, cmap='Blues', alpha=0.6)
ax.scatter(X[:, 0], X[:, 1], c='red', s=30, alpha=0.6, label='Données originales')
ax.set_title('Données originales + Densité estimée')
ax.legend()

# Échantillons générés
ax = axes[1]
ax.contourf(xx, yy, Z, levels=20, cmap='Blues', alpha=0.6)
ax.scatter(X_new[:, 0], X_new[:, 1], c=y_new, cmap='viridis', s=30, alpha=0.6, label='Échantillons générés')
ax.set_title('Nouveaux échantillons générés par GMM')
ax.legend()

plt.tight_layout()
plt.show()

print(f"Généré {n_samples} nouveaux échantillons")
```

---

### 4. Sélection du nombre de composantes (BIC/AIC)

**Code pour choisir K optimal** :
```python
from sklearn.mixture import GaussianMixture

# Données
X, _ = make_blobs(n_samples=300, centers=4, random_state=42)

# Tester différents nombres de composantes
n_components_range = range(1, 10)
bic_scores = []
aic_scores = []

for n_components in n_components_range:
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    gmm.fit(X)
    bic_scores.append(gmm.bic(X))
    aic_scores.append(gmm.aic(X))

# Visualisation
plt.figure(figsize=(10, 5))
plt.plot(n_components_range, bic_scores, 'o-', label='BIC')
plt.plot(n_components_range, aic_scores, 's-', label='AIC')
plt.xlabel('Nombre de composantes')
plt.ylabel('Score (plus bas = meilleur)')
plt.title('Sélection du nombre de composantes GMM')
plt.legend()
plt.grid(True)
plt.show()

optimal_bic = n_components_range[np.argmin(bic_scores)]
optimal_aic = n_components_range[np.argmin(aic_scores)]
print(f"Optimal (BIC): {optimal_bic} composantes")
print(f"Optimal (AIC): {optimal_aic} composantes")
```

---

### 5. Types de covariance

**Code pour comparer les types de matrices de covariance** :
```python
# Données
X, _ = make_blobs(n_samples=300, centers=3, random_state=42)

# 4 types de covariance
covariance_types = ['full', 'tied', 'diag', 'spherical']
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for idx, cov_type in enumerate(covariance_types):
    gmm = GaussianMixture(n_components=3, covariance_type=cov_type, random_state=42)
    y_pred = gmm.fit_predict(X)
    
    ax = axes[idx]
    ax.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', s=30, alpha=0.6)
    
    # Tracer ellipses si 'full' ou 'tied'
    if cov_type in ['full', 'tied']:
        for i in range(3):
            if cov_type == 'full':
                cov = gmm.covariances_[i]
            else:  # tied
                cov = gmm.covariances_
            draw_ellipse(gmm.means_[i], cov, ax, 
                        alpha=0.2, edgecolor='red', linewidth=2, facecolor='none')
    
    ax.plot(gmm.means_[:, 0], gmm.means_[:, 1], 'rx', markersize=15, markeredgewidth=3)
    ax.set_title(f'{cov_type.capitalize()}\nBIC: {gmm.bic(X):.1f}')

plt.tight_layout()
plt.show()
```

**Explication des types** :
- **full** : Chaque composante a sa propre matrice de covariance complète (K matrices de d×d)
- **tied** : Toutes les composantes partagent la même matrice de covariance (1 matrice de d×d)
- **diag** : Matrices diagonales (variances seulement, pas de corrélations) (K matrices diagonales)
- **spherical** : Une seule variance par composante (K scalaires)

---

### 6. Anomaly Detection avec GMM

**Code pour détecter des outliers** :
```python
# 1. Données normales
X_normal, _ = make_blobs(n_samples=300, centers=2, random_state=42)

# 2. Ajouter des outliers
np.random.seed(42)
X_outliers = np.random.uniform(low=-10, high=10, size=(20, 2))
X = np.vstack([X_normal, X_outliers])

# 3. Entraîner GMM
gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
gmm.fit(X_normal)  # Entraîner seulement sur données normales

# 4. Calculer densités (log-likelihood)
densities = gmm.score_samples(X)

# 5. Seuil pour anomalies (percentile)
threshold = np.percentile(densities, 5)  # 5% les plus faibles
is_anomaly = densities < threshold

# 6. Visualisation
plt.figure(figsize=(10, 6))
plt.scatter(X[~is_anomaly, 0], X[~is_anomaly, 1], 
            c='blue', s=30, alpha=0.6, label='Normal')
plt.scatter(X[is_anomaly, 0], X[is_anomaly, 1], 
            c='red', s=100, marker='X', label='Anomalies', edgecolors='black', linewidth=2)

# Contour de densité
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = np.exp(gmm.score_samples(np.c_[xx.ravel(), yy.ravel()])).reshape(xx.shape)
plt.contour(xx, yy, Z, levels=10, colors='green', alpha=0.3)

plt.title('Anomaly Detection avec GMM')
plt.legend()
plt.show()

print(f"Anomalies détectées: {is_anomaly.sum()}/{len(X)}")
```

## Quand l'utiliser

- ✅ **Clustering soft** : Besoin de probabilités d'appartenance (pas juste labels)
- ✅ **Clusters non-sphériques** : Ellipses, formes allongées (covariances)
- ✅ **Density estimation** : Modéliser la distribution des données
- ✅ **Génération de données** : Créer des échantillons synthétiques
- ✅ **Anomaly detection** : Identifier outliers (faible densité)
- ✅ **Données avec incertitude** : Soft assignments utiles

**Cas d'usage typiques** :
- 🎨 **Segmentation d'images** : Couleurs, régions
- 🔊 **Traitement audio** : Reconnaissance de phonèmes
- 📊 **Analyse de données** : Identifier sous-populations
- 🏥 **Médecine** : Groupes de patients, sous-types de maladies
- 💰 **Finance** : Segmentation de clients, régimes de marché

**Quand NE PAS utiliser** :
- ❌ Besoin de hard assignments uniquement → K-Means plus rapide
- ❌ Très grandes données (>100k) → K-Means ou Mini-Batch K-Means
- ❌ Clusters non-gaussiens (formes complexes) → DBSCAN, Hierarchical
- ❌ Haute dimensionnalité (>50 features) → Coûteux en mémoire

## Forces

✅ **Soft clustering** : Probabilités d'appartenance (incertitude)  
✅ **Flexibilité formes** : Ellipses via matrices de covariance  
✅ **Modèle génératif** : Peut générer de nouveaux échantillons  
✅ **Density estimation** : Modélise p(x) complètement  
✅ **Base théorique** : Maximum de vraisemblance bien défini  
✅ **Anomaly detection** : Naturel via densité faible

**Exemple de soft clustering** :
```python
# Point à la frontière entre 2 clusters
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(X)

# Point ambiguë
x_ambiguous = np.array([[0, 0]])
probas = gmm.predict_proba(x_ambiguous)[0]

print(f"Probabilités:")
print(f"  Cluster 0: {probas[0]:.1%}")  # 45%
print(f"  Cluster 1: {probas[1]:.1%}")  # 55%
# → GMM capture l'incertitude !

# K-Means donne un choix binaire
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
label = kmeans.predict(x_ambiguous)[0]
print(f"\nK-Means: Cluster {label}")  # 0 ou 1 catégorique
```

## Limites

❌ **Sensible à l'initialisation** : Peut converger vers optimum local  
❌ **Choix de K difficile** : Nombre de composantes à déterminer (BIC/AIC)  
❌ **Complexité O(K·d²·n)** : Lent pour grandes données ou haute dimension  
❌ **Hypothèse gaussienne** : Clusters non-gaussiens mal modélisés  
❌ **Singularités** : Covariances peuvent devenir singulières  
❌ **Mémoire** : Stockage de K matrices de covariance (d×d)

**Problème d'initialisation** :
```python
# Même données, initialisations différentes → résultats différents
scores = []
for i in range(10):
    gmm = GaussianMixture(n_components=3, random_state=i)
    gmm.fit(X)
    scores.append(gmm.score(X))

print(f"Log-likelihood min: {min(scores):.2f}")
print(f"Log-likelihood max: {max(scores):.2f}")
print(f"Différence: {max(scores) - min(scores):.2f}")

# Solution: Plusieurs initialisations (n_init)
gmm = GaussianMixture(n_components=3, n_init=10)  # Essayer 10 fois
```

**Singularités de covariance** :
```python
# Si tous les points d'un cluster sont identiques
# → Covariance = 0 → Singularité !

# Solution: Régularisation
gmm = GaussianMixture(
    n_components=3, 
    reg_covar=1e-6  # Ajouter une petite constante à la diagonale
)
```

## Variantes / liens

### Hyperparamètres clés

```python
GaussianMixture(
    n_components=3,           # Nombre de composantes gaussiennes
    covariance_type='full',   # 'full', 'tied', 'diag', 'spherical'
    tol=1e-3,                 # Seuil de convergence EM
    max_iter=100,             # Nombre max d'itérations EM
    n_init=1,                 # Nombre d'initialisations différentes
    init_params='kmeans',     # 'kmeans', 'random', 'k-means++'
    reg_covar=1e-6,           # Régularisation pour éviter singularités
    random_state=42
)
```

**Recommandations** :
- **n_components** : Utiliser BIC/AIC pour sélectionner
- **covariance_type** : 
  - `'full'` : Maximum de flexibilité (défaut)
  - `'diag'` : Si features indépendantes, plus rapide
  - `'spherical'` : Si clusters sphériques, très rapide
- **n_init** : Au moins 10 pour robustesse
- **reg_covar** : Augmenter si erreurs de singularité

### Algorithme EM détaillé

**E-step (Expectation)** :
```python
# Pour chaque point x_i et composante k:
# Calculer responsabilité γ(z_k) = p(k | x_i)

γ[i, k] = π[k] * N(x[i] | μ[k], Σ[k]) / Σ_j π[j] * N(x[i] | μ[j], Σ[j])
```

**M-step (Maximization)** :
```python
# Mettre à jour paramètres avec maximum de vraisemblance

N_k = Σ_i γ[i, k]  # Nombre effectif de points dans cluster k

μ[k] = (1/N_k) * Σ_i γ[i, k] * x[i]
Σ[k] = (1/N_k) * Σ_i γ[i, k] * (x[i] - μ[k])(x[i] - μ[k])^T
π[k] = N_k / n
```

### Critères de sélection

**BIC (Bayesian Information Criterion)** :
```
BIC = -2 * log-likelihood + p * log(n)

p = nombre de paramètres libres
n = nombre d'exemples

→ Plus petit = meilleur (pénalise complexité)
```

**AIC (Akaike Information Criterion)** :
```
AIC = -2 * log-likelihood + 2 * p

→ Pénalise moins la complexité que BIC
```

### Relations avec d'autres modèles

- **K-Means** : Cas particulier de GMM (covariances sphériques, identiques)
- **EM Algorithm** : Algorithme général (GMM est une application)
- **Naive Bayes** : Utilise aussi gaussiennes (mais pour classification)
- **Hidden Markov Models** : GMM pour modéliser émissions
- **Factor Analysis** : Réduction de dimension probabiliste
- **Variational Autoencoders** : Extension deep learning de GMM

### Variantes avancées

**1. Bayesian GMM** :
```python
from sklearn.mixture import BayesianGaussianMixture

# Approche bayésienne avec prior Dirichlet
bgmm = BayesianGaussianMixture(
    n_components=10,         # Max composantes
    weight_concentration_prior=1e-3,  # Prior sur poids (favorise peu de composantes)
    covariance_type='full'
)
bgmm.fit(X)

# Détermine automatiquement le nombre effectif de composantes
effective_components = (bgmm.weights_ > 0.01).sum()
print(f"Composantes effectives: {effective_components}")
```

**2. GMM avec features manquantes** :
```python
# Imputation via EM
from sklearn.impute import IterativeImputer

imputer = IterativeImputer(estimator=GaussianMixture(n_components=2))
X_imputed = imputer.fit_transform(X_with_missing)
```

## Références

### Documentation et tutoriels
- **Scikit-learn** : [Gaussian Mixture](https://scikit-learn.org/stable/modules/mixture.html)
- **StatQuest** : [EM Algorithm](https://www.youtube.com/watch?v=REypj2sy_5U) (YouTube)

### Livres
- **"Pattern Recognition and Machine Learning"** (Bishop, 2006) - Chapitre 9
- **"The Elements of Statistical Learning"** (Hastie et al., 2009) - Chapitre 14
- **"Machine Learning: A Probabilistic Perspective"** (Murphy, 2012) - Chapitre 11

### Papers fondamentaux
- **EM Algorithm** : Dempster, Laird & Rubin, 1977 - "Maximum Likelihood from Incomplete Data via the EM Algorithm"
- **GMM** : McLachlan & Peel, 2000 - "Finite Mixture Models"
- **Bayesian GMM** : Rasmussen, 2000 - "The Infinite Gaussian Mixture Model"

### Théorie

**Algorithme EM** :
```
Objectif: Maximiser log p(X | θ)

E-step:  Q(θ | θ^old) = E[log p(X, Z | θ) | X, θ^old]
M-step:  θ^new = argmax Q(θ | θ^old)

Garantie: log p(X | θ^new) ≥ log p(X | θ^old)

→ Converge vers optimum local
```

**Comparaison de performance** :
```
Dataset: 10,000 points, 5 clusters

Algorithme     Temps    Mémoire    Soft?
K-Means        0.02s    1 MB       Non
GMM (full)     0.5s     5 MB       Oui
GMM (diag)     0.2s     2 MB       Oui
DBSCAN         1.5s     3 MB       Non

→ GMM: Bon compromis si soft clustering nécessaire
```

### Tuning rapide (règles empiriques)

**Workflow recommandé** :
```python
# 1. Tester K-Means d'abord (baseline)
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

# 2. Si besoins soft clustering ou clusters ellipsoïdaux
# Sélectionner K avec BIC
from sklearn.mixture import GaussianMixture

bic_scores = []
for k in range(1, 10):
    gmm = GaussianMixture(n_components=k, n_init=10)
    gmm.fit(X)
    bic_scores.append(gmm.bic(X))

optimal_k = np.argmin(bic_scores) + 1

# 3. Entraîner modèle final
gmm_final = GaussianMixture(
    n_components=optimal_k,
    covariance_type='full',
    n_init=10,
    random_state=42
)
gmm_final.fit(X)
```

**Choix du type de covariance** :
```python
# Si features indépendantes (pas de corrélations)
→ covariance_type='diag' (plus rapide)

# Si clusters de même forme
→ covariance_type='tied' (partage une covariance)

# Si clusters sphériques
→ covariance_type='spherical' (comme K-Means)

# Sinon (flexibilité maximale)
→ covariance_type='full' (défaut)
```
