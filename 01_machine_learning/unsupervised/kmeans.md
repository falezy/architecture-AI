# K-means

Clustering par centroïdes, rapide, nécessite K.

## Idée clé

**K-Means** est un algorithme de **clustering par partitionnement** qui divise les données en K groupes (clusters) en assignant chaque point au centroïde le plus proche. C'est l'algorithme de clustering le plus populaire grâce à sa simplicité et sa rapidité.

**Principe** :
1. Initialiser K centroïdes aléatoirement
2. **Assignment** : Assigner chaque point au centroïde le plus proche
3. **Update** : Recalculer les centroïdes comme la moyenne des points assignés
4. Répéter 2-3 jusqu'à convergence (centroïdes ne bougent plus)

**Algorithme de Lloyd** :
```
Input: X (données), K (nombre de clusters)

1. Initialiser K centroïdes μ₁, ..., μₖ aléatoirement

2. Répéter jusqu'à convergence:
   
   a. Assignment step:
      Pour chaque point xᵢ:
        cᵢ = argmin ||xᵢ - μₖ||²
             k
   
   b. Update step:
      Pour chaque cluster k:
        μₖ = moyenne{xᵢ : cᵢ = k}

3. Retourner centroïdes μ et labels c
```

**Fonction objectif (inertie)** :
```
J = Σ Σ ||xᵢ - μₖ||²
    k xᵢ∈Cₖ

Minimiser la somme des distances au carré
→ Lloyd garantit convergence vers optimum local
```

**Visualisation** :
```
Iteration 0:        Iteration 1:        Iteration 5:
   •  •  •             •──•──•             •  •  •
  •  ×  •            •   ×   •           • × × • 
   •  •  •             •──•──•             •  •  •
      ×                   ×                   ×

× = centroïdes      Assignment            Convergé!
• = points          + Update
```

**Différence avec autres méthodes** :
| Aspect | K-Means | Hierarchical | GMM | DBSCAN |
|--------|---------|--------------|-----|--------|
| **Besoin K?** | Oui | Non (dendrogramme) | Oui | Non |
| **Forme clusters** | Sphériques | Flexible | Ellipsoïdales | Arbitraire |
| **Vitesse** | Très rapide | Lent O(n³) | Moyen | Moyen |
| **Soft/Hard** | Hard | Hard | Soft | Hard |
| **Gère bruit** | Non | Non | Peu | Oui |

## Exemples concrets

### 1. K-Means de base : Clustering 2D

**Scénario** : Grouper des données 2D en 3 clusters.

**Code Python avec K-Means** :
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 1. Générer données avec 3 clusters
X, y_true = make_blobs(
    n_samples=300, 
    centers=3,
    cluster_std=0.6,
    random_state=42
)

# 2. K-Means
kmeans = KMeans(
    n_clusters=3,
    init='k-means++',  # Initialisation intelligente (défaut)
    n_init=10,         # Nombre d'initialisations différentes
    max_iter=300,      # Nombre max d'itérations
    random_state=42
)
kmeans.fit(X)

# 3. Prédictions
y_pred = kmeans.predict(X)
centroids = kmeans.cluster_centers_

# 4. Métriques
inertia = kmeans.inertia_  # Somme des distances au carré
print(f"Inertie (within-cluster sum of squares): {inertia:.2f}")
print(f"Nombre d'itérations: {kmeans.n_iter_}")
print(f"Centroïdes:\n{centroids}")

# 5. Visualisation
plt.figure(figsize=(12, 5))

# Données originales
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', s=50, alpha=0.6)
plt.title('Données originales (3 clusters)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# K-Means résultat
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', s=50, alpha=0.6)
plt.scatter(
    centroids[:, 0], 
    centroids[:, 1], 
    c='red', 
    s=300, 
    marker='X',
    edgecolors='black',
    linewidth=2,
    label='Centroïdes'
)
plt.title(f'K-Means (K=3, inertie={inertia:.0f})')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

plt.tight_layout()
plt.show()
```

---

### 2. Méthode du coude (Elbow Method) : Choisir K optimal

**Code pour déterminer le nombre de clusters** :
```python
# Données
X, _ = make_blobs(n_samples=300, centers=4, random_state=42)

# Tester différents K
K_range = range(1, 11)
inertias = []
silhouette_scores = []

from sklearn.metrics import silhouette_score

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)
    
    if k > 1:  # Silhouette nécessite au moins 2 clusters
        score = silhouette_score(X, kmeans.labels_)
        silhouette_scores.append(score)
    else:
        silhouette_scores.append(0)

# Visualisation
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Méthode du coude (inertie)
ax = axes[0]
ax.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
ax.set_xlabel('Nombre de clusters (K)')
ax.set_ylabel('Inertie (within-cluster SS)')
ax.set_title('Méthode du coude')
ax.grid(True, alpha=0.3)
ax.axvline(x=4, color='r', linestyle='--', label='K optimal suggéré')
ax.legend()

# Silhouette score
ax = axes[1]
ax.plot(K_range, silhouette_scores, 'gs-', linewidth=2, markersize=8)
ax.set_xlabel('Nombre de clusters (K)')
ax.set_ylabel('Silhouette Score')
ax.set_title('Silhouette Score (plus haut = meilleur)')
ax.grid(True, alpha=0.3)
ax.axvline(x=4, color='r', linestyle='--', label='K optimal suggéré')
ax.legend()

plt.tight_layout()
plt.show()

print(f"Inertie pour K=4: {inertias[3]:.2f}")
print(f"Silhouette score pour K=4: {silhouette_scores[3]:.3f}")
```

---

### 3. Visualiser les itérations de K-Means

**Code pour montrer la convergence** :
```python
# Générer données simples
np.random.seed(42)
X_simple = np.vstack([
    np.random.randn(30, 2) * 0.5 + [0, 0],
    np.random.randn(30, 2) * 0.5 + [3, 3],
    np.random.randn(30, 2) * 0.5 + [0, 3]
])

# K-Means avec max_iter=1 pour voir chaque étape
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for i, max_iter in enumerate([0, 1, 2, 3, 5, 10]):
    kmeans = KMeans(n_clusters=3, init='random', max_iter=max_iter, n_init=1, random_state=42)
    
    if max_iter == 0:
        # Initialisation seulement
        kmeans = KMeans(n_clusters=3, init='random', max_iter=0, n_init=1, random_state=42)
        kmeans.fit(X_simple)
        centroids = kmeans.cluster_centers_
        labels = np.zeros(len(X_simple))
    else:
        kmeans.fit(X_simple)
        centroids = kmeans.cluster_centers_
        labels = kmeans.labels_
    
    ax = axes[i]
    ax.scatter(X_simple[:, 0], X_simple[:, 1], c=labels, cmap='viridis', s=50, alpha=0.6)
    ax.scatter(centroids[:, 0], centroids[:, 1], c='red', s=300, marker='X', 
               edgecolors='black', linewidth=2)
    ax.set_title(f'Itération {max_iter}' if max_iter > 0 else 'Initialisation')
    ax.set_xlim(-2, 5)
    ax.set_ylim(-2, 5)

plt.tight_layout()
plt.show()
```

---

### 4. K-Means++ vs initialisation aléatoire

**Code pour comparer les initialisations** :
```python
X, _ = make_blobs(n_samples=300, centers=3, random_state=42)

# Initialisation aléatoire (mauvaise)
kmeans_random = KMeans(n_clusters=3, init='random', n_init=1, random_state=0)
kmeans_random.fit(X)
inertia_random = kmeans_random.inertia_

# K-Means++ (intelligente)
kmeans_pp = KMeans(n_clusters=3, init='k-means++', n_init=1, random_state=0)
kmeans_pp.fit(X)
inertia_pp = kmeans_pp.inertia_

# Visualisation
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Random init
ax = axes[0]
ax.scatter(X[:, 0], X[:, 1], c=kmeans_random.labels_, cmap='viridis', s=50, alpha=0.6)
ax.scatter(kmeans_random.cluster_centers_[:, 0], kmeans_random.cluster_centers_[:, 1],
           c='red', s=300, marker='X', edgecolors='black', linewidth=2)
ax.set_title(f'Init Random\nInertie: {inertia_random:.0f}')

# K-Means++ init
ax = axes[1]
ax.scatter(X[:, 0], X[:, 1], c=kmeans_pp.labels_, cmap='viridis', s=50, alpha=0.6)
ax.scatter(kmeans_pp.cluster_centers_[:, 0], kmeans_pp.cluster_centers_[:, 1],
           c='red', s=300, marker='X', edgecolors='black', linewidth=2)
ax.set_title(f'Init K-Means++\nInertie: {inertia_pp:.0f}')

plt.tight_layout()
plt.show()

print(f"Amélioration avec K-Means++: {(inertia_random - inertia_pp) / inertia_random * 100:.1f}%")
```

---

### 5. Mini-Batch K-Means pour grandes données

**Code pour comparer vitesse** :
```python
from sklearn.cluster import MiniBatchKMeans
import time

# Grandes données
X_large = np.random.randn(100000, 50)

# K-Means standard
start = time.time()
kmeans = KMeans(n_clusters=10, n_init=3)
kmeans.fit(X_large)
time_kmeans = time.time() - start

# Mini-Batch K-Means
start = time.time()
mbkmeans = MiniBatchKMeans(n_clusters=10, batch_size=1000, n_init=3)
mbkmeans.fit(X_large)
time_mbkmeans = time.time() - start

print(f"K-Means:            {time_kmeans:.2f}s, inertie: {kmeans.inertia_:.0f}")
print(f"Mini-Batch K-Means: {time_mbkmeans:.2f}s, inertie: {mbkmeans.inertia_:.0f}")
print(f"Speedup: {time_kmeans / time_mbkmeans:.1f}x")
```

---

### 6. Application : Compression d'image

**Code pour réduire le nombre de couleurs** :
```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Créer une image simple (ou charger une vraie image)
image = np.zeros((200, 200, 3))
image[:100, :100] = [1, 0, 0]  # Rouge
image[:100, 100:] = [0, 1, 0]  # Vert
image[100:, :100] = [0, 0, 1]  # Bleu
image[100:, 100:] = [1, 1, 0]  # Jaune
# Ajouter variations
image += np.random.randn(200, 200, 3) * 0.1

# Reshape pour K-Means
pixels = image.reshape(-1, 3)

# Compression avec différents K
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

k_values = [2, 4, 8, 16, 32, 64]

for idx, k in enumerate(k_values):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=3)
    labels = kmeans.fit_predict(pixels)
    compressed_pixels = kmeans.cluster_centers_[labels]
    compressed_image = compressed_pixels.reshape(200, 200, 3)
    
    ax = axes[idx]
    ax.imshow(np.clip(compressed_image, 0, 1))
    ax.set_title(f'K={k} couleurs')
    ax.axis('off')

plt.tight_layout()
plt.show()

# Calcul taux de compression
original_size = 200 * 200 * 3
compressed_size = k_values[-1] * 3 + 200 * 200  # K centroïdes + labels
compression_ratio = original_size / compressed_size
print(f"Taux de compression (K={k_values[-1]}): {compression_ratio:.1f}x")
```

## Quand l'utiliser

- ✅ **Grandes données** : O(n·K·d·i) très rapide (linéaire en n)
- ✅ **Clusters sphériques** : Assume tailles et densités similaires
- ✅ **K connu** : Nombre de clusters estimé à l'avance
- ✅ **Simplicité** : Facile à implémenter et interpréter
- ✅ **Scalabilité** : Mini-Batch K-Means pour millions de points
- ✅ **Première approche** : Baseline rapide pour clustering

**Cas d'usage typiques** :
- 🎨 **Compression d'images** : Réduction de couleurs
- 📊 **Segmentation client** : Groupes de comportements
- 🏷️ **Prétraitement** : Features pour autre modèle
- 🗺️ **Géolocalisation** : Zones géographiques
- 📝 **Text mining** : Clustering de documents (avec TF-IDF)

**Quand NE PAS utiliser** :
- ❌ Clusters non-sphériques (formes allongées) → Hierarchical, DBSCAN
- ❌ Densités variables → DBSCAN, OPTICS
- ❌ Présence de bruit/outliers → DBSCAN robuste
- ❌ K inconnu → Hierarchical (dendrogramme)
- ❌ Besoin soft assignments → GMM (probabilités)

## Forces

✅ **Très rapide** : O(n·K·d·i) linéaire en n  
✅ **Scalable** : Mini-Batch pour millions de points  
✅ **Simple** : Facile à comprendre et implémenter  
✅ **Déterministe** : Avec même initialisation → même résultat  
✅ **Convergence garantie** : Vers optimum local  
✅ **Memory efficient** : Stocke seulement K centroïdes

**Exemple de vitesse** :
```python
import time

# 1 million de points
X_huge = np.random.randn(1000000, 10)

start = time.time()
kmeans = KMeans(n_clusters=10, n_init=1)
kmeans.fit(X_huge)
print(f"K-Means sur 1M points: {time.time() - start:.2f}s")

# vs Hierarchical (impossible)
# AgglomerativeClustering prendrait des heures!
```

## Limites

❌ **Nécessite K** : Nombre de clusters à spécifier à l'avance  
❌ **Optimum local** : Sensible à l'initialisation  
❌ **Assume sphères** : Clusters de tailles/densités similaires  
❌ **Sensible outliers** : Points aberrants affectent centroïdes  
❌ **Hard clustering** : Pas de probabilités d'appartenance  
❌ **Métrique euclidienne** : Distance L2 seulement (standard)

**Problème de forme** :
```python
from sklearn.datasets import make_moons

# Données en forme de lunes (non-sphériques)
X, _ = make_moons(n_samples=200, noise=0.05, random_state=42)

# K-Means échoue
kmeans = KMeans(n_clusters=2)
y_kmeans = kmeans.fit_predict(X)

# DBSCAN réussit
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)
y_dbscan = dbscan.fit_predict(X)

# Visualisation
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')
axes[0].set_title('K-Means (échec sur formes non-sphériques)')

axes[1].scatter(X[:, 0], X[:, 1], c=y_dbscan, cmap='viridis')
axes[1].set_title('DBSCAN (succès)')

plt.show()
```

**Sensibilité aux outliers** :
```python
# Données avec outliers
X_clean = np.random.randn(100, 2)
X_outliers = np.random.randn(5, 2) * 5 + 10  # Outliers loin
X_with_outliers = np.vstack([X_clean, X_outliers])

# K-Means tire les centroïdes vers les outliers
kmeans = KMeans(n_clusters=1)
kmeans.fit(X_with_outliers)
centroid = kmeans.cluster_centers_[0]

print(f"Centroïde (avec outliers): {centroid}")
print(f"Moyenne vraies données: {X_clean.mean(axis=0)}")
# → Centroïde biaisé par outliers
```

## Variantes / liens

### Hyperparamètres clés

```python
KMeans(
    n_clusters=8,           # Nombre de clusters K
    init='k-means++',       # 'k-means++', 'random', ou array de centroïdes
    n_init=10,              # Nombre d'initialisations différentes (garde meilleur)
    max_iter=300,           # Nombre max d'itérations par run
    tol=1e-4,               # Seuil de convergence
    random_state=None,      # Seed pour reproductibilité
    algorithm='lloyd'       # 'lloyd', 'elkan' (plus rapide pour certains cas)
)
```

**Recommandations** :
- **n_clusters** : Utiliser elbow method ou silhouette score
- **init** : Toujours `'k-means++'` (meilleur que random)
- **n_init** : Au moins 10 pour robustesse (essayer 10 fois, garder meilleur)
- **algorithm** : `'elkan'` si K petit et d élevé

### Algorithmes d'initialisation

**1. Random** :
```python
# Choisir K points aléatoirement
centroids = X[np.random.choice(len(X), K, replace=False)]
```
- Simple mais peut donner mauvais résultats

**2. K-Means++ (Arthur & Vassilvitskii, 2007)** :
```python
# 1. Choisir premier centroïde aléatoirement
# 2. Pour chaque nouveau centroïde:
#    - Calculer distance min de chaque point aux centroïdes existants
#    - Choisir nouveau centroïde avec probabilité ∝ distance²
```
- Meilleur: Spread out initial centroids
- Défaut dans scikit-learn

**3. Manual** :
```python
# Spécifier manuellement
initial_centroids = np.array([[0, 0], [5, 5], [10, 0]])
kmeans = KMeans(n_clusters=3, init=initial_centroids, n_init=1)
```

### Métriques de qualité

**1. Inertie (within-cluster sum of squares)** :
```python
inertia = kmeans.inertia_
# Plus bas = meilleur (mais décroît toujours avec K)
```

**2. Silhouette Score** :
```python
from sklearn.metrics import silhouette_score

score = silhouette_score(X, labels)
# Entre -1 et 1, plus haut = meilleur
# > 0.5 : clusters bien séparés
# < 0.2 : clusters se chevauchent
```

**3. Davies-Bouldin Index** :
```python
from sklearn.metrics import davies_bouldin_score

dbi = davies_bouldin_score(X, labels)
# Plus bas = meilleur
```

**4. Calinski-Harabasz Index** :
```python
from sklearn.metrics import calinski_harabasz_score

chi = calinski_harabasz_score(X, labels)
# Plus haut = meilleur
```

### Variantes de K-Means

**1. Mini-Batch K-Means** :
```python
from sklearn.cluster import MiniBatchKMeans

# 10-100x plus rapide sur grandes données
mbkmeans = MiniBatchKMeans(
    n_clusters=10,
    batch_size=1000,  # Nombre de points par batch
    n_init=3
)
mbkmeans.fit(X_large)
```

**2. K-Medoids (PAM)** :
```python
from sklearn_extra.cluster import KMedoids

# Utilise points réels comme centroïdes (plus robuste aux outliers)
kmedoids = KMedoids(n_clusters=3, method='pam')
kmedoids.fit(X)
```

**3. Fuzzy C-Means** :
```python
# Soft clustering (chaque point a probabilité pour chaque cluster)
# Similaire à GMM mais avec contrainte sur somme = 1
```

**4. K-Means avec contraintes** :
```python
# Contraintes: must-link, cannot-link
# Généralement implémentation custom requise
```

### Relations avec d'autres modèles

- **GMM** : K-Means = cas spécial de GMM (covariances sphériques identiques)
- **Vector Quantization** : K-Means utilisé pour compression
- **K-NN** : Utilise distances comme K-Means mais pour classification
- **Hierarchical** : Peut initialiser avec K-Means pour accélérer
- **DBSCAN** : Density-based, ne nécessite pas K

### Preprocessing recommandé

**Normalisation importante** :
```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Pipeline: normalisation + K-Means
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Important pour K-Means!
    ('kmeans', KMeans(n_clusters=3))
])

pipeline.fit(X)
labels = pipeline.predict(X)
```

**Pourquoi normaliser** :
```python
# Sans normalisation: features avec grandes valeurs dominent
X = np.random.randn(100, 2)
X[:, 0] *= 1000  # Feature 1: 0-1000
X[:, 1] *= 1     # Feature 2: 0-1

# K-Means se base principalement sur feature 1
# → Normaliser pour que chaque feature contribue équitablement
```

## Références

### Documentation et tutoriels
- **Scikit-learn** : [K-Means](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- **StatQuest** : [K-Means Clustering](https://www.youtube.com/watch?v=4b5d3muPQmA) (YouTube)
- **Andrew Ng** : [K-Means Algorithm](https://www.coursera.org/learn/machine-learning) (Coursera)

### Livres
- **"Pattern Recognition and Machine Learning"** (Bishop, 2006) - Chapitre 9
- **"The Elements of Statistical Learning"** (Hastie et al., 2009) - Chapitre 14.3
- **"Introduction to Data Mining"** (Tan et al., 2018) - Chapitre 8

### Papers fondamentaux
- **K-Means** : Lloyd, 1982 (publié) / MacQueen, 1967 - "Least Squares Quantization in PCM"
- **K-Means++** : Arthur & Vassilvitskii, 2007 - "k-means++: The Advantages of Careful Seeding"
- **Mini-Batch K-Means** : Sculley, 2010 - "Web-Scale K-Means Clustering"

### Théorie

**Complexité** :
```
Temps: O(n · K · d · i)
  n = nombre de points
  K = nombre de clusters
  d = dimension
  i = nombre d'itérations (souvent < 20)

Espace: O(n · d + K · d)

→ Linéaire en n (très scalable)
```

**Convergence** :
```
Lloyd garantit:
- Inertie diminue à chaque itération
- Convergence vers optimum local (pas global!)
- Généralement < 20 itérations
```

**Benchmark de performance** :
```
Dataset: 100,000 points, 10 features, K=5

Algorithme           Temps    Inertie
K-Means (lloyd)      0.8s     12,450
K-Means (elkan)      0.5s     12,450
Mini-Batch (b=1000)  0.1s     12,680
GMM (full cov)       15s      N/A
Hierarchical         ∞        N/A (trop lent)

→ K-Means: Imbattable en vitesse
```

### Tuning rapide (règles empiriques)

**Workflow recommandé** :
```python
# 1. Normaliser
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Elbow method pour choisir K
inertias = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

plt.plot(range(1, 11), inertias, 'bo-')
plt.xlabel('K')
plt.ylabel('Inertie')
plt.title('Elbow Method')
plt.show()

# 3. Entraîner modèle final
optimal_k = 4  # Choisir visuellement
kmeans_final = KMeans(n_clusters=optimal_k, n_init=10, random_state=42)
labels = kmeans_final.fit_predict(X_scaled)
```

**Choix de K** :
```python
# Méthode 1: Elbow method (chercher "coude")
# Méthode 2: Silhouette score (maximiser)
# Méthode 3: Domain knowledge (nombre attendu)
# Méthode 4: Gap statistic (comparer avec null model)
```

**Optimisation pour grandes données** :
```python
# Si n > 100,000 → Mini-Batch K-Means
from sklearn.cluster import MiniBatchKMeans

mbkmeans = MiniBatchKMeans(
    n_clusters=10,
    batch_size=1000,
    n_init=3,
    random_state=42
)
labels = mbkmeans.fit_predict(X_large)

# Si n > 10 millions → Utiliser Spark MLlib ou Dask-ML
```
