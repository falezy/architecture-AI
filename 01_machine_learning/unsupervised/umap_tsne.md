# UMAP / t-SNE

Réduction de dimension non-linéaire (visualisation).

## Idée clé

**t-SNE** (t-distributed Stochastic Neighbor Embedding) et **UMAP** (Uniform Manifold Approximation and Projection) sont des techniques de **réduction de dimension non-linéaire** optimisées pour **visualisation**. Contrairement à PCA (linéaire), elles préservent les structures **locales** (voisinages) des données haute dimension.

### t-SNE (2008)

**Principe** :
1. Construire une distribution de probabilités pour les paires de points en haute dimension (similarités)
2. Construire une distribution similaire en basse dimension (2D/3D)
3. Minimiser la divergence KL entre les deux distributions
4. Utilise une distribution Student-t en basse dimension (d'où le "t")

**Formule** :
```
Haute dimension (probabilité que i et j sont voisins):
pⱼ|ᵢ = exp(-||xᵢ - xⱼ||² / 2σᵢ²) / Σₖ exp(-||xᵢ - xₖ||² / 2σᵢ²)

Basse dimension (distribution Student-t):
qᵢⱼ = (1 + ||yᵢ - yⱼ||²)⁻¹ / Σₖₗ (1 + ||yₖ - yₗ||²)⁻¹

Objectif: Minimiser KL(P||Q) = Σᵢⱼ pᵢⱼ log(pᵢⱼ/qᵢⱼ)
```

### UMAP (2018)

**Principe** :
1. Construire un graphe de voisinage en haute dimension
2. Optimiser un layout en basse dimension qui préserve la topologie
3. Utilise la théorie des variétés (manifold) et topologie algébrique
4. Plus rapide et préserve mieux la structure globale que t-SNE

**Différences clés** :
```
                t-SNE               UMAP
Vitesse         Lent O(n²)          Rapide O(n log n)
Structure       Locale ++           Locale + Globale +
Distance        Pas de sens         ≈ préservée
Nouveau point   Impossible          Possible (transform)
Déterministe    Non                 Non (mais + stable)
```

**Visualisation** :
```
Haute dimension (Ex: 64D)    →    t-SNE/UMAP    →    2D
   • • • • •                                        Cluster 1: • • •
  • Features •                 Préserve                        • •
   corrélées                   voisinages          Cluster 2:  • • •
  • • • • •                                                    • •
```

**Tableau comparatif** :
| Aspect | PCA | t-SNE | UMAP |
|--------|-----|-------|------|
| **Type** | Linéaire | Non-linéaire | Non-linéaire |
| **Vitesse** | Très rapide | Lent | Rapide |
| **Préserve** | Variance globale | Structure locale | Local + global |
| **Déterministe** | Oui | Non | Quasi |
| **Transform nouveau** | ✅ | ❌ | ✅ |
| **Distances** | ✅ | ❌ | ≈ |
| **Clustering** | Après | Après | Après |

## Exemples concrets

### 1. t-SNE de base : MNIST digits

**Scénario** : Visualiser 784D (28×28 pixels) en 2D.

**Code Python avec t-SNE** :
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
import time

# 1. Charger données
digits = load_digits()
X = digits.data  # 1797 images, 64 features
y = digits.target

print(f"Données originales: {X.shape}")

# 2. t-SNE
start = time.time()
tsne = TSNE(
    n_components=2,
    perplexity=30,      # Balance local/global (5-50 typique)
    learning_rate=200,  # 'auto' ou 10-1000
    n_iter=1000,        # Nombre d'itérations
    random_state=42
)
X_tsne = tsne.fit_transform(X)
elapsed = time.time() - start

print(f"t-SNE temps: {elapsed:.2f}s")
print(f"KL divergence finale: {tsne.kl_divergence_:.3f}")

# 3. Visualisation
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    X_tsne[:, 0], 
    X_tsne[:, 1], 
    c=y, 
    cmap='tab10',
    s=30,
    alpha=0.7,
    edgecolors='black',
    linewidth=0.5
)
plt.colorbar(scatter, label='Digit')
plt.title(f't-SNE Visualization of Digits (perplexity={tsne.perplexity})')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

### 2. Comparaison t-SNE vs UMAP

**Code pour comparer les deux méthodes** :
```python
from sklearn.manifold import TSNE
import umap
import time

# Données
digits = load_digits()
X = digits.data
y = digits.target

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# t-SNE
start = time.time()
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X)
time_tsne = time.time() - start

ax = axes[0]
scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', s=30, alpha=0.7)
ax.set_title(f't-SNE ({time_tsne:.1f}s)')
ax.set_xlabel('Component 1')
ax.set_ylabel('Component 2')
plt.colorbar(scatter, ax=ax, label='Digit')

# UMAP
start = time.time()
reducer = umap.UMAP(
    n_components=2,
    n_neighbors=15,     # Taille du voisinage local
    min_dist=0.1,       # Distance min en 2D
    random_state=42
)
X_umap = reducer.fit_transform(X)
time_umap = time.time() - start

ax = axes[1]
scatter = ax.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='tab10', s=30, alpha=0.7)
ax.set_title(f'UMAP ({time_umap:.1f}s - {time_tsne/time_umap:.1f}x plus rapide)')
ax.set_xlabel('Component 1')
ax.set_ylabel('Component 2')
plt.colorbar(scatter, ax=ax, label='Digit')

plt.tight_layout()
plt.show()

print(f"\nt-SNE: {time_tsne:.2f}s")
print(f"UMAP:  {time_umap:.2f}s (speedup: {time_tsne/time_umap:.1f}x)")
```

---

### 3. Impact de perplexity (t-SNE)

**Code pour montrer l'effet crucial de perplexity** :
```python
# Données
digits = load_digits()
X = digits.data[:500]  # Sous-ensemble pour rapidité
y = digits.target[:500]

# Tester différentes perplexités
perplexities = [5, 15, 30, 50]
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

for idx, perp in enumerate(perplexities):
    tsne = TSNE(
        n_components=2,
        perplexity=perp,
        random_state=42,
        n_iter=1000
    )
    X_tsne = tsne.fit_transform(X)
    
    ax = axes[idx]
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', s=30, alpha=0.7)
    ax.set_title(f'Perplexity = {perp}\nKL divergence = {tsne.kl_divergence_:.3f}')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')

plt.tight_layout()
plt.show()

print("\nPerplexity guide:")
print("  5-15:  Focus très local, petits clusters")
print("  30:    Défaut, bon équilibre")
print("  50+:   Plus global, structures larges")
```

---

### 4. UMAP : n_neighbors et min_dist

**Code pour comprendre les hyperparamètres UMAP** :
```python
import umap

# Données
digits = load_digits()
X = digits.data[:500]
y = digits.target[:500]

# Impact de n_neighbors
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

n_neighbors_list = [5, 15, 50, 100]

for idx, n_neighbors in enumerate(n_neighbors_list):
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        random_state=42
    )
    X_umap = reducer.fit_transform(X)
    
    ax = axes[idx]
    scatter = ax.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='tab10', s=30, alpha=0.7)
    ax.set_title(f'n_neighbors = {n_neighbors}')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')

plt.suptitle('Impact de n_neighbors (taille du voisinage)', fontsize=14)
plt.tight_layout()
plt.show()

# Impact de min_dist
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

min_dist_list = [0.0, 0.1, 0.5, 0.99]

for idx, min_dist in enumerate(min_dist_list):
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=min_dist,
        random_state=42
    )
    X_umap = reducer.fit_transform(X)
    
    ax = axes[idx]
    scatter = ax.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='tab10', s=30, alpha=0.7)
    ax.set_title(f'min_dist = {min_dist}')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')

plt.suptitle('Impact de min_dist (compacité des clusters)', fontsize=14)
plt.tight_layout()
plt.show()

print("\nHyperparamètres UMAP:")
print("  n_neighbors petit (5): Très local, petits clusters")
print("  n_neighbors grand (50): Plus global, grosses structures")
print("  min_dist petit (0.0): Clusters compacts")
print("  min_dist grand (0.99): Clusters dispersés")
```

---

### 5. UMAP : Transform sur nouveaux points

**Code montrant l'avantage de UMAP (t-SNE ne peut pas)** :
```python
import umap
from sklearn.model_selection import train_test_split

# Données
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.3, random_state=42
)

# UMAP: Fit sur train
reducer = umap.UMAP(n_components=2, random_state=42)
X_train_umap = reducer.fit_transform(X_train)

# Transform sur test (IMPOSSIBLE avec t-SNE!)
X_test_umap = reducer.transform(X_test)

# Visualisation
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Train
ax = axes[0]
scatter = ax.scatter(X_train_umap[:, 0], X_train_umap[:, 1], 
                    c=y_train, cmap='tab10', s=30, alpha=0.7, label='Train')
ax.set_title('UMAP - Training set')
ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
plt.colorbar(scatter, ax=ax, label='Digit')

# Test projeté
ax = axes[1]
scatter = ax.scatter(X_test_umap[:, 0], X_test_umap[:, 1], 
                    c=y_test, cmap='tab10', s=30, alpha=0.7, label='Test')
ax.set_title('UMAP - Test set (transformed)')
ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
plt.colorbar(scatter, ax=ax, label='Digit')

plt.tight_layout()
plt.show()

print("✅ UMAP peut transformer de nouveaux points")
print("❌ t-SNE nécessite re-fit complet sur toutes les données")
```

---

### 6. Pipeline complet : PCA → t-SNE/UMAP

**Code pour réduire d'abord avec PCA (accélération)** :
```python
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import time

# Grandes données
digits = load_digits()
X = digits.data
y = digits.target

# Sans PCA préalable
start = time.time()
tsne_direct = TSNE(n_components=2, random_state=42)
X_tsne_direct = tsne_direct.fit_transform(X)
time_direct = time.time() - start

# Avec PCA d'abord (64 → 20 → 2)
start = time.time()
pca = PCA(n_components=20)
X_pca = pca.fit_transform(X)
tsne_pca = TSNE(n_components=2, random_state=42)
X_tsne_pca = tsne_pca.fit_transform(X_pca)
time_pca = time.time() - start

# Visualisation
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]
scatter = ax.scatter(X_tsne_direct[:, 0], X_tsne_direct[:, 1], 
                    c=y, cmap='tab10', s=30, alpha=0.7)
ax.set_title(f't-SNE Direct ({time_direct:.1f}s)')
plt.colorbar(scatter, ax=ax)

ax = axes[1]
scatter = ax.scatter(X_tsne_pca[:, 0], X_tsne_pca[:, 1], 
                    c=y, cmap='tab10', s=30, alpha=0.7)
ax.set_title(f'PCA(20) → t-SNE ({time_pca:.1f}s, {time_direct/time_pca:.1f}x plus rapide)')
plt.colorbar(scatter, ax=ax)

plt.tight_layout()
plt.show()

print(f"\nDirect:     {time_direct:.2f}s")
print(f"PCA→t-SNE:  {time_pca:.2f}s (speedup: {time_direct/time_pca:.1f}x)")
print(f"\n✅ Recommandation: Toujours faire PCA(50) avant t-SNE si d > 50")
```

## Quand l'utiliser

- ✅ **Visualisation exploratoire** : Explorer structures dans données haute dimension
- ✅ **Découverte de clusters** : Voir groupements naturels
- ✅ **Après PCA** : Si PCA 2D insuffisant (relations non-linéaires)
- ✅ **Interprétation** : Comprendre ce que le modèle a appris
- ✅ **Publication** : Figures pour papers (clusters, diversité)
- ✅ **Validation** : Vérifier que données font sens

**Cas d'usage typiques** :
- 🧬 **Génomique** : Cellules single-cell RNA-seq (t-SNE/UMAP standard)
- 📝 **NLP** : Visualiser word embeddings, document clusters
- 🖼️ **Vision** : Features CNN, image similarity
- 🎵 **Audio** : Spectrogrammes, embeddings musicaux
- 📊 **ML général** : Comprendre features apprises

**Quand NE PAS utiliser** :
- ❌ Preprocessing pour ML → PCA (t-SNE/UMAP perdent distances)
- ❌ Besoin distances exactes → PCA, MDS
- ❌ Clustering direct → K-Means, DBSCAN sur features originaux
- ❌ Interprétation quantitative → Distances 2D pas fiables
- ❌ Production/temps réel → Trop lent (sauf UMAP pré-fit)

## Forces

### t-SNE

✅ **Préserve structure locale** : Clusters très nets  
✅ **Visualisation excellente** : Standard pour exploration  
✅ **Robuste** : Fonctionne sur beaucoup de types de données  
✅ **Flexible** : Perplexity ajustable

### UMAP

✅ **Très rapide** : 10-100x plus rapide que t-SNE  
✅ **Structure locale ET globale** : Meilleur équilibre  
✅ **Transform nouveau point** : Pas besoin de re-fit  
✅ **Distances ≈ préservées** : Plus fiable que t-SNE  
✅ **Scalable** : Millions de points possibles  
✅ **Théorie solide** : Basé sur topologie

**Exemple de vitesse** :
```python
import time
import umap
from sklearn.manifold import TSNE

# 10,000 points, 50 features
X_large = np.random.randn(10000, 50)

# t-SNE
start = time.time()
tsne = TSNE(n_components=2)
# X_tsne = tsne.fit_transform(X_large)  # Trop lent!
print(f"t-SNE: >5 minutes pour 10k points")

# UMAP
start = time.time()
reducer = umap.UMAP(n_components=2)
X_umap = reducer.fit_transform(X_large)
time_umap = time.time() - start
print(f"UMAP: {time_umap:.2f}s (100x+ plus rapide)")
```

## Limites

### t-SNE

❌ **Très lent** : O(n²) impossible pour >10k points  
❌ **Non-déterministe** : Résultats différents à chaque run  
❌ **Pas de transform** : Nouveaux points nécessitent re-fit complet  
❌ **Distances perdues** : Distances 2D sans signification  
❌ **Perplexity critique** : Choix difficile, impact énorme  
❌ **Structure globale perdue** : Focus sur local uniquement

### UMAP

❌ **Non-déterministe** : Moins que t-SNE mais pas parfait  
❌ **Distances approx** : Pas exactes, juste mieux que t-SNE  
❌ **Hyperparamètres** : n_neighbors, min_dist à tuner  
❌ **Moins étudié** : Plus récent que t-SNE (2018 vs 2008)

**Problème de distances** :
```python
# DANGER: Ne pas utiliser distances 2D pour clustering!
X_tsne = tsne.fit_transform(X)

# ❌ MAUVAIS
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
labels = kmeans.fit_predict(X_tsne)  # Clusters sur 2D déformé!

# ✅ BON
labels = kmeans.fit_predict(X)  # Clusters sur features originaux
# Puis visualiser avec t-SNE
X_vis = tsne.fit_transform(X)
plt.scatter(X_vis[:, 0], X_vis[:, 1], c=labels)
```

**Non-déterminisme** :
```python
# Même données, résultats différents
for run in range(3):
    tsne = TSNE(n_components=2, random_state=run)
    X_tsne = tsne.fit_transform(X)
    # Visualisations complètement différentes!
    
# Solution: Fixer random_state
tsne = TSNE(n_components=2, random_state=42)
```

## Variantes / liens

### Hyperparamètres t-SNE

```python
TSNE(
    n_components=2,         # Toujours 2 ou 3 pour visualisation
    perplexity=30,          # 5-50 (équilibre local/global)
    early_exaggeration=12,  # Facteur initial (espacement clusters)
    learning_rate=200,      # 10-1000 ou 'auto'
    n_iter=1000,            # 250-1000 (convergence)
    random_state=None
)
```

**Perplexity** : Le plus important!
- **5-15** : Très local, petits clusters nets
- **30** : Défaut, bon équilibre (recommandé)
- **50+** : Plus global, structures larges

### Hyperparamètres UMAP

```python
umap.UMAP(
    n_components=2,         # Dimension de sortie
    n_neighbors=15,         # 2-100 (taille voisinage)
    min_dist=0.1,           # 0.0-0.99 (compacité clusters)
    metric='euclidean',     # 'euclidean', 'cosine', 'manhattan'...
    random_state=None
)
```

**n_neighbors** :
- **5-10** : Très local, détails fins
- **15** : Défaut, bon équilibre
- **50+** : Plus global, structure large

**min_dist** :
- **0.0** : Clusters très compacts
- **0.1** : Défaut (recommandé)
- **0.5+** : Clusters plus dispersés

### Variantes

**1. openTSNE** (t-SNE plus rapide) :
```python
from openTSNE import TSNE

# Implémentation optimisée
tsne = TSNE(
    n_components=2,
    perplexity=30,
    n_jobs=-1  # Parallélisation
)
X_tsne = tsne.fit(X)
```

**2. Parametric UMAP** (avec neural network) :
```python
import umap.parametric_umap

# Utilise un encoder neural
embedder = umap.parametric_umap.ParametricUMAP()
embedding = embedder.fit_transform(X)
```

**3. Supervised UMAP** (utilise labels) :
```python
# UMAP peut utiliser labels pour meilleure séparation
reducer = umap.UMAP(n_components=2)
embedding = reducer.fit_transform(X, y=labels)
```

### Relations avec d'autres modèles

- **PCA** : Linéaire, à faire AVANT t-SNE/UMAP si d > 50
- **MDS (Multidimensional Scaling)** : Préserve distances, linéaire
- **Isomap** : Non-linéaire, préserve distances géodésiques
- **LLE (Locally Linear Embedding)** : Non-linéaire, préserve local
- **Autoencoders** : Deep learning, apprentissage de représentation

### Preprocessing recommandé

**Pipeline standard** :
```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# 1. Normaliser
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. PCA si d > 50 (accélération massive)
if X.shape[1] > 50:
    pca = PCA(n_components=50)
    X_scaled = pca.fit_transform(X_scaled)

# 3. t-SNE ou UMAP
reducer = umap.UMAP(n_components=2, random_state=42)
X_2d = reducer.fit_transform(X_scaled)
```

## Références

### Documentation et tutoriels
- **t-SNE** : [Original Paper](https://jmlr.org/papers/v9/vandermaaten08a.html) (van der Maaten & Hinton, 2008)
- **UMAP** : [Documentation](https://umap-learn.readthedocs.io/)
- **Distill.pub** : [How to Use t-SNE Effectively](https://distill.pub/2016/misread-tsne/) (Excellent!)
- **StatQuest** : [t-SNE Explained](https://www.youtube.com/watch?v=NEaUSP4YerM)

### Papers fondamentaux
- **t-SNE** : van der Maaten & Hinton, 2008 - "Visualizing Data using t-SNE"
- **UMAP** : McInnes et al., 2018 - "UMAP: Uniform Manifold Approximation and Projection"
- **Barnes-Hut t-SNE** : van der Maaten, 2014 - Accélération O(n log n)

### Benchmarks

**Dataset: 10,000 points, 50 features**
```
Méthode             Temps    Préserve      Transform?
PCA                 0.5s     Variance      ✅
t-SNE (exact)       >300s    Local++       ❌
t-SNE (Barnes-Hut)  60s      Local++       ❌
UMAP                5s       Local+Global+ ✅
openTSNE            20s      Local++       ❌

→ UMAP: Meilleur compromis vitesse/qualité
```

### Tuning rapide (règles empiriques)

**Workflow t-SNE** :
```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 1. Normaliser
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. PCA si nécessaire
if X.shape[1] > 50:
    pca = PCA(n_components=50)
    X_scaled = pca.fit_transform(X_scaled)

# 3. t-SNE avec perplexity par défaut
tsne = TSNE(
    n_components=2,
    perplexity=30,  # Bon défaut
    learning_rate='auto',
    n_iter=1000,
    random_state=42
)
X_tsne = tsne.fit_transform(X_scaled)

# 4. Si résultats mauvais, essayer perplexity=5, 15, 50
```

**Workflow UMAP** :
```python
import umap

# 1. UMAP avec défauts (souvent suffisant)
reducer = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    random_state=42
)
X_umap = reducer.fit_transform(X)

# 2. Tuner si nécessaire:
# - Clusters trop dispersés → min_dist=0.0
# - Trop local → n_neighbors=50
# - Trop global → n_neighbors=5
```

**Choix t-SNE vs UMAP** :
```python
# Utiliser t-SNE si:
# - Dataset petit (<5k points)
# - Visualisation publication (plus établi)
# - Focus structure locale uniquement

# Utiliser UMAP si:
# - Dataset grand (>5k points) → 10-100x plus rapide
# - Besoin transform nouveaux points
# - Structure globale importante
# - Production/interactif
# - Recommandé par défaut maintenant!
```
