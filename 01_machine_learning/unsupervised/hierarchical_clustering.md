# Hierarchical Clustering

Clustering hiérarchique (dendrogramme).

## Idée clé

**Hierarchical Clustering** est un algorithme de clustering qui construit une **hiérarchie de clusters** représentée par un **dendrogramme** (arbre). Contrairement à K-Means qui nécessite de spécifier K à l'avance, le clustering hiérarchique produit une structure complète permettant de choisir n'importe quel nombre de clusters en coupant l'arbre à différents niveaux.

**Deux approches** :

1. **Agglomerative (bottom-up)** : ⭐ Plus courant
   - Départ : Chaque point est un cluster
   - Répéter : Fusionner les 2 clusters les plus proches
   - Fin : Un seul cluster contenant tous les points

2. **Divisive (top-down)** :
   - Départ : Tous les points dans un cluster
   - Répéter : Diviser le cluster le plus hétérogène
   - Fin : Chaque point est son propre cluster

**Algorithme agglomératif** :
```
1. Initialiser: N clusters (1 point par cluster)
2. Répéter jusqu'à 1 cluster:
   a. Calculer distances entre tous les paires de clusters
   b. Fusionner les 2 clusters les plus proches (selon linkage)
   c. Mettre à jour la matrice de distances
3. Résultat: Dendrogramme
```

**Dendrogramme** :
```
         ┌────────┐
         │        │
      ┌──┴──┐  ┌──┴──┐
      │     │  │     │
    ┌─┴─┐ ┌─┴─┐│   ┌─┴─┐
    •   • •   ••   •   •
   p1  p2 p3 p4p5  p6  p7

Height = distance de fusion
Couper à hauteur h → K clusters
```

**Linkage criteria (critères de fusion)** :
- **Single** : Distance minimale entre points
- **Complete** : Distance maximale entre points
- **Average** : Distance moyenne entre tous les points
- **Ward** : Minimise la variance intra-cluster (meilleur en général)

## Exemples concrets

### 1. Clustering hiérarchique avec dendrogramme

**Scénario** : Grouper des données 2D et visualiser la hiérarchie.

**Code Python avec AgglomerativeClustering** :
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import dendrogram, linkage

# 1. Générer données
X, y_true = make_blobs(n_samples=50, centers=3, random_state=42)

# 2. Calculer matrice de distances et linkage pour dendrogramme
Z = linkage(X, method='ward')  # 'single', 'complete', 'average', 'ward'

# 3. Visualiser dendrogramme
plt.figure(figsize=(12, 5))
dendrogram(
    Z,
    truncate_mode='lastp',  # Montrer seulement les p dernières fusions
    p=12,
    leaf_rotation=90,
    leaf_font_size=10,
    show_contracted=True
)
plt.title('Dendrogramme - Ward Linkage')
plt.xlabel('Index du cluster ou (nombre de points)')
plt.ylabel('Distance')
plt.axhline(y=10, color='r', linestyle='--', label='Seuil de coupure')
plt.legend()
plt.show()

# 4. Clustering avec nombre de clusters fixé
n_clusters = 3
model = AgglomerativeClustering(
    n_clusters=n_clusters,
    linkage='ward'
)
y_pred = model.fit_predict(X)

# 5. Visualisation des clusters
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', s=100, edgecolors='black')
plt.title(f'Hierarchical Clustering (Ward, K={n_clusters})')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Cluster')
plt.show()

print(f"Clusters trouvés: {n_clusters}")
print(f"Labels: {np.unique(y_pred)}")
```

---

### 2. Comparaison des linkage methods

**Code pour voir l'impact du critère de liaison** :
```python
from sklearn.datasets import make_moons

# Données avec forme non-convexe
X, _ = make_moons(n_samples=200, noise=0.05, random_state=42)

# Tester 4 linkage methods
linkages = ['single', 'complete', 'average', 'ward']
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

for idx, linkage_method in enumerate(linkages):
    # Clustering
    model = AgglomerativeClustering(n_clusters=2, linkage=linkage_method)
    y_pred = model.fit_predict(X)
    
    # Visualisation
    ax = axes[idx]
    ax.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', s=50, edgecolors='black')
    ax.set_title(f'Linkage: {linkage_method.capitalize()}')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')

plt.tight_layout()
plt.show()
```

**Résultats typiques** :
- **Single** : Bon pour formes allongées/non-convexes (effet "chaîne")
- **Complete** : Clusters compacts, sphériques
- **Average** : Compromis
- **Ward** : Meilleur en général, minimise variance

---

### 3. Dendrogramme détaillé pour petits datasets

**Code pour comprendre la structure hiérarchique** :
```python
# Petit dataset pour visualisation claire
np.random.seed(42)
X_small = np.random.randn(10, 2) * 0.5
X_small[:3] += [2, 2]   # Cluster 1
X_small[3:6] += [0, 0]  # Cluster 2
X_small[6:] += [4, 0]   # Cluster 3

# Calcul linkage
Z = linkage(X_small, method='ward')

# Dendrogramme complet (tous les points)
plt.figure(figsize=(12, 6))
dendrogram(
    Z,
    labels=[f'P{i}' for i in range(len(X_small))],  # Labels des points
    leaf_rotation=0,
    leaf_font_size=12
)
plt.title('Dendrogramme Complet - 10 Points')
plt.xlabel('Points')
plt.ylabel('Distance de Ward')
plt.grid(True, alpha=0.3)
plt.show()

# Afficher la matrice de linkage
print("Matrice de linkage (Z):")
print("Cluster1  Cluster2  Distance  Nombre de points")
for i, row in enumerate(Z):
    print(f"{int(row[0]):8d}  {int(row[1]):8d}  {row[2]:8.2f}  {int(row[3]):8d}")
```

---

### 4. Choisir le nombre de clusters optimal

**Code avec méthode du coude sur la distance** :
```python
# Données
X, _ = make_blobs(n_samples=300, centers=5, random_state=42)

# Calculer linkage
Z = linkage(X, method='ward')

# Distances de fusion (dernières K fusions)
last_merges = Z[-20:, 2]  # 20 dernières distances

# Méthode du coude
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(last_merges)+1), last_merges[::-1], 'bo-')
plt.xlabel('Nombre de clusters')
plt.ylabel('Distance de fusion')
plt.title('Méthode du coude - Distances de fusion')
plt.grid(True)
plt.show()

# Grand saut = nombre optimal de clusters
acceleration = np.diff(last_merges, 2)
optimal_clusters = len(last_merges) - np.argmax(acceleration[::-1])
print(f"Nombre optimal de clusters suggéré: {optimal_clusters}")
```

---

### 5. Clustering avec contrainte de connectivité

**Code pour forcer des contraintes de voisinage** :
```python
from sklearn.neighbors import kneighbors_graph

# Données
X, _ = make_blobs(n_samples=100, centers=3, random_state=42)

# Sans contrainte
model_no_constraint = AgglomerativeClustering(n_clusters=3, linkage='ward')
y_no_constraint = model_no_constraint.fit_predict(X)

# Avec contrainte de connectivité (seulement voisins peuvent fusionner)
connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)
model_constraint = AgglomerativeClustering(
    n_clusters=3, 
    linkage='ward',
    connectivity=connectivity
)
y_constraint = model_constraint.fit_predict(X)

# Visualisation
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.scatter(X[:, 0], X[:, 1], c=y_no_constraint, cmap='viridis', s=50)
ax.set_title('Sans contrainte')

ax = axes[1]
ax.scatter(X[:, 0], X[:, 1], c=y_constraint, cmap='viridis', s=50)
ax.set_title('Avec contrainte de connectivité')

plt.tight_layout()
plt.show()
```

---

### 6. Application : Clustering d'images

**Code pour regrouper des pixels par couleur** :
```python
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

# Créer une image simple
image = np.zeros((100, 100, 3))
image[:50, :50] = [1, 0, 0]  # Rouge
image[:50, 50:] = [0, 1, 0]  # Vert
image[50:, :50] = [0, 0, 1]  # Bleu
image[50:, 50:] = [1, 1, 0]  # Jaune
# Ajouter du bruit
image += np.random.randn(100, 100, 3) * 0.1

# Reshape en (n_pixels, 3)
pixels = image.reshape(-1, 3)

# Clustering hiérarchique
model = AgglomerativeClustering(n_clusters=4, linkage='ward')
labels = model.fit_predict(pixels)

# Reconstruire image segmentée
segmented = labels.reshape(100, 100)

# Visualisation
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(image)
axes[0].set_title('Image originale')
axes[0].axis('off')

axes[1].imshow(segmented, cmap='viridis')
axes[1].set_title('Segmentation (4 clusters)')
axes[1].axis('off')

plt.tight_layout()
plt.show()
```

## Quand l'utiliser

- ✅ **Nombre de clusters inconnu** : Dendrogramme permet de choisir K après
- ✅ **Hiérarchie utile** : Structure multi-niveaux informative
- ✅ **Petits/moyens datasets** : < 10,000 points (O(n³) en temps)
- ✅ **Formes non-convexes** : Single linkage gère bien
- ✅ **Interprétabilité** : Dendrogramme facile à expliquer
- ✅ **Déterministe** : Pas d'initialisation aléatoire

**Cas d'usage typiques** :
- 🧬 **Biologie** : Phylogénie, taxonomie, clustering de gènes
- 📊 **Analyse de données** : Exploration de structure
- 🗂️ **Organisation** : Hiérarchie de documents, catégories
- 🏥 **Médecine** : Classification de maladies
- 🛒 **Marketing** : Segmentation client avec sous-segments

**Quand NE PAS utiliser** :
- ❌ Très grandes données (>10k) → K-Means ou Mini-Batch
- ❌ Besoin de rapidité → K-Means beaucoup plus rapide
- ❌ Clusters de densité variable → DBSCAN
- ❌ Mémoire limitée → Complexité O(n²) en espace

## Forces

✅ **Pas de K à spécifier** : Dendrogramme montre toutes les options  
✅ **Hiérarchie informative** : Structure multi-niveaux  
✅ **Déterministe** : Toujours même résultat (pas d'aléatoire)  
✅ **Plusieurs linkages** : Ward, single, complete, average  
✅ **Visualisation claire** : Dendrogramme facile à interpréter  
✅ **Formes flexibles** : Single linkage gère formes allongées

**Exemple d'exploration de K** :
```python
# Avec K-Means: besoin de tester plusieurs K
for k in [2, 3, 4, 5]:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    # Comparer...

# Avec Hierarchical: un seul dendrogramme
Z = linkage(X, method='ward')
dendrogram(Z)
# Choisir K visuellement en coupant à différentes hauteurs
```

## Limites

❌ **Complexité O(n³)** : Très lent pour grandes données  
❌ **Complexité O(n²) mémoire** : Matrice de distances complète  
❌ **Pas de réaffectation** : Décisions irréversibles (greedy)  
❌ **Sensible au bruit** : Outliers affectent la structure  
❌ **Choix du linkage** : Résultats très différents selon critère  
❌ **Pas de probabilités** : Hard assignment uniquement

**Temps d'exécution** :
```python
import time

for n in [100, 500, 1000, 2000]:
    X = np.random.randn(n, 10)
    
    start = time.time()
    model = AgglomerativeClustering(n_clusters=5)
    model.fit(X)
    elapsed = time.time() - start
    
    print(f"n={n:5d}: {elapsed:.2f}s")

# Output typique:
# n=  100: 0.05s
# n=  500: 0.5s
# n= 1000: 3.5s
# n= 2000: 25s  (croissance cubique!)
```

**Problème de décisions irréversibles** :
```python
# Si 2 points fusionnent trop tôt (erreur)
# → Impossible de les séparer plus tard
# → Propagation de l'erreur dans tout l'arbre

# K-Means peut réassigner à chaque itération
```

## Variantes / liens

### Hyperparamètres clés

```python
AgglomerativeClustering(
    n_clusters=2,           # Nombre de clusters (ou None)
    linkage='ward',         # 'ward', 'complete', 'average', 'single'
    distance_threshold=None,# Couper à cette distance (si n_clusters=None)
    connectivity=None,      # Matrice de connectivité (contraintes)
    compute_full_tree=False # Si True, construit arbre complet
)
```

**Recommandations** :
- **linkage** : 
  - `'ward'` : Meilleur en général (minimise variance)
  - `'single'` : Pour formes allongées, non-convexes
  - `'complete'` : Pour clusters compacts, sphériques
  - `'average'` : Compromis entre single et complete
- **n_clusters vs distance_threshold** : 
  - Spécifier n_clusters OU distance_threshold (pas les deux)
- **connectivity** : Utiliser si contraintes spatiales

### Linkage criteria détaillés

**1. Single (nearest neighbor)** :
```python
d(A, B) = min{d(a, b) : a ∈ A, b ∈ B}
```
- Force : Détecte formes allongées
- Faible : Sensible au "chaînage" (chaining effect)

**2. Complete (farthest neighbor)** :
```python
d(A, B) = max{d(a, b) : a ∈ A, b ∈ B}
```
- Force : Clusters compacts
- Faible : Sensible aux outliers

**3. Average (UPGMA)** :
```python
d(A, B) = (1/|A||B|) Σ Σ d(a, b)
                     a∈A b∈B
```
- Force : Compromis équilibré
- Faible : Pas d'optimisation claire

**4. Ward** :
```python
d(A, B) = Δ(variance intra-cluster après fusion)
```
- Force : Minimise variance (critère optimal)
- Faible : Assume clusters gaussiens

**Visualisation des différences** :
```python
# Données avec forme allongée
from sklearn.datasets import make_moons
X, _ = make_moons(n_samples=100, noise=0.05)

linkages = ['single', 'complete', 'ward']
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, linkage in zip(axes, linkages):
    model = AgglomerativeClustering(n_clusters=2, linkage=linkage)
    y = model.fit_predict(X)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
    ax.set_title(f'Linkage: {linkage}')

plt.show()
```

### Relations avec d'autres modèles

- **K-Means** : Clustering plat (pas de hiérarchie), plus rapide
- **DBSCAN** : Density-based, gère bruit et formes complexes
- **GMM** : Clustering probabiliste (soft assignments)
- **BIRCH** : Hierarchical scalable (pour grandes données)
- **OPTICS** : Density-based hiérarchique

### Scipy vs Scikit-learn

**Scipy (pour dendrogrammes)** :
```python
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# Calcul linkage
Z = linkage(X, method='ward')

# Dendrogramme
dendrogram(Z)

# Couper à une hauteur
clusters = fcluster(Z, t=10, criterion='distance')
```

**Scikit-learn (pour prédictions)** :
```python
from sklearn.cluster import AgglomerativeClustering

# Clustering
model = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = model.fit_predict(X)
```

### Variantes avancées

**1. BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies)** :
```python
from sklearn.cluster import Birch

# Scalable pour grandes données
birch = Birch(n_clusters=3, threshold=0.5, branching_factor=50)
labels = birch.fit_predict(X)

# Beaucoup plus rapide que Agglomerative sur grandes données
```

**2. Dendrogramme circulaire** :
```python
from scipy.cluster.hierarchy import dendrogram

# Dendrogramme circulaire (pour beauté)
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='polar')
dendrogram(Z, ax=ax, orientation='left')
plt.show()
```

## Références

### Documentation et tutoriels
- **Scikit-learn** : [Hierarchical Clustering](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering)
- **Scipy** : [Hierarchical Clustering](https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html)
- **StatQuest** : [Hierarchical Clustering](https://www.youtube.com/watch?v=7xHsRkOdVwo) (YouTube)

### Livres
- **"The Elements of Statistical Learning"** (Hastie et al., 2009) - Chapitre 14.3
- **"Pattern Recognition and Machine Learning"** (Bishop, 2006) - Chapitre 9
- **"An Introduction to Statistical Learning"** (James et al., 2021) - Chapitre 12

### Papers fondamentaux
- **Single Linkage** : Florek et al., 1951
- **Complete Linkage** : Sørensen, 1948
- **Ward's Method** : Ward, 1963 - "Hierarchical Grouping to Optimize an Objective Function"
- **BIRCH** : Zhang et al., 1996 - "BIRCH: An Efficient Data Clustering Method for Very Large Databases"

### Théorie

**Complexité** :
```
Temps: O(n³) pour agglomeratif standard
       O(n² log n) avec optimisations
Espace: O(n²) pour matrice de distances complète
```

**Matrice de linkage (Scipy)** :
```
Z[i] = [cluster1, cluster2, distance, n_points]

cluster1, cluster2: indices des clusters fusionnés
distance: distance de Ward (ou autre)
n_points: nombre total de points dans nouveau cluster
```

**Comparaison de performance** :
```
Dataset: 1,000 points, 2D

Algorithme           Temps    Mémoire
K-Means              0.01s    1 MB
Hierarchical (Ward)  0.8s     8 MB
DBSCAN               0.15s    2 MB
GMM                  0.3s     3 MB

→ Hierarchical: Lent mais informatif
```

### Tuning rapide (règles empiriques)

**Workflow recommandé** :
```python
# 1. Calculer linkage et visualiser dendrogramme
from scipy.cluster.hierarchy import linkage, dendrogram

Z = linkage(X, method='ward')
plt.figure(figsize=(12, 6))
dendrogram(Z)
plt.axhline(y=..., color='r')  # Ligne de coupure
plt.show()

# 2. Choisir K visuellement en regardant les grands sauts

# 3. Clustering avec K choisi
from sklearn.cluster import AgglomerativeClustering

model = AgglomerativeClustering(n_clusters=K, linkage='ward')
labels = model.fit_predict(X)
```

**Choix du linkage** :
```python
# Formes allongées, non-convexes → 'single'
# Clusters compacts, sphériques → 'complete' ou 'ward'
# Compromis général → 'average'
# Minimiser variance (meilleur souvent) → 'ward'

# Tester visuellement
for linkage in ['single', 'complete', 'average', 'ward']:
    model = AgglomerativeClustering(n_clusters=3, linkage=linkage)
    model.fit(X)
    # Visualiser et comparer
```

**Optimisation pour grandes données** :
```python
# Si n > 10,000 → Utiliser BIRCH au lieu de Agglomerative
from sklearn.cluster import Birch

birch = Birch(n_clusters=5)
labels = birch.fit_predict(X_large)

# Ou sous-échantillonner
X_sample = X[np.random.choice(len(X), 5000, replace=False)]
model = AgglomerativeClustering(n_clusters=5)
model.fit(X_sample)
```
