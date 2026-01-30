# Decision Trees

Modèle interprétable basé sur des règles de split.

## Idée clé

Un **arbre de décision** est un modèle qui apprend une séquence de **questions binaires** (if/else) pour prédire une valeur. Il divise récursivement l'espace des features en régions homogènes.

**Fonctionnement** :
1. **Root node** : Choisir la meilleure feature pour diviser les données
2. **Split** : Créer deux branches (gauche/droite) selon une condition
3. **Répéter** : Pour chaque branche, choisir la prochaine meilleure question
4. **Leaf nodes** : Arrêter quand critère de pureté atteint (ou profondeur max)

**Visualisation conceptuelle** :
```
                    [Surface < 70m²?]
                    /              \
                 OUI               NON
                /                    \
        [Prix < 200k€]         [Chambres < 3?]
         /        \              /          \
      OUI        NON          OUI          NON
       /          \           /             \
   Petit      Moyen      Grand          Très Grand
```

**Critères de split** :
- **Classification** : Gini impurity, Entropy (Information Gain)
- **Régression** : MSE, MAE

**Formule Gini Impurity** :
```
Gini = 1 - Σ(pᵢ)²
```
où `pᵢ` = proportion de la classe i

**Formule Entropy** :
```
Entropy = -Σ(pᵢ · log₂(pᵢ))
```

## Exemples concrets

### 1. Classification : Prédire l'approbation d'un prêt bancaire

**Scénario** : Une banque veut automatiser l'approbation de prêts selon le revenu, l'âge et l'historique de crédit.

**Données d'exemple** :
```
Revenu (k€) | Âge | Crédit | Prêt approuvé?
30          | 25  | Bon    | Non
70          | 35  | Bon    | Oui
50          | 45  | Mauvais| Non
90          | 50  | Bon    | Oui
```

**Code Python avec scikit-learn** :
```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. Données d'entraînement
X = np.array([
    [30, 25, 1],  # Revenu, Âge, Crédit (1=Bon, 0=Mauvais)
    [70, 35, 1],
    [50, 45, 0],
    [90, 50, 1],
    [40, 30, 0],
    [80, 40, 1],
    [35, 28, 1],
    [60, 38, 0],
])
y = np.array([0, 1, 0, 1, 0, 1, 0, 0])  # 0=Refusé, 1=Approuvé

# 2. Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 3. Créer et entraîner l'arbre
model = DecisionTreeClassifier(
    max_depth=3,           # Profondeur maximale
    criterion='gini',      # ou 'entropy'
    min_samples_split=2,   # Min samples pour split
    random_state=42
)
model.fit(X_train, y_train)

# 4. Prédire pour un nouveau client
nouveau_client = np.array([[55, 32, 1]])  # 55k€, 32 ans, bon crédit
prediction = model.predict(nouveau_client)
proba = model.predict_proba(nouveau_client)

print(f"Prédiction: {'Approuvé' if prediction[0] == 1 else 'Refusé'}")
print(f"Probabilité: {proba[0][1]:.2%}")

# 5. Évaluation
from sklearn.metrics import accuracy_score, classification_report
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print(classification_report(y_test, y_pred, target_names=['Refusé', 'Approuvé']))

# 6. Feature importance
features = ['Revenu', 'Âge', 'Crédit']
importances = model.feature_importances_
for feature, importance in zip(features, importances):
    print(f"{feature}: {importance:.3f}")
```

**Visualisation de l'arbre** :
```python
plt.figure(figsize=(15, 8))
plot_tree(
    model, 
    feature_names=['Revenu', 'Âge', 'Crédit'],
    class_names=['Refusé', 'Approuvé'],
    filled=True,           # Couleurs selon classe
    rounded=True,
    fontsize=10
)
plt.title("Arbre de décision - Approbation de prêt")
plt.show()
```

**Interprétation** :
```
Si Revenu > 60k€ → Approuvé
Sinon:
    Si Crédit = Bon ET Âge > 30 → Approuvé
    Sinon → Refusé
```

---

### 2. Régression : Prédire le prix d'une maison

**Scénario** : Prédire le prix d'une maison selon sa surface et le nombre de chambres.

**Code Python** :
```python
from sklearn.tree import DecisionTreeRegressor
import numpy as np

# 1. Données
X = np.array([
    [50, 1],   # 50m², 1 chambre
    [80, 2],   # 80m², 2 chambres
    [120, 3],  # 120m², 3 chambres
    [150, 4],
    [70, 2],
    [100, 3],
    [60, 1],
    [140, 4],
])
y = np.array([150, 240, 360, 450, 210, 300, 180, 420])  # Prix en k€

# 2. Créer l'arbre de régression
model = DecisionTreeRegressor(
    max_depth=3,
    min_samples_leaf=2,
    random_state=42
)
model.fit(X, y)

# 3. Prédire pour une nouvelle maison
nouvelle_maison = np.array([[90, 2]])  # 90m², 2 chambres
prix_predit = model.predict(nouvelle_maison)
print(f"Prix prédit pour 90m², 2 chambres: {prix_predit[0]:.0f}k€")

# 4. Visualiser l'arbre
plt.figure(figsize=(15, 8))
plot_tree(
    model,
    feature_names=['Surface', 'Chambres'],
    filled=True,
    rounded=True
)
plt.show()

# 5. Feature importance
print(f"Importance Surface: {model.feature_importances_[0]:.3f}")
print(f"Importance Chambres: {model.feature_importances_[1]:.3f}")
```

---

### 3. Visualisation des frontières de décision

**Code pour visualiser comment l'arbre divise l'espace** :
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification

# Données 2D pour visualisation
X, y = make_classification(
    n_samples=100, 
    n_features=2, 
    n_redundant=0, 
    n_clusters_per_class=1,
    random_state=42
)

# Entraîner l'arbre
model = DecisionTreeClassifier(max_depth=3)
model.fit(X, y)

# Créer une grille pour visualiser les frontières
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.02),
    np.arange(y_min, y_max, 0.02)
)

# Prédire pour chaque point de la grille
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Visualisation
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Frontières de décision (lignes rectangulaires)')
plt.show()
```

**Observation** : Les arbres créent des **frontières rectangulaires** (perpendiculaires aux axes).

## Quand l'utiliser

- ✅ **Interprétabilité requise** : Besoin d'expliquer les décisions (médecine, finance, juridique)
- ✅ **Données mixtes** : Features numériques ET catégorielles (pas de preprocessing nécessaire)
- ✅ **Relations non-linéaires** : Capture automatiquement les interactions complexes
- ✅ **Peu de preprocessing** : Pas besoin de normalisation ou d'encodage one-hot
- ✅ **Baseline rapide** : Modèle simple à mettre en place pour tester rapidement
- ✅ **Feature importance** : Identifier les variables les plus importantes

**Cas d'usage typiques** :
- 🏥 **Médecine** : Diagnostic basé sur symptômes (arbre de décision clinique)
- 💳 **Finance** : Approbation de crédit, détection de fraude
- 🎯 **Marketing** : Segmentation client, prédiction de churn
- 🏭 **Industrie** : Maintenance prédictive, contrôle qualité
- 📊 **Sciences** : Classification d'espèces, analyse d'images

## Forces

✅ **Très interprétable** : Visualisation simple, règles if/else compréhensibles  
✅ **Pas de preprocessing** : Gère directement les features catégorielles et numériques  
✅ **Pas de normalisation** : Insensible à l'échelle des features  
✅ **Capture non-linéarités** : Relations complexes sans feature engineering  
✅ **Feature importance** : Identifie automatiquement les variables importantes  
✅ **Rapide à entraîner** : Complexité O(n·d·log(n))  
✅ **Robuste aux outliers** : Basé sur des splits, pas des distances

**Exemple de force** :
```python
# Pas besoin de preprocessing !
X = pd.DataFrame({
    'Surface': [50, 80, 120],        # Échelle 0-200
    'Distance_centre': [1, 10, 50],  # Échelle 0-100
    'Type': ['Appartement', 'Maison', 'Villa']  # Catégoriel
})

# Fonctionne directement avec :
from sklearn.tree import DecisionTreeRegressor
# Après encodage basique des catégories
```

## Limites

❌ **Surapprentissage** : Très sujet à l'overfitting (mémorise le bruit)  
❌ **Instabilité** : Petite variation des données → arbre complètement différent  
❌ **Frontières rectangulaires** : Inefficace pour frontières diagonales/circulaires  
❌ **Biais vers features à forte cardinalité** : Préfère les features avec beaucoup de valeurs  
❌ **Problème XOR** : Difficile de capturer certaines relations géométriques  
❌ **Prédictions discontinues** : Changements brusques aux frontières  
❌ **Pas d'extrapolation** : Prédit uniquement des valeurs vues (régression)

**Exemple de surapprentissage** :
```python
# Arbre sans contrainte → overfitting
model_overfit = DecisionTreeClassifier()  # Pas de max_depth
model_overfit.fit(X_train, y_train)
print(f"Train accuracy: {model_overfit.score(X_train, y_train):.2%}")  # 100%
print(f"Test accuracy: {model_overfit.score(X_test, y_test):.2%}")    # ~70%

# Arbre régularisé → meilleur
model_regularized = DecisionTreeClassifier(
    max_depth=5,         # Limiter la profondeur
    min_samples_leaf=10  # Min samples par feuille
)
model_regularized.fit(X_train, y_train)
print(f"Test accuracy: {model_regularized.score(X_test, y_test):.2%}")  # ~85%
```

**Hyperparamètres pour contrôler l'overfitting** :
```python
DecisionTreeClassifier(
    max_depth=5,              # Profondeur max (défaut: None)
    min_samples_split=20,     # Min samples pour split (défaut: 2)
    min_samples_leaf=10,      # Min samples par feuille (défaut: 1)
    max_features='sqrt',      # Nb features à considérer (défaut: all)
    max_leaf_nodes=50,        # Nb max de feuilles
    min_impurity_decrease=0.01  # Min gain d'impureté pour split
)
```

## Variantes / liens

### Ensembles d'arbres (solutions au surapprentissage)

**1. Random Forest** : Moyenne de plusieurs arbres aléatoires
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,    # Nombre d'arbres
    max_depth=10,
    random_state=42
)
model.fit(X_train, y_train)
```

**Avantages** :
- ✅ Moins de surapprentissage qu'un seul arbre
- ✅ Meilleure généralisation
- ✅ Robuste au bruit
- ❌ Perd en interprétabilité

**2. Gradient Boosting** : Arbres séquentiels qui corrigent les erreurs
```python
from sklearn.ensemble import GradientBoostingClassifier
# Ou XGBoost, LightGBM, CatBoost (plus performants)

model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3
)
```

**Librairies modernes** :
```python
# XGBoost (populaire en compétitions Kaggle)
import xgboost as xgb
model = xgb.XGBClassifier(n_estimators=100, max_depth=5)

# LightGBM (très rapide)
import lightgbm as lgb
model = lgb.LGBMClassifier(n_estimators=100)

# CatBoost (meilleur pour features catégorielles)
from catboost import CatBoostClassifier
model = CatBoostClassifier(iterations=100, verbose=0)
```

### Relations avec d'autres modèles

- **CART** : Classification And Regression Trees (algorithme standard)
- **ID3, C4.5** : Arbres basés sur Information Gain (plus anciens)
- **Extra Trees** : Arbres avec splits aléatoires
- **Isolation Forest** : Arbres pour détection d'anomalies

### Visualisation avancée

**Export en format texte** :
```python
from sklearn.tree import export_text
tree_rules = export_text(model, feature_names=['Surface', 'Chambres'])
print(tree_rules)
```

**Export en Graphviz** :
```python
from sklearn.tree import export_graphviz
import graphviz

dot_data = export_graphviz(
    model,
    feature_names=['Surface', 'Chambres'],
    class_names=['Petit', 'Grand'],
    filled=True,
    rounded=True
)
graph = graphviz.Source(dot_data)
graph.render("decision_tree")  # Sauvegarde en PDF
```

## Références

### Documentation et tutoriels
- **Scikit-learn** : [Decision Trees](https://scikit-learn.org/stable/modules/tree.html)
- **StatQuest** : [Decision Trees Explained](https://www.youtube.com/watch?v=_L39rN6gz7Y) (YouTube)
- **Visualisation interactive** : [R2D3 Visual Intro](http://www.r2d3.us/visual-intro-to-machine-learning-part-1/)

### Livres
- **"An Introduction to Statistical Learning"** (James et al., 2021) - Chapitre 8
- **"The Elements of Statistical Learning"** (Hastie et al., 2009) - Chapitre 9
- **"Hands-On Machine Learning"** (Géron, 2019) - Chapitre 6

### Algorithmes et implémentations
- **CART** : Breiman et al., 1984 (algorithme de base)
- **ID3** : Quinlan, 1986 (Information Gain)
- **C4.5** : Quinlan, 1993 (amélioration de ID3)
- **Scikit-learn** : Implementation optimisée en Cython

### Librairies pour ensembles
```python
# Scikit-learn (basique)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# XGBoost (performance)
pip install xgboost

# LightGBM (rapidité)
pip install lightgbm

# CatBoost (catégories)
pip install catboost
```

### Métriques d'évaluation des arbres
- **Gini Impurity** : Mesure de "pureté" des nœuds
- **Entropy / Information Gain** : Mesure de réduction d'incertitude
- **Variance Reduction** : Pour la régression (MSE)
