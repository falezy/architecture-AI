# Random Forest

Ensemble d'arbres (bagging) robuste, peu de tuning.

## Idée clé

**Random Forest** est un **ensemble de nombreux arbres de décision** entraînés indépendamment sur des sous-ensembles aléatoires des données et des features. La prédiction finale est obtenue par **vote majoritaire** (classification) ou **moyenne** (régression).

**Principe (Bagging + Feature Randomness)** :
1. **Bootstrap** : Créer N échantillons aléatoires avec remise (même taille que dataset)
2. **Entraîner** : Pour chaque échantillon, entraîner un arbre de décision
3. **Feature Randomness** : À chaque split, considérer seulement √d features aléatoires (au lieu de toutes)
4. **Agréger** : Vote majoritaire (classification) ou moyenne (régression)

**Formule** :
```
Classification : ŷ = mode(h₁(x), h₂(x), ..., hₙ(x))
Régression     : ŷ = (1/N) Σ hᵢ(x)
```
- `hᵢ(x)` : prédiction de l'arbre i
- `N` : nombre d'arbres (typiquement 100-500)

**Pourquoi ça fonctionne ?**
- **Bootstrap** : Chaque arbre voit des données différentes → diversité
- **Feature Randomness** : Arbres apprennent des patterns différents → décorrélation
- **Moyenne** : Réduit la variance (overfitting) sans augmenter le biais

**Différence avec un seul arbre** :
| Aspect | Arbre unique | Random Forest |
|--------|--------------|---------------|
| **Overfitting** | Élevé | Faible (moyenne de N arbres) |
| **Variance** | Élevée | Faible |
| **Biais** | Faible | Faible |
| **Stabilité** | Instable | Très stable |
| **Interprétabilité** | Élevée | Moyenne (feature importance) |
| **Vitesse** | Rapide | Plus lent (N arbres) |

## Exemples concrets

### 1. Classification : Prédire la survie sur le Titanic

**Scénario** : Prédire si un passager survit selon l'âge, la classe, le sexe, et le tarif.

**Code Python avec Random Forest** :
```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# 1. Données simulées (style Titanic)
data = {
    'Age': [22, 38, 26, 35, 28, 45, 31, 50, 18, 60, 25, 40],
    'Pclass': [3, 1, 3, 1, 3, 2, 1, 2, 3, 1, 2, 3],
    'Sex': [1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1],  # 1=Male, 0=Female
    'Fare': [7, 71, 8, 53, 8, 13, 50, 15, 7, 30, 25, 10],
    'Survived': [0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0]
}
df = pd.DataFrame(data)

X = df.drop('Survived', axis=1)
y = df['Survived']

# 2. Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 3. Créer et entraîner Random Forest
model = RandomForestClassifier(
    n_estimators=100,        # Nombre d'arbres
    max_depth=10,            # Profondeur max par arbre
    min_samples_split=2,     # Min samples pour split
    min_samples_leaf=1,      # Min samples par feuille
    max_features='sqrt',     # √d features par split (défaut)
    bootstrap=True,          # Bootstrap sampling
    oob_score=True,          # Out-of-bag score (validation automatique)
    random_state=42,
    n_jobs=-1                # Utiliser tous les CPU
)
model.fit(X_train, y_train)

# 4. Prédictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

print(f"Accuracy (train): {model.score(X_train, y_train):.2%}")
print(f"Accuracy (test): {accuracy_score(y_test, y_pred):.2%}")
print(f"OOB Score: {model.oob_score_:.2%}")  # Validation automatique !

print("\nMatrice de confusion:")
print(confusion_matrix(y_test, y_pred))
print("\nRapport de classification:")
print(classification_report(y_test, y_pred, target_names=['Décédé', 'Survécu']))

# 5. Feature importance
features = X.columns
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

print("\nFeature Importance:")
for i in range(len(features)):
    print(f"{i+1}. {features[indices[i]]}: {importances[indices[i]]:.3f}")

# Visualisation
plt.figure(figsize=(10, 6))
plt.title("Feature Importance - Random Forest")
plt.bar(range(len(features)), importances[indices])
plt.xticks(range(len(features)), [features[i] for i in indices])
plt.ylabel('Importance')
plt.show()

# 6. Prédire pour un nouveau passager
nouveau_passager = pd.DataFrame({
    'Age': [30],
    'Pclass': [1],
    'Sex': [0],  # Female
    'Fare': [50]
})
prediction = model.predict(nouveau_passager)[0]
proba = model.predict_proba(nouveau_passager)[0]
print(f"\nNouveau passager: {nouveau_passager.iloc[0].to_dict()}")
print(f"Prédiction: {'Survécu' if prediction == 1 else 'Décédé'}")
print(f"Probabilité de survie: {proba[1]:.2%}")
```

---

### 2. Régression : Prédire le prix d'une maison

**Scénario** : Prédire le prix d'une maison selon ses caractéristiques.

**Code Python** :
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

# 1. Données
data = {
    'Surface': [50, 80, 120, 150, 70, 100, 60, 140, 90, 110, 75, 130],
    'Chambres': [1, 2, 3, 4, 2, 3, 1, 4, 2, 3, 2, 3],
    'Age': [10, 5, 2, 1, 15, 8, 20, 3, 12, 6, 18, 4],
    'Distance_centre': [5, 2, 1, 1, 10, 3, 15, 2, 8, 4, 12, 2],
    'Prix': [150, 240, 360, 450, 210, 300, 180, 420, 270, 330, 200, 380]
}
df = pd.DataFrame(data)

X = df.drop('Prix', axis=1)
y = df['Prix']

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 3. Random Forest Regressor
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=2,
    max_features='sqrt',
    oob_score=True,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# 4. Prédictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print(f"R² (train): {r2_score(y_train, y_pred_train):.3f}")
print(f"R² (test): {r2_score(y_test, y_pred_test):.3f}")
print(f"RMSE (test): {np.sqrt(mean_squared_error(y_test, y_pred_test)):.2f} k€")
print(f"MAE (test): {mean_absolute_error(y_test, y_pred_test):.2f} k€")
print(f"OOB Score: {model.oob_score_:.3f}")

# 5. Feature importance
importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)
print("\nFeature Importance:")
print(importances)

# 6. Prédire pour une nouvelle maison
nouvelle_maison = pd.DataFrame({
    'Surface': [95],
    'Chambres': [3],
    'Age': [7],
    'Distance_centre': [3]
})
prix_predit = model.predict(nouvelle_maison)[0]
print(f"\nNouvelle maison: {nouvelle_maison.iloc[0].to_dict()}")
print(f"Prix prédit: {prix_predit:.0f} k€")
```

---

### 3. Comparaison : Single Tree vs Random Forest

**Code pour montrer la réduction d'overfitting** :
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Données avec du bruit
X, y = make_classification(
    n_samples=500, 
    n_features=20, 
    n_informative=15,
    n_redundant=5,
    random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 1. Arbre unique (sans contrainte → overfitting)
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)
print(f"Decision Tree:")
print(f"  Train accuracy: {tree.score(X_train, y_train):.2%}")
print(f"  Test accuracy: {tree.score(X_test, y_test):.2%}")
print(f"  → Overfitting: {tree.score(X_train, y_train) - tree.score(X_test, y_test):.2%}")

# 2. Random Forest (résistant à l'overfitting)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print(f"\nRandom Forest:")
print(f"  Train accuracy: {rf.score(X_train, y_train):.2%}")
print(f"  Test accuracy: {rf.score(X_test, y_test):.2%}")
print(f"  → Overfitting: {rf.score(X_train, y_train) - rf.score(X_test, y_test):.2%}")
```

**Output typique** :
```
Decision Tree:
  Train accuracy: 100.00%
  Test accuracy: 82.00%
  → Overfitting: 18.00%

Random Forest:
  Train accuracy: 99.00%
  Test accuracy: 91.00%
  → Overfitting: 8.00%
```

---

### 4. Hyperparameter Tuning avec GridSearchCV

**Code pour trouver les meilleurs hyperparamètres** :
```python
from sklearn.model_selection import GridSearchCV

# Grille de paramètres
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# Grid Search avec Cross-Validation
rf = RandomForestClassifier(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(
    rf, 
    param_grid, 
    cv=5,                    # 5-fold cross-validation
    scoring='accuracy',
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(f"Meilleurs paramètres: {grid_search.best_params_}")
print(f"Meilleur score CV: {grid_search.best_score_:.3f}")

# Entraîner avec les meilleurs paramètres
best_rf = grid_search.best_estimator_
test_score = best_rf.score(X_test, y_test)
print(f"Accuracy (test): {test_score:.2%}")
```

---

### 5. Out-of-Bag (OOB) Score : Validation gratuite

**Code pour utiliser OOB comme validation** :
```python
# OOB Score = validation automatique (pas besoin de validation set)
model = RandomForestClassifier(
    n_estimators=100,
    oob_score=True,  # Activer OOB
    random_state=42
)
model.fit(X_train, y_train)

# OOB score ≈ validation score (gratuit !)
print(f"OOB Score: {model.oob_score_:.2%}")
print(f"Test Score: {model.score(X_test, y_test):.2%}")

# OOB predictions (pour chaque exemple du train set)
oob_predictions = model.oob_decision_function_  # Probabilités OOB
print(f"OOB predictions shape: {oob_predictions.shape}")
```

**Explication OOB** :
- Chaque arbre est entraîné sur ~63% des données (bootstrap)
- Les 37% restants sont "out-of-bag" pour cet arbre
- On peut évaluer l'arbre sur ces données OOB
- Moyenne des évaluations OOB = OOB score (validation gratuite !)

## Quand l'utiliser

- ✅ **Baseline solide** : Excellent point de départ, souvent difficile à battre
- ✅ **Peu de tuning** : Fonctionne bien avec paramètres par défaut
- ✅ **Données avec bruit** : Robuste aux outliers et données manquantes
- ✅ **Feature importance** : Identifier les variables importantes
- ✅ **Classification ET régression** : Modèle polyvalent
- ✅ **Données tabulaires** : Très bon sur données structurées (CSV, bases de données)
- ✅ **Pas de normalisation** : Insensible à l'échelle des features

**Cas d'usage typiques** :
- 💳 **Finance** : Scoring de crédit, détection de fraude
- 🏥 **Santé** : Diagnostic médical, prédiction de risque
- 🎯 **Marketing** : Prédiction de churn, segmentation client
- 🏭 **Industrie** : Maintenance prédictive, contrôle qualité
- 🌾 **Agriculture** : Prédiction de rendement, classification de maladies

**Quand NE PAS utiliser** :
- ❌ Compétitions Kaggle top performance → XGBoost/LightGBM
- ❌ Images/audio/vidéo → Deep Learning (CNN, RNN)
- ❌ Texte brut → Transformers (BERT, GPT)
- ❌ Besoin d'interprétabilité totale → Decision Tree unique, Regression linéaire

## Forces

✅ **Très robuste** : Résistant à l'overfitting (moyenne de N arbres)  
✅ **Peu de tuning** : Fonctionne bien "out of the box"  
✅ **Pas de normalisation** : Insensible à l'échelle des features  
✅ **Gère données manquantes** : Peut gérer NaN (avec stratégie)  
✅ **Feature importance** : Identifie variables importantes  
✅ **Parallélisable** : Entraînement rapide avec n_jobs=-1  
✅ **OOB Score** : Validation gratuite sans split séparé

**Exemple de robustesse** :
```python
# Ajouter du bruit (outliers)
X_noisy = X.copy()
X_noisy[0, 0] = 1000  # Outlier extrême

# Decision Tree → sensible
tree = DecisionTreeClassifier()
tree.fit(X_noisy, y)
print(f"Tree accuracy: {tree.score(X_test, y_test):.2%}")  # ~75%

# Random Forest → robuste
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_noisy, y)
print(f"RF accuracy: {rf.score(X_test, y_test):.2%}")  # ~89%
```

## Limites

❌ **Moins performant que XGBoost** : Sur données tabulaires complexes  
❌ **Mémoire** : N arbres = N fois plus de mémoire  
❌ **Lent en prédiction** : Doit interroger N arbres (vs 1 seul)  
❌ **Extrapolation** : Ne prédit que dans la plage des valeurs d'entraînement  
❌ **Interprétabilité** : Moins qu'un seul arbre (100+ arbres)  
❌ **Biais pour features à haute cardinalité** : Préfère features avec beaucoup de valeurs  
❌ **Pas adapté aux séries temporelles** : Sans features temporelles explicites

**Exemple d'extrapolation** :
```python
# Train sur prix 100-500k€
X_train = np.array([[100], [200], [300], [400], [500]])
y_train = np.array([100, 200, 300, 400, 500])

rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train, y_train)

# Prédire pour 1000k€ (hors plage)
print(rf.predict([[1000]]))  # ~500 (max vu en train, pas 1000!)
```

**Temps de prédiction** :
```python
import time

# Single tree
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
start = time.time()
tree.predict(X_test)
print(f"Tree predict: {time.time() - start:.4f}s")

# Random Forest (100 trees)
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
start = time.time()
rf.predict(X_test)
print(f"RF predict: {time.time() - start:.4f}s")  # ~100x plus lent
```

## Variantes / liens

### Hyperparamètres clés

```python
RandomForestClassifier(
    # Nombre d'arbres
    n_estimators=100,        # Plus = mieux (mais diminishing returns après 100-200)
    
    # Profondeur et complexité
    max_depth=None,          # None = arbres profonds (défaut)
    min_samples_split=2,     # Min samples pour split
    min_samples_leaf=1,      # Min samples par feuille
    max_leaf_nodes=None,     # Limiter nombre de feuilles
    
    # Feature sampling
    max_features='sqrt',     # √d pour classification (défaut)
                             # 'log2', 'auto', None, int, float
    
    # Bootstrap
    bootstrap=True,          # Bootstrap sampling (défaut)
    oob_score=False,         # Calculer OOB score
    
    # Parallélisation
    n_jobs=-1,               # Utiliser tous les CPU
    random_state=42          # Reproductibilité
)
```

**Recommandations** :
- **n_estimators** : 100-200 (bon compromis vitesse/performance)
- **max_features** : 'sqrt' (classification), 'auto' ou 1/3 (régression)
- **max_depth** : None (laisser pousser) ou 10-30 si overfitting
- **min_samples_leaf** : 1-5 (augmenter si overfitting)

### Relations avec d'autres modèles

**1. Bagging (Bootstrap Aggregating)** :
```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# Random Forest = Bagging + Feature Randomness
bagging = BaggingClassifier(
    DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=1.0,  # Bootstrap 100% des données
    bootstrap=True
)
```

**2. Extra Trees (Extremely Randomized Trees)** :
```python
from sklearn.ensemble import ExtraTreesClassifier

# Extra Trees = Random Forest + splits aléatoires
extra_trees = ExtraTreesClassifier(
    n_estimators=100,
    max_features='sqrt',
    random_state=42
)
# Plus rapide que RF, parfois meilleur
```

**3. Gradient Boosting** (XGBoost, LightGBM) :
- **RF** : Arbres en parallèle, indépendants
- **Boosting** : Arbres séquentiels, correctifs
- **Performance** : Boosting > RF (mais plus sensible au tuning)

**4. Isolation Forest** (détection d'anomalies) :
```python
from sklearn.ensemble import IsolationForest

# Utilise Random Forest pour anomaly detection
iso = IsolationForest(n_estimators=100, contamination=0.1)
anomalies = iso.fit_predict(X)
```

### Feature Importance avancée

**Permutation Importance** (plus fiable que feature_importances_) :
```python
from sklearn.inspection import permutation_importance

# Entraîner le modèle
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

# Permutation importance
perm_importance = permutation_importance(
    rf, X_test, y_test, 
    n_repeats=10,
    random_state=42
)

# Afficher
for i, feature in enumerate(X.columns):
    print(f"{feature}: {perm_importance.importances_mean[i]:.3f} "
          f"± {perm_importance.importances_std[i]:.3f}")
```

### Calibration des probabilités

```python
from sklearn.calibration import CalibratedClassifierCV

# Random Forest non calibré
rf = RandomForestClassifier(n_estimators=100)

# Calibration
calibrated_rf = CalibratedClassifierCV(rf, cv=5, method='sigmoid')
calibrated_rf.fit(X_train, y_train)

# Probabilités mieux calibrées
probas = calibrated_rf.predict_proba(X_test)
```

## Références

### Documentation et tutoriels
- **Scikit-learn** : [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest)
- **StatQuest** : [Random Forest Explained](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ) (YouTube)

### Livres
- **"An Introduction to Statistical Learning"** (James et al., 2021) - Chapitre 8
- **"The Elements of Statistical Learning"** (Hastie et al., 2009) - Chapitre 15
- **"Hands-On Machine Learning"** (Géron, 2019) - Chapitre 7

### Papers fondamentaux
- **Random Forests** : Breiman, 2001 - "Random Forests" (paper original)
- **Bagging** : Breiman, 1996 - "Bagging Predictors"
- **Feature Importance** : Breiman, 2001 - Mesure d'impureté

### Comparaison de performance

**Benchmark (Dataset : Credit Card Fraud Detection)** :
```
Algorithme              Accuracy    AUC-ROC    Temps
Logistic Regression       92%        0.85      1s
Decision Tree             85%        0.78      2s
Random Forest (100)       97%        0.95      15s
XGBoost                   98%        0.97      25s

→ RF: Excellent compromis performance/simplicité
```

### Tuning rapide (règles empiriques)

**Si overfitting** :
```python
# Augmenter régularisation
RandomForestClassifier(
    n_estimators=100,
    max_depth=10,          # Limiter profondeur
    min_samples_leaf=5,    # Augmenter min samples
    max_features='sqrt'    # Moins de features
)
```

**Si underfitting** :
```python
# Réduire régularisation
RandomForestClassifier(
    n_estimators=200,      # Plus d'arbres
    max_depth=None,        # Arbres profonds
    min_samples_leaf=1,    # Moins de contraintes
    max_features='auto'    # Plus de features
)
```

**Si trop lent** :
```python
# Accélérer
RandomForestClassifier(
    n_estimators=50,       # Moins d'arbres
    max_depth=10,          # Limiter profondeur
    max_samples=0.8,       # Sous-échantillonner
    n_jobs=-1              # Paralléliser
)
```
