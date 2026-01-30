# Gradient Boosting

Ensemble séquentiel (XGBoost/LightGBM/CatBoost).

## Idée clé

Le **Gradient Boosting** construit un ensemble d'arbres de décision **séquentiellement** : chaque nouvel arbre corrige les erreurs des arbres précédents en se concentrant sur les exemples mal prédits.

**Fonctionnement** :
1. **Initialiser** : Commencer avec une prédiction simple (moyenne pour régression, classe majoritaire pour classification)
2. **Calculer les résidus** : Erreurs = Vraie valeur - Prédiction actuelle
3. **Entraîner un arbre** : Pour prédire ces résidus (pas les valeurs originales)
4. **Mettre à jour** : Nouvelle prédiction = Ancienne prédiction + (learning_rate × prédiction_arbre)
5. **Répéter** : Ajouter des arbres jusqu'à convergence ou nombre d'arbres max

**Formule** :
```
F₀(x) = valeur_initiale
F₁(x) = F₀(x) + α·h₁(x)
F₂(x) = F₁(x) + α·h₂(x)
...
Fₙ(x) = Σ α·hᵢ(x)
```
- `hᵢ(x)` : arbre i (prédit les résidus)
- `α` : learning rate (0.01-0.3 typiquement)

**Analogie** : Comme un étudiant qui révise un examen
- Arbre 1 : Apprend les concepts de base
- Arbre 2 : Se concentre sur les erreurs de l'arbre 1
- Arbre 3 : Corrige les erreurs restantes
- → Amélioration progressive

**Différence avec Random Forest** :
| Aspect | Random Forest | Gradient Boosting |
|--------|---------------|-------------------|
| Construction | Parallèle (indépendant) | Séquentielle (corrective) |
| Arbres | Profonds | Peu profonds (stumps) |
| Overfitting | Moins sensible | Plus sensible |
| Performance | Bonne | **Excellent** (SOTA) |
| Vitesse | Rapide | Plus lent |

## Exemples concrets

### 1. Classification : Prédire la survie sur le Titanic (XGBoost)

**Scénario** : Prédire si un passager survit selon son âge, classe, sexe, et tarif.

**Code Python avec XGBoost** :
```python
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# 1. Données simulées (style Titanic)
data = {
    'Age': [22, 38, 26, 35, 28, 45, 31, 50, 18, 60],
    'Pclass': [3, 1, 3, 1, 3, 2, 1, 2, 3, 1],  # Classe (1=1st, 3=3rd)
    'Sex': [1, 0, 0, 0, 1, 1, 0, 1, 0, 1],     # 1=Male, 0=Female
    'Fare': [7, 71, 8, 53, 8, 13, 50, 15, 7, 30],
    'Survived': [0, 1, 1, 1, 0, 0, 1, 0, 1, 1]
}
df = pd.DataFrame(data)

X = df.drop('Survived', axis=1)
y = df['Survived']

# 2. Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Créer et entraîner XGBoost
model = XGBClassifier(
    n_estimators=100,      # Nombre d'arbres
    max_depth=3,           # Profondeur par arbre
    learning_rate=0.1,     # α (taux d'apprentissage)
    subsample=0.8,         # % de données par arbre
    colsample_bytree=0.8,  # % de features par arbre
    random_state=42,
    eval_metric='logloss'  # Métrique d'évaluation
)

# Entraînement avec early stopping
eval_set = [(X_train, y_train), (X_test, y_test)]
model.fit(
    X_train, y_train,
    eval_set=eval_set,
    early_stopping_rounds=10,  # Arrêter si pas d'amélioration
    verbose=False
)

# 4. Prédictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.3f}")
print("\n", classification_report(y_test, y_pred))

# 5. Feature importance
import matplotlib.pyplot as plt
from xgboost import plot_importance

plt.figure(figsize=(10, 6))
plot_importance(model, max_num_features=10)
plt.title("Feature Importance - XGBoost")
plt.show()

# 6. Courbe d'apprentissage
results = model.evals_result()
plt.figure(figsize=(10, 6))
plt.plot(results['validation_0']['logloss'], label='Train')
plt.plot(results['validation_1']['logloss'], label='Test')
plt.xlabel('Nombre d\'arbres')
plt.ylabel('Log Loss')
plt.legend()
plt.title('Courbe d\'apprentissage')
plt.show()
```

---

### 2. Régression : Prédire le prix d'une maison (LightGBM)

**Scénario** : Prédire le prix d'une maison avec LightGBM (plus rapide que XGBoost).

**Code Python avec LightGBM** :
```python
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. Données
data = {
    'Surface': [50, 80, 120, 150, 70, 100, 60, 140, 90, 110],
    'Chambres': [1, 2, 3, 4, 2, 3, 1, 4, 2, 3],
    'Age': [10, 5, 2, 1, 15, 8, 20, 3, 12, 6],
    'Distance_centre': [5, 2, 1, 1, 10, 3, 15, 2, 8, 4],
    'Prix': [150, 240, 360, 450, 210, 300, 180, 420, 270, 330]  # k€
}
df = pd.DataFrame(data)

X = df.drop('Prix', axis=1)
y = df['Prix']

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Créer dataset LightGBM (format optimisé)
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# 4. Paramètres
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',  # Gradient Boosting Decision Tree
    'num_leaves': 31,         # Complexité de l'arbre
    'learning_rate': 0.05,
    'feature_fraction': 0.9,  # % features par arbre
    'bagging_fraction': 0.8,  # % données par arbre
    'bagging_freq': 5,
    'verbose': -1
}

# 5. Entraînement avec early stopping
model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,     # Max arbres
    valid_sets=[train_data, test_data],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=100)
    ]
)

# 6. Prédictions
y_pred = model.predict(X_test, num_iteration=model.best_iteration)

print(f"R² Score: {r2_score(y_test, y_pred):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f} k€")

# 7. Feature importance
lgb.plot_importance(model, max_num_features=10)
plt.title("Feature Importance - LightGBM")
plt.show()
```

---

### 3. Comparaison XGBoost vs LightGBM vs CatBoost

**Code pour comparer les 3 librairies** :
```python
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

# Données
X, y = make_classification(n_samples=10000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Dictionnaire pour stocker les résultats
results = {}

# 1. XGBoost
from xgboost import XGBClassifier
start = time.time()
xgb_model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_time = time.time() - start
results['XGBoost'] = {
    'Accuracy': accuracy_score(y_test, xgb_pred),
    'Time (s)': xgb_time
}

# 2. LightGBM
import lightgbm as lgb
start = time.time()
lgb_model = lgb.LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.1)
lgb_model.fit(X_train, y_train)
lgb_pred = lgb_model.predict(X_test)
lgb_time = time.time() - start
results['LightGBM'] = {
    'Accuracy': accuracy_score(y_test, lgb_pred),
    'Time (s)': lgb_time
}

# 3. CatBoost
from catboost import CatBoostClassifier
start = time.time()
cat_model = CatBoostClassifier(
    iterations=100, 
    depth=5, 
    learning_rate=0.1,
    verbose=0
)
cat_model.fit(X_train, y_train)
cat_pred = cat_model.predict(X_test)
cat_time = time.time() - start
results['CatBoost'] = {
    'Accuracy': accuracy_score(y_test, cat_pred),
    'Time (s)': cat_time
}

# Afficher les résultats
df_results = pd.DataFrame(results).T
print(df_results)
```

**Résultats typiques** :
```
          Accuracy  Time (s)
XGBoost      0.87      1.2
LightGBM     0.88      0.6   ← Plus rapide
CatBoost     0.89      2.5   ← Meilleure accuracy
```

---

### 4. Hyperparameter Tuning avec Optuna

**Code pour trouver les meilleurs hyperparamètres** :
```python
import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

def objective(trial):
    # Définir l'espace de recherche
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
    }
    
    # Créer le modèle
    model = XGBClassifier(**params, random_state=42)
    
    # Cross-validation
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    return score.mean()

# Lancer l'optimisation
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print("Meilleurs paramètres:", study.best_params)
print(f"Meilleur score: {study.best_value:.3f}")

# Entraîner avec les meilleurs paramètres
best_model = XGBClassifier(**study.best_params, random_state=42)
best_model.fit(X_train, y_train)
```

## Quand l'utiliser

- ✅ **Compétitions Kaggle** : Gradient Boosting domine les leaderboards (90%+ des solutions gagnantes)
- ✅ **Données tabulaires** : Meilleur pour données structurées (CSV, bases de données)
- ✅ **Performance maximale** : Quand on veut le meilleur score possible
- ✅ **Features catégorielles** : CatBoost gère nativement sans encodage
- ✅ **Grandes données** : LightGBM très rapide sur millions de lignes
- ✅ **Interprétabilité partielle** : Feature importance + SHAP values

**Cas d'usage typiques** :
- 💳 **Finance** : Scoring de crédit, détection de fraude, prédiction de défaut
- 🎯 **Marketing** : Prédiction de churn, recommandations, CLV (Customer Lifetime Value)
- 🏥 **Santé** : Diagnostic médical, prédiction de réadmission
- 🏭 **Industrie** : Maintenance prédictive, optimisation de production
- 🛒 **E-commerce** : Prédiction de conversion, pricing dynamique

**Quand NE PAS utiliser** :
- ❌ Images/audio/vidéo → Utilisez Deep Learning (CNN, RNN)
- ❌ Texte brut → Transformers (BERT, GPT)
- ❌ Besoin d'interprétabilité totale → Régression linéaire, Decision Tree simple

## Forces

✅ **Performance SOTA** : Meilleur modèle pour données tabulaires (Kaggle)  
✅ **Capture non-linéarités** : Relations complexes sans feature engineering  
✅ **Robuste au bruit** : Gère bien les outliers et données bruitées  
✅ **Gère données manquantes** : Pas besoin d'imputation (surtout XGBoost/LightGBM)  
✅ **Feature importance** : Identifie variables importantes  
✅ **Régularisation intégrée** : L1/L2, contrôle de la complexité  
✅ **Scalable** : LightGBM gère millions de lignes rapidement

**Exemple de performance** :
```
Dataset: Prédiction de churn (10,000 clients)

Logistic Regression:  AUC = 0.72
Random Forest:        AUC = 0.81
XGBoost:             AUC = 0.89  ← Meilleur !
```

## Limites

❌ **Temps d'entraînement** : Plus lent que Random Forest (séquentiel)  
❌ **Sensible au surapprentissage** : Nécessite tuning des hyperparamètres  
❌ **Nombreux hyperparamètres** : learning_rate, max_depth, subsample, etc.  
❌ **Moins interprétable** : 100+ arbres difficiles à visualiser  
❌ **Pas adapté aux images/texte** : Deep Learning meilleur  
❌ **Besoin de GPU** : Pour grandes données (optionnel mais recommandé)  
❌ **Dépendance à l'ordre** : Ordre des features peut affecter performance

**Problème de surapprentissage** :
```python
# Sans régularisation → overfitting
model = XGBClassifier(n_estimators=1000, max_depth=10)
model.fit(X_train, y_train)
print(f"Train: {model.score(X_train, y_train):.2%}")  # 99%
print(f"Test: {model.score(X_test, y_test):.2%}")    # 78%

# Avec régularisation → meilleur
model = XGBClassifier(
    n_estimators=100,      # Moins d'arbres
    max_depth=5,           # Arbres moins profonds
    learning_rate=0.05,    # Learning rate plus faible
    subsample=0.8,         # Bagging
    colsample_bytree=0.8,  # Feature sampling
    reg_alpha=0.1,         # L1 regularization
    reg_lambda=1.0         # L2 regularization
)
model.fit(X_train, y_train)
print(f"Test: {model.score(X_test, y_test):.2%}")    # 86%
```

## Variantes / liens

### Principales librairies

**1. XGBoost** (eXtreme Gradient Boosting)
```python
from xgboost import XGBClassifier, XGBRegressor

model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    objective='binary:logistic',  # ou 'multi:softmax', 'reg:squarederror'
    tree_method='hist',           # ou 'gpu_hist' pour GPU
    enable_categorical=True       # Support natif des catégories
)
```

**Avantages** :
- ✅ Plus mature, bien documenté
- ✅ Support GPU excellent
- ✅ Gère données manquantes nativement

**2. LightGBM** (Light Gradient Boosting Machine)
```python
import lightgbm as lgb

model = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    boosting_type='gbdt',    # ou 'dart', 'goss'
    num_leaves=31,           # Clé pour la performance
    device='gpu'             # Support GPU
)
```

**Avantages** :
- ✅ **Plus rapide** : 10x plus rapide que XGBoost
- ✅ Moins de mémoire
- ✅ Meilleur pour grandes données (millions de lignes)

**3. CatBoost** (Categorical Boosting)
```python
from catboost import CatBoostClassifier

model = CatBoostClassifier(
    iterations=100,
    depth=5,
    learning_rate=0.1,
    cat_features=['Sex', 'Embarked'],  # Features catégorielles
    task_type='GPU'                     # Support GPU
)
```

**Avantages** :
- ✅ **Meilleur pour catégories** : Encodage natif, pas besoin de one-hot
- ✅ Moins de tuning nécessaire
- ✅ Souvent meilleure accuracy "out of the box"

### Tableau comparatif

| Critère | XGBoost | LightGBM | CatBoost |
|---------|---------|----------|----------|
| **Vitesse** | Moyen | ⚡ Très rapide | Moyen |
| **Accuracy** | ★★★★ | ★★★★ | ★★★★★ |
| **Features catégorielles** | Basique | Basique | ⭐ Excellent |
| **Tuning requis** | Élevé | Élevé | Faible |
| **Maturité** | ★★★★★ | ★★★★ | ★★★ |
| **Grandes données** | Bon | ⚡ Excellent | Bon |

### Relations avec d'autres modèles

- **AdaBoost** : Ancêtre du Gradient Boosting (moins performant)
- **Gradient Boosted Trees (GBT)** : Nom générique, scikit-learn
- **HistGradientBoosting** : Version scikit-learn (inspirée de LightGBM)
- **NGBoost** : Gradient Boosting probabiliste (incertitude)

### Interprétabilité avec SHAP

**Expliquer les prédictions** :
```python
import shap

# Créer l'explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Graphique pour une prédiction
shap.force_plot(
    explainer.expected_value, 
    shap_values[0], 
    X_test.iloc[0]
)

# Importance globale
shap.summary_plot(shap_values, X_test)
```

## Références

### Documentation officielle
- **XGBoost** : https://xgboost.readthedocs.io/
- **LightGBM** : https://lightgbm.readthedocs.io/
- **CatBoost** : https://catboost.ai/docs/

### Tutoriels et cours
- **Kaggle Learn** : [Intro to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning)
- **StatQuest** : [Gradient Boost Explained](https://www.youtube.com/watch?v=3CC4N4z3GJc) (YouTube)
- **Awesome XGBoost** : https://github.com/dmlc/xgboost/tree/master/demo

### Livres
- **"Hands-On Machine Learning"** (Géron, 2019) - Chapitre 7
- **"Applied Predictive Modeling"** (Kuhn & Johnson, 2013)
- **"Introduction to Boosted Trees"** (Tianqi Chen, 2014) - XGBoost creator

### Papers fondamentaux
- **XGBoost** : Chen & Guestrin, 2016 - "XGBoost: A Scalable Tree Boosting System"
- **LightGBM** : Ke et al., 2017 - "LightGBM: A Highly Efficient Gradient Boosting Decision Tree"
- **CatBoost** : Prokhorenkova et al., 2018 - "CatBoost: unbiased boosting with categorical features"

### Installation
```bash
# XGBoost
pip install xgboost

# LightGBM
pip install lightgbm

# CatBoost
pip install catboost

# SHAP (pour interprétabilité)
pip install shap

# Optuna (pour hyperparameter tuning)
pip install optuna
```

### Hyperparamètres clés à tuner

**Priorité 1 (plus impactants)** :
- `learning_rate` : 0.01-0.3 (0.1 = bon départ)
- `max_depth` : 3-10 (5 = bon départ)
- `n_estimators` : 100-1000 (plus = mieux, mais risque overfitting)

**Priorité 2** :
- `subsample` : 0.5-1.0 (0.8 recommandé)
- `colsample_bytree` : 0.5-1.0 (0.8 recommandé)

**Régularisation** :
- `reg_alpha` (L1) : 0-10
- `reg_lambda` (L2) : 0-10
