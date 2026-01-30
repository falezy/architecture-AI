# Logistic Regression

Classification linéaire (probabilités) pour binaire/multi-classe.

## Idée clé

La **régression logistique** est un modèle de **classification** (pas de régression malgré son nom !) qui prédit la **probabilité** qu'un exemple appartienne à une classe. Elle utilise la fonction **sigmoïde** pour transformer une combinaison linéaire en probabilité entre 0 et 1.

**Formule** :
```
z = β₀ + β₁·x₁ + β₂·x₂ + ... + βₙ·xₙ
P(y=1|x) = σ(z) = 1 / (1 + e^(-z))
```

- `z` : score linéaire (logit)
- `σ(z)` : fonction sigmoïde
- `P(y=1|x)` : probabilité que y=1 sachant x

**Fonction sigmoïde** :
```
      1 |           ________
        |         /
  P(y=1)|       /
        |     /
      0 |___/________________
        -∞    0    +∞
             z (logit)
```

**Propriétés** :
- Si `z → +∞` alors `P(y=1) → 1`
- Si `z → -∞` alors `P(y=1) → 0`
- Si `z = 0` alors `P(y=1) = 0.5`

**Décision** :
```
Si P(y=1) ≥ 0.5 → Classe 1
Sinon → Classe 0
```

**Différence avec régression linéaire** :
| Aspect | Régression linéaire | Régression logistique |
|--------|-------------------|----------------------|
| **Tâche** | Régression (prédire valeur continue) | Classification (prédire classe) |
| **Output** | Valeur réelle (-∞ à +∞) | Probabilité (0 à 1) |
| **Fonction** | Linéaire : `y = βx` | Sigmoïde : `P = σ(βx)` |
| **Loss** | MSE | Log Loss (Cross-Entropy) |

**Fonction de coût (Log Loss)** :
```
Loss = -[y·log(p) + (1-y)·log(1-p)]
```
- Pénalise fortement les mauvaises prédictions confiantes

## Exemples concrets

### 1. Classification binaire : Détection de spam

**Scénario** : Classifier un email comme spam (1) ou non-spam (0) selon le nombre de mots suspects et la longueur.

**Code Python avec scikit-learn** :
```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# 1. Données d'exemple
X = np.array([
    [5, 100],   # 5 mots suspects, 100 mots au total
    [2, 50],
    [15, 200],
    [1, 30],
    [20, 250],
    [3, 80],
    [18, 180],
    [0, 40],
    [25, 300],
    [8, 150]
])
y = np.array([0, 0, 1, 0, 1, 0, 1, 0, 1, 1])  # 0=non-spam, 1=spam

# 2. Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. Créer et entraîner le modèle
model = LogisticRegression(
    solver='lbfgs',      # Algorithme d'optimisation (L-BFGS)
    max_iter=1000,
    random_state=42
)
model.fit(X_train, y_train)

# 4. Coefficients
print(f"Intercept (β₀): {model.intercept_[0]:.3f}")
print(f"Coefficients: {model.coef_[0]}")
print(f"  β₁ (mots suspects): {model.coef_[0][0]:.3f}")
print(f"  β₂ (longueur): {model.coef_[0][1]:.3f}")

# 5. Prédire pour un nouvel email
nouvel_email = np.array([[10, 120]])  # 10 mots suspects, 120 mots
probabilite = model.predict_proba(nouvel_email)[0]
prediction = model.predict(nouvel_email)[0]

print(f"\nNouvel email: {nouvel_email[0]}")
print(f"Probabilité spam: {probabilite[1]:.2%}")
print(f"Prédiction: {'SPAM' if prediction == 1 else 'NON-SPAM'}")

# 6. Évaluation
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.2%}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.3f}")
print("\nMatrice de confusion:")
print(confusion_matrix(y_test, y_pred))
print("\nRapport de classification:")
print(classification_report(y_test, y_pred, target_names=['Non-spam', 'Spam']))
```

**Visualisation de la frontière de décision** :
```python
# Fonction pour tracer la frontière
def plot_decision_boundary(X, y, model):
    # Créer une grille
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 10, X[:, 1].max() + 10
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.1),
        np.arange(y_min, y_max, 1)
    )
    
    # Prédire pour chaque point
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Tracer
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='k')
    plt.xlabel('Nombre de mots suspects')
    plt.ylabel('Longueur du message')
    plt.title('Frontière de décision - Régression Logistique')
    plt.colorbar(label='Classe')
    plt.show()

plot_decision_boundary(X, y, model)
```

---

### 2. Courbe ROC et choix du seuil

**Code pour analyser les performances et ajuster le seuil** :
```python
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# 1. Obtenir les probabilités
y_proba = model.predict_proba(X_test)[:, 1]

# 2. Calculer la courbe ROC
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)

# 3. Tracer la courbe ROC
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Aléatoire')
plt.xlabel('Taux de faux positifs (FPR)')
plt.ylabel('Taux de vrais positifs (TPR)')
plt.title('Courbe ROC')
plt.legend()
plt.grid(True)
plt.show()

# 4. Ajuster le seuil (au lieu de 0.5 par défaut)
seuil_optimal = 0.3  # Exemple: favoriser le rappel (détecter plus de spam)
y_pred_seuil = (y_proba >= seuil_optimal).astype(int)

print(f"Avec seuil = 0.5:")
print(classification_report(y_test, model.predict(X_test)))

print(f"\nAvec seuil = {seuil_optimal}:")
print(classification_report(y_test, y_pred_seuil))
```

**Interprétation** :
- **AUC = 1.0** : Modèle parfait
- **AUC = 0.5** : Modèle aléatoire (ligne diagonale)
- **Seuil** : Ajuster selon le coût des faux positifs vs faux négatifs

---

### 3. Classification multi-classe : Diagnostic médical

**Scénario** : Classifier une maladie (A, B, C) selon la température et le pouls.

**Code Python** :
```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
import numpy as np

# 1. Données simulées (3 classes)
X, y = make_classification(
    n_samples=300,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_classes=3,
    n_clusters_per_class=1,
    random_state=42
)

# Labels: 0=Maladie A, 1=Maladie B, 2=Maladie C

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. Régression logistique multi-classe
model = LogisticRegression(
    multi_class='multinomial',  # One-vs-Rest ou multinomial
    solver='lbfgs',
    max_iter=1000
)
model.fit(X_train, y_train)

# 4. Prédiction avec probabilités pour chaque classe
nouveau_patient = np.array([[0.5, 1.2]])
probas = model.predict_proba(nouveau_patient)[0]
prediction = model.predict(nouveau_patient)[0]

print("Probabilités:")
for i, p in enumerate(probas):
    print(f"  Maladie {chr(65+i)}: {p:.2%}")
print(f"\nDiagnostic: Maladie {chr(65+prediction)}")

# 5. Évaluation
from sklearn.metrics import accuracy_score, classification_report
y_pred = model.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.2%}")
print("\nRapport de classification:")
print(classification_report(
    y_test, y_pred, 
    target_names=['Maladie A', 'Maladie B', 'Maladie C']
))
```

**Visualisation des frontières multi-classes** :
```python
def plot_multiclass_decision_boundary(X, y, model):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.02),
        np.arange(y_min, y_max, 0.02)
    )
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Frontières de décision - Classification multi-classe')
    plt.colorbar(label='Classe')
    plt.show()

plot_multiclass_decision_boundary(X, y, model)
```

---

### 4. Régularisation : Ridge (L2) et Lasso (L1)

**Code pour éviter le surapprentissage** :
```python
from sklearn.linear_model import LogisticRegression

# Données avec beaucoup de features (risque d'overfitting)
X, y = make_classification(n_samples=100, n_features=50, n_informative=10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 1. Sans régularisation
model_none = LogisticRegression(penalty=None, max_iter=1000)
model_none.fit(X_train, y_train)
print(f"Sans régularisation:")
print(f"  Train: {model_none.score(X_train, y_train):.2%}")
print(f"  Test: {model_none.score(X_test, y_test):.2%}")

# 2. Avec L2 (Ridge) - par défaut
model_l2 = LogisticRegression(penalty='l2', C=1.0, max_iter=1000)
model_l2.fit(X_train, y_train)
print(f"\nAvec L2 (C=1.0):")
print(f"  Train: {model_l2.score(X_train, y_train):.2%}")
print(f"  Test: {model_l2.score(X_test, y_test):.2%}")

# 3. Avec L1 (Lasso) - sélection de features
model_l1 = LogisticRegression(penalty='l1', C=1.0, solver='saga', max_iter=1000)
model_l1.fit(X_train, y_train)
print(f"\nAvec L1 (C=1.0):")
print(f"  Train: {model_l1.score(X_train, y_train):.2%}")
print(f"  Test: {model_l1.score(X_test, y_test):.2%}")
print(f"  Features sélectionnées: {np.sum(model_l1.coef_[0] != 0)}/{X.shape[1]}")
```

**Paramètre C** :
- `C` grand (ex: 100) → Peu de régularisation (peut overfitter)
- `C` petit (ex: 0.01) → Forte régularisation (peut underfitter)
- `C = 1.0` → Bon point de départ

## Quand l'utiliser

- ✅ **Classification binaire** : Spam/non-spam, fraude/légal, malade/sain
- ✅ **Probabilités nécessaires** : Besoin de `P(y=1)` plutôt qu'une simple classe
- ✅ **Interprétabilité** : Comprendre l'impact de chaque feature (coefficients)
- ✅ **Baseline** : Modèle simple et rapide pour commencer
- ✅ **Données linéairement séparables** : Classes séparables par une ligne/hyperplan
- ✅ **Peu de données** : Fonctionne bien avec petits datasets (contrairement aux deep learning)

**Cas d'usage typiques** :
- 🏥 **Médecine** : Diagnostic (malade/sain), risque de réadmission
- 💳 **Finance** : Approbation de prêt, détection de fraude, défaut de paiement
- 📧 **Marketing** : Classification spam, prédiction de clic (CTR), churn
- 🎓 **Éducation** : Prédiction de réussite/échec d'un étudiant
- 🔐 **Sécurité** : Détection d'intrusion, authentification

## Forces

✅ **Simple et rapide** : Entraînement très rapide, peu de ressources  
✅ **Interprétable** : Coefficients indiquent l'impact de chaque feature  
✅ **Probabilités calibrées** : Donne des probabilités (pas juste des classes)  
✅ **Peu de données** : Fonctionne avec petits datasets  
✅ **Régularisation intégrée** : L1/L2 pour éviter overfitting  
✅ **Multi-classe natif** : One-vs-Rest ou Multinomial  
✅ **Pas de tuning** : Peu d'hyperparamètres (contrairement à XGBoost)

**Exemple d'interprétabilité** :
```python
# Comprendre l'impact des features
coefficients = model.coef_[0]
features = ['Mots suspects', 'Longueur']

for feature, coef in zip(features, coefficients):
    impact = "augmente" if coef > 0 else "diminue"
    print(f"{feature}: {impact} la probabilité de spam de {abs(coef):.3f}")
    
# Output:
# Mots suspects: augmente la probabilité de spam de 0.245
# Longueur: augmente la probabilité de spam de 0.018
```

## Limites

❌ **Hypothèse de linéarité** : Assume une frontière linéaire (ligne droite)  
❌ **Features engineering** : Nécessite de créer des features pertinentes  
❌ **Pas pour relations complexes** : XOR, frontières circulaires difficiles  
❌ **Sensible aux outliers** : Peut biaiser la frontière de décision  
❌ **Multicolinéarité** : Problème si features corrélées  
❌ **Pas adapté aux images/texte brut** : Mieux avec features extraites  
❌ **Déséquilibre de classes** : Requiert `class_weight='balanced'`

**Exemple de limitation (XOR problem)** :
```python
# Problème XOR (non-linéairement séparable)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])  # XOR

model = LogisticRegression()
model.fit(X, y)
print(f"Accuracy: {model.score(X, y):.2%}")  # ~50% (aléatoire!)

# Solution: Ajouter des features non-linéaires
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)  # Ajoute x1*x2, x1², x2²

model_poly = LogisticRegression()
model_poly.fit(X_poly, y)
print(f"Accuracy avec features polynomiales: {model_poly.score(X_poly, y):.2%}")  # 100%
```

**Gérer le déséquilibre de classes** :
```python
# Dataset déséquilibré: 90% classe 0, 10% classe 1
model = LogisticRegression(
    class_weight='balanced',  # Pénalise plus les erreurs sur classe minoritaire
    max_iter=1000
)
model.fit(X_train, y_train)
```

## Variantes / liens

### Solveurs (algorithmes d'optimisation)

```python
LogisticRegression(
    solver='...',  # Choix du solveur
    max_iter=1000
)
```

| Solveur | Régularisation | Vitesse | Multi-classe | Quand l'utiliser |
|---------|---------------|---------|--------------|------------------|
| **lbfgs** | L2 | Rapide | ✅ | **Défaut** : petites/moyennes données |
| **liblinear** | L1, L2 | Moyen | ❌ (OvR) | Grandes données + binaire |
| **saga** | L1, L2, ElasticNet | Lent | ✅ | Très grandes données |
| **newton-cg** | L2 | Rapide | ✅ | Peu de features |
| **sag** | L2 | Rapide | ✅ | Grandes données |

**Recommandation** :
- Données < 10,000 → `lbfgs` (défaut)
- Données > 100,000 → `saga` ou `sag`
- Besoin de L1 (feature selection) → `saga` ou `liblinear`

### Multi-classe : One-vs-Rest vs Multinomial

```python
# One-vs-Rest (OvR): N modèles binaires (1 par classe)
model_ovr = LogisticRegression(multi_class='ovr')

# Multinomial: 1 modèle avec softmax
model_multi = LogisticRegression(multi_class='multinomial', solver='lbfgs')
```

**Différences** :
- **OvR** : Plus simple, plus rapide, mais probabilités non calibrées
- **Multinomial** : Meilleur pour probabilités, plus lent

### Relations avec d'autres modèles

- **Régression linéaire** : Version régression (output continu)
- **Perceptron** : Ancêtre sans probabilités (classification binaire)
- **SVM** : Classification avec marge (plus robuste aux outliers)
- **Naive Bayes** : Classification probabiliste (suppose indépendance)
- **Neural Network** : Généralisation avec couches cachées
- **Softmax Regression** : Extension multi-classe (équivalent à multinomial logistic)

### Métriques d'évaluation

```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss
)

# Prédictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Métriques
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall: {recall_score(y_test, y_pred):.3f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.3f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.3f}")
print(f"Log Loss: {log_loss(y_test, y_proba):.3f}")
```

**Quelle métrique choisir ?**
- **Accuracy** : Données équilibrées
- **Precision** : Minimiser faux positifs (ex: spam)
- **Recall** : Minimiser faux négatifs (ex: détection de cancer)
- **F1-Score** : Compromis precision/recall
- **AUC-ROC** : Performance globale (invariant au seuil)

## Références

### Documentation et tutoriels
- **Scikit-learn** : [Logistic Regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
- **StatQuest** : [Logistic Regression Explained](https://www.youtube.com/watch?v=yIYKR4sgzI8) (YouTube)
- **Andrew Ng** : [ML Course - Classification](https://www.coursera.org/learn/machine-learning)

### Livres
- **"An Introduction to Statistical Learning"** (James et al., 2021) - Chapitre 4
- **"The Elements of Statistical Learning"** (Hastie et al., 2009) - Chapitre 4
- **"Pattern Recognition and Machine Learning"** (Bishop, 2006) - Chapitre 4

### Papers et théorie
- **Logistic function** : Pierre François Verhulst, 1838 (fonction sigmoïde)
- **Maximum Likelihood Estimation** : R.A. Fisher, 1922
- **Cross-Entropy Loss** : Kullback-Leibler divergence

### Outils Python
```python
# Scikit-learn (le plus populaire)
from sklearn.linear_model import LogisticRegression

# Statsmodels (plus de statistiques)
import statsmodels.api as sm
model = sm.Logit(y, X).fit()
print(model.summary())  # P-values, odds ratios, etc.

# PyTorch (deep learning framework)
import torch.nn as nn
model = nn.Sequential(
    nn.Linear(n_features, 1),
    nn.Sigmoid()
)
```

### Hyperparamètres clés

```python
LogisticRegression(
    penalty='l2',           # 'l1', 'l2', 'elasticnet', None
    C=1.0,                  # Inverse de la régularisation (plus grand = moins de régularisation)
    solver='lbfgs',         # 'lbfgs', 'liblinear', 'saga', 'newton-cg', 'sag'
    max_iter=100,           # Nombre max d'itérations
    multi_class='auto',     # 'ovr', 'multinomial', 'auto'
    class_weight=None,      # 'balanced' pour classes déséquilibrées
    random_state=42
)
```

**Tuning de C** (régularisation) :
```python
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(X_train, y_train)
print(f"Meilleur C: {grid.best_params_['C']}")
```
