# Naive Bayes

Classifieur probabiliste simple, efficace sur texte.

## Idée clé

**Naive Bayes** est un classifieur probabiliste basé sur le **théorème de Bayes** avec l'hypothèse "naïve" que toutes les features sont **indépendantes** entre elles. Malgré cette hypothèse souvent fausse, il fonctionne remarquablement bien en pratique, surtout pour la classification de texte.

**Théorème de Bayes** :
```
P(Classe|Features) = P(Features|Classe) · P(Classe) / P(Features)
```

Simplifié pour la classification :
```
P(C|x₁,x₂,...,xₙ) ∝ P(C) · P(x₁|C) · P(x₂|C) · ... · P(xₙ|C)
```

- `P(C)` : **Prior** (probabilité a priori de la classe)
- `P(xᵢ|C)` : **Likelihood** (probabilité de la feature sachant la classe)
- `P(C|x)` : **Posterior** (probabilité de la classe sachant les features)

**Hypothèse "naïve" (indépendance conditionnelle)** :
```
P(x₁,x₂,...,xₙ|C) = P(x₁|C) · P(x₂|C) · ... · P(xₙ|C)
```

**Décision** :
```
Classe prédite = argmax P(C) · ∏ P(xᵢ|C)
                    C        i
```

### Exemple simple : Méteo et Tennis

**Question** : Jouer au tennis selon la météo ?

| Météo | Jouer Tennis |
|-------|--------------|
| Soleil | Oui |
| Pluie | Non |
| Nuageux | Oui |
| Soleil | Oui |
| Pluie | Non |

**Calculer** : P(Oui | Soleil) vs P(Non | Soleil)

```
P(Oui | Soleil) ∝ P(Soleil | Oui) · P(Oui)
                = (2/3) · (3/5) = 0.4

P(Non | Soleil) ∝ P(Soleil | Non) · P(Non)
                = (0/2) · (2/5) = 0

→ Prédiction : OUI
```

## Exemples concrets

### 1. Classification de texte : Détection de spam

**Scénario** : Classifier  emails comme spam/non-spam selon les mots présents.

**Code Python avec Multinomial Naive Bayes** :
```python
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Données d'exemple
emails = [
    "Win free money now",
    "Meeting at 3pm tomorrow",
    "Claim your prize today",
    "Project deadline next week",
    "Congratulations you won",
    "Lunch with team on Friday",
    "Limited offer act now",
    "Review the quarterly report",
    "Get rich quick scheme",
    "Conference call at 2pm"
]
labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1=spam, 0=non-spam

# 2. Convertir texte en features (bag-of-words)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)
y = np.array(labels)

print(f"Vocabulaire: {vectorizer.get_feature_names_out()[:10]}...")
print(f"Matrice features shape: {X.shape}")

# 3. Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 4. Entraîner Multinomial Naive Bayes
model = MultinomialNB(alpha=1.0)  # alpha = lissage de Laplace
model.fit(X_train, y_train)

# 5. Prédire pour un nouvel email
nouvel_email = ["Win a free vacation now"]
X_new = vectorizer.transform(nouvel_email)
prediction = model.predict(X_new)[0]
proba = model.predict_proba(X_new)[0]

print(f"\nNouvel email: {nouvel_email[0]}")
print(f"Prédiction: {'SPAM' if prediction == 1 else 'NON-SPAM'}")
print(f"Probabilité spam: {proba[1]:.2%}")

# 6. Évaluation
y_pred = model.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.2%}")
print("\nMatrice de confusion:")
print(confusion_matrix(y_test, y_pred))
print("\nRapport de classification:")
print(classification_report(y_test, y_pred, target_names=['Non-spam', 'Spam']))

# 7. Probabilités par classe (log probabilities)
print("\nLog probabilités des mots (spam):")
feature_names = vectorizer.get_feature_names_out()
log_probs = model.feature_log_prob_[1]  # Classe spam
top_indices = np.argsort(log_probs)[-5:]  # Top 5 mots
for idx in top_indices:
    print(f"  {feature_names[idx]}: {np.exp(log_probs[idx]):.3f}")
```

---

### 2. Sentiment Analysis avec TF-IDF

**Scénario** : Classifier des avis clients comme positifs ou négatifs.

**Code Python** :
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# 1. Données
avis = [
    "Ce produit est excellent, je recommande",
    "Très déçu, qualité médiocre",
    "Parfait, correspond à mes attentes",
    "Arnaque, ne fonctionne pas du tout",
    "Superbe qualité, excellent rapport qualité-prix",
    "Service client horrible, produit cassé",
    "Incroyable, meilleur achat de l'année",
    "À éviter absolument, perte de temps",
]
sentiments = [1, 0, 1, 0, 1, 0, 1, 0]  # 1=positif, 0=négatif

# 2. Pipeline: TF-IDF + Naive Bayes
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=100)),  # TF-IDF au lieu de CountVectorizer
    ('nb', MultinomialNB())
])

# 3. Entraîner
pipeline.fit(avis, sentiments)

# 4. Prédire
nouveaux_avis = [
    "Produit de très bonne qualité",
    "Complètement nul, je regrette"
]

for avis_text in nouveaux_avis:
    prediction = pipeline.predict([avis_text])[0]
    proba = pipeline.predict_proba([avis_text])[0]
    sentiment = "POSITIF" if prediction == 1 else "NÉGATIF"
    print(f"\nAvis: {avis_text}")
    print(f"Sentiment: {sentiment} (confiance: {max(proba):.2%})")
```

---

### 3. Gaussian Naive Bayes : Classification numérique

**Scénario** : Classifier des fleurs (Iris dataset) selon longueur/largeur des pétales.

**Code Python** :
```python
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 1. Charger dataset Iris
iris = load_iris()
X = iris.data[:, :2]  # Utiliser seulement 2 features pour visualisation
y = iris.target

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. Gaussian Naive Bayes (pour features continues)
model = GaussianNB()
model.fit(X_train, y_train)

# 4. Prédire
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print("\nRapport de classification:")
print(classification_report(
    y_test, y_pred, 
    target_names=iris.target_names
))

# 5. Probabilités pour une nouvelle fleur
nouvelle_fleur = [[5.1, 3.5]]  # Longueur/largeur sépale
probas = model.predict_proba(nouvelle_fleur)[0]
print("\nNouvelle fleur:", nouvelle_fleur[0])
for i, classe in enumerate(iris.target_names):
    print(f"  P({classe}): {probas[i]:.2%}")
```

**Visualisation des frontières de décision** :
```python
import matplotlib.pyplot as plt
import numpy as np

def plot_decision_boundary_gaussian(X, y, model):
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
    plt.title('Gaussian Naive Bayes - Frontières de décision')
    plt.colorbar()
    plt.show()

plot_decision_boundary_gaussian(X, y, model)
```

---

### 4. Comparaison des 3 variantes

**Code pour comparer Gaussian, Multinomial, Bernoulli** :
```python
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.preprocessing import MinMaxScaler

# Données synthétiques
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Normaliser pour Multinomial (features positives)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Comparer les 3 variantes
models = {
    'Gaussian': GaussianNB(),
    'Multinomial': MultinomialNB(),
    'Bernoulli': BernoulliNB()
}

for name, model in models.items():
    if name == 'Gaussian':
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
    else:
        model.fit(X_train_scaled, y_train)
        score = model.score(X_test_scaled, y_test)
    
    print(f"{name} NB: {score:.2%}")
```

## Quand l'utiliser

- ✅ **Classification de texte** : Spam, sentiment analysis, catégorisation de documents
- ✅ **Données haute dimensionnalité** : Nombreuses features (ex: bag-of-words)
- ✅ **Baseline rapide** : Entraînement très rapide, bon point de départ
- ✅ **Temps réel** : Prédictions instantanées, faible latence
- ✅ **Peu de données d'entraînement** : Fonctionne avec petits datasets
- ✅ **Interprétabilité** : Probabilités faciles à comprendre
- ✅ **Online learning** : Mise à jour incrémentale possible (`partial_fit`)

**Cas d'usage typiques** :
- 📧 **Email** : Filtrage de spam, classification automatique
- 💬 **NLP** : Sentiment analysis, classification de topics, détection de langue
- 📰 **Médias** : Catégorisation d'articles, recommandations
- 🏥 **Médecine** : Diagnostic basé sur symptômes (si indépendance raisonnable)
- 🔐 **Sécurité** : Détection d'intrusion, classification de malware

**Quand NE PAS utiliser** :
- ❌ Features fortement corrélées (viole l'hypothèse d'indépendance)
- ❌ Besoin de performance maximale sur données tabulaires → XGBoost
- ❌ Relations complexes non-linéaires → Deep Learning

## Forces

✅ **Très rapide** : Entraînement et prédiction quasi-instantanés  
✅ **Peu de données** : Fonctionne bien avec petits datasets  
✅ **Scalable** : Gère bien grandes dimensions (millions de features)  
✅ **Pas d'hyperparamètres** : Juste le lissage de Laplace (alpha)  
✅ **Probabilités** : Fournit des probabilités calibrées  
✅ **Multi-classe natif** : Pas besoin de One-vs-Rest  
✅ **Online learning** : Mise à jour incrémentale avec `partial_fit()`

**Exemple de vitesse** :
```python
import time
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Dataset texte (18,000 documents)
newsgroups = fetch_20newsgroups(subset='all')
X = TfidfVectorizer(max_features=10000).fit_transform(newsgroups.data)
y = newsgroups.target

# Entraînement
start = time.time()
model = MultinomialNB()
model.fit(X, y)
print(f"Temps d'entraînement: {time.time() - start:.2f}s")  # ~0.1s !

# Prédiction
start = time.time()
predictions = model.predict(X[:1000])
print(f"Temps prédiction (1000 docs): {time.time() - start:.4f}s")  # ~0.001s !
```

## Limites

❌ **Hypothèse d'indépendance** : Rarement vraie (features souvent corrélées)  
❌ **Performance limitée** : Moins bon que XGBoost/Neural Nets sur données tabulaires  
❌ **Sensible aux features inutiles** : Toutes les features affectent le score  
❌ **Zero-frequency problem** : Mot jamais vu → P=0 (résolu par lissage)  
❌ **Pas de régularisation** : Peut overfitter avec trop de features  
❌ **Relations complexes** : Ne capture pas interactions entre features  
❌ **Calibration des probabilités** : Probas peuvent être biaisées

**Exemple du problème zero-frequency** :
```python
# Si "gratuit" n'apparaît jamais dans les emails non-spam
# → P("gratuit" | non-spam) = 0
# → P(non-spam | email avec "gratuit") = 0 (même si autres mots légitimes)

# Solution: Lissage de Laplace (alpha)
model = MultinomialNB(alpha=1.0)  # Ajoute pseudo-count de 1 partout

# alpha=0 → Pas de lissage (risque zero-frequency)
# alpha=1 → Lissage de Laplace (défaut, recommandé)
# alpha>1 → Lissage fort (plus conservateur)
```

**Impact de l'hypothèse d'indépendance** :
```python
# Exemple: "Bon" et "Excellent" sont corrélés (souvent ensemble)
# Naive Bayes compte leur co-occurrence 2 fois (surpoids)
# → Probabilités biaisées mais souvent classification correcte quand même !
```

## Variantes / liens

### Les 3 variantes principales

**1. Gaussian Naive Bayes** - Features continues (distribution normale)
```python
from sklearn.naive_bayes import GaussianNB

model = GaussianNB(
    var_smoothing=1e-9  # Lissage de la variance (stabilité numérique)
)
```

**Quand** : Features numériques continues (température, taille, poids, etc.)  
**Hypothèse** : Chaque feature suit une distribution normale (Gaussienne)

**2. Multinomial Naive Bayes** - Comptages discrets (texte)
```python
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB(
    alpha=1.0,       # Lissage de Laplace (défaut: 1.0)
    fit_prior=True   # Apprendre P(C) des données (défaut: True)
)
```

**Quand** : Bag-of-words, TF-IDF, fréquences de mots  
**Hypothèse** : Features représentent des comptages (entiers positifs)

**3. Bernoulli Naive Bayes** - Features binaires
```python
from sklearn.naive_bayes import BernoulliNB

model = BernoulliNB(
    alpha=1.0,
    binarize=0.0  # Seuil pour binariser les features (0.0 = déjà binaires)
)
```

**Quand** : Features binaires (présence/absence de mots)  
**Hypothèse** : Chaque feature est 0 ou 1

### Tableau comparatif

| Variante | Type de features | Distribution | Use case | Exemple |
|----------|-----------------|--------------|----------|---------|
| **Gaussian** | Continues | Normale | Données numériques | Iris, température |
| **Multinomial** | Comptages | Multinomiale | Fréquences de mots | TF-IDF, bag-of-words |
| **Bernoulli** | Binaires | Bernoulli | Présence/absence | Document contient "free"? |

### Online Learning (partial_fit)

**Entraînement incrémental sur flux de données** :
```python
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()

# Entraîner par batches
for batch_X, batch_y in data_stream:
    model.partial_fit(
        batch_X, 
        batch_y, 
        classes=np.array([0, 1])  # Classes possibles (requis au 1er appel)
    )

# Prédire en continu
predictions = model.predict(new_data)
```

**Avantage** : Mise à jour du modèle sans tout ré-entraîner (adaptatif)

### Relations avec d'autres modèles

- **Régression logistique** : Modèle discriminatif (vs NB = génératif)
- **LDA** (Linear Discriminant Analysis) : Similaire mais assume covariance partagée
- **K-Nearest Neighbors** : Autre baseline rapide mais plus lent en prédiction
- **TF-IDF + Cosine Similarity** : Alternative pour classification texte
- **Deep Learning (BERT)** : Meilleur pour texte mais beaucoup plus lent

### Preprocessing pour texte

**Pipeline complet** :
```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=5000,      # Top 5000 mots
        stop_words='english',   # Retirer mots vides
        ngram_range=(1, 2),     # Unigrammes + bigrammes
        min_df=2,               # Ignorer mots trop rares
        max_df=0.8              # Ignorer mots trop fréquents
    )),
    ('nb', MultinomialNB(alpha=1.0))
])

pipeline.fit(texts_train, y_train)
predictions = pipeline.predict(texts_test)
```

## Références

### Documentation
- **Scikit-learn** : [Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html)
- **StatQuest** : [Naive Bayes Explained](https://www.youtube.com/watch?v=O2L2Uv9pdDA) (YouTube)

### Livres
- **"An Introduction to Statistical Learning"** (James et al., 2021) - Chapitre 4
- **"Pattern Recognition and Machine Learning"** (Bishop, 2006) - Chapitre 8
- **"Speech and Language Processing"** (Jurafsky & Martin, 2023) - Chapitre 4

### Papers fondamentaux
- **Bayes' Theorem** : Thomas Bayes, 1763 (posthume)
- **"Naive Bayes at Forty"** (Lewis, 1998) - Analyse de performance
- **"Spam Filtering with Naive Bayes"** (Sahami et al., 1998)

### Théorème de Bayes

**Formulation complète** :
```
P(A|B) = P(B|A) · P(A) / P(B)

Où:
- P(A|B) : Posterior (ce qu'on cherche)
- P(B|A) : Likelihood (vraisemblance)
- P(A) : Prior (probabilité a priori)
- P(B) : Evidence (normalisation)
```

**Exemple médical** :
```
Maladie M, Test T positif
P(M|T+) = P(T+|M) · P(M) / P(T+)

- P(M) = 0.01 (1% population malade)
- P(T+|M) = 0.99 (sensibilité: 99%)
- P(T+|¬M) = 0.05 (5% faux positifs)

P(T+) = P(T+|M)·P(M) + P(T+|¬M)·P(¬M)
      = 0.99·0.01 + 0.05·0.99 = 0.0594

P(M|T+) = 0.99·0.01 / 0.0594 = 0.167 (16.7%)

→ Même avec test positif, seulement 16.7% de chance d'être malade !
```

### Tuning des hyperparamètres

```python
from sklearn.model_selection import GridSearchCV

# Grid search pour alpha (lissage)
param_grid = {
    'alpha': [0.1, 0.5, 1.0, 2.0, 5.0]
}

grid = GridSearchCV(
    MultinomialNB(), 
    param_grid, 
    cv=5, 
    scoring='accuracy'
)
grid.fit(X_train, y_train)

print(f"Meilleur alpha: {grid.best_params_['alpha']}")
print(f"Meilleur score: {grid.best_score_:.3f}")
```

### Comparaison performance (dataset 20newsgroups)

```
Algorithme                 Accuracy    Temps
Naive Bayes (Multinomial)    77%      0.1s
Logistic Regression          82%      2.5s
SVM (Linear)                 83%      8.2s
Random Forest                75%      15s
XGBoost                      84%      25s

→ NB: 2e meilleur rapport performance/vitesse !
```
