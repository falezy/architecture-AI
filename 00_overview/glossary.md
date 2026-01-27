# Glossary — AI / ML / DL / RL

Glossaire pratique des termes qui reviennent le plus souvent quand on parle de modèles et méthodes en intelligence artificielle.

---

## A

### Accuracy
Proportion de prédictions correctes (attention : trompeur si classes déséquilibrées).

### Activation (fonction d’activation)
Fonction non-linéaire dans un réseau (ReLU, sigmoid, tanh…) qui permet d’apprendre des relations complexes.

### Agent
Système qui perçoit un environnement, choisit des actions et poursuit un objectif (souvent en RL).

### AUC (ROC-AUC)
Mesure de performance pour classification binaire basée sur la courbe ROC (robuste au seuil).

---

## B

### Backpropagation
Algorithme de calcul de gradients (rétropropagation) pour entraîner les réseaux neuronaux.

### Bagging
Technique d’ensemble : entraîner plusieurs modèles sur des échantillons bootstrap et agréger (ex. Random Forest).

### Batch / Mini-batch
Sous-ensemble de données utilisé pour calculer un gradient à chaque étape d’entraînement.

### Bias (biais)
- **Biais statistique** : erreur systématique d’un modèle.
- **Biais d’équité** : performance inégale selon des groupes (fairness).

### Bias–Variance tradeoff
Compromis : modèles simples → plus de biais, modèles complexes → plus de variance (surapprentissage).

---

## C

### Calibration
Qualité des probabilités prédites (ex : une prédiction à 0.8 doit être correcte ~80% du temps).

### Categorical cross-entropy
Fonction de perte utilisée en classification multi-classe.

### Classification
Prédire une étiquette (binaire, multi-classe, multi-label).

### Clustering
Regrouper des données non étiquetées (k-means, GMM, clustering hiérarchique).

### Confusion matrix
Table des vrais positifs, faux positifs, vrais négatifs, faux négatifs.

### Cross-validation (CV)
Évaluer un modèle en le testant sur plusieurs splits (k-fold, stratified…).

---

## D

### Data leakage
Le modèle “voit” de l’info du test pendant l’entraînement (via features, preprocessing, split incorrect). Garantit de mauvaises surprises en prod.

### Deep Learning (DL)
Sous-domaine du ML basé sur des réseaux neuronaux profonds.

### Diffusion model
Modèle génératif entraîné à débruiter progressivement (très utilisé pour images).

### Dimensionality reduction
Réduction de dimension (PCA, UMAP, t-SNE) pour compression/visualisation.

---

## E

### Embedding
Représentation vectorielle dense d’un objet (mot, phrase, image, utilisateur…). Sert à la similarité, clustering, retrieval.

### Ensemble
Combinaison de plusieurs modèles (bagging, boosting, stacking).

### Epoch
Un passage complet sur les données d’entraînement.

### Exploration vs Exploitation (RL)
- **Exploration** : tester des actions pour apprendre.
- **Exploitation** : choisir la meilleure action connue.

---

## F

### Feature
Variable d’entrée utilisée par le modèle (signal, attribut, transformation).

### Fine-tuning
Ré-entraînement d’un modèle pré-entraîné sur une tâche/domaine spécifique.

### F1-score
Moyenne harmonique de précision et rappel (utile avec classes déséquilibrées).

---

## G

### GAN (Generative Adversarial Network)
Modèle génératif avec générateur vs discriminateur.

### Generalization
Capacité à bien fonctionner sur des données nouvelles (non vues).

### Gradient
Dérivée de la loss par rapport aux paramètres (poids). Utilisé pour l’optimisation.

### Gradient Boosting
Ensemble séquentiel où chaque modèle corrige les erreurs du précédent (XGBoost, LightGBM…).

### GNN (Graph Neural Network)
Réseau neuronal pour données en graphe (nœuds/arêtes), via message passing.

---

## H

### Hyperparameters
Paramètres choisis par l’humain (learning rate, profondeur, régularisation, etc.), pas appris directement par le modèle.

---

## L

### Learning rate
Taille du pas de mise à jour des paramètres (critique pour stabilité/rapidité).

### Loss (fonction de perte)
Mesure à minimiser pendant l’entraînement (MSE, cross-entropy…).

### LLM (Large Language Model)
Grand modèle de langage (souvent basé sur Transformers) entraîné sur de gros corpus texte.

---

## M

### Machine Learning (ML)
Méthodes qui apprennent des patterns à partir de données (supervisé, non-supervisé, RL…).

### MDP (Markov Decision Process)
Cadre RL : états, actions, transitions, récompenses, politique.

### Model-based RL
RL où l’agent apprend/utilise un modèle de la dynamique pour planifier.

### Momentum
Technique d’optimisation pour lisser et accélérer la descente de gradient.

---

## N

### Normalization / Standardization
Mise à l’échelle des features (z-score, min-max) pour stabiliser l’apprentissage.

---

## O

### Overfitting (surapprentissage)
Modèle trop adapté aux données d’entraînement, performance faible en généralisation.

### Optimizer
Algorithme qui met à jour les poids (SGD, Adam, RMSProp…).

---

## P

### Precision / Recall
- **Precision** : parmi les prédits positifs, combien sont corrects ?
- **Recall** : parmi les vrais positifs, combien sont détectés ?

### Pretraining
Entraîner un modèle sur une tâche générale (souvent auto-supervisée) avant adaptation.

### POMDP
Version RL avec observabilité partielle : l’agent maintient une croyance sur l’état.

### Policy (politique)
Stratégie de l’agent en RL : mapping état → action.

### PPO
Algorithme RL policy-gradient stable, très utilisé.

---

## R

### Random Forest
Ensemble d’arbres (bagging) robuste, baseline solide.

### Regularization
Techniques pour réduire le surapprentissage (L1/L2, dropout, early stopping).

### Reinforcement Learning (RL)
Apprentissage par interaction : l’agent maximise une récompense cumulée.

### Representation learning
Apprendre des représentations (embeddings) utiles à plusieurs tâches.

---

## S

### Self-supervised learning
Apprentissage sans labels manuels via tâches de prétexte (masked token, contrastif…).

### Softmax
Transforme des scores en distribution de probabilité multi-classe.

### SVM
Classifieur à marge maximale (linéaire ou kernel).

---

## T

### Transformer
Architecture de réseau neuronal à attention (self-attention) pour séquences (texte, audio, vision…).

### Training / Validation / Test
- **Train** : apprendre les paramètres
- **Val** : choisir hyperparamètres / early stopping
- **Test** : évaluer final (une seule fois idéalement)

---

## U

### Unsupervised learning
Apprendre sans labels (clustering, réduction de dimension, densité…).

---

## V

### Validation set
Données séparées pour contrôler le tuning et éviter de tricher sur le test.

---

## Notes rapides
- **IA** : terme large qui englobe ML, DL, RL, planification, systèmes symboliques…
- **ML** : apprend depuis les données.
- **DL** : ML avec réseaux profonds.
- **RL** : apprentissage par récompense via interaction.
