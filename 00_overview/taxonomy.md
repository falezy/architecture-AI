# Taxonomy — AI / ML / DL / RL
Cette taxonomie classe les familles de modèles IA et propose une grille simple pour les comparer (performance, données, coûts, interprétabilité).

---

## 1) Classification par paradigme d’apprentissage

### 1.1 Apprentissage supervisé
**But :** apprendre une fonction entrée → sortie à partir de données étiquetées.  
**Exemples :**
- Régression : Linear Regression, Random Forest, Gradient Boosting, MLP
- Classification : Logistic Regression, SVM, Trees/Boosting, CNN/Transformers

**Quand :** tu as des labels fiables et une métrique claire.  
**Forces :** efficace et direct.  
**Limites :** dépend fortement de la qualité des labels et du dataset.

### 1.2 Apprentissage non supervisé
**But :** trouver structure/compactage sans labels.  
**Exemples :** k-means, GMM, clustering hiérarchique, PCA, UMAP/t-SNE.

**Quand :** exploration, segmentation, réduction de dimension.  
**Forces :** utile pour découvrir.  
**Limites :** validation moins directe, résultats sensibles aux paramètres.

### 1.3 Apprentissage auto-supervisé / self-supervised
**But :** créer un signal d’apprentissage sans labels manuels.  
**Exemples :**
- NLP : masked language modeling
- Vision : contrastive learning
- Pretraining Transformers / embeddings

**Quand :** gros volumes de données brutes + besoin de représentations réutilisables.

### 1.4 Apprentissage par renforcement (RL)
**But :** apprendre à agir via récompense (séquences décisionnelles).  
**Exemples :** Q-learning, DQN, PPO, SAC, TD3; cadres MDP/POMDP.

**Quand :** problème “action → conséquence” (robot, jeux, allocation dynamique).  
**Limites :** coûteux en données/simulation, stabilité, tuning.

---

## 2) Classification par type de données

### 2.1 Tabulaire
**Meilleurs baselines :** Gradient Boosting, Random Forest, Logistic/Linear.  
**Deep learning :** utile si très gros dataset ou features apprises, sinon souvent moins bon que boosting.

### 2.2 Texte
- Compréhension : Transformers encodeurs (BERT-like)
- Génération : Transformers décodeurs (GPT-like)
- Similarité : embeddings + recherche vectorielle

### 2.3 Images / vision
- CNN / Vision Transformers
- Génération : diffusion

### 2.4 Séries temporelles
- Classiques : ARIMA, Prophet
- Deep : RNN/Transformers time-series

### 2.5 Graphes
- GNN si la structure relationnelle est essentielle

---

## 3) Classification par objectif

### 3.1 Prédiction (predictive)
Régression / classification : modèle prédit une sortie.

### 3.2 Description / structure (descriptive)
Clustering, réduction de dimension, estimation de densité.

### 3.3 Génération (generative)
GAN, VAE, diffusion, LLM : produire données similaires (texte/images/etc.).

### 3.4 Décision séquentielle (control)
RL, planification, POMDP : choisir des actions au fil du temps.

---

## 4) Classification par “famille de modèles”

### 4.1 Modèles linéaires
Linear/Logistic Regression, linear SVM.  
**+** rapides, interprétables, solides en baseline.  
**-** limites sur relations non linéaires sans features.

### 4.2 Arbres et ensembles
Decision Trees, Random Forest (bagging), Gradient Boosting (boosting).  
**+** très fort sur tabulaire, robuste.  
**-** moins adapté aux données brutes (texte/image) sans embeddings.

### 4.3 Méthodes à noyaux / marges
SVM kernels.  
**+** bien sur petits datasets.  
**-** scale parfois difficile.

### 4.4 Réseaux neuronaux (DL)
MLP, CNN, RNN/LSTM/GRU, Transformers, GNN.  
**+** excellent sur données non structurées (texte, image, audio).  
**-** coût, tuning, interprétabilité.

### 4.5 Modèles probabilistes
Bayesian Networks, HMM, Kalman Filters, Gaussian Processes.  
**+** incertitude explicite, interprétation probabiliste.  
**-** complexité de modélisation, hypothèses.

---

# Grille de comparaison (critères)

## A) Données
1. Type (tabulaire/texte/image/série/graph)
2. Taille dataset (petit/moyen/très grand)
3. Qualité labels (bruit, biais, coût)

## B) Objectif
4. Régression vs classification vs génération vs contrôle
5. Coût d’erreur (faux positifs/faux négatifs, risques)

## C) Performance & robustesse
6. Métriques (AUC/F1/RMSE, calibration)
7. Robustesse au shift (distribution change)
8. Sensibilité au bruit / outliers

## D) Contraintes produit
9. Latence (temps réel ?)
10. Ressources (CPU/GPU, mémoire)
11. Scalabilité (volume, throughput)

## E) Interprétabilité & gouvernance
12. Explicabilité nécessaire ?
13. Audit/fairness/biais
14. Reproductibilité (pipeline, seeds, versions)

## F) Déploiement & maintenance
15. Fréquence de retraining
16. Monitoring (drift, perf, calibration)
17. Coût opérationnel (MLOps)

---

## Conseils rapides
- Si **tabulaire** → commence par **Gradient Boosting**.
- Si **texte** → embeddings/Transformers selon le besoin.
- Si **décision séquentielle** → RL seulement si c’est vraiment nécessaire (sinon heuristiques/planification).
- Si **incertitude** critique → probabiliste + calibration.
