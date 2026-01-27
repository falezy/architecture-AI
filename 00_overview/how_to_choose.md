# How to choose — Quel modèle / approche IA choisir ?

Ce guide te donne une méthode simple et pratique pour choisir une famille de modèles (ML, DL, RL, probabiliste) selon ton problème, tes données, tes contraintes et ton objectif (prototype rapide vs production vs recherche).

---

## 1) Commence par clarifier le type de problème

### A) Prédire une valeur (régression)
Exemples : prix, demande, durée, score.
- Baselines : **Linear Regression**, **Random Forest**, **Gradient Boosting**
- Si séries temporelles : **ARIMA/Prophet**, modèles deep time series

### B) Prédire une classe (classification)
Exemples : spam/non-spam, défaut de paiement, diagnostic.
- Baselines : **Logistic Regression**, **SVM**, **Random Forest**, **Gradient Boosting**
- Si texte : **Transformers (BERT-like)** ou embeddings + classifieur

### C) Regrouper / découvrir de la structure (non-supervisé)
Exemples : segmentation clients, clustering documents.
- Clustering : **k-means**, **GMM**, **hierarchical**
- Visualisation : **PCA**, **UMAP/t-SNE**

### D) Générer du contenu (génératif)
Exemples : images, texte, augmentation de données.
- Texte : **LLM / Transformers (GPT-like)**
- Images : **Diffusion**, **GAN** (plus rare aujourd’hui), **VAE** (plutôt représentations)

### E) Décider et agir dans un environnement (contrôle / RL)
Exemples : robot, jeu, stratégie, allocation dynamique.
- Si état observable : **MDP / RL standard**
- Si partiellement observable : **POMDP**
- Algo (selon action space) : **DQN** (discret), **PPO/SAC/TD3** (souvent continu)

---

## 2) Regarde tes données (c’est le vrai “GPS”)

### A) Tabulaire (table, features)
Exemples : finance, CRM, capteurs agrégés.
- Meilleurs baselines production : **Gradient Boosting (XGBoost/LightGBM/CatBoost)**
- Interprétable : **Logistic/Linear**, **Decision Trees**
- Robuste et simple : **Random Forest**

### B) Texte
- Compréhension/classification : **Encoder Transformers (BERT-like)**
- Génération/rédaction : **Decoder Transformers (GPT-like)**
- Recherche/similarité : **Embeddings** + index (vector search)

### C) Images / vision
- Classification/détection : **CNN** ou **Vision Transformers**
- Génération : **Diffusion**
- Petits datasets : préférer **transfer learning**

### D) Séries temporelles
- Baselines : **ARIMA**, **Prophet**
- Complexe / multi-signaux : modèles deep time series (RNN/Transformers)

### E) Graphes (réseaux, relations)
- **GNN** si la structure graphe est essentielle (social, molécules, liens)

---

## 3) Contraintes : explique / scale / temps / coût

### A) Besoin d’explicabilité forte ?
- Oui : **Linear/Logistic**, **Decision Trees**, **rule-based**
- Moyen : **Random Forest**, **Gradient Boosting** (avec SHAP, etc.)
- Faible : **Deep Learning / Transformers**

### B) Latence en production (temps réel) ?
- Ultra faible : modèles simples (linéaires, arbres petits)
- Faible : Gradient boosting optimisé
- Plus lourd : deep/transformers (souvent besoin GPU ou quantization)

### C) Taille dataset
- Petit : baselines classiques + régularisation + data cleaning
- Moyen : gradient boosting, CNN/transformers fine-tunés
- Très grand : deep learning + pré-entraînement

### D) Données bruitées / incertitude importante ?
- Considère **probabiliste/bayésien** (GP, Bayes nets, Kalman)
- Ou ensembles + calibration

---

## 4) Une stratégie “pro” en 6 étapes (très efficace)

1. **Définis la métrique** (F1, AUC, RMSE, revenue uplift, etc.)  
2. **Fais une baseline simple** (Logistic/Linear ou Random Forest)  
3. **Fais une baseline forte tabulaire** (Gradient Boosting)  
4. **Ajoute deep learning** seulement si :
   - données non tabulaires (texte/image/audio) ou
   - gain clair attendu
5. **Traite l’incertitude** (calibration, probabiliste, ensembles) si nécessaire  
6. **Teste la robustesse** (shift, fairness, bruit, edge cases)

---

## 5) “Cheat sheet” — choix rapide

### Problèmes tabulaires (le plus fréquent en entreprise)
- **1er choix** : Gradient Boosting  
- **Si interprétable** : Logistic/Linear + features claires  
- **Si baseline rapide** : Random Forest

### Texte
- Classification : BERT-like
- Génération : GPT-like
- Similarité/retrieval : embeddings

### Images
- Classification : CNN / ViT
- Génération : Diffusion

### Décision / contrôle (séquentiel)
- Discret : DQN
- Continu : PPO / SAC / TD3
- Partiellement observable : POMDP + croyance/mémoire

### Incertitude / estimation d’état
- Dynamique : Kalman / filtres bayésiens
- Peu de données + incertitude : Gaussian Processes

---

## 6) Pièges classiques (à éviter)
- Confondre **accuracy** avec bonne perf (classes déséquilibrées → préfère F1/AUC)
- Oublier le **data leakage**
- Pas de vraie séparation train/val/test
- Aller direct sur deep/LLM sans baseline tabulaire solide
- Sous-estimer la **qualité des features** et du preprocessing

---

## 7) Template “fiche modèle” (à copier dans chaque .md)
Pour chaque algo/modèle, tu peux remplir :

- **Quand l’utiliser**
- **Hypothèses**
- **Inputs attendus**
- **Forces / limites**
- **Complexité**
- **Hyperparams clés**
- **Métriques**
- **Bonnes pratiques**
- **Références**

---