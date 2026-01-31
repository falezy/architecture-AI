# Architecture Cognitive vs Modèles IA — Différences et quand choisir

Ce document explique les **différences fondamentales** entre les **architectures cognitives** et les **modèles d'IA/ML classiques**, et fournit un guide pratique pour savoir **quand utiliser l'un, l'autre, ou les deux ensemble**.

---

## 1) Qu'est-ce qu'une Architecture Cognitive ?

Une **architecture cognitive** est un système computationnel qui vise à **modéliser le fonctionnement de la cognition humaine** de manière intégrée. Elle ne se contente pas de résoudre un problème spécifique, mais cherche à reproduire les **mécanismes généraux** de la pensée : mémoire, raisonnement, apprentissage, perception, prise de décision, etc.

### Exemples d'architectures cognitives
- **ACT-R** (Adaptive Control of Thought—Rational) : mémoire déclarative + procédurale, production rules
- **SOAR** (State, Operator, And Result) : résolution de problèmes, chunking, apprentissage par renforcement
- **CLARION** (Connectionist Learning with Adaptive Rule Induction ON-line) : niveaux implicite/explicite, bottom-up/top-down
- **SIGMA** : unification symbolique + probabiliste + RL
- **LIDA** (Learning Intelligent Distribution Agent) : architecture inspirée de la théorie de la conscience globale

### Caractéristiques clés
1. **Généralité** : conçues pour des tâches multiples (pas un seul objectif)
2. **Modularité cognitive** : perception, mémoire, attention, raisonnement, action
3. **Apprentissage progressif** : accumulation de connaissances et compétences
4. **Interprétabilité** : processus explicite, pas boîte noire
5. **Inspiration cognitive** : modélisation basée sur la psychologie/neuroscience

---

## 2) Qu'est-ce qu'un Modèle IA/ML Classique ?

Un **modèle d'IA/ML** est conçu pour **résoudre un problème spécifique** (classement, prédiction, génération, etc.) de manière **optimale** sur un jeu de données donné, en apprenant via des algorithmes statistiques ou deep learning.

### Exemples de modèles IA
- **Machine Learning** : Gradient Boosting, Random Forest, SVM, Logistic Regression
- **Deep Learning** : CNN, Transformers, RNN/LSTM
- **Reinforcement Learning** : DQN, PPO, SAC
- **Probabiliste** : Bayesian Networks, Gaussian Processes, Kalman Filters

### Caractéristiques clés
1. **Spécialisation** : optimisé pour une tâche spécifique (vision, NLP, prédiction, etc.)
2. **Performance** : maximiser une métrique (accuracy, F1, RMSE, reward)
3. **Data-driven** : apprend à partir de données annotées ou de récompenses
4. **Flexibilité architecturale** : choix adapté au type de données et objectif
5. **Scalabilité** : peut traiter de très grands volumes de données

---

## 3) Tableau Comparatif — Cognitive Architecture vs AI Model

| **Critère**                  | **Architecture Cognitive**                                  | **Modèle IA/ML Classique**                                  |
|------------------------------|-------------------------------------------------------------|-------------------------------------------------------------|
| **Objectif**                 | Modéliser la cognition humaine de façon générale           | Résoudre une tâche spécifique efficacement                  |
| **Portée**                   | Multi-tâches, généraliste                                   | Mono-tâche, spécialisé                                      |
| **Apprentissage**            | Progressif, cumulatif (chunks, rules)                       | Entraînement sur dataset défini, puis figé                  |
| **Mémoire**                  | Mémoire déclarative + procédurale, long terme + court terme | Poids du réseau, pas de séparation explicite mémoire        |
| **Raisonnement**             | Raisonnement symbolique explicite (rules, inférences)      | Raisonnement implicite (features apprises)                  |
| **Interprétabilité**         | Haute (règles, traces d'exécution)                          | Variable (faible pour DL, modérée pour trees/linear)        |
| **Adaptabilité**             | Peut transférer connaissances entre tâches                  | Transfer learning possible, mais limité                     |
| **Performance brute**        | Généralement inférieure à un modèle spécialisé              | Peut atteindre SOTA sur tâches spécifiques                  |
| **Complexité implémentation**| Haute (multiples modules, orchestration)                    | Modérée à haute (selon architecture)                        |
| **Besoin en données**        | Peut fonctionner avec peu de données (raisonnement)        | Souvent nécessite beaucoup de données (surtout DL)          |
| **Cas d'usage**              | Agents autonomes, simulation cognitive, tuteurs             | Prédiction, classification, génération, reconnaissance      |

---

## 4) Quand Choisir une Architecture Cognitive ?

### ✅ Utilise une architecture cognitive si :

1. **Généralité requise**  
   - Tu veux un agent capable de réaliser **plusieurs tâches** différentes, pas seulement une prédiction
   - Exemple : un assistant personnel qui doit raisonner, se souvenir, planifier, apprendre

2. **Besoin de raisonnement explicite**  
   - Tu as besoin de **traces d'exécution**, d'explications claires sur pourquoi une décision a été prise
   - Exemple : tuteur intelligent, système d'aide à la décision médicale

3. **Apprentissage cumulatif et transfert**  
   - L'agent doit **apprendre en continu** et transférer des compétences entre domaines
   - Exemple : robot qui apprend de nouvelles tâches sans oublier les anciennes

4. **Modélisation de la cognition humaine**  
   - Objectif de **recherche en sciences cognitives** ou psychologie
   - Exemple : simuler des processus mentaux, tester des théories cognitives

5. **Données limitées mais connaissances a priori**  
   - Tu n'as pas beaucoup de données, mais tu peux **modéliser des règles** et du raisonnement
   - Exemple : système expert avec règles métier

### ✅ Exemples d'applications
- Agents autonomes complexes (robots, NPCs intelligents)
- Tuteurs intelligents (ITS - Intelligent Tutoring Systems)
- Simulation cognitive et neuroscience computationnelle
- Assistants personnels généraux (vs assistants spécialisés)
- Systèmes de décision multi-étapes avec mémoire

---

## 5) Quand Choisir un Modèle IA/ML Classique ?

### ✅ Utilise un modèle IA/ML si :

1. **Tâche spécifique et bien définie**  
   - Tu veux classifier, prédire, générer sur un domaine précis
   - Exemple : détection de fraude, prédiction de prix, génération de texte

2. **Données abondantes et annotées**  
   - Tu disposes d'un **gros dataset** avec labels ou signal de récompense
   - Exemple : ImageNet pour vision, corpus de texte pour LLM

3. **Performance maximale cruciale**  
   - Tu cherches à atteindre le **meilleur score** sur une métrique
   - Exemple : compétition Kaggle, production où chaque point de précision compte

4. **Pas besoin de généralité**  
   - L'agent n'a pas besoin de raisonner sur plusieurs domaines ou tâches
   - Exemple : un chatbot spécialisé FAQ, un modèle de recommandation

5. **Scalabilité et déploiement industriel**  
   - Infrastructure pour **servir des millions de requêtes** par jour
   - Exemple : moteur de recherche, recommandation Netflix, traduction automatique

### ✅ Exemples d'applications
- Classification d'images (CNN, Vision Transformers)
- NLP : sentiment analysis, NER, génération (Transformers)
- Prédiction tabulaire (Gradient Boosting)
- Recommandation (matrix factorization, collaborative filtering)
- Jeux vidéo optimisation stratégique (RL spécialisé)

---

## 6) Quand Combiner les Deux ? (Approche Hybride)

### 🔄 Architecture Cognitive + Modèles IA/ML

L'approche **hybride** consiste à utiliser une **architecture cognitive comme orchestrateur**, et des **modèles IA spécialisés comme modules** pour des sous-tâches.

### Pourquoi combiner ?
- **Raisonnement haut-niveau** (cognitive) + **performance sur tâches spécifiques** (ML)
- **Mémoire et planification** (cognitive) + **perception robuste** (deep learning)
- **Interprétabilité** (cognitive) + **apprentissage data-driven** (ML)

### Architecture hybride type

```
┌──────────────────────────────────────────────┐
│      Architecture Cognitive (Orchestrateur)  │
│  ┌─────────────┐  ┌──────────┐  ┌─────────┐ │
│  │  Mémoire    │  │Raisonnement│ │Planning │ │
│  │(Déclarative)│  │ (Règles)  │ │ (Goals) │ │
│  └─────────────┘  └──────────┘  └─────────┘ │
│           ▲                                   │
│           │                                   │
│           │ requêtes/décisions                │
│           ▼                                   │
├──────────────────────────────────────────────┤
│         Modules IA/ML Spécialisés             │
│  ┌───────┐  ┌───────────┐  ┌──────────────┐ │
│  │ Vision│  │    NLP    │  │ Prédiction   │ │
│  │ (CNN) │  │(Transformer)│ │(Boosting/RL) │ │
│  └───────┘  └───────────┘  └──────────────┘ │
└──────────────────────────────────────────────┘
```

### Exemples concrets

#### Exemple 1 : Robot autonome
- **Architecture cognitive** : planifie les tâches, gère la mémoire des lieux visités, raisonne sur les priorités
- **Modèles ML** :
  - **Vision (CNN)** : détection d'objets
  - **RL (PPO)** : contrôle moteur bas-niveau
  - **NLP (Transformer)** : compréhension des commandes vocales

#### Exemple 2 : Assistant virtuel intelligent
- **Architecture cognitive (ex: CLARION, SIGMA)** : gestion du contexte, intentions, croyances utilisateur, planification multi-étapes
- **Modèles ML** :
  - **Transformers** : compréhension et génération de langage naturel
  - **Classification** : détection d'intention (intent classifier)
  - **Recommandation** : suggérer des actions (ML)

#### Exemple 3 : Système de tuteur intelligent (ITS)
- **Architecture cognitive (ex: ACT-R)** : modélise l'état cognitif de l'élève, adaptation pédagogique
- **Modèles ML** :
  - **Gradient Boosting** : prédire la difficulté des exercices
  - **RNN/Transformers** : analyser les réponses textuelles
  - **Reinforcement Learning** : optimiser la séquence de contenu

#### Exemple 4 : Jeu vidéo NPC intelligent
- **Architecture cognitive (ex: SOAR)** : décisions stratégiques, mémoire des interactions, roleplay
- **Modèles ML** :
  - **RL (DQN/PPO)** : tactiques de combat optimales
  - **Pathfinding ML** : navigation apprise
  - **NLG** : génération de dialogues réalistes

---

## 7) Guide de Décision Rapide

### Arbre de décision pratique

```
┌───────────────────────────────────────┐
│   Quel est ton objectif principal ?   │
└─────────────┬─────────────────────────┘
              │
      ┌───────┴───────┐
      │               │
      ▼               ▼
┌─────────────┐  ┌──────────────────┐
│ Performance │  │ Modéliser une    │
│ maximale    │  │ cognition        │
│ sur tâche   │  │ générale/agent   │
│ spécifique  │  │ autonome         │
└─────┬───────┘  └────────┬─────────┘
      │                   │
      │                   │
      ▼                   ▼
┌─────────────┐  ┌──────────────────┐
│ Modèle ML   │  │ Architecture     │
│ Classique   │  │ Cognitive        │
│             │  │                  │
│ Exemples:   │  │ Exemples:        │
│ • Boosting  │  │ • ACT-R          │
│ • CNN       │  │ • SOAR           │
│ • Transform.│  │ • CLARION        │
│ • RL        │  │ • SIGMA          │
└─────────────┘  └──────────────────┘
                         │
                         │
                ┌────────┴────────────┐
                │ Besoin de modules   │
                │ haute-performance ? │
                │ (vision, NLP, etc.) │
                └────────┬────────────┘
                         │
                         ▼
                ┌──────────────────┐
                │  Hybride !       │
                │  Cognitive +     │
                │  ML modules      │
                └──────────────────┘
```

---

## 8) Checklist de Sélection

### Pour une Architecture Cognitive ✅
- [ ] Besoin de généralité (multi-tâches)
- [ ] Raisonnement explicite et interprétable
- [ ] Mémoire long-terme et transfert de connaissances
- [ ] Apprentissage progressif et cumulatif
- [ ] Simulation cognitive / recherche en sciences cognitives
- [ ] Peu de données, mais règles métier disponibles

### Pour un Modèle IA/ML Classique ✅
- [ ] Tâche unique bien définie (classification, prédiction, génération)
- [ ] Dataset large et annoté
- [ ] Optimisation d'une métrique cible (AUC, F1, RMSE, etc.)
- [ ] Pas besoin de transférer entre domaines
- [ ] Infrastructure de déploiement scalable (millions de requêtes)
- [ ] Performance brute prioritaire

### Pour une Approche Hybride ✅
- [ ] Agent autonome complexe (robot, NPC, assistant)
- [ ] Besoin de raisonnement haut-niveau + modules spécialisés
- [ ] Combinaison de mémoire/planification + perception/prédiction
- [ ] Système tuteur intelligent ou décision médicale
- [ ] Interaction humaine riche (dialogue, adaptation)

---

## 9) Avantages et Inconvénients

### Architecture Cognitive

| ✅ **Avantages**                                   | ❌ **Inconvénients**                                  |
|----------------------------------------------------|------------------------------------------------------|
| Généralité et flexibilité (multi-domaines)         | Performance inférieure aux modèles spécialisés       |
| Interprétabilité élevée (règles, traces)           | Complexité d'implémentation                          |
| Transfert de connaissances efficace                | Peu d'outils standards / frameworks matures          |
| Apprentissage progressif sans oublier              | Scalabilité limitée (lourdeur computationnelle)      |
| Modélisation cognitive réaliste (recherche)        | Tuning difficile (nombreux paramètres)               |

### Modèle IA/ML

| ✅ **Avantages**                                   | ❌ **Inconvénients**                                  |
|----------------------------------------------------|------------------------------------------------------|
| Performance SOTA sur tâches spécifiques            | Spécialisation (pas de généralité)                   |
| Large écosystème (frameworks, outils)              | Catastrophic forgetting (oublie en apprenant)        |
| Scalabilité industrielle                           | Besoin de beaucoup de données (surtout DL)           |
| Transfer learning possible (fine-tuning)           | Interprétabilité limitée (surtout DL)                |
| Recherche très active, nouvelles architectures     | Peut sur-apprendre, sensible au shift de distribution|

---

## 10) Ressources et Références

### Architectures Cognitives
- **ACT-R** : [http://act-r.psy.cmu.edu/](http://act-r.psy.cmu.edu/)
- **SOAR** : [https://soar.eecs.umich.edu/](https://soar.eecs.umich.edu/)
- **CLARION** : [http://www.cogsci.rpi.edu/~rsun/clarion.html](http://www.cogsci.rpi.edu/~rsun/clarion.html)
- **SIGMA** : Rosenbloom et al. (2013), *An Implementation of the Sigma Cognitive Architecture*
- **LIDA** : Franklin et al., *The LIDA Architecture*

### Livres de référence
- *Unified Theories of Cognition* — Allen Newell
- *How Can the Human Mind Occur in the Physical Universe?* — John Anderson (ACT-R)
- *The Cambridge Handbook of Computational Psychology* — Ron Sun (éditeur)

### Comparaisons Cognitive vs Standard AI
- Laird, Lebiere, Rosenbloom (2017), *A Standard Model of the Mind*
- Kotseruba & Tsotsos (2020), *40 Years of Cognitive Architectures: Core Cognitive Abilities and Practical Applications*

### Modèles IA/ML (références dans vos fichiers existants)
- `01_machine_learning/` : Gradient Boosting, SVM, etc.
- `02_deep_learning/` : CNN, Transformers, etc.
- `03_reinforcement_learning/` : DQN, PPO, SAC, etc.

---

## 11) FAQ — Questions Fréquentes

### Q: Est-ce qu'une architecture cognitive peut remplacer tous les modèles ML ?
**R:** Non. Les architectures cognitives excellent dans la **généralité et l'interprétabilité**, mais les modèles ML spécialisés atteignent des **performances supérieures** sur des tâches spécifiques (ex: classification d'images, NLP). L'idéal est souvent **hybride**.

### Q: Les architectures cognitives sont-elles encore pertinentes aujourd'hui ?
**R:** Oui, surtout pour :
- **Agents autonomes** (robots, NPCs)
- **Recherche en sciences cognitives**
- **Systèmes nécessitant interprétabilité et raisonnement explicite**
- **Apprentissage continuel** sans catastrophic forgetting

### Q: Peut-on utiliser du Deep Learning dans une architecture cognitive ?
**R:** Absolument ! C'est justement l'approche **hybride** recommandée : l'architecture cognitive orchestre, et des modules DL gèrent perception/prédiction.

### Q: Quel langage pour implémenter une architecture cognitive ?
**R:** 
- **ACT-R** : Lisp, Python (via pyactr)
- **SOAR** : C++, Java, Python bindings
- **CLARION** : Java, Python
- **Custom** : Python (le plus flexible pour intégration ML)

### Q: Combien de données faut-il ?
- **Architecture cognitive** : peut fonctionner avec **peu de données** (raisonnement symbolique)
- **Modèle ML classique** : dépend (Boosting = moyen, DL = beaucoup)
- **Hybride** : modéré (cognitive réduit besoin en données pour certaines tâches)

---

## 12) Conclusion et Recommandations

### Stratégie recommandée

1. **Commence simple** : si ton problème est bien défini (classification, prédiction), utilise un **modèle ML classique**.
   
2. **Si généralité requise** : explore les **architectures cognitives**, surtout si tu as besoin de :
   - Raisonnement explicite
   - Mémoire et transfert de connaissances
   - Agent multi-tâches

3. **Combine pour le meilleur des deux mondes** :
   - Architecture cognitive = **cerveau** (orchestration, raisonnement, mémoire)
   - Modèles ML = **sens et muscles** (perception, prédiction, action optimale)

4. **Itère et évalue** :
   - Mesure **performance, interprétabilité, coût de développement**
   - Fais des prototypes rapides avant de t'engager dans une architecture complexe

---

## Résumé en une phrase

> **"Utilise un modèle IA/ML pour résoudre une tâche spécifique efficacement, une architecture cognitive pour construire un agent général qui raisonne et apprend progressivement, et combine les deux pour des agents autonomes complexes ayant besoin de raisonnement haut-niveau ET de modules haute-performance."**

---

**Auteur** : Documentation AI/ML Catalog  
**Dernière mise à jour** : 2026-01-30
