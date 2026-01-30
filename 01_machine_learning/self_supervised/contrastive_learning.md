# Contrastive Learning

Apprentissage de représentations via paires positives/négatives.

## Idée clé

L'apprentissage contrastif apprend des représentations en **rapprochant** les embeddings de paires positives (similaires) et en **éloignant** les embeddings de paires négatives (différentes).

**Principe** :
1. Pour chaque donnée, créer des **vues positives** (augmentations de la même donnée)
2. Utiliser les autres données du batch comme **vues négatives**
3. Optimiser une fonction de perte qui :
   - Maximise la similarité entre positifs
   - Minimise la similarité entre négatifs

**Formule générale (NT-Xent Loss)** :
```
L = -log( exp(sim(z_i, z_j) / τ) / Σ_k exp(sim(z_i, z_k) / τ) )
```
- `z_i, z_j` : embeddings de la paire positive
- `z_k` : embeddings des paires négatives
- `τ` : température (hyperparamètre)
- `sim` : similarité cosine

## Exemples concrets

### 1. Vision : SimCLR (images)

**Scénario** : 10,000 images non-étiquetées de chiens, chats, oiseaux → apprendre des représentations sans labels.

**Pipeline** :
1. **Augmentation** : Créer 2 versions de chaque image
   - Image : un chat orange
   - Vue 1 : crop + saturation réduite
   - Vue 2 : rotation 15° + flou léger

2. **Encodage** : Passer les 2 vues dans un ResNet
   ```python
   z1 = resnet(augment(img))  # [128]
   z2 = resnet(augment(img))  # [128]
   ```

3. **Loss contrastive** :
   - Positif : z1 (chat) ≈ z2 (même chat augmenté)
   - Négatif : z1 (chat) ≠ z_chien, z_oiseau, etc.

**Code Python** :
```python
import torch
import torch.nn.functional as F

# Deux versions augmentées
img1 = augment(original_image)  # crop, rotation
img2 = augment(original_image)  # flip, color jitter

# Encoder
z1 = encoder(img1)  # [128]
z2 = encoder(img2)  # [128]

# Similarité cosine
similarity = F.cosine_similarity(z1, z2)

# NT-Xent loss (simplifié)
# Rapprocher z1 et z2, éloigner des autres images du batch
```

**Résultat** :
- 2 photos du même chat → embeddings proches
- Chat vs chien → embeddings éloignés
- Utilisation : classification avec peu de labels, recherche d'images similaires

---

### 2. NLP : SimCSE (texte)

**Scénario** : 100,000 phrases en français → apprendre des représentations sémantiques sans supervision.

**Pipeline** :
1. **Augmentation via dropout** : Même phrase, 2 passages dans BERT
   - Phrase : "Le chat dort sur le canapé"
   - Vue 1 : BERT(phrase, dropout=0.1)
   - Vue 2 : BERT(phrase, dropout=0.1) → différent à cause du dropout stochastique

2. **Paires négatives** : Autres phrases du batch
   - Positive : "Le chat dort..." (2 vues)
   - Négatives : "La voiture roule vite", "J'aime les pommes"

3. **Loss contrastive** :
   - Positif : z1 (chat dort) ≈ z2 (même phrase, dropout différent)
   - Négatif : z1 (chat dort) ≠ z_voiture, z_pommes

**Code Python** :
```python
from transformers import BertModel
import torch.nn.functional as F

# Même phrase, 2 passages avec dropout
sentence = "Le chat dort sur le canapé"
z1 = bert(sentence, dropout=0.1)  # [768]
z2 = bert(sentence, dropout=0.1)  # [768] (différent)

# Phrases négatives
negatives = ["La voiture roule", "J'aime les pommes"]
z_neg = [bert(s) for s in negatives]

# Loss : rapprocher z1/z2, éloigner z1/z_neg
```

**Résultat** :
- "Le chat dort" ≈ "Un félin se repose" (sémantiquement proche)
- "Le chat dort" ≠ "La voiture roule" (sémantiquement éloigné)
- Applications : recherche sémantique, détection de paraphrases, Q&A

## Quand l'utiliser

- **Peu ou pas de labels** : Exploiter de grandes quantités de données non-étiquetées
- **Pré-entraînement** : Créer un modèle de base avant fine-tuning supervisé
- **Recherche de similarité** : Images similaires, phrases sémantiquement proches
- **Clustering** : Grouper automatiquement des données sans labels
- **Transfert learning** : Apprendre des représentations génériques réutilisables

**Domaines d'application** :
- Vision : ImageNet, détection d'objets, segmentation
- NLP : Sentence embeddings, recherche sémantique, RAG
- Audio : Reconnaissance de sons, classification de musique
- Multi-modal : CLIP (vision + texte)

## Forces

✅ **Pas besoin de labels** : Exploite des données non-étiquetées massives  
✅ **Représentations génériques** : Transférables à de multiples tâches  
✅ **Performances SOTA** : Comparable au supervisé sur ImageNet (SimCLR, MoCo)  
✅ **Augmentation automatique** : Pas besoin d'annotations manuelles  
✅ **Scalable** : Parallélisable sur plusieurs GPUs  
✅ **Robustesse** : Moins de surapprentissage qu'en supervisé pur

## Limites

❌ **Coût computationnel** : Besoin de grandes batches (4096+ pour SimCLR)  
❌ **Choix des augmentations** : Critique et dépendant du domaine (vision ≠ NLP)  
❌ **Hyperparamètres sensibles** : Température τ, taille du batch, architecture  
❌ **Paires négatives** : Peut induire des biais (faux négatifs dans le batch)  
❌ **Interprétabilité** : Difficile de comprendre ce que le modèle apprend  
❌ **Besoin de fine-tuning** : Rarement utilisé directement sans adaptation

## Variantes / liens

### Méthodes contrastives populaires
- **SimCLR** (2020) : Simple framework, large batch, projection head
- **MoCo** (Momentum Contrast, 2020) : Queue de négatifs, encoder momentum
- **BYOL** (2020) : Pas de paires négatives (predictor + momentum)
- **SimCSE** (2021) : Contrastive learning pour NLP via dropout
- **CLIP** (2021) : Vision + texte (images-captions contrastifs)

### Relations avec d'autres approches
- **Self-supervised learning** : Contrastive learning est une sous-catégorie
- **Metric learning** : Triplet loss, Siamese networks
- **Masked modeling** : BERT (masking) vs SimCSE (contrastive)
- **Clustering-based** : SwAV, DeepCluster

### Liens avec supervision
- **Semi-supervisé** : Contrastive + peu de labels (FixMatch)
- **Weakly-supervised** : Contrastive avec labels bruités

## Références

### Papers fondamentaux
- **SimCLR** : [A Simple Framework for Contrastive Learning](https://arxiv.org/abs/2002.05709) (Chen et al., 2020)
- **MoCo** : [Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/abs/1911.05722) (He et al., 2020)
- **SimCSE** : [SimCSE: Simple Contrastive Learning of Sentence Embeddings](https://arxiv.org/abs/2104.08821) (Gao et al., 2021)
- **CLIP** : [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) (Radford et al., 2021)

### Tutoriels et implémentations
- [PyTorch SimCLR](https://github.com/sthalles/SimCLR)
- [Sentence-Transformers (SimCSE)](https://www.sbert.net/)
- [OpenAI CLIP](https://github.com/openai/CLIP)

### Surveys
- [Self-supervised Learning: Generative or Contrastive](https://arxiv.org/abs/2006.08218) (Liu et al., 2021)
