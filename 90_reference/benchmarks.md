# Benchmarks & Datasets de référence

Guide des benchmarks standards pour évaluer les modèles ML/AI par domaine.

---

## 🖼️ Computer Vision

### Classification d'images

| Dataset | Description | Taille | Métrique | SOTA |
|---------|-------------|--------|----------|------|
| **MNIST** | Chiffres manuscrits | 70K images (28×28) | Accuracy | >99.8% |
| **CIFAR-10** | 10 classes objets | 60K images (32×32) | Accuracy | ~99% |
| **CIFAR-100** | 100 classes objets | 60K images (32×32) | Top-1 Accuracy | ~95% |
| **ImageNet** | 1000 classes | 14M images | Top-1/Top-5 Accuracy | 90%/98% (ViT) |

### Détection d'objets

| Dataset | Description | Métrique | SOTA |
|---------|-------------|----------|------|
| **COCO** | 80 classes, 200K images | mAP@50-95 | ~65 mAP (DINO, 2023) |
| **Pascal VOC** | 20 classes | mAP | ~90 mAP |
| **Open Images** | 600 classes, 9M images | mAP | Variable |

### Segmentation

| Dataset | Type | Métrique | 
|---------|------|----------|
| **ADE20K** | Semantic segmentation | mIoU |
| **Cityscapes** | Urban scenes | mIoU |
| **COCO-Stuff** | Segmentation + stuff | mIoU |

---

## 📝 Natural Language Processing

### Compréhension (GLUE, SuperGLUE)

| Benchmark | Tasks | Métrique | SOTA |
|-----------|-------|----------|------|
| **GLUE** | 9 tâches (sentiment, entailment, etc.) | Avg score | 90+ (GPT-4) |
| **SuperGLUE** | Tâches plus difficiles | Avg score | 90+ (GPT-4) |
| **SQuAD** | Question answering | F1/EM | 95/90 (Human: 91/82) |
| **RACE** | Reading comprehension | Accuracy | ~95% |

### Génération de texte

| Dataset | Task | Métrique |
|---------|------|----------|
| **Penn Treebank** | Language modeling | Perplexity (PPL) |
| **WikiText-103** | Language modeling | PPL |
| **CNN/DailyMail** | Summarization | ROUGE-L |
| **WMT** | Translation | BLEU |

### Métriques NLP courantes

- **BLEU** : Translation quality (0-100)
- **ROUGE** : Summarization (ROUGE-1, ROUGE-2, ROUGE-L)
- **METEOR** : Translation/generation
- **Perplexity (PPL)** : Language modeling (lower is better)
- **F1 Score** : Named Entity Recognition, QA
- **Exact Match (EM)** : Question answering

---

## 🎮 Reinforcement Learning

### Environnements classiques

| Environment | Description | Métrique | SOTA |
|-------------|-------------|----------|------|
| **CartPole** | Balance pole | Episode reward | 500 (perfect) |
| **MountainCar** | Reach flag | Episode reward | -110 to 0 |
| **Lunar Lander** | Safe landing | Episode reward | 200+ |

### Atari 2600

| Benchmark | # Games | Métrique | SOTA |
|-----------|---------|----------|------|
| **Atari-57** | 57 jeux Atari | Human-normalized score | 1000%+ (MuZero, Agent57) |

**Metrics** :
- **Human-normalized score** : `(Agent - Random) / (Human - Random) × 100%`
- **Median/Mean score** : Across all games

### Contrôle continu

| Benchmark | Description | Métrique |
|-----------|-------------|----------|
| **MuJoCo** | Robotics simulation (Ant, Humanoid, etc.) | Episode reward |
| **dm_control** | DeepMind control suite | Episode reward |
| **Meta-World** | Manipulation tasks | Success rate |

---

## 📊 Recommendation Systems

### Datasets

| Dataset | Domain | Size | Métrique |
|---------|--------|------|----------|
| **MovieLens** | Movies | 100K-25M ratings | RMSE, MAE, Precision@K |
| **Netflix Prize** | Movies | 100M ratings | RMSE |
| **Amazon Reviews** | Products | Millions | NDCG, HR@K |
| **Last.fm** | Music | User-artist plays | Precision@K, Recall@K |

### Métriques

- **RMSE/MAE** : Rating prediction error
- **Precision@K / Recall@K** : Top-K recommendations
- **NDCG@K** : Normalized Discounted Cumulative Gain
- **Hit Rate (HR@K)** : % users with ≥1 relevant item in top-K
- **MRR** : Mean Reciprocal Rank

---

## ⏱️ Time Series

### Datasets

| Dataset | Domain | Métrique |
|---------|--------|----------|
| **M4 Competition** | Forecasting (100K series) | SMAPE, MASE |
| **Electricity** | Power consumption | MAE, RMSE |
| **Traffic** | Road occupancy | MAE, RMSE |
| **ETT** | Electricity transformer (hourly) | MSE, MAE |

### Métriques

- **MSE/RMSE** : Mean Squared Error
- **MAE** : Mean Absolute Error
- **MAPE** : Mean Absolute Percentage Error
- **SMAPE** : Symmetric MAPE
- **MASE** : Mean Absolute Scaled Error

---

## 🧠 Probabilistic Models

### Benchmarks

| Task | Dataset | Métrique |
|------|---------|----------|
| **Bayesian Optimization** | Synthetic functions, HPO | Regret, convergence |
| **Gaussian Processes** | UCI regression datasets | NLPD, RMSE |
| **Kalman Filtering** | Tracking benchmarks | RMSE, tracking error |

---

## 📈 Graph Neural Networks

### Datasets

| Dataset | Type | # Nodes | # Edges | Task |
|---------|------|---------|---------|------|
| **Cora** | Citation network | 2.7K | 5.4K | Node classification |
| **Citeseer** | Citation network | 3.3K | 4.7K | Node classification |
| **PubMed** | Citation network | 19K | 44K | Node classification |
| **PROTEINS** | Protein structures | ~40/graph | - | Graph classification |
| **ogbn-arxiv** | Large citation | 169K | 1.2M | Node classification |

### Métriques

- **Node classification** : Accuracy, F1-score
- **Link prediction** : AUC, Hits@K
- **Graph classification** : Accuracy

---

## 🔍 Ressources utiles

### Leaderboards

- **Papers with Code** : [paperswithcode.com](https://paperswithcode.com/) - Tous domaines
- **HuggingFace Leaderboard** : NLP, Vision models
- **OpenAI Gym Leaderboard** : RL environnements
- **Kaggle** : Competitions actives

### Benchmarking Libraries

```python
# Computer Vision
from torchvision.datasets import CIFAR10, ImageNet
from torchmetrics import Accuracy

# NLP
from datasets import load_dataset  # HuggingFace
dataset = load_dataset("glue", "sst2")

# RL
import gym
env = gym.make("CartPole-v1")

# Time Series
from gluonts.dataset.repository import get_dataset
dataset = get_dataset("electricity")
```

---

## 📌 Notes

- **SOTA** = State-of-the-art (meilleurs résultats publiés)
- Les scores évoluent constamment avec les nouveaux modèles
- Toujours comparer avec les mêmes protocoles d'évaluation
- Attention au **overfitting** sur les benchmarks publics
- Utiliser plusieurs métriques pour une évaluation complète

**Dernière mise à jour** : Janvier 2026
