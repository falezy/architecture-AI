#!/usr/bin/env bash
set -euo pipefail

ROOT="ai-models-catalog"

mkdir -p \
  "$ROOT/00_overview" \
  "$ROOT/01_machine_learning/supervised" \
  "$ROOT/01_machine_learning/unsupervised" \
  "$ROOT/01_machine_learning/self_supervised" \
  "$ROOT/02_deep_learning/transformers" \
  "$ROOT/02_deep_learning/generative" \
  "$ROOT/02_deep_learning/graph" \
  "$ROOT/03_reinforcement_learning/basics" \
  "$ROOT/03_reinforcement_learning/value_based" \
  "$ROOT/03_reinforcement_learning/policy_based" \
  "$ROOT/03_reinforcement_learning/actor_critic" \
  "$ROOT/03_reinforcement_learning/model_based_rl" \
  "$ROOT/04_probabilistic_models" \
  "$ROOT/05_recommendation_and_ranking" \
  "$ROOT/06_time_series" \
  "$ROOT/90_reference"

# README racine + overview
cat > "$ROOT/README.md" <<'EOF'
# AI Models Catalog

Catalogue de familles, modèles et algorithmes en IA/ML/DL/RL.
Chaque fichier .md contient : idée clé, quand l’utiliser, forces/faiblesses, variantes, références.
EOF

: > "$ROOT/00_overview/glossary.md"
: > "$ROOT/00_overview/taxonomy.md"
: > "$ROOT/00_overview/how_to_choose.md"
: > "$ROOT/90_reference/reading_list.md"
: > "$ROOT/90_reference/benchmarks.md"

# Helper pour créer un .md avec titre + 1-liner
write_md () {
  local path="$1"
  local title="$2"
  local desc="$3"
  cat > "$path" <<EOF
# $title

$desc

## Idée clé
- TODO

## Quand l'utiliser
- TODO

## Forces
- TODO

## Limites
- TODO

## Variantes / liens
- TODO

## Références
- TODO
EOF
}

# 01 ML — supervised
write_md "$ROOT/01_machine_learning/supervised/linear_regression.md" "Linear Regression" "Régression linéaire pour prédire une variable continue."
write_md "$ROOT/01_machine_learning/supervised/logistic_regression.md" "Logistic Regression" "Classification linéaire (probabilités) pour binaire/multi-classe."
write_md "$ROOT/01_machine_learning/supervised/svm.md" "SVM" "Classification/régression via marges maximales, kernels possibles."
write_md "$ROOT/01_machine_learning/supervised/decision_trees.md" "Decision Trees" "Modèle interprétable basé sur des règles de split."
write_md "$ROOT/01_machine_learning/supervised/random_forest.md" "Random Forest" "Ensemble de arbres (bagging) robuste, peu de tuning."
write_md "$ROOT/01_machine_learning/supervised/gradient_boosting.md" "Gradient Boosting" "Ensemble séquentiel (XGBoost/LightGBM/CatBoost)."
write_md "$ROOT/01_machine_learning/supervised/naive_bayes.md" "Naive Bayes" "Classifieur probabiliste simple, efficace sur texte."

# 01 ML — unsupervised
write_md "$ROOT/01_machine_learning/unsupervised/kmeans.md" "K-means" "Clustering par centroïdes, rapide, nécessite K."
write_md "$ROOT/01_machine_learning/unsupervised/gmm.md" "Gaussian Mixture Models (GMM)" "Clustering probabiliste via mélanges gaussiens (EM)."
write_md "$ROOT/01_machine_learning/unsupervised/hierarchical_clustering.md" "Hierarchical Clustering" "Clustering hiérarchique (dendrogramme)."
write_md "$ROOT/01_machine_learning/unsupervised/pca.md" "PCA" "Réduction de dimension linéaire par variance maximale."
write_md "$ROOT/01_machine_learning/unsupervised/umap_tsne.md" "UMAP / t-SNE" "Réduction de dimension non-linéaire (visualisation)."

# 01 ML — self-supervised
write_md "$ROOT/01_machine_learning/self_supervised/contrastive_learning.md" "Contrastive Learning" "Apprentissage de représentations via paires positives/négatives."

# 02 Deep learning
write_md "$ROOT/02_deep_learning/mlp.md" "MLP (Fully Connected)" "Réseau dense, base pour tabulaire ou petits signaux."
write_md "$ROOT/02_deep_learning/cnn.md" "CNN" "Réseaux convolutionnels pour images/vision et signaux locaux."
write_md "$ROOT/02_deep_learning/rnn_lstm_gru.md" "RNN / LSTM / GRU" "Réseaux séquentiels historiques (avant Transformers)."

# 02 Transformers
write_md "$ROOT/02_deep_learning/transformers/transformer.md" "Transformer" "Architecture à attention (self-attention) pour séquences."
write_md "$ROOT/02_deep_learning/transformers/encoder_models_bert.md" "Encoder models (BERT-like)" "Encodeurs pour compréhension (classification, embeddings)."
write_md "$ROOT/02_deep_learning/transformers/decoder_models_gpt.md" "Decoder models (GPT-like)" "Décodeurs autoregressifs pour génération."
write_md "$ROOT/02_deep_learning/transformers/encoder_decoder_t5_bart.md" "Encoder-Decoder (T5/BART-like)" "Seq2seq pour traduction, résumé, etc."

# 02 Generative
write_md "$ROOT/02_deep_learning/generative/autoencoders.md" "Autoencoders" "Compression/reconstruction : encodeur + décodeur."
write_md "$ROOT/02_deep_learning/generative/vae.md" "VAE" "Autoencodeur variationnel, génération probabiliste."
write_md "$ROOT/02_deep_learning/generative/gan.md" "GAN" "Génération via adversarial training (générateur vs discriminateur)."
write_md "$ROOT/02_deep_learning/generative/diffusion_models.md" "Diffusion Models" "Génération par débruitage progressif (images, audio, etc.)."

# 02 Graph
write_md "$ROOT/02_deep_learning/graph/gnn.md" "GNN" "Réseaux pour graphes (message passing)."

# 03 RL
write_md "$ROOT/03_reinforcement_learning/basics/mdp.md" "MDP" "Cadre standard RL : états, actions, transitions, récompenses."
write_md "$ROOT/03_reinforcement_learning/basics/pomdp.md" "POMDP" "RL sous observabilité partielle (croyance sur l’état)."
write_md "$ROOT/03_reinforcement_learning/value_based/q_learning.md" "Q-learning" "Apprentissage de la valeur d’action (tabulaire)."
write_md "$ROOT/03_reinforcement_learning/value_based/dqn.md" "DQN" "Q-learning avec réseau neuronal (Deep Q-Network)."
write_md "$ROOT/03_reinforcement_learning/policy_based/reinforce.md" "REINFORCE" "Policy gradient simple basé sur retours."
write_md "$ROOT/03_reinforcement_learning/policy_based/ppo.md" "PPO" "Policy gradient stable (clipped objective), très utilisé."
write_md "$ROOT/03_reinforcement_learning/actor_critic/a2c_a3c.md" "A2C / A3C" "Actor-critic synchrones/asynchrones."
write_md "$ROOT/03_reinforcement_learning/actor_critic/sac_td3_ddpg.md" "SAC / TD3 / DDPG" "Actor-critic pour actions continues (off-policy)."
write_md "$ROOT/03_reinforcement_learning/model_based_rl/model_based_rl.md" "Model-based RL" "Apprend un modèle du monde pour planifier/imaginer."

# 04 Probabilistic
write_md "$ROOT/04_probabilistic_models/bayesian_networks.md" "Bayesian Networks" "Graphes probabilistes pour inférence causale/probabiliste."
write_md "$ROOT/04_probabilistic_models/hmm.md" "HMM" "Modèle de Markov caché pour séquences (probabiliste)."
write_md "$ROOT/04_probabilistic_models/kalman_filters.md" "Kalman Filters" "Filtrage bayésien linéaire (estimation d’état)."
write_md "$ROOT/04_probabilistic_models/gaussian_processes.md" "Gaussian Processes" "Modèle non-paramétrique avec incertitude (régression/classif)."

# 05 Reco/Ranking
write_md "$ROOT/05_recommendation_and_ranking/collaborative_filtering.md" "Collaborative Filtering" "Recommandation basée sur similarités utilisateurs/items."
write_md "$ROOT/05_recommendation_and_ranking/matrix_factorization.md" "Matrix Factorization" "Recommandation via facteurs latents (SVD-like)."
write_md "$ROOT/05_recommendation_and_ranking/learning_to_rank.md" "Learning to Rank" "Optimise le classement (search/reco)."

# 06 Time series
write_md "$ROOT/06_time_series/arima.md" "ARIMA" "Prévision séries temporelles (classique)."
write_md "$ROOT/06_time_series/prophet.md" "Prophet" "Prévision avec saisonnalités/tendances (pratique)."
write_md "$ROOT/06_time_series/deep_time_series.md" "Deep Time Series" "Prévision via RNN/Transformers, etc."

echo "✅ Repo créé: $(pwd)/$ROOT"
