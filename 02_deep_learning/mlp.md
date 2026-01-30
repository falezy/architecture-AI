# MLP (Fully Connected)

Réseau dense, base pour tabulaire ou petits signaux.

## Idée clé

**MLP (Multi-Layer Perceptron)** ou **Fully Connected Network** est le réseau de neurones le plus simple : chaque neurone d'une couche est connecté à **tous** les neurones de la couche suivante. C'est la brique de base du deep learning.

**Architecture** :
```
Input Layer → Hidden Layer(s) → Output Layer

Input:  x₁ ───┐
        x₂ ───┼──→ h₁ ───┐
        x₃ ───┘    h₂ ───┼──→ ŷ
                   h₃ ───┘
                   
Chaque connexion a un poids w
Chaque neurone a un biais b
```

**Forward pass** (neurone j de couche l) :
```
zⱼ⁽ˡ⁾ = Σᵢ wᵢⱼ⁽ˡ⁾ aᵢ⁽ˡ⁻¹⁾ + bⱼ⁽ˡ⁾
aⱼ⁽ˡ⁾ = σ(zⱼ⁽ˡ⁾)

où:
- aᵢ⁽ˡ⁻¹⁾ : activations de la couche précédente
- wᵢⱼ⁽ˡ⁾ : poids de connexion
- bⱼ⁽ˡ⁾ : biais
- σ : fonction d'activation (ReLU, sigmoid, tanh)
```

**Backpropagation** :
```
1. Forward: calculer toutes les activations
2. Loss: L = loss(ŷ, y_true)
3. Backward: calculer gradients via chain rule
4. Update: w ← w - η ∇L/∂w
```

**Code minimal** :
```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),   # Fully connected
            nn.ReLU(),                           # Activation
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)

# Usage
model = MLP(input_dim=10, hidden_dim=64, output_dim=1)
```

**Comparaison avec autres architectures** :
| Architecture | Connexions | Paramètres | Use Case |
|--------------|------------|------------|----------|
| **MLP** | Fully connected | O(n·m) élevé | Tabulaire, petit |
| **CNN** | Locales + shared | O(k²·c) réduit | Images, séquences |
| **RNN** | Récurrentes | O(h²) moyen | Séquences |
| **Transformer** | Attention | O(n²·d) | Séquences longues |

## Exemples concrets

### 1. Classification binaire : Titanic

**Scénario** : Prédire survie des passagers (tabulaire).

**Code complet avec PyTorch** :
```python
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# 1. Charger données Titanic
titanic = fetch_openml('titanic', version=1, as_frame=True, parser='auto')
X = titanic.data[['pclass', 'age', 'sibsp', 'parch', 'fare']].fillna(0)
y = (titanic.target == '1').astype(int)

# 2. Split et normalisation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convertir en tensors
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train.values).unsqueeze(1)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.FloatTensor(y_test.values).unsqueeze(1)

# 3. Définir MLP
class TitanicMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Pour probabilités [0, 1]
        )
    
    def forward(self, x):
        return self.network(x)

model = TitanicMLP()
criterion = nn.BCELoss()  # Binary Cross Entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Entraînement
epochs = 100
for epoch in range(epochs):
    # Forward
    predictions = model(X_train_t)
    loss = criterion(predictions, y_train_t)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

# 5. Évaluation
model.eval()
with torch.no_grad():
    test_predictions = model(X_test_t)
    test_loss = criterion(test_predictions, y_test_t)
    accuracy = ((test_predictions > 0.5) == y_test_t).float().mean()
    
print(f'\nTest Loss: {test_loss:.4f}')
print(f'Test Accuracy: {accuracy:.2%}')
```

---

### 2. Régression : Prix maison

**Code avec scikit-learn (simple)** :
```python
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# 1. Données
housing = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(
    housing.data, housing.target, test_size=0.2, random_state=42
)

# 2. Normalisation
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. MLP Regressor
mlp = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32),  # 3 couches cachées
    activation='relu',
    solver='adam',
    alpha=0.001,  # L2 regularization
    batch_size=128,
    learning_rate_init=0.001,
    max_iter=200,
    random_state=42,
    verbose=True
)

# 4. Entraînement
mlp.fit(X_train, y_train)

# 5. Évaluation
y_pred = mlp.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'\nMSE: {mse:.4f}')
print(f'R²: {r2:.4f}')
print(f'Nombre de couches: {len(mlp.hidden_layer_sizes) + 1}')
print(f'Iterations: {mlp.n_iter_}')
```

---

### 3. Architecture profonde avec PyTorch

**MLP profond avec batch norm et dropout** :
```python
import torch
import torch.nn as nn

class DeepMLP(nn.Module):
    def __init__(self, input_dim=10, hidden_dims=[128, 256, 256, 128], 
                 output_dim=1, dropout=0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Couches cachées
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),  # Normalisation
                nn.ReLU(),
                nn.Dropout(dropout)           # Régularisation
            ])
            prev_dim = hidden_dim
        
        # Couche de sortie
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Créer modèle
model = DeepMLP(input_dim=20, hidden_dims=[256, 512, 512, 256], output_dim=10)

# Compter paramètres
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total paramètres: {total_params:,}")
print(f"Trainable: {trainable_params:,}")
print(f"\nArchitecture:\n{model}")
```

---

### 4. Visualiser gradients et activations

**Code pour debugging** :
```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Modèle simple
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)

# Fausse données
X = torch.randn(100, 10)
y = torch.randn(100, 1)

# Forward pass
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Entraîner et collecter gradients
gradient_norms = []

for epoch in range(50):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    
    # Collecter norme des gradients
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    gradient_norms.append(total_norm)
    
    optimizer.step()

# Visualiser
plt.figure(figsize=(10, 5))
plt.plot(gradient_norms)
plt.xlabel('Epoch')
plt.ylabel('Gradient Norm')
plt.title('Gradient Magnitude pendant Training')
plt.grid(True)
plt.show()

print(f"Gradient norm moyen: {np.mean(gradient_norms):.4f}")
print(f"Gradient norm final: {gradient_norms[-1]:.4f}")
```

---

### 5. Comparaison fonctions d'activation

**Code pour tester ReLU, Tanh, Sigmoid** :
```python
import torch
import torch.nn as nn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Données
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.LongTensor(y_train)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.LongTensor(y_test)

# Tester différentes activations
activations = {
    'ReLU': nn.ReLU(),
    'Tanh': nn.Tanh(),
    'Sigmoid': nn.Sigmoid(),
    'LeakyReLU': nn.LeakyReLU(0.1),
    'ELU': nn.ELU()
}

results = {}

for name, activation in activations.items():
    # Modèle
    model = nn.Sequential(
        nn.Linear(20, 64),
        activation,
        nn.Linear(64, 32),
        activation,
        nn.Linear(32, 2)
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Entraîner
    losses = []
    for epoch in range(100):
        optimizer.zero_grad()
        output = model(X_train_t)
        loss = criterion(output, y_train_t)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    # Évaluer
    model.eval()
    with torch.no_grad():
        test_output = model(X_test_t)
        _, predicted = torch.max(test_output, 1)
        accuracy = (predicted == y_test_t).float().mean()
    
    results[name] = {'losses': losses, 'accuracy': accuracy.item()}
    print(f"{name}: Accuracy = {accuracy:.2%}")

# Visualiser convergence
plt.figure(figsize=(12, 6))
for name, data in results.items():
    plt.plot(data['losses'], label=f"{name} (acc={data['accuracy']:.2%})")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Convergence par fonction d\'activation')
plt.legend()
plt.grid(True)
plt.show()
```

---

### 6. Learning rate scheduling

**Code avec différentes stratégies** :
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR, ReduceLROnPlateau

# Modèle simple
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

# Fausse données
X = torch.randn(1000, 10)
y = torch.randn(1000, 1)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

# Tester différents schedulers
# 1. StepLR: réduit LR tous les N epochs
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

# 2. ExponentialLR: décroissance exponentielle
# scheduler = ExponentialLR(optimizer, gamma=0.95)

# 3. ReduceLROnPlateau: réduit si loss plateau
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

lrs = []
losses = []

for epoch in range(100):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    
    # Update scheduler
    scheduler.step()  # Pour ReduceLROnPlateau: scheduler.step(loss)
    
    lrs.append(optimizer.param_groups[0]['lr'])
    losses.append(loss.item())
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}, LR: {lrs[-1]:.6f}, Loss: {loss.item():.4f}")

# Visualiser
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(lrs)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Learning Rate')
ax1.set_title('Learning Rate Schedule')
ax1.grid(True)

ax2.plot(losses)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_title('Training Loss')
ax2.grid(True)

plt.tight_layout()
plt.show()
```

## Quand l'utiliser

- ✅ **Données tabulaires** : CSV, bases de données, features structurées
- ✅ **Petit dataset** : Quelques milliers d'exemples
- ✅ **Baseline rapide** : Premier modèle à tester
- ✅ **Features pré-extraites** : Après PCA, TF-IDF, embeddings
- ✅ **Problèmes simples** : Classification/régression standard
- ✅ **Pas de structure spatiale/temporelle** : Ordre des features n'importe pas

**Cas d'usage typiques** :
- 📊 **Prédiction tabulaire** : Prix, churn, fraude
- 🏥 **Médical** : Diagnostic à partir de features extraites
- 💰 **Finance** : Scoring de crédit, prédiction de prix
- 🎯 **Recommandation** : Embeddings → score
- 🔬 **Science** : Features mesurées → prédiction

**Quand NE PAS utiliser** :
- ❌ Images → CNN (structure spatiale)
- ❌ Séquences/texte → RNN/Transformer (structure temporelle)
- ❌ Graphes → GNN (structure relationnelle)
- ❌ Très grand dataset (>1M) → Méthodes plus efficaces
- ❌ Beaucoup de features (>1000) → Réduction dimension d'abord

## Forces

✅ **Universel** : Approxime toute fonction (universal approximator)  
✅ **Simple** : Facile à implémenter et comprendre  
✅ **Flexible** : Adapte à classification, régression, multi-task  
✅ **Rapide** : Entraînement parallélisable sur GPU  
✅ **Bien étudié** : Beaucoup de documentation et outils  
✅ **Baseline** : Bon point de départ

**Théorème d'approximation universelle** :
```
Un MLP avec 1 couche cachée et suffisamment de neurones
peut approximer n'importe quelle fonction continue sur
un compact à une précision arbitraire.

Mais: "Suffisamment de neurones" peut être énorme!
→ En pratique: plusieurs couches profondes = plus efficace
```

## Limites

❌ **Beaucoup de paramètres** : O(n·m) pour n→m connexions  
❌ **Overfitting facile** : Surtout si peu de données  
❌ **Ignore structure** : Pas de notion de localité spatiale/temporelle  
❌ **Pas de shift invariance** : Sensible à position des features  
❌ **Difficile à interpréter** : Boîte noire  
❌ **Vanishing gradients** : Si trop profond sans attention

**Exemple de overfitting** :
```python
# Trop de paramètres pour peu de données
X_small = torch.randn(50, 10)   # 50 exemples seulement
y_small = torch.randn(50, 1)

# Modèle avec 100k paramètres!
huge_model = nn.Sequential(
    nn.Linear(10, 1000),
    nn.ReLU(),
    nn.Linear(1000, 1000),
    nn.ReLU(),
    nn.Linear(1000, 1)
)

# Résultat: overfitting massif
# Train loss → 0, Test loss élevé
```

**Comparaison avec autres modèles** :
```python
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import time

X, y = make_classification(n_samples=10000, n_features=20, random_state=42)

models = {
    'MLP': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=50),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'SVM': SVC()
}

for name, model in models.items():
    start = time.time()
    model.fit(X, y)
    elapsed = time.time() - start
    score = model.score(X, y)
    print(f"{name}: {elapsed:.2f}s, accuracy={score:.2%}")

# MLP: rapide mais peut overfit
# RF: robust, pas besoin normalisation
# SVM: lent, besoin normalisation
```

## Variantes / liens

### Architectures classiques

**1. Simple MLP** :
```python
nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, output_dim)
)
```

**2. Deep MLP** :
```python
nn.Sequential(
    nn.Linear(input_dim, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 512),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, output_dim)
)
```

**3. Residual MLP** :
```python
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
    
    def forward(self, x):
        return x + self.block(x)  # Skip connection

model = nn.Sequential(
    nn.Linear(input_dim, 256),
    ResidualBlock(256),
    ResidualBlock(256),
    nn.Linear(256, output_dim)
)
```

### Fonctions d'activation

| Activation | Formule | Avantages | Désavantages |
|------------|---------|-----------|--------------|
| **ReLU** | max(0, x) | Rapide, pas de vanishing | Dying ReLU |
| **LeakyReLU** | max(0.01x, x) | Pas de dying | Choix de α |
| **Tanh** | tanh(x) | Centré sur 0 | Vanishing gradients |
| **Sigmoid** | 1/(1+e⁻ˣ) | [0,1] pour proba | Vanishing gradients |
| **ELU** | x si x>0, α(eˣ-1) sinon | Smooth | Plus lent |
| **GELU** | x·Φ(x) | Utilisé dans transformers | Coûteux |

### Régularisation

**1. Dropout** :
```python
nn.Dropout(p=0.5)  # Drop 50% neurones pendant training
```

**2. Batch Normalization** :
```python
nn.BatchNorm1d(hidden_dim)  # Normalise activations
```

**3. Layer Normalization** :
```python
nn.LayerNorm(hidden_dim)  # Utilisé dans transformers
```

**4. Weight Decay (L2)** :
```python
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
```

### Optimizers

```python
# SGD classique
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Adam (recommandé par défaut)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# AdamW (Adam avec weight decay corrigé)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# RMSprop
optimizer = optim.RMSprop(model.parameters(), lr=0.001)
```

### Loss functions

**Classification** :
```python
# Binaire
nn.BCELoss()              # Binary Cross Entropy (avec sigmoid)
nn.BCEWithLogitsLoss()    # Combiné sigmoid + BCE (plus stable)

# Multi-classe
nn.CrossEntropyLoss()    # Combiné softmax + NLL (recommandé)
nn.NLLLoss()             # Negative Log Likelihood
```

**Régression** :
```python
nn.MSELoss()        # Mean Squared Error
nn.L1Loss()         # Mean Absolute Error
nn.SmoothL1Loss()   # Huber loss (robuste aux outliers)
```

## Références

### Documentation
- **PyTorch** : [nn.Module](https://pytorch.org/docs/stable/nn.html)
- **TensorFlow/Keras** : [Dense Layer](https://keras.io/api/layers/core_layers/dense/)
- **Scikit-learn** : [MLPClassifier/Regressor](https://scikit-learn.org/stable/modules/neural_networks_supervised.html)

### Papers fondamentaux
- **Perceptron** : Rosenblatt, 1958 - "The Perceptron: A Probabilistic Model"
- **Backpropagation** : Rumelhart et al., 1986 - "Learning Internal Representations by Error Propagation"
- **Universal Approximation** : Cybenko, 1989 / Hornik et al., 1989
- **Dropout** : Srivastava et al., 2014 - "Dropout: A Simple Way to Prevent Overfitting"
- **Batch Normalization** : Ioffe & Szegedy, 2015
- **Adam Optimizer** : Kingma & Ba, 2014

### Livres
- **Deep Learning** (Goodfellow et al., 2016) - Chapitres 6-8
- **Neural Networks and Deep Learning** (Nielsen) - [En ligne gratuit](http://neuralnetworksanddeeplearning.com/)

### Best practices

**1. Normalisation des données** :
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

**2. Choix architecture** :
```
Règle empirique:
- Largeur: 2-3x input_dim pour première couche
- Profondeur: 2-4 couches cachées
- Diminuer taille progressivement

Exemple: input=100 → [256, 128, 64, output]
```

**3. Hyperparamètres typiques** :
```python
# Bon point de départ
model = MLP(
    hidden_layers=[256, 128, 64],
    activation='relu',
    dropout=0.2,
    batch_norm=True
)

optimizer = optim.Adam(model.parameters(), lr=0.001)
batch_size = 32 ou 64
epochs = 100-200
```

**4. Early stopping** :
```python
best_loss = float('inf')
patience = 10
counter = 0

for epoch in range(max_epochs):
    val_loss = validate(model, val_loader)
    
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pt')
        counter = 0
    else:
        counter += 1
    
    if counter >= patience:
        print("Early stopping!")
        break
```
