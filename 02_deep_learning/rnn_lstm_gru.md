# RNN / LSTM / GRU

Réseaux séquentiels historiques (avant Transformers).

## Idée clé

**RNN (Recurrent Neural Network)** traite des **séquences** en maintenant un **état caché** qui se propage dans le temps. LSTM et GRU sont des variantes avancées qui résolvent le problème du **vanishing gradient**.

**Architecture RNN basique** :
```
Séquence: x₁, x₂, x₃, ...

h₀  →  h₁  →  h₂  →  h₃
       ↑      ↑      ↑
       x₁     x₂     x₃
       ↓      ↓      ↓
       y₁     y₂     y₃

hₜ = tanh(Wₓₕ·xₜ + Wₕₕ·hₜ₋₁ + bₕ)
yₜ = Wₕᵧ·hₜ + bᵧ
```

**Problème : Vanishing Gradient** :
```
∂L/∂h₀ = ∂L/∂hₜ · ∏(∂hᵢ/∂hᵢ₋₁)
                  i=1→t

Si ∂hᵢ/∂hᵢ₋₁ < 1 → produit → 0 (vanishing)
Si ∂hᵢ/∂hᵢ₋₁ > 1 → produit → ∞ (exploding)

→ Difficile d'apprendre dépendances long terme
```

**Solution 1 : LSTM (Long Short-Term Memory)** :
```
Gates qui contrôlent flux d'information:

fₜ = σ(Wf·[hₜ₋₁, xₜ])  Forget gate
iₜ = σ(Wi·[hₜ₋₁, xₜ])  Input gate
c̃ₜ = tanh(Wc·[hₜ₋₁, xₜ])  Candidate
oₜ = σ(Wo·[hₜ₋₁, xₜ])  Output gate

cₜ = fₜ⊙cₜ₋₁ + iₜ⊙c̃ₜ   Cell state (mémoire)
hₜ = oₜ⊙tanh(cₜ)        Hidden state

⊙ = element-wise product
```

**Solution 2 : GRU (Gated Recurrent Unit)** :
```
Version simplifiée de LSTM (2 gates au lieu de 3):

zₜ = σ(Wz·[hₜ₋₁, xₜ])  Update gate
rₜ = σ(Wr·[hₜ₋₁, xₜ])  Reset gate
h̃ₜ = tanh(W·[rₜ⊙hₜ₋₁, xₜ])
hₜ = (1-zₜ)⊙hₜ₋₁ + zₜ⊙h̃ₜ

→ Moins de paramètres que LSTM
```

**Comparaison** :
| Modèle | Paramètres | Complexité | Performance |
|--------|------------|------------|-------------|
| RNN | 4·d·h | Simple | Vanishing gradient |
| LSTM | 4·(4·d·h) | Complexe | Excellent long terme |
| GRU | 3·(3·d·h) | Moyen | Comparable LSTM |

## Exemples concrets

### 1. Sentiment Analysis avec LSTM

**Code PyTorch complet** :
```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 1. Définir LSTM pour classification
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc ​= nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)        # (batch, seq_len, embed_dim)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Prendre dernière sortie
        last_output = lstm_out[:, -1, :]   # (batch, hidden_dim)
        
        # Classification
        out = self.dropout(last_output)
        out = self.fc(out)                  # (batch, num_classes)
        return out

# 2. Fausses données (exemple simplifié)
vocab_size = 1000
seq_length = 50
num_samples = 1000

# Simulate text data (token indices)
X_train = torch.randint(0, vocab_size, (num_samples, seq_length))
y_train = torch.randint(0, 2, (num_samples,))  # Binary sentiment

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 3. Entraînement
model = SentimentLSTM(vocab_size=vocab_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}')

# 4. Prédiction
model.eval()
with torch.no_grad():
    sample = X_train[0:1]
    prediction = model(sample)
    sentiment = torch.argmax(prediction, dim=1)
    print(f"Sentiment: {'Positive' if sentiment == 1 else 'Negative'}")
```

---

### 2. Time Series Forecasting avec LSTM

**Code pour prédiction séquentielle** :
```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 1. Générer données temporelles
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Série temporelle synthétique (sinusoïde + bruit)
t = np.linspace(0, 100, 1000)
data = np.sin(t) + 0.1 * np.random.randn(1000)

seq_length = 20
X, y = create_sequences(data, seq_length)

# Train/test split
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Convert to tensors
X_train_t = torch.FloatTensor(X_train).unsqueeze(-1)  # (N, seq, 1)
y_train_t = torch.FloatTensor(y_train)
X_test_t = torch.FloatTensor(X_test).unsqueeze(-1)
y_test_t = torch.FloatTensor(y_test)

# 2. Modèle LSTM
class TimeSeriesLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]  # Take last timestep
        prediction = self.fc(last_output)
        return prediction.squeeze()

model = TimeSeriesLSTM(hidden_size=64, num_layers=2)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 3. Entraînement
epochs = 50
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    predictions = model(X_train_t)
    loss = criterion(predictions, y_train_t)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}')

# 4. Évaluation
model.eval()
with torch.no_grad():
    test_predictions = model(X_test_t).numpy()

# Visualisation
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='True', alpha=0.7)
plt.plot(test_predictions, label='Predicted', alpha=0.7)
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.title('Time Series Forecasting with LSTM')
plt.legend()
plt.grid(True)
plt.show()

mse = np.mean((y_test - test_predictions)**2)
print(f'Test MSE: {mse:.6f}')
```

---

### 3. Text Generation avec GRU

**Code pour générer du texte caractère par caractère** :
```python
import torch
import torch.nn as nn

# 1. Préparer données
text = "hello world this is a simple example of character level language modeling"
chars = sorted(set(text))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for ch, i in char_to_idx.items()}

# Encoder le texte
encoded = [char_to_idx[ch] for ch in text]

# Créer séquences
seq_length = 10
X, y = [], []
for i in range(len(encoded) - seq_length):
    X.append(encoded[i:i+seq_length])
    y.append(encoded[i+seq_length])

X = torch.LongTensor(X)
y = torch.LongTensor(y)

# 2. Modèle GRU
class CharGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim=32, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        gru_out, hidden = self.gru(embedded, hidden)
        last_out = gru_out[:, -1, :]
        output = self.fc(last_out)
        return output, hidden

vocab_size = len(chars)
model = CharGRU(vocab_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 3. Entraînement
for epoch in range(100):
    optimizer.zero_grad()
    output, _ = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

# 4. Génération de texte
def generate_text(model, start_str, length=50):
    model.eval()
    with torch.no_grad():
        # Encoder start string
        current = [char_to_idx[ch] for ch in start_str[-seq_length:]]
        generated = start_str
        hidden = None
        
        for _ in range(length):
            x = torch.LongTensor([current])
            output, hidden = model(x, hidden)
            probs = torch.softmax(output[0], dim=0)
            next_idx = torch.multinomial(probs, 1).item()
            next_char = idx_to_char[next_idx]
            generated += next_char
            current = current[1:] + [next_idx]
        
        return generated

generated = generate_text(model, "hello ", length=50)
print(f"Generated: {generated}")
```

---

### 4. Bidirectional LSTM

**Code pour BiLSTM (contexte passé + futur)** :
```python
class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            bidirectional=True,  # ← Clé
            batch_first=True
        )
        
        # 2x hidden_dim car bidirectionnel
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        
        # Concat forward + backward du dernier timestep
        # lstm_out: (batch, seq, hidden*2)
        last_output = lstm_out[:, -1, :]
        
        output = self.fc(last_output)
        return output

# Utilisation identique à LSTM unidirectionnel
model = BiLSTM(vocab_size=1000)
print(f"Paramètres: {sum(p.numel() for p in model.parameters()):,}")
```

---

### 5. Attention Mechanism (pré-Transformer)

**Code pour attention sur LSTM** :
```python
class LSTMWithAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        
        # Attention layer
        self.attention = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)  # (batch, seq, hidden)
        
        # Calculer attention weights
        attention_scores = self.attention(lstm_out)  # (batch, seq, 1)
        attention_weights = torch.softmax(attention_scores, dim=1)
        
        # Weighted sum des outputs
        context = torch.sum(lstm_out * attention_weights, dim=1)  # (batch, hidden)
        
        output = self.fc(context)
        return output, attention_weights

model = LSTMWithAttention(vocab_size=1000)

# Exemple forward
x = torch.randint(0, 1000, (32, 50))  # batch=32, seq_len=50
output, weights = model(x)

print(f"Output shape: {output.shape}")          # (32, 2)
print(f"Attention weights: {weights.shape}")    # (32, 50, 1)
```

---

### 6. Comparison RNN vs LSTM vs GRU

**Code benchmark** :
```python
import time

class VanillaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

# Comparer
input_size, hidden_size, output_size = 50, 128, 10
models = {
    'RNN': VanillaRNN(input_size, hidden_size, output_size),
    'LSTM': LSTMModel(input_size, hidden_size, output_size),
    'GRU': GRUModel(input_size, hidden_size, output_size)
}

# Benchmark
x = torch.randn(64, 100, input_size)  # batch, seq_len, features

for name, model in models.items():
    params = sum(p.numel() for p in model.parameters())
    
    start = time.time()
    for _ in range(100):
        _ = model(x)
    elapsed = time.time() - start
    
    print(f"{name}: {params:,} params, {elapsed:.3f}s")

# Output typique:
# RNN: 24,330 params, 0.234s
# LSTM: 90,890 params, 0.456s  ← 4x plus de params
# GRU: 69,130 params, 0.378s   ← Compromis
```

## Quand l'utiliser

- ✅ **Séquences courtes/moyennes** : <500 tokens
- ✅ **Données séquentielles** : Texte, audio, vidéo, time series
- ✅ **Petit dataset** : Quelques milliers d'exemples
- ✅ **Mémoire limitée** : Moins de RAM que Transformers
- ✅ **Online/streaming** : Traitement token par token

**Cas d'usage typiques** :
- 📝 **Sentiment analysis** : Critiques, tweets
- 📈 **Time series** : Prédiction stock, météo
- 🎵 **Audio** : Reconnaissance vocale
- 💬 **Chatbots simples** : Réponses courtes
- 🔤 **Named Entity Recognition** : BiLSTM-CRF
- 🎙️ **Speech-to-text** : Anciennes architectures

**Quand NE PAS utiliser** :
- ❌ Séquences longues (>500) → **Transformers** (meilleur)
- ❌ Parallélisation nécessaire → **Transformers** (plus rapide)
- ❌ Long-range dependencies → **Transformers**
- ❌ Tabulaire → MLP, XGBoost
- ❌ Images → CNN, Vision Transformers

## Forces

✅ **Traite séquences** : Ordre important  
✅ **Mémoire** : État caché conserve information  
✅ **Petit modèle** : Moins de params que Transformers  
✅ **Online processing** : Token par token  
✅ **LSTM/GRU** : Résout vanishing gradient  
✅ **Bien compris** : Beaucoup de recherche

**Avantage streaming** :
```python
# RNN peut traiter token par token
hidden = None
for token in stream:
    output, hidden = model(token, hidden)
    yield output

# Transformers besoin de toute la séquence
```

## Limites

❌ **Vanishing/exploding gradients** : RNN vanilla  
❌ **Séquences longues** : LSTM limité à ~500 tokens  
❌ **Lent** : Séquentiel, pas parallélisable  
❌ **Long-range dependencies** : Difficile même pour LSTM  
❌ **Transformers meilleurs** : Depuis 2017  
❌ **Difficile à entraîner** : Gradient clipping nécessaire

**Comparaison LSTM vs Transformer** :
```python
# LSTM: O(n) séquentiel
for t in range(seq_len):
    h[t] = lstm(x[t], h[t-1])  # Doit attendre h[t-1]

# Transformer: O(1) parallèle
# Tous les tokens traités en parallèle
Q, K, V = linear(X)
attention = softmax(Q @ K.T) @ V  # Matrix ops parallèles
```

## Variantes / liens

### Types de RNN

**1. Vanilla RNN** :
```python
nn.RNN(input_size, hidden_size, num_layers)
# Simple mais vanishing gradient
```

**2. LSTM** :
```python
nn.LSTM(input_size, hidden_size, num_layers, bidirectional=False)
# 3 gates: forget, input, output
# Cell state + hidden state
```

**3. GRU** :
```python
nn.GRU(input_size, hidden_size, num_layers)
# 2 gates: reset, update
# Moins de params que LSTM
```

**4. Bidirectional** :
```python
nn.LSTM(..., bidirectional=True)
# Traite séquence dans les deux sens
# Utilise forward + backward context
```

### Architectures

**Many-to-One** (Sentiment):
```
x₁ → x₂ → x₃ → x₄
              ↓
              y
```

**Many-to-Many** (Traduction):
```
x₁ → x₂ → x₃ (Encoder)
              ↓
         y₁ → y₂ → y₃ (Decoder)
```

**One-to-Many** (Image captioning):
```
x → y₁ → y₂ → y₃ → ...
```

**Seq-to-Seq** :
```python
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, src, trg):
        # Encoder
        encoder_outputs, hidden = self.encoder(src)
        
        # Decoder avec context du encoder
        outputs = self.decoder(trg, hidden)
        return outputs
```

### Techniques importantes

**1. Gradient Clipping** :
```python
# Éviter exploding gradients
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
```

**2. Teacher Forcing** :
```python
# Entraînement: utiliser true target comme input
for t in range(seq_len):
    output = decoder(trg[t], hidden)  # trg[t] = vrai token
    
# Inference: utiliser prediction précédente
output = decoder(predicted_token, hidden)
```

**3. Scheduled Sampling** :
```python
# Mix teacher forcing et predictions
use_teacher = random.random() < teacher_forcing_ratio
input_token = trg[t] if use_teacher else predicted_token
```

**4. Layer Normalization** :
```python
class LSTMWithLayerNorm(nn.Module):
    def __init__(self, ...):
        self.lstm = nn.LSTM(...)
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x):
        out, hidden = self.lstm(x)
        out = self.layer_norm(out)
        return out, hidden
```

### PyTorch RNN API

```python
# Basique
rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

# Avec dropout entre layers
lstm = nn.LSTM(input_size, hidden_size, num_layers=3, dropout=0.2)

# Bidirectionnel
lstm = nn.LSTM(input_size, hidden_size, bidirectional=True)

# Forward pass
output, (hidden, cell) = lstm(x)  # LSTM
output, hidden = gru(x)           # GRU/RNN

# x shape: (batch, seq_len, input_size) si batch_first=True
# output: (batch, seq_len, hidden_size * num_directions)
# hidden: (num_layers * num_directions, batch, hidden_size)
```

## Références

### Papers fondamentaux
- **RNN** : Rumelhart et al., 1986 - "Learning representations by back-propagating errors"
- **LSTM** : Hochreiter & Schmidhuber, 1997 - "Long Short-Term Memory"
- **GRU** : Cho et al., 2014 - "Learning Phrase Representations using RNN Encoder-Decoder"
- **Seq2Seq** : Sutskever et al., 2014 - "Sequence to Sequence Learning with Neural Networks"
- **Attention** : Bahdanau et al., 2014 - "Neural Machine Translation by Jointly Learning to Align"
- **Bidirectional RNN** : Schuster & Paliwal, 1997

### Documentation
- **PyTorch** : [RNN](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html), [LSTM](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html), [GRU](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html)
- **Keras** : [LSTM](https://keras.io/api/layers/recurrent_layers/lstm/), [GRU](https://keras.io/api/layers/recurrent_layers/gru/)

### Best practices

**Architecture tips** :
```
Règles empiriques:
- Hidden size: 128-512 (selon complexité)
- Num layers: 1-3 (2 est bon défaut)
- Dropout: 0.2-0.5 entre layers
- Bidirectional: si pas online processing
```

**Training tips** :
```python
# 1. Gradient clipping (CRITICAL!)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

# 2. Learning rate schedule
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

# 3. Initialisation
for name, param in model.named_parameters():
    if 'weight' in name:
        nn.init.xavier_uniform_(param)
    elif 'bias' in name:
        nn.init.zeros_(param)
```

**Quand utiliser LSTM vs GRU** :
```
LSTM:
- Long sequences
- Complex dependencies
- Plus de données

GRU:
- Moins de données
- Plus rapide
- Performances souvent similaires

→ Essayer les deux, GRU d'abord (plus simple)
```

**Transition vers Transformers** :
```python
# RNN/LSTM: ordre historique
# Maintenant: Transformers dominant

Pour nouvelles tâches:
- Texte: BERT, GPT (Transformers)
- Time series: LSTMLSTM encore OK, ou Temporal Fusion Transformer
- Audio: Wav2Vec2, Whisper (Transformers)

RNN/LSTM utile si:
- Ressources limitées
- Online/streaming critical
- Dataset très petit
```
