# Transformer

Architecture à attention (self-attention) pour séquences.

## Idée clé

**Transformer** remplace la récurrence (RNN/LSTM) par l'**attention** pour traiter les séquences en **parallèle**. C'est l'architecture qui a révolutionné le NLP et au-delà (2017).

**Self-Attention : l'innovation clé** :
```
Pour chaque token, calculer sa relation avec TOUS les autres tokens

Attention(Q, K, V) = softmax(QK^T / √d_k) V

Q = Query (ce que je cherche)
K = Key (suis-je pertinent?)
V = Value (information à transmettre)

Exemple "The cat sat on the mat":
Token "sat" attend à:
- "cat" (qui fait l'action?)
- "mat" (où?)
→ Apprend les dépendances automatiquement!
```

**Architecture complète** :
```
Encoder:
Input → Embedding + Positional Encoding
      → [Multi-Head Attention + Add&Norm
      → Feed-Forward + Add&Norm] ×N
      
Decoder:
Output → Embedding + Positional Encoding
       → [Masked Multi-Head Attention + Add&Norm
       → Cross Attention (with encoder) + Add&Norm
       → Feed-Forward + Add&Norm] ×N
       → Linear → Softmax
```

**Multi-Head Attention** :
```python
# Au lieu d'une seule attention:
Attention(Q, K, V)

# Utiliser H "têtes" en parallèle:
head_i = Attention(Q·W_i^Q, K·W_i^K, V·W_i^V)
MultiHead = Concat(head_1, ..., head_H) · W^O

→ Chaque tête apprend différents patterns
```

**Positional Encoding** :
```
Transformer n'a pas de notion d'ordre!
→ Ajouter encodage de position

PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

→ Permet de distinguer position dans séquence
```

**Avantages vs RNN** :
- **Parallélisable** : O(1) au lieu de O(n)
- **Long-range** : Attention directe entre tokens éloignés
- **Interprétable** : Visualiser attention weights

## Exemples concrets

### 1. Transformer from scratch (simple)

**Code PyTorch implémentation minimale** :
```python
import torch
import torch.nn as nn
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k
    
    def forward(self, Q, K, V, mask=None):
        # Q, K, V: (batch, seq_len, d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(self.d_k)
    
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # Linear projections and split into heads
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        x, attention_weights = self.attention(Q, K, V, mask)
        
        # Concat heads
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final linear
        output = self.W_o(x)
        
        return output, attention_weights

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class FeedForward(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention + residual
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward + residual
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

# Test
d_model, num_heads = 512, 8
model = TransformerEncoderLayer(d_model, num_heads)

# Input: (batch, seq_len, d_model)
x = torch.randn(32, 100, d_model)
output = model(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Paramètres: {sum(p.numel() for p in model.parameters()):,}")
```

---

### 2. Translation avec PyTorch Transformer

**Code utilisant nn.Transformer** :
```python
import torch
import torch.nn as nn

class TranslationTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, 
                 nhead=8, num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()
        
        self.d_model = d_model
        
        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        
        # Output projection
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Embed and add positional encoding
        src = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt = self.pos_encoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        
        # Transformer
        output = self.transformer(src, tgt, 
                                 src_mask=src_mask,
                                 tgt_mask=tgt_mask)
        
        # Project to vocab
        return self.fc_out(output)
    
    def generate_square_subsequent_mask(self, sz):
        """Mask pour empêcher attention sur tokens futurs"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask

# Usage
src_vocab, tgt_vocab = 10000, 8000
model = TranslationTransformer(src_vocab, tgt_vocab)

# Exemple forward
src = torch.randint(0, src_vocab, (32, 50))  # batch=32, src_len=50
tgt = torch.randint(0, tgt_vocab, (32, 40))  # tgt_len=40

# Créer masque pour décoder
tgt_mask = model.generate_square_subsequent_mask(40)

output = model(src, tgt, tgt_mask=tgt_mask)
print(f"Output shape: {output.shape}")  # (32, 40, tgt_vocab)

print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
```

---

### 3. Visualiser Attention Weights

**Code pour voir ce que le modèle attend** :
```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention(attention_weights, src_tokens, tgt_tokens):
    """
    attention_weights: (seq_len_tgt, seq_len_src)
    """
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(
        attention_weights.detach().cpu().numpy(),
        xticklabels=src_tokens,
        yticklabels=tgt_tokens,
        cmap='viridis',
        cbar=True
    )
    
    plt.xlabel('Source Tokens')
    plt.ylabel('Target Tokens')
    plt.title('Attention Weights')
    plt.tight_layout()
    plt.show()

# Exemple
src_tokens = ["The", "cat", "sat", "on", "the", "mat", "<EOS>"]
tgt_tokens = ["Le", "chat", "était", "assis", "<EOS>"]

# Simuler attention weights
attention = torch.randn(len(tgt_tokens), len(src_tokens))
attention = torch.softmax(attention, dim=-1)

visualize_attention(attention, src_tokens, tgt_tokens)
```

---

### 4. Text Classification avec Transformer Encoder

**Code pour sentiment analysis** :
```python
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=2, num_classes=2):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Classification head
        self.fc = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        # Transformer
        x = self.transformer(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Classification
       return self.fc(x)

# Usage
vocab_size = 10000
model = TransformerClassifier(vocab_size)

# Training loop (simplifié)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Fausse données
X_train = torch.randint(0, vocab_size, (1000, 100))
y_train = torch.randint(0, 2, (1000,))

for epoch in range(5):
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    
    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
```

---

### 5. Attention Patterns visuels

**Code pour analyser les patterns d'attention** :
```python
def analyze_attention_heads(model, tokens):
    """Visualiser ce que chaque attention head apprend"""
    model.eval()
    
    with torch.no_grad():
        # Get attention weights from all heads
        _, attention_weights = model.attention(tokens, tokens, tokens)
        # attention_weights: (batch, num_heads, seq_len, seq_len)
    
    num_heads = attention_weights.size(1)
    seq_len = attention_weights.size(2)
    
    fig, axes = plt.subplots(2, num_heads//2, figsize=(20, 8))
    axes = axes.ravel()
    
    for head_idx in range(num_heads):
        attn = attention_weights[0, head_idx].cpu().numpy()
        
        ax = axes[head_idx]
        im = ax.imshow(attn, cmap='viridis')
        ax.set_title(f'Head {head_idx+1}')
        ax.set_xlabel('Key')
        ax.set_ylabel('Query')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.show()

# Test avec modèle simple
d_model, num_heads = 128, 8
attention = MultiHeadAttention(d_model, num_heads)

tokens = torch.randn(1, 20, d_model)  # 20 tokens
analyze_attention_heads(attention, tokens)
```

---

### 6. Learning Rate Warmup (crucial pour Transformers)

**Code avec warmup scheduler** :
```python
class WarmupScheduler:
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0
    
    def step(self):
        self.step_num += 1
        lr = self.d_model ** (-0.5) * min(
            self.step_num ** (-0.5),
            self.step_num * self.warmup_steps ** (-1.5)
        )
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr

# Usage
model = TranslationTransformer(10000, 8000, d_model=512)
optimizer = torch.optim.Adam(model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9)
scheduler = WarmupScheduler(optimizer, d_model=512, warmup_steps=4000)

# Training loop
lrs = []
for step in range(10000):
    # Training step...
    lr = scheduler.step()
    lrs.append(lr)

# Visualiser learning rate schedule
plt.figure(figsize=(10, 6))
plt.plot(lrs)
plt.xlabel('Training Steps')
plt.ylabel('Learning Rate')
plt.title('Transformer Learning Rate Schedule with Warmup')
plt.grid(True)
plt.show()
```

## Quand l'utiliser

- ✅ **NLP** : Texte, traduction, summarization
- ✅ **Séquences longues** : >500 tokens (vs RNN)
- ✅ **Parallélisation** : Many GPUs disponibles
- ✅ **Transfer learning** : BERT, GPT pré-entraînés
- ✅ **Vision** : ViT (Vision Transformer)
- ✅ **Multimodal** : Texte + image (CLIP, Flamingo)

**Cas d'usage** :
- 📝 **NLP** : BERT, GPT, T5 (state-of-the-art)
- 🖼️ **Vision** : ViT, DINO, Swin Transformer
- 🎵 **Audio** : Wav2Vec2, Whisper
- 🧬 ** Bio** : AlphaFold, ProteinBERT
- 🎮 **RL** : Decision Transformer
- 🎨 **Generation** : Stable Diffusion, DALL-E

**Quand NE PAS utiliser** :
- ❌ Petit dataset (<10k) sans pré-training
- ❌ Ressources limitées (CPU only)
- ❌ Online/streaming (RNN meilleur)
- ❌ Tabulaire (XGBoost meilleur)

## Forces

✅ **Parallélisable** : Training rapide sur GPU  
✅ **Long-range dependencies** : Attention directe  
✅ **State-of-the-art** : Domine NLP/Vision  
✅ **Transfer learning** : Pré-training puissant  
✅ **Interprétable** : Attention weights visualisables  
✅ **Flexible** : Encoder-only, Decoder-only, Enc-Dec

**Scaling laws** :
```
Performances ∝ (Taille modèle, Données, Compute)

GPT-3: 175B paramètres
PaLM: 540B paramètres
→ Bigger is better (jusqu'à présent)
```

## Limites

❌ **Quadratique en mémoire** : O(n²) pour attention  
❌ **Beaucoup de données** : Nécessite pré-training  
❌ **Compute intensif** : Coûteux à entraîner  
❌ **Séquences très longues** : >4096 tokens difficile  
❌ **Overfitting facile** : Si pas assez de données

**Complexité attention** :
```python
# Self-attention: O(n² · d)
QK^T = (n × d) @ (d × n) = O(n² · d)

Pour n=1000, d=512:
→ 512M opérations par layer!

Solutions:
- Linformer: O(n · k)
- Longformer: O(n · w) (window attention)
- Flash Attention: Optimisation GPU
```

## Variantes / liens

### Types d'architectures

**1. Encoder-only** (BERT):
```python
nn.TransformerEncoder(encoder_layer, num_layers)

# Use case: Classification, NER, Q&A
```

**2. Decoder-only** (GPT):
```python
nn.TransformerDecoder(decoder_layer, num_layers)

# Use case: Generation, completion
# Masked attention (causal)
```

**3. Encoder-Decoder** (T5, BART):
```python
nn.Transformer(...)

# Use case: Translation, summarization
```

### Efficient Transformers

**Linformer** :
```
Attention linéaire: O(n) au lieu de O(n²)
Q @ (K^T · E) @ V où E projette n→k
```

**Longformer** :
```
Window + global attention
(512 tokens) + (quelques tokens globaux)
```

**BigBird** :
```
Sparse attention:
- Local window
- Global tokens
- Random connections
```

**Flash Attention** :
```
Optimisation GPU/mémoire
Même complexité mais 3-10x plus rapide
```

### Vision Transformers

```python
from timm import create_model

# Vision Transformer
vit = create_model('vit_base_patch16_224', pretrained=True)

# Swin Transformer (hierarchical)
swin = create_model('swin_base_patch4_window7_224', pretrained=True)

# Image:
# 1. Split en patches 16×16
# 2. Linear projection
# 3. Transformer encoder
# 4. Classification head
```

### Key Components en détail

**Layer Normalization** :
```python
# Avant (Pre-LN, standard maintenant):
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))

# Original (Post-LN):
x = LayerNorm(x + Attention(x))
x = LayerNorm(x + FFN(x))
```

**Feed-Forward Network** :
```python
# Simple 2-layer MLP
FFN(x) = max(0, x·W1 + b1)·W2 + b2

# GLU variant (GPT-style):
FFN(x) = (x·W1) ⊙ σ(x·W2) · W3
```

**Attention variants** :
```python
# Scaled Dot-Product (standard)
Attention = softmax(QK^T / √d_k) V

# Additive (Bahd​anau)
Attention = softmax(v^T · tanh(W[Q;K])) V

# Relative Position (T5, Transformer-XL)
Attention avec biais de position relatifs
```

## Références

### Papers fondamentaux
- **Transformer** : Vaswani et al., 2017 - "Attention is All You Need"
- **BERT** : Devlin et al., 2018
- **GPT** : Radford et al., 2018
- **GPT-2/3** : Radford et al., 2019 / Brown et al., 2020
- **T5** : Raffel et al., 2019
- **ViT** : Dosovitskiy et al., 2020
- **Flash Attention** : Dao et al., 2022

### Documentation
- **PyTorch** : [nn.Transformer](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html)
- **Hugging Face**: [Transformers library](https://huggingface.co/docs/transformers)
- **Annotated Transformer** : [Harvard NLP](http://nlp.seas.harvard.edu/annotated-transformer/)

### Best practices

**Training tips** :
```python
# 1. Warmup LR schedule (CRITICAL)
scheduler = WarmupScheduler(optimizer, d_model=512, warmup_steps=4000)

# 2. Label smoothing
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# 3. Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 4. Mixed precision
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    output = model(x)
    loss = criterion(output, y)
scaler.scale(loss).backward()
scaler.step(optimizer)
```

**Architecture choices** :
```
Standard setups:
- Base: d=512, heads=8, layers=6
- Large: d=1024, heads=16, layers=12
- XL: d=2048, heads=32, layers=24

d_ff = 4 × d_model (standard)
dropout = 0.1
```

**Pour commencer** :
```python
# Ne PAS entraîner from scratch!
# Utiliser modèles pré-entraînés:

from transformers import AutoModel, AutoTokenizer

# BERT
model = AutoModel.from_pretrained('bert-base-uncased')

# GPT-2
model = AutoModel.from_pretrained('gpt2')

# T5
model = AutoModel.from_pretrained('t5-base')

# Fine-tune sur votre tâche
```
