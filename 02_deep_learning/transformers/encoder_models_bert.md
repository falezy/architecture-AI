# Encoder models (BERT-like)

Encodeurs pour compréhension (classification, embeddings).

## Idée clé

**BERT (Bidirectional Encoder Representations from Transformers)** utilise uniquement l'**encoder** du Transformer pour créer des **représentations contextuelles bidirectionnelles**. Révolution NLP en 2018.

**Pré-entraînement bidirectionnel** :
```
RNN/GPT: → → → (unidirectionnel)
BERT:    ← → (bidirectionnel via masking)

"The cat sat on the mat"
Pour token "sat":
- Voit "cat" (gauche) ET "mat" (droite)
→ Compréhension contextuelle complète
```

**Deux tâches de pré-training** :

1. **MLM (Masked Language Model)** :
```
Input:  "The [MASK] sat on the mat"
Target: Prédire "cat"

15% tokens masqués:
- 80% → [MASK]
- 10% → token aléatoire  
- 10% → unchanged

→ Force à apprendre contexte bidirectionnel
```

2. **NSP (Next Sentence Prediction)** :
```
[CLS] Sentence A [SEP] Sentence B [SEP]
→ Prédire si B suit A

Exemples:
✓ "I love dogs. [SEP] They are friendly." → IsNext
✗ "I love dogs. [SEP] Paris is beautiful." → NotNext

→ Apprend relations entre phrases
```

**Architecture** :
```
Input: [CLS] token1 token2 ... [SEP]
       ↓
Token Embeddings + Position Embeddings + Segment Embeddings
       ↓
Transformer Encoder ×12 (base) ou ×24 (large)
       ↓
[CLS] token → Classification
Autres tokens → Token-level tasks (NER, QA)
```

**Fine-tuning** :
```
Après pré-training, ajouter une tête spécifique:
- Classification: Linear([CLS])
- NER: Linear(each token)
- QA: Start/End span prediction
```

## Exemples concrets

### 1. Classification avec BERT (HuggingFace)

**Code pour sentiment analysis** :
```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# 1. Charger modèle pré-entraîné
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2  # Binary classification
)

# 2. Préparer données
texts = [
    "This movie is amazing!",
    "Terrible waste of time."
]
labels = torch.tensor([1, 0])  # Positive, Negative

# Tokenize
encoded = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors='pt'
)

# 3. Fine-tuning
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

model.train()
for epoch in range(3):
    optimizer.zero_grad()
    
    outputs = model(
        input_ids=encoded['input_ids'],
        attention_mask=encoded['attention_mask'],
        labels=labels
    )
    
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    
    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

# 4. Prédiction
model.eval()
with torch.no_grad():
    test_text = ["I really enjoyed this film!"]
    test_encoded = tokenizer(test_text, return_tensors='pt', padding=True)
    
    outputs = model(**test_encoded)
    prediction = torch.argmax(outputs.logits, dim=-1)
    
    print(f"Sentiment: {'Positive' if prediction == 1 else 'Negative'}")
```

---

### 2. Named Entity Recognition (NER)

**Code pour extraction d'entités** :
```python
from transformers import BertTokenizerFast, BertForTokenClassification
from transformers import pipeline

# 1. Utiliser pipeline (simple)
ner_pipeline = pipeline(
    "ner",
    model="dbmdz/bert-large-cased-finetuned-conll03-english",
    aggregation_strategy="simple"
)

text = "Apple CEO Tim Cook announced new product in Cupertino."
entities = ner_pipeline(text)

for entity in entities:
    print(f"{entity['word']}: {entity['entity_group']} ({entity['score']:.2f})")

# Output:
# Apple: ORG (0.99)
# Tim Cook: PER (0.99)
# Cupertino: LOC (0.99)

# 2. Manuel (plus de contrôle)
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
model = BertForTokenClassification.from_pretrained(
    'bert-base-cased',
    num_labels=9  # B-PER, I-PER, B-LOC, I-LOC, B-ORG, I-ORG, B-MISC, I-MISC, O
)

# Tokenize
tokens = tokenizer(text, return_tensors='pt')

# Predict
with torch.no_grad():
    outputs = model(**tokens)
    predictions = torch.argmax(outputs.logits, dim=-1)

# Decoder
label_map = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-LOC', ...}
predicted_labels = [label_map[p.item()] for p in predictions[0]]
```

---

### 3. Question Answering

**Code pour extraction de réponses** :
```python
from transformers import BertForQuestionAnswering, BertTokenizer

# Charger modèle fine-tuné sur SQuAD
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

# Question et contexte
question = "What is the capital of France?"
context = "France is a country in Europe. Its capital is Paris, which is known for the Eiffel Tower."

# Tokenize
inputs = tokenizer(question, context, return_tensors='pt')

# Predict start/end positions
with torch.no_grad():
    outputs = model(**inputs)
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits
    
    start_idx = torch.argmax(start_scores)
    end_idx = torch.argmax(end_scores)

# Extract answer
answer_tokens = inputs['input_ids'][0][start_idx:end_idx+1]
answer = tokenizer.decode(answer_tokens)

print(f"Question: {question}")
print(f"Answer: {answer}")  # "Paris"
```

---

### 4. Feature Extraction (Embeddings)

**Code pour obtenir embeddings contextuels** :
```python
from transformers import BertModel, BertTokenizer
import torch

# Charger BERT
model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Texte
text = "BERT creates powerful embeddings"
inputs = tokenizer(text, return_tensors='pt')

# Forward pass
model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    
    # Dernière couche cachée
    last_hidden_state = outputs.last_hidden_state  # (batch, seq_len, 768)
    
    # [CLS] token embedding (pour classification)
    cls_embedding = last_hidden_state[:, 0, :]  # (batch, 768)
    
    # Mean pooling (pour similarity)
    mean_embedding = last_hidden_state.mean(dim=1)  # (batch, 768)

print(f"[CLS] embedding shape: {cls_embedding.shape}")
print(f"Mean embedding shape: {mean_embedding.shape}")

# Utiliser pour similarity
from torch.nn.functional import cosine_similarity

text1 = "The cat sits on the mat"
text2 = "A cat is on the mat"

emb1 = get_embedding(model, tokenizer, text1)
emb2 = get_embedding(model, tokenizer, text2)

similarity = cosine_similarity(emb1, emb2)
print(f"Similarity: {similarity.item():.4f}")
```

---

### 5. Training from scratch (MLM)

**Code pour pré-entraîner BERT sur votre corpus** :
```python
from transformers import BertConfig, BertForMaskedLM, BertTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# 1. Configuration
config = BertConfig(
    vocab_size=30522,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    max_position_embeddings=512
)

# 2. Initialiser modèle random
model = BertForMaskedLM(config)
print(f"Paramètres: {model.num_parameters():,}")

# 3. Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 4. Data collator pour MLM (masque automatiquement 15% tokens)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

# 5. Préparer dataset
texts = ["Your corpus here...", "More text...", ...]
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

# from datasets import Dataset
# dataset = Dataset.from_dict({"text": texts})
# tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 6. Training
training_args = TrainingArguments(
    output_dir="./bert-pretrained",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
    learning_rate=5e-5,
    warmup_steps=10_000,
)

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_dataset,
#     data_collator=data_collator,
# )
# trainer.train()
```

---

### 6. Visualiser attention

**Code pour voir patterns d'attention** :
```python
from transformers import BertModel, BertTokenizer
import matplotlib.pyplot as plt
import seaborn as sns

model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

text = "The quick brown fox jumps over the lazy dog"
inputs = tokenizer(text, return_tensors='pt')

with torch.no_grad():
    outputs = model(**inputs)
    attentions = outputs.attentions  # Tuple of (batch, heads, seq, seq)

# Visualiser layer 0, head 0
attn = attentions[0][0, 0].numpy()  # Premier layer, premier head
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

plt.figure(figsize=(10, 8))
sns.heatmap(attn, xticklabels=tokens, yticklabels=tokens, cmap='viridis')
plt.title('BERT Attention Layer 0, Head 0')
plt.tight_layout()
plt.show()
```

## Quand l'utiliser

- ✅ **Classification de texte** : Sentiment, topic, intent
- ✅ **NER** : Extraction d'entités nommées
- ✅ **Question Answering** : SQuAD, extractive QA
- ✅ **Similarity/Retrieval** : Semantic search
- ✅ **Feature extraction** : Embeddings pour downstream tasks
- ✅ **Multi-lingual** : mBERT pour 100+ langues

**Cas d'usage** :
- 📧 **Email classification** : Spam, catégories
- 🏥 **Medical NER** : Extraire diagnostics, médicaments
- 💼 **Document understanding** : Contracts, legal
- 🔍 **Search** : Semantic search avec BERT embeddings
- 🌍 **Translation alternatives** : Embeddings cross-lingual

**Quand NE PAS utiliser** :
- ❌ **Génération de texte** → GPT (decoder)
- ❌ **Seq2Seq** → T5, BART (encoder-decoder)
- ❌ **Très long contexte** (>512 tokens) → Longformer, BigBird
- ❌ **Real-time** → DistilBERT (plus rapide)

## Forces

✅ **Bidirectionnel** : Contexte complet gauche+droite  
✅ **Transfer learning** : Pré-entraîné sur énormes corpus  
✅ **State-of-the-art** : Dominant en NLU (2018-2020)  
✅ **Versatile** : Fine-tune pour many tasks  
✅ **Multi-lingual** : mBERT, XLM-R  
✅ **Interprétable** : Attention weights visualisables

**Performance boost** :
```
Avant BERT (2017):
- SQuAD: F1 = 82%
- GLUE benchmark: 70%

Après BERT (2018):
- SQuAD: F1 = 93%
- GLUE: 82%

→ +10-20% sur la plupart des tâches!
```

## Limites

❌ **Ne génère pas** : Encoder-only  
❌ **Max 512 tokens** : Séquences longues difficile  
❌ **Lent** : Inference coûteuse  
❌ **NSP peu utile** : RoBERTa l'abandonne  
❌ **Nécessite fine-tuning** : Pas zero-shot comme GPT-3

**Comparaison tailles** :
```
BERT-Base:  110M params, 12 layers, 768 hidden
BERT-Large: 340M params, 24 layers, 1024 hidden

Inference time (batch=1):
- Base: ~100ms/sequence
- Large: ~350ms/sequence
→ DistilBERT: ~40ms (97% performance, 40% faster)
```

## Variantes / liens

### Modèles de la famille BERT

**RoBERTa** (2019):
```python
from transformers import RobertaModel

# Améliorations:
# - Pas de NSP
# - Batch size plus grand
# - Plus de données
# - Dynamic masking
# → +2-3% vs BERT

model = RobertaModel.from_pretrained('roberta-base')
```

**DistilBERT** (2019):
```python
from transformers import DistilBertModel

# Knowledge distillation:
# - 6 layers au lieu de 12
# - 40% plus rapide
# - 40% moins de params
# - 97% des performances BERT

model = DistilBertModel.from_pretrained('distilbert-base-uncased')
```

**ALBERT** (2019):
```python
# Parameter sharing:
# - Partage poids entre layers
# - Factorized embeddings
# - 18x moins de params que BERT-large
# - Performances similaires

from transformers import AlbertModel
model = AlbertModel.from_pretrained('albert-base-v2')
```

**ELECTRA** (2020):
```python
# Replaced token detection au lieu de MLM:
# - Plus efficace en samples
# - Meilleur avec peu de données

from transformers import ElectraModel
model = ElectraModel.from_pretrained('google/electra-base-discriminator')
```

**DeBERTa** (2021):
```python
# Disentangled attention:
# - Attention position + content séparées
# - State-of-the-art sur SuperGLUE

from transformers import DebertaV2Model
model = DebertaV2Model.from_pretrained('microsoft/deberta-v3-base')
```

### Domain-specific BERT

```python
# SciBERT (scientific papers)
from transformers import AutoModel
sci_bert = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

# BioBERT (biomedical)
bio_bert = AutoModel.from_pretrained('dmis-lab/biobert-v1.1')

# ClinicalBERT (medical records)
clinical = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')

# FinBERT (finance)
fin_bert = AutoModel.from_pretrained('ProsusAI/finbert')

# Legal-BERT
legal = AutoModel.from_pretrained('nlpaueb/legal-bert-base-uncased')
```

### Training tricks

**Whole Word Masking** :
```
Standard MLM:
"play ##ing foot ##ball" → "play [MASK] foot ##ball"

WWM:
"play ##ing foot ##ball" → "[MASK] [MASK] foot ##ball"
→ Masque le mot entier
```

**Dynamic Masking** (RoBERTa):
```python
# Au lieu de masquer une fois pendant preprocessing:
# → Masquer différemment à chaque epoch
# → Modèle voit plus de variations
```

**Gradient Accumulation** :
```python
# Pour simuler gros batch avec GPU limité
accumulation_steps = 4
optimizer.zero_grad()

for i, batch in enumerate(train_loader):
    loss = model(batch).loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## Références

### Papers
- **BERT** : Devlin et al., 2018 - "BERT: Pre-training of Deep Bidirectional Transformers"
- **RoBERTa** : Liu et al., 2019
- **DistilBERT** : Sanh et al., 2019
- **ALBERT** : Lan et al., 2019
- **ELECTRA** : Clark et al., 2020
- **DeBERTa** : He et al., 2021

### Documentation
- **HuggingFace** : [BERT docs](https://huggingface.co/docs/transformers/model_doc/bert)
- **Original paper** : [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
- **Illustrated BERT** : [Jay Alammar](http://jalammar.github.io/illustrated-bert/)

### Best practices

**Fine-tuning tips** :
```python
# 1. Learning rate: 2e-5, 3e-5, 5e-5
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

# 2. Warmup
from transformers import get_linear_schedule_with_warmup
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=500,
    num_training_steps=total_steps
)

# 3. Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 4. Epochs: 2-4 (souvent 3)
# 5. Batch size: 16 ou 32
```

**Quick start** :
```python
from transformers import pipeline

# Sentiment
classifier = pipeline("sentiment-analysis")
classifier("I love this!")

# NER
ner = pipeline("ner")
ner("Apple CEO Tim Cook")

# Question Answering
qa = pipeline("question-answering")
qa(question="Who?", context="Tim Cook is CEO")

# Feature extraction
extractor = pipeline("feature-extraction", model="bert-base-uncased")
embeddings = extractor("Hello world")
```
