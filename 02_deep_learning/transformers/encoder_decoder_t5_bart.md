# Encoder-Decoder (T5/BART-like)

Seq2seq pour traduction, résumé, etc.

## Idée clé

**T5/BART** : Encoder-Decoder Transformer pour **séquence-à-séquence**. Combine vision bidirectionnelle (encoder) et génération autoregressive (decoder).

**Architecture** :
```
Input → Encoder (bidirectional) → Context
                                    ↓
Context → Decoder (causal) → Output

Utilise cross-attention: Decoder attend sur encoder outputs
```

**T5 (Text-to-Text Transfer Transformer)** :
```
Tout est text-to-text:
- Translation: "translate English to French: Hello" → "Bonjour"
- Summarization: "summarize: [article]" → [summary]
- QA: "question: [q] context: [c]" → [answer]

→ Framework unifié pour toutes tâches NLP
```

**BART** : Denoising autoencoder (pré-training avec corruption)

## Exemples concrets

### 1. Summarization avec T5

```python
from transformers import T5ForConditional Generation, T5Tokenizer

model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')

article = """The tower is 324 metres tall, about the same height as an 81-storey building. 
It was the first structure to reach a height of 300 metres."""

input_text = "summarize: " + article
inputs = tokenizer(input_text, return_tensors='pt', max_length=512, truncation=True)

outputs = model.generate(inputs['input_ids'], max_length=50, num_beams=4, early_stopping=True)
summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(summary)
```

### 2. Translation

```python
# T5 multi-task
input_text = "translate English to German: How are you?"
inputs = tokenizer(input_text, return_tensors='pt')
outputs = model.generate(inputs['input_ids'])
print(tokenizer.decode(outputs[0]))
```

### 3. BART pour paraphrasing

```python
from transformers import BartForConditionalGeneration, BartTokenizer

model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

text = "Artificial intelligence is transforming our world."
inputs = tokenizer(text, return_tensors='pt')
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0]))
```

## Quand l'utiliser

- ✅ **Translation** : Seq2seq naturel
- ✅ **Summarization** : Abstractive summaries
- ✅ **Question Answering** : Generative QA
- ✅ **Paraphrasing** : Réécriture
- ✅ **Data-to-text** : Tableaux → descriptions

**Quand NE PAS utiliser** :
- ❌ Classification → BERT
- ❌ Text generation only → GPT
- ❌ Embeddings → BERT, Sentence-BERT

## Forces

✅ **Seq2seq optimal** : Encoder+Decoder  
✅ **Versatile** : Translation, summarization, QA  
✅ **Strong baselines** : Pre-trained sur C4, CNN/DM  
✅ **Text-to-text paradigm** (T5) : Unifié

## Limites

❌ **Plus lourd** : Encoder + Decoder  
❌ **Lent** : Génération séquentielle  
❌ **Moins populaire que GPT** : Pour génération pure

## Variantes / liens

**T5** : Text-to-text, pré-entraîné span corruption  
**BART** : Denoising autoencoder  
**mT5/mBART** : Multilingual  
**PEGASUS** : Optimisé summarization  
**MarianMT** : Translation spécialisé

## Références

- **T5** : Raffel et al., 2019 - "Exploring the Limits of Transfer Learning"
- **BART** : Lewis et al., 2019
- **HuggingFace** : [T5 docs](https://huggingface.co/docs/transformers/model_doc/t5)
