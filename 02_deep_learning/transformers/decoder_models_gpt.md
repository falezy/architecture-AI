# Decoder models (GPT-like)

Décodeurs autoregressifs pour génération.

## Idée clé

**GPT (Generative Pre-trained Transformer)** utilise uniquement le **decoder** du Transformer pour générer du texte de manière **autoregressive** (token par token). Dominant pour la génération de texte.

**Génération autoregressive** :
```
Prédire le prochain token basé sur les précédents:

Given: "The cat"
Predict: "sat"

Given: "The cat sat"
Predict: "on"

→ Génération séquentielle: P(word_t | word_<t)
```

**Causal Masking** :
```
BERT: Voit tous les tokens (bidirectionnel)
GPT:  Voit seulement tokens précédents (causal)

"The cat sat on the mat"
Pour prédire "sat":
- ✓ Voit: "The cat"
- ✗ NE voit PAS: "on the mat"

→ Masque attention sur tokens futurs
```

**Pré-entraînement** :
```
Objectif: Causal Language Modeling (CLM)

Input:  "The cat sat on"
Target: "cat sat on the"

Loss = CrossEntropy(predicted, target)

→ Apprend à prédire token suivant
```

**GPT vs BERT** :
| | GPT | BERT |
|---|---|---|
| **Architecture** | Decoder-only | Encoder-only |
| **Attention** | Causal (unidirectional) | Bidirectional |
| **Training** | CLM (next token) | MLM (masked tokens) |
| **Use case** | Génération | Compréhension |
| **Zero-shot** | Oui (GPT-3+) | Non |

## Exemples concrets

### 1. Text Generation

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

prompt = "Once upon a time"
inputs = tokenizer(prompt, return_tensors='pt')

outputs = model.generate(
    inputs['input_ids'],
    max_length=50,
    temperature=0.8,
    top_p=0.95,
    do_sample=True
)

print(tokenizer.decode(outputs[0]))
```

### 2. Sampling Strategies

```python
# Greedy, Beam, Top-k, Top-p
greedy = model.generate(inputs['input_ids'], max_length=30)
beam = model.generate(inputs['input_ids'], num_beams=5)
topk = model.generate(inputs['input_ids'], top_k=50, do_sample=True)
topp = model.generate(inputs['input_ids'], top_p=0.95, do_sample=True)
```

### 3. Chat/Dialogue

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-medium')
tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')

chat_history = None
user_input = "Hello, how are you?"
new_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

if chat_history is not None:
    bot_input = torch.cat([chat_history, new_ids], dim=-1)
else:
    bot_input = new_ids

chat_history = model.generate(bot_input, max_length=1000, pad_token_id=tokenizer.eos_token_id)
response = tokenizer.decode(chat_history[:, bot_input.shape[-1]:][0])
print(f"Bot: {response}")
```

## Quand l'utiliser

- ✅ **Génération de texte** : Articles, histoires, code
- ✅ **Chatbots** : Dialogue, assistance
- ✅ **Code generation** : Codex, GitHub Copilot
- ✅ **Few-shot learning** : GPT-3+

**Quand NE PAS utiliser** :
- ❌ Classification → BERT
- ❌ NER → BERT
- ❌ Extractive QA → BERT

## Forces

✅ **Génération fluide** : Texte cohérent  
✅ **Zero/few-shot** : GPT-3+ sans fine-tuning  
✅ **Versatile** : Texte, code, dialogue  
✅ **Scaling** : Plus grand = meilleur

## Limites

❌ **Unidirectionnel** : Ne voit pas le futur  
❌ **Hallucinations** : Invente des faits  
❌ **Répétitions** : Boucles parfois  
❌ **Coût** : GPT-3/4 API coûteuse

## Variantes / liens

**GPT-2** (2019):
```python
gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')  # 117M
gpt2_xl = GPT2LMHeadModel.from_pretrained('gpt2-xl')  # 1.5B
```

**GPT-3** (2020): 175B params, API only

**GPT-4** (2023): Multimodal

**Open alternatives**:
```python
gptj = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
neo = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")
bloom = AutoModelForCausalLM.from_pretrained("bigscience/bloom")
```

**Techniques** :
- Temperature: contrôle randomness
- Top-p (nucleus): sampling adaptatif
- Repetition penalty: évite boucles

## Références

- **GPT** : Radford et al., 2018
- **GPT-2** : Radford et al., 2019
- **GPT-3** : Brown et al., 2020
- **HuggingFace** : [GPT-2 docs](https://huggingface.co/docs/transformers/model_doc/gpt2)
- **OpenAI API** : [platform.openai.com](https://platform.openai.com/)
