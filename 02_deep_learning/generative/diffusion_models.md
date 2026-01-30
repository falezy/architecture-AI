# Diffusion Models

Génération par débruitage progressif (images, audio, etc.).

## Idée clé

**Diffusion Models** : Génère données en **reverse** d'un processus de **noising progressif**. State-of-the-art pour images (Stable Diffusion, DALL-E 2, Imagen).

**Forward process** (ajout de bruit) :
```
x₀ → x₁ → x₂ → ... → xₜ → ... → noise
Clean   Progressivement bruité    Pure noise

xₜ = √(αₜ)·x₀ + √(1-αₜ)·ε,  ε ~ N(0,1)
```

**Reverse process** (génération) :
```
noise → xₜ₋₁ → ... → x₁ → x₀
            Débruitage progressif

Apprend pₐ(xₜ₋₁ | xₜ) pour chaque step
```

**Training** :
```python
# Prédire le bruit à chaque step
loss = ||ε - ε_θ(xₜ, t)||²

où ε = bruit ajouté, ε_θ = réseau qui prédit bruit
```

## Exemples concrets

### 1. DDPM simplifié

```python
import torch
import torch.nn as nn

class SimpleDiffusion:
    def __init__(self, num_steps=1000):
        self.num_steps = num_steps
        # Variance schedule (beta)
        self.betas = torch.linspace(1e-4, 0.02, num_steps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    
    def add_noise(self, x0, t):
        """Forward process: ajoute bruit"""
        noise = torch.randn_like(x0)
        alpha_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        
        # xₜ = √(αₜ)·x₀ + √(1-αₜ)·ε
        xt = torch.sqrt(alpha_t) * x0 + torch.sqrt(1 - alpha_t) * noise
        return xt, noise
    
    def denoise_step(self, model, xt, t):
        """Reverse process: enlève bruit"""
        # Prédire bruit
        noise_pred = model(xt, t)
        
        alpha_t = self.alphas_cumprod[t]
        beta_t = self.betas[t]
        
        # Reconstruire x_{t-1}
        xt_1 = (xt - beta_t / torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(1 - beta_t)
        
        # Ajouter un peu de bruit (sauf au dernier step)
        if t > 0:
            noise = torch.randn_like(xt)
            xt_1 += torch.sqrt(beta_t) * noise
        
        return xt_1

# Training
model = UNet()  # U-Net pour prédire bruit
diffusion = SimpleDiffusion()

for images in dataloader:
    # Random timestep
    t = torch.randint(0, diffusion.num_steps, (images.size(0),))
    
    # Add noise
    xt, noise = diffusion.add_noise(images, t)
    
    # Predict noise
    noise_pred = model(xt, t)
    
    # Loss
    loss = nn.functional.mse_loss(noise_pred, noise)
    loss.backward()
```

### 2. Génération d'images

```python
@torch.no_grad()
def generate(model, diffusion, shape):
    # Start from pure noise
    xt = torch.randn(shape)
    
    # Iterative denoising
    for t in reversed(range(diffusion.num_steps)):
        xt = diffusion.denoise_step(model, xt, t)
    
    return xt

# Générer
generated_images = generate(model, diffusion, (16, 3, 32, 32))
```

### 3. Stable Diffusion (text-to-image)

```python
from diffusers import StableDiffusionPipeline

# Charger modèle pré-entraîné
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2")
pipe = pipe.to("cuda")

# Générer à partir de prompt
prompt = "A beautiful sunset over mountains, digital art"
image = pipe(prompt, num_inference_steps=50).images[0]

image.save("sunset.png")
```

## Quand l'utiliser

- ✅ **Image generation** : State-of-the-art qualité
- ✅ **Text-to-image** : DALL-E 2, Stable Diffusion, Imagen
- ✅ **Audio synthesis** : WaveGrad
- ✅ **Inpainting** : Remplir parties manquantes
- ✅ **Super-resolution** : SR3

**Quand NE PAS utiliser** :
- ❌ Real-time → Trop lent (50-1000 steps)
- ❌ Besoin latent space structuré → VAE
- ❌ Ressources limitées → Très gourmand

## Forces

✅ **Meilleure qualité** : State-of-the-art pour images  
✅ **Stable training** : Pas de mode collapse (vs GAN)  
✅ **Diversité** : Pas de mode collapse  
✅ **High resolution** : 512x512، 1024x1024+

## Limites

❌ **Très lent** : 50-1000 denoising steps  
❌ **Gourmand** : Beaucoup de compute/mémoire  
❌ **Difficile à contrôler** : Sans guidance

## Variantes / liens

**DDPM** : Denoising Diffusion Probabilistic Models (2020)  
**DDIM** : Deterministic sampling (plus rapide)  
**Stable Diffusion** : Latent diffusion + text conditioning  
**DALL-E 2** : Text→image de OpenAI  
**Imagen** : Text→image de Google  
**Latent Diffusion** : Diffusion dans latent space (plus rapide)

## Références

- **DDPM** : Ho et al., 2020 - "Denoising Diffusion Probabilistic Models"
- **Stable Diffusion** : Rombach et al., 2022 - "High-Resolution Image Synthesis with Latent Diffusion"
- **DALL-E 2** : Ramesh et al., 2022
- **HuggingFace Diffusers** : [huggingface.co/docs/diffusers](https://huggingface.co/docs/diffusers)
