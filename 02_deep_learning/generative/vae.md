# VAE

Autoencodeur variationnel, génération probabiliste.

## Idée clé

**VAE (Variational Autoencoder)** apprend un **espace latent probabiliste** pour générer de nouvelles données. Combine autoencoder avec modélisation bayésienne.

**Architecture** :
```
Input x → Encoder → μ, σ (latent params)
                     ↓
               z ~ N(μ, σ²) (sampling)
                     ↓
               Decoder → x̂ (reconstruction)

Loss = Reconstruction + KL Divergence
```

**Reparameterization trick** :
```python
# Au lieu de sampling directement (non-différentiable):
z = sample(N(μ, σ²))  # ❌ Can't backprop

# Reparameterization:
ε ~ N(0, 1)
z = μ + σ ⊙ ε  # ✓ Différentiable!

→ Permet backpropagation
```

**ELBO Loss** :
```
L = E[log p(x|z)] - KL(q(z|x) || p(z))
  = Reconstruction - KL divergence

Reconstruction: -||x - x̂||²
KL: Régularise latent space → N(0,1)
```

## Exemples concrets

### 1. VAE pour MNIST

```python
import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_logvar = nn.Linear(400, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Loss function
def vae_loss(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Training
model = VAE(latent_dim=20)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    for batch_x, _ in train_loader:
        optimizer.zero_grad()
        recon, mu, logvar = model(batch_x)
        loss = vae_loss(recon, batch_x, mu, logvar)
        loss.backward()
        optimizer.step()
```

### 2. Génération d'images

```python
# Générer nouvelles images
model.eval()
with torch.no_grad():
    # Sample from prior N(0,1)
    z = torch.randn(64, 20)  # 64 images
    generated = model.decode(z).view(-1, 1, 28, 28)
    
    # Visualiser
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(8, 8, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(generated[i, 0], cmap='gray')
        ax.axis('off')
    plt.show()
```

### 3. Interpolation dans latent space

```python
# Interpoler entre deux images
def interpolate(model, img1, img2, steps=10):
    model.eval()
    with torch.no_grad():
        mu1, _ = model.encode(img1.view(-1, 784))
        mu2, _ = model.encode(img2.view(-1, 784))
        
        interpolations = []
        for alpha in torch.linspace(0, 1, steps):
            z = (1 - alpha) * mu1 + alpha * mu2
            recon = model.decode(z)
            interpolations.append(recon.view(28, 28))
        
        return interpolations

# Visualiser
interp = interpolate(model, img_a, img_b)
fig, axes = plt.subplots(1, len(interp), figsize=(15, 2))
for i, (ax, img) in enumerate(zip(axes, interp)):
    ax.imshow(img, cmap='gray')
    ax.axis('off')
```

## Quand l'utiliser

- ✅ **Génération d'images** : Visages, digits, etc.
- ✅ **Anomaly detection** : Reconstruction error
- ✅ **Data augmentation** : Nouvelles samples
- ✅ **Dimensionality reduction** : Latent space visualisation
- ✅ **Semi-supervised learning** : Avec labels partiels

**Quand NE PAS utiliser** :
- ❌ Images haute résolution → Diffusion Models, StyleGAN
- ❌ Génération photo-réaliste → GAN meilleur
- ❌ Tabulaire → Autres méthodes

## Forces

✅ **Latent space structuré** : Interpolation smooth  
✅ **Probabiliste** : Incertitude modélisée  
✅ **Stable training** : Moins de mode collapse que GAN  
✅ **Anomaly detection** : Via reconstruction error

## Limites

❌ **Images floues** : Reconstruction moins sharp que GAN  
❌ **KL collapse** : Latent space inutilisé parfois  
❌ **Hyperparams sensibles** : Balance BCE/KL difficile  
❌ **Moins réaliste** : GAN/Diffusion meilleurs

## Variantes / liens

**β-VAE** :
```python
loss = BCE + β * KLD  # β > 1 pour disentanglement
```

**Conditional VAE** :
```python
# Encoder + label
z = encode(concat(x, y))
# Decoder + label
x̂ = decode(concat(z, y))
```

**VQ-VAE** : Discrete latent space (Codebook)

**Hierarchical VAE** : Multiple latent levels

## Références

- **VAE** : Kingma & Welling, 2013 - "Auto-Encoding Variational Bayes"
- **β-VAE** : Higgins et al., 2017
- **PyTorch Tutorial** : [pytorch.org/tutorials](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
