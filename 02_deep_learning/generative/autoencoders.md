# Autoencoders

Compression/reconstruction : encodeur + décodeur.

## Idée clé

**Autoencoder** : Apprend à **compresser** (encoder) puis **reconstruire** (decoder) les données via un **bottleneck**. Apprentissage non-supervisé de représentations.

**Architecture** :
```
Input x → Encoder → z (latent/bottleneck) → Decoder → x̂
Loss = ||x - x̂||²  (reconstruction error)
```

**Bottleneck** : Force compression → apprend features importantes

## Exemples concrets

### 1. Autoencoder simple (MNIST)

```python
import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),  # Bottleneck
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

model = Autoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    for images, _ in train_loader:
        images = images.view(-1, 784)
        recon = model(images)
        loss = criterion(recon, images)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 2. Denoising Autoencoder

```python
# Ajouter bruit pendant training
def add_noise(images, noise_factor=0.3):
    noisy = images + noise_factor * torch.randn_like(images)
    return torch.clamp(noisy, 0., 1.)

for images, _ in train_loader:
    images = images.view(-1, 784)
    noisy_images = add_noise(images)
    
    recon = model(noisy_images)
    loss = criterion(recon, images)  # Reconstruit clean image!
```

### 3. Convolutional Autoencoder

```python
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # 28x28 → 14x14
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 14x14 → 7x7
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
```

## Quand l'utiliser

- ✅ **Dimensionality reduction** : PCA alternatif non-linéaire
- ✅ **Anomaly detection** : Reconstruction error élevé = anomalie
- ✅ **Denoising** : Nettoyer images/signaux
- ✅ **Feature learning** : Pré-training non-supervisé
- ✅ **Compression** : Images, audio

## Forces

✅ **Non-supervisé** : Pas besoin de labels  
✅ **Versatile** : Images, texte, audio  
✅ **Anomaly detection** : Via reconstruction error  
✅ **Simple** : Architecture straightforward

## Limites

❌ **Latent space non-structuré** : Pas de sampling comme VAE  
❌ **Overfitting** : Peut mémoriser training data  
❌ **Less sharp** : GAN/Diffusion meilleurs pour génération

## Variantes / liens

**Denoising AE** : Reconstruit clean à partir de noisy  
**Sparse AE** : Régularisation L1 sur latent  
**Contractive AE** : Pénalise sensitivité aux inputs  
**VAE** : Probabilistic (voir vae.md)  
**VQ-VAE** : Discrete latent codes

## Références

- **Autoencoder** : Hinton & Salakhutdinov, 2006
- **Denoising AE** : Vincent et al., 2008
- **PyTorch Tutorial** : [pytorch.org/tutorials](https://pytorch.org/tutorials)
