# GAN

Génération via adversarial training (générateur vs discriminateur).

## Idée clé

**GAN (Generative Adversarial Network)** : Deux réseaux en compétition - **Generator** crée des fakes, **Discriminator** les détecte. Jeu à somme nulle (minimax game).

**Architecture** :
```
Noise z ~ N(0,1)
    ↓
Generator G(z) → Fake image
    ↓
Discriminator D → Real/Fake?
    ↑
Real image

G veut tromper D
D veut différencier real/fake
```

**Minimax game** :
```
min_G max_D V(D,G) = E[log D(x)] + E[log(1 - D(G(z)))]

D maximise : Identifier real vs fake
G minimise : Trom​per D

→ Équilibre de Nash quand D(G(z)) = 0.5
```

**Training alternée** :
```python
for epoch in epochs:
    # 1. Train Discriminator
    real_loss = -log(D(real_images))
    fake_loss = -log(1 - D(G(noise)))
    d_loss = real_loss + fake_loss
    d_loss.backward()
    
    # 2. Train Generator
    g_loss = -log(D(G(noise)))  # Veut D(fake) = 1
    g_loss.backward()
```

## Exemples concrets

### 1. DCGAN (Deep Convolutional GAN)

```python
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.main = nn.Sequential(
            # latent_dim x 1 x 1 → 512 x 4 x 4
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            # 512 x 4 x 4 → 256 x 8 x 8
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # 256 x 8 x 8 → 128 x 16 x 16
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # 128 x 16 x 16 → 3 x 32 x 32
            nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.main(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # 3 x 32 x 32 → 128 x 16 x 16
            nn.Conv2d(3, 128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 128 x 16 x 16 → 256 x 8 x 8
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 256 x 8 x 8 → 512 x 4 x 4
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 512 x 4 x 4 → 1
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        return self.main(img).view(-1, 1)

# Training loop
G = Generator()
D = Discriminator()

criterion = nn.BCELoss()
optimizerD = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))

for epoch in range(100):
    for real_images, _ in dataloader:
        batch_size = real_images.size(0)
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)
        
        # Train Discriminator
        D.zero_grad()
        output_real = D(real_images)
        loss_real = criterion(output_real, real_labels)
        
        noise = torch.randn(batch_size, 100, 1, 1)
        fake_images = G(noise)
        output_fake = D(fake_images.detach())
        loss_fake = criterion(output_fake, fake_labels)
        
        d_loss = loss_real + loss_fake
        d_loss.backward()
        optimizerD.step()
        
        # Train Generator
        G.zero_grad()
        output = D(fake_images)
        g_loss = criterion(output, real_labels)  # Veut D(fake)=1
        g_loss.backward()
        optimizerG.step()
    
    print(f"Epoch [{epoch+1}/100] D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}")
```

### 2. Génération d'images

```python
# Générer nouvelles images
G.eval()
with torch.no_grad():
    noise = torch.randn(64, 100, 1, 1)
    fake_images = G(noise)
    
    import torchvision
    grid = torchvision.utils.make_grid(fake_images, nrow=8, normalize=True)
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.show()
```

## Quand l'utiliser

- ✅ **Image generation** : Visages, artwork
- ✅ **Style transfer** : CycleGAN, Pix2Pix
- ✅ **Data augmentation** : Générer training data
- ✅ **Super-resolution** : SRGAN
- ✅ **Image-to-image** : Pix2Pix, facades→photos

**Quand NE PAS utiliser** :
- ❌ Training instable acceptable → VAE safer
- ❌ Besoin latent space structuré → VAE
- ❌ Génération très haute résolution → Diffusion Models

## Forces

✅ **Sharp images** : Plus réaliste que VAE  
✅ **Photo-réaliste** : StyleGAN, BigGAN  
✅ **Creative applications** : Art, design  
✅ **Transfer learning** : Pré-trained generators

## Limites

❌ **Mode collapse** : Génère peu de variété  
❌ **Training instable** : Oscillations D vs G  
❌ **Difficile à converger** : Balance G/D critique  
❌ **Pas de latent space structure** : Interpolation moins smooth

## Variantes / liens

**DCGAN** : Deep Convolutional GAN (2015)

**WGAN** : Wasserstein GAN (plus stable)
```python
# Remplace BCE par Wasserstein distance
d_loss = -torch.mean(D(real)) + torch.mean(D(fake))
g_loss = -torch.mean(D(fake))
# + Gradient penalty ou weight clipping
```

**StyleGAN** : State-of-the-art visages (2018-2020)

**CycleGAN** : Image-to-image sans paired data
```python
# Horse → Zebra et Zebra → Horse
# Cycle consistency: G_AB(G_BA(x)) ≈ x
```

**Pix2Pix** : Paired image-to-image translation

**BigGAN** : Large-scale high-resolution (512x512)

## Références

- **GAN** : Goodfellow et al., 2014 - "Generative Adversarial Nets"
- **DCGAN** : Radford et al., 2015
- **WGAN** : Arjovsky et al., 2017
- **StyleGAN** : Karras et al., 2018
- **PyTorch Tutorial** : [pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
