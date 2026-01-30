# CNN

Réseaux convolutionnels pour images/vision et signaux locaux.

## Idée clé

**CNN (Convolutional Neural Network)** utilise des **convolutions** au lieu de connexions fully-connected pour exploiter la **structure locale** et la **translation invariance** des images. C'est l'architecture dominante en vision par ordinateur.

**Opération de convolution** :
```
Image (H×W×C) * Kernel (K×K×C) = Feature map (H'×W'×F)

Exemple 3×3:
Input:           Kernel:         Output:
1  2  3          -1  0  1        
4  5  6    *     -2  0  2   =    (somme pondérée)
7  8  9          -1  0  1

→ Détecte patterns locaux (edges, textures, formes)
```

**Architecture typique** :
```
Input Image
    ↓
[Conv + ReLU + Pool] ×N  ← Feature extraction
    ↓
[Flatten]
    ↓
[FC Layers]              ← Classification
    ↓
Output (classes)

Exemple:
28×28×1 → Conv(3×3, 32) → 26×26×32
        → MaxPool(2×2)  → 13×13×32
        → Conv(3×3, 64) → 11×11×64
        → MaxPool(2×2)  → 5×5×64
        → Flatten       → 1600
        → FC(128)       → 128
        → FC(10)        → 10 classes
```

**Composants clés** :
1. **Convolution** : Détecte features locales
2. **Activation** (ReLU) : Non-linéarité
3. **Pooling** : Réduction dimension + invariance
4. **Fully Connected** : Classification finale

**Avantages vs MLP** :
- **Moins de paramètres** : Poids partagés (shared weights)
- **Translation invariance** : Détecte features partout
- **Hiérarchie** : Features simples → complexes

**Calcul des paramètres** :
```python
# Conv layer: kernel_size × kernel_size × in_channels × out_channels + bias
Conv(3×3, 32→64) = 3×3×32×64 + 64 = 18,496

# FC layer: in_features × out_features + bias  
FC(1600→128) = 1600×128 + 128 = 204,928
```

## Exemples concrets

### 1. CNN simple : MNIST digits

**Code PyTorch complet** :
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Définir CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolution layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)   # 28×28×1 → 26×26×32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)  # 13×13×32 → 11×11×64
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)
        
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        # Conv block 1
        x = F.relu(self.conv1(x))           # 28×28×1 → 26×26×32
        x = F.max_pool2d(x, 2)              # → 13×13×32
        
        # Conv block 2
        x = F.relu(self.conv2(x))           # 13×13×32 → 11×11×64
        x = F.max_pool2d(x, 2)              # → 5×5×64
        
        # Flatten
        x = x.view(-1, 64 * 5 * 5)          # → 1600
        
        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 2. Charger données
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000)

# 3. Entraînement
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')

# 4. Évaluation
model.eval()
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()

accuracy = 100. * correct / len(test_loader.dataset)
print(f'\nTest Accuracy: {accuracy:.2f}%')
```

---

### 2. CNN moderne : CIFAR-10 avec ResNet blocks

**Code avec skip connections** :
```python
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Skip connection
        out = F.relu(out)
        return out

class ResNetCIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Initial conv
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual blocks
        self.layer1 = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ResidualBlock(128),
            ResidualBlock(128)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            ResidualBlock(256),
            ResidualBlock(256)
        )
        
        # Global average pooling + FC
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # 32×32×3 → 32×32×64
        x = self.layer1(x)                    # 32×32×64
        x = self.layer2(x)                    # 16×16×128
        x = self.layer3(x)                    # 8×8×256
        x = self.avgpool(x)                   # 1×1×256
        x = x.view(x.size(0), -1)            # 256
        x = self.fc(x)                        # num_classes
        return x

model = ResNetCIFAR(num_classes=10)
print(f"Paramètres: {sum(p.numel() for p in model.parameters()):,}")
```

---

### 3. Visualiser feature maps

**Code pour voir ce que le CNN apprend** :
```python
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# Charger modèle entraîné
model = SimpleCNN()
# model.load_state_dict(torch.load('model.pth'))
model.eval()

# Charger une image
transform = transforms.ToTensor()
dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
image, label = dataset[0]
image_batch = image.unsqueeze(0)  # Ajouter batch dimension

# Hook pour capturer activations
activations = {}
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

model.conv1.register_forward_hook(get_activation('conv1'))
model.conv2.register_forward_hook(get_activation('conv2'))

# Forward pass
with torch.no_grad():
    output = model(image_batch)

# Visualiser feature maps
fig, axes = plt.subplots(4, 8, figsize=(15, 8))
axes = axes.ravel()

# Première couche conv (32 feature maps)
act = activations['conv1'][0]  # Shape: (32, 26, 26)
for i in range(min(32, len(axes))):
    axes[i].imshow(act[i].cpu(), cmap='viridis')
    axes[i].axis('off')
    axes[i].set_title(f'Filter {i}')

plt.suptitle('Feature Maps de Conv1')
plt.tight_layout()
plt.show()

print(f"Conv1 output shape: {activations['conv1'].shape}")
print(f"Conv2 output shape: {activations['conv2'].shape}")
```

---

### 4. Transfer Learning avec modèles pré-entraînés

**Code utilisant ResNet pre-trained** :
```python
import torch
import torchvision.models as models
import torch.nn as nn

# 1. Charger ResNet18 pré-entraîné sur ImageNet
resnet = models.resnet18(pretrained=True)

# 2. Remplacer dernière couche pour notre tâche (10 classes)
num_features = resnet.fc.in_features
resnet.fc = nn.Linear(num_features, 10)

# 3. Geler les couches pré-entraînées (optionnel)
for param in resnet.parameters():
    param.requires_grad = False

# Dégeler seulement la dernière couche
for param in resnet.fc.parameters():
    param.requires_grad = True

# 4. Entraîner seulement FC layer
optimizer = torch.optim.Adam(resnet.fc.parameters(), lr=0.001)

# Alternative: Fine-tuning complet avec petit LR
# for param in resnet.parameters():
#     param.requires_grad = True
# optimizer = torch.optim.Adam(resnet.parameters(), lr=0.0001)

print(f"Modèle modifié:")
print(f"  Trainable params: {sum(p.numel() for p in resnet.parameters() if p.requires_grad):,}")
print(f"  Total params: {sum(p.numel() for p in resnet.parameters()):,}")
```

---

### 5. Data Augmentation

**Code avec transformations** :
```python
from torchvision import transforms

# Training transforms (avec augmentation)
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Test transforms (sans augmentation)
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Visualiser augmentation
import matplotlib.pyplot as plt
from PIL import Image

img = Image.open('sample.jpg')
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
axes = axes.ravel()

for i in range(8):
    augmented = train_transform(img)
    axes[i].imshow(augmented.permute(1, 2, 0))
    axes[i].axis('off')

plt.suptitle('Exemples de Data Augmentation')
plt.tight_layout()
plt.show()
```

---

### 6. Architectures classiques comparées

**Code comparant LeNet, AlexNet style, VGG style** :
```python
class LeNet5(nn.Module):
    """LeNet-5 (1998) - Premier CNN successful"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 16*4*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class VGGStyle(nn.Module):
    """VGG-like: blocs répétés de Conv + Pool"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 64 filters
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 2: 128 filters
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 3: 256 filters
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Comparer tailles
models_dict = {
    'LeNet-5': LeNet5(),
    'VGG-Style': VGGStyle(),
}

for name, model in models_dict.items():
    params = sum(p.numel() for p in model.parameters())
    print(f"{name}: {params:,} paramètres")
```

## Quand l'utiliser

- ✅ **Images** : Classification, détection, segmentation
- ✅ **Signaux 1D** : Audio, séries temporelles, ECG
- ✅ **Données avec structure locale** : Pixels voisins corrélés
- ✅ **Translation invariance** : Features doivent être détectées partout
- ✅ **Hiérarchie de features** : Edges → textures → objets

**Cas d'usage typiques** :
- 🖼️ **Vision** : Classification d'images (ImageNet)
- 🚗 **Véhicules autonomes** : Détection d'objets
- 🏥 **Médical** : Analysis de radiographies, IRM
- 📸 **Reconnaissance faciale** : FaceNet, DeepFace
- 🎨 **Style transfer** : Neural style, GANs
- 🔊 **Audio** : Spectrogrammes → CNN 2D

**Quand NE PAS utiliser** :
- ❌ Données tabulaires → MLP, XGBoost
- ❌ Séquences longues → Transformers
- ❌ Graphes → GNN
- ❌ Très petites images (<20×20) → MLP suffit
- ❌ Besoin interpretabilité forte → Decision trees

## Forces

✅ **Translation invariance** : Détecte features n'importe où  
✅ **Paramètres partagés** : Moins de params que MLP  
✅ **Hiérarchie** : Features simples → complexes  
✅ **Prouvé** : État de l'art en vision depuis 2012  
✅ **Transfer learning** : Pré-entraînement ImageNet  
✅ **GPU-friendly** : Parallélisation efficace

**Exemple de réduction de paramètres** :
```python
# MLP sur image 28×28
fc_params = 28*28 * 128 = 100,352

# CNN équivalent
conv_params = 3*3*1*32 + 3*3*32*64 = 18,720
# → 5x moins de paramètres!
```

## Limites

❌ **Beaucoup de données** : Nécessite milliers d'exemples  
❌ **Computationally expensive** : Training lent  
❌ **Hyperparamètres** : Kernel size, nb layers, filters  
❌ **Interprétabilité** : Boîte noire  
❌ **Adversarial attacks** : Vulnérable à perturbations  
❌ **Rotation/scale** : Pas naturellement invariant

**Limitation : Position sensible** :
```python
# CNN détecte "chat" au centre
# Mais si on translate l'image, peut échouer si:
# - Pooling trop agressif
# - Pas assez de data augmentation
# → Besoin de beaucoup d'exemples ou augmentation
```

## Variantes / liens

### Architectures historiques

**1. LeNet-5 (1998)** :
```
32×32 → Conv(5×5, 6) → Pool → Conv(5×5, 16) → Pool → FC(120) → FC(84) → FC(10)
```

**2. AlexNet (2012)** - ImageNet breakthrough :
```
227×227 → Conv(11×11, 96, stride=4) → Pool → ... → FC(4096) → FC(4096) → FC(1000)
- ReLU activation
- Dropout
- Data augmentation
```

**3. VGG-16/19 (2014)** :
```
Stacks de Conv 3×3 + Pool
- Simple et deep
- 138M paramètres
```

**4. ResNet (2015)** :
```python
# Skip connections résolvent vanishing gradient
x_out = F.relu(conv(x) + x)  # Residual connection
```

**5. EfficientNet (2019)** :
```
Compound scaling: depth, width, resolution
État de l'art accuracy/efficiency
```

### Types de couches

**Convolution** :
```python
nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)

# Exemples
nn.Conv2d(3, 64, 3, padding=1)      # 3×3, preserve size
nn.Conv2d(64, 128, 1)                # 1×1, change channels
nn.Conv2d(128, 256, 3, stride=2)    # Downsample
```

**Pooling** :
```python
nn.MaxPool2d(kernel_size, stride=None)    # Max pooling
nn.AvgPool2d(kernel_size)                  # Average pooling
nn.AdaptiveAvgPool2d((1, 1))              # Global pooling
```

**Normalization** :
```python
nn.BatchNorm2d(num_features)    # Batch norm (standard)
nn.GroupNorm(num_groups, num_features)  
nn.InstanceNorm2d(num_features) # Style transfer
```

### Techniques importantes

**1. Batch Normalization** :
```python
# Normalise activations par batch
x = conv(x)
x = bn(x)    # mean=0, std=1 par batch
x = relu(x)
```

**2. Dropout** :
```python
nn.Dropout2d(p=0.25)  # Drop 25% de feature maps
```

**3. Global Average Pooling** :
```python
# Remplace FC layers
x = AdaptiveAvgPool2d((1, 1))(x)  # H×W×C → 1×1×C
x = x.view(x.size(0), -1)          # → C
```

### Modèles pré-entraînés PyTorch

```python
from torchvision import models

# Classification
resnet18 = models.resnet18(pretrained=True)
resnet50 = models.resnet50(pretrained=True)
vgg16 = models.vgg16(pretrained=True)
efficientnet_b0 = models.efficientnet_b0(pretrained=True)
mobilenet_v2 = models.mobilenet_v2(pretrained=True)

# Détection d'objets
faster_rcnn = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# Segmentation
fcn = models.segmentation.fcn_resnet50(pretrained=True)
```

## Références

### Papers fondamentaux
- **LeNet** : LeCun et al., 1998 - "Gradient-based learning applied to document recognition"
- **AlexNet** : Krizhevsky et al., 2012 - "ImageNet Classification with Deep CNNs"
- **VGG** : Simonyan & Zisserman, 2014 - "Very Deep Convolutional Networks"
- **ResNet** : He et al., 2015 - "Deep Residual Learning for Image Recognition"
- **Batch Normalization** : Ioffe & Szegedy, 2015
- **EfficientNet** : Tan & Le, 2019

### Documentation
- **PyTorch** : [nn.Conv2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
- **TensorFlow/Keras** : [Conv2D](https://keras.io/api/layers/convolution_layers/convolution2d/)
- **CS231n** : [Stanford CNN course](http://cs231n.stanford.edu/)

### Best practices

**Architecture design** :
```
Règles empiriques:
- Kernel size: 3×3 (standard), 5×5, 7×7 (première couche)
- Filters: doubler après chaque pool (32→64→128→256)
- Stride: 1 pour conv, 2 pour pool
- Padding: 'same' pour préserver taille
```

**Training tips** :
```python
# 1. Data augmentation (critical!)
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# 2. Learning rate schedule
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# 3. Early stopping
# 4. Gradient clipping si instable
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Combien de données** :
```
Règle empirique:
- From scratch: 50k-100k+ images
- Fine-tuning: 1k-10k images
- Transfer learning (freeze): 100-1k images

→ Toujours utiliser pre-trained models si possible!
```
