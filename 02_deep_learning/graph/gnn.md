# GNN

Réseaux pour graphes (message passing).

## Idée clé

**GNN (Graph Neural Network)** : Apprend sur structures de **graphes** (réseaux sociaux, molécules, etc.) via **message passing** entre nœuds.

**Message Passing** :
```
Pour chaque node v:
1. Agrège messages desvoisins: m_v = AGG({h_u | u ∈ N(v)})
2. Update: h_v' = UPDATE(h_v, m_v)

Répète K fois (K layers)
```

**Architecture générale** :
```
Input: Graph G = (V, E), features X
↓
[Message Passing ×K layers]
↓
Node embeddings h_v ou Graph embedding h_G
↓
Task-specific head (classification, regression, etc.)
```

## Exemples concrets

### 1. GCN (Graph Convolutional Network)

```python
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid

# Dataset (Cora: citation network)
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

class GCN(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Layer 1
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = torch.dropout(x, p=0.5, train=self.training)
        
        # Layer 2
        x = self.conv2(x, edge_index)
        
        return torch.log_softmax(x, dim=1)

model = GCN(dataset.num_features, dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = nn.functional.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
```

### 2. GAT (Graph Attention Network)

```python
from torch_geometric.nn import GATConv

class GAT(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.conv1 = GATConv(num_features, 8, heads=8, dropout=0.6)
        self.conv2 = GATConv(8*8, num_classes, heads=1, concat=False, dropout=0.6)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.dropout(x, p=0.6, train=self.training)
        x = torch.elu(self.conv1(x, edge_index))
        x = torch.dropout(x, p=0.6, train=self.training)
        x = self.conv2(x, edge_index)
        return torch.log_softmax(x, dim=1)
```

### 3. Graph Classification

```python
from torch_geometric.nn import global_mean_pool
from torch_geometric.datasets import TUDataset

dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')

class GraphClassifier(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, 64)
        self.conv2 = GCNConv(64, 64)
        self.conv3 = GCNConv(64, 64)
        self.fc = nn.Linear(64, num_classes)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.relu(self.conv3(x, edge_index))
        
        # Global pooling (graph-level)
        x = global_mean_pool(x, batch)
        
        x = self.fc(x)
        return x
```

## Quand l'utiliser

- ✅ **Molecular property prediction** : Molécules = graphes
- ✅ **Social networks** : Recommandation, influence
- ✅ **Knowledge graphs** : Reasoning, link prediction
- ✅ **Traffic/routing** : Réseaux routiers
- ✅ **Citation networks** : Paper classification

**Quand NE PAS utiliser** :
- ❌ Grids régulières → CNN
- ❌ Séquences → RNN, Transformer
- ❌ Tabulaire → XGBoost, MLP

## Forces

✅ **Gère graphes** : Structure non-euclidienne  
✅ **Versatile** : Noeuds, arêtes, graphes entiers  
✅ **Inductive** : Généralise à nouveaux graphs  
✅ **State-of-the-art** : Molécules, social networks

## Limites

❌ **Over-smoothing** : Trop de layers → tous nodes similaires  
❌ **Scalability** : Grands graphes difficiles  
❌ **Expressiveness** : WL test limitations

## Variantes / liens

**GCN** : Graph Convolutional Network  
**GAT** : Graph Attention (attention weights sur voisins)  
**GraphSAGE** : Sampling de voisins (scalable)  
**GIN** : Graph Isomorphism Network (plus expressif)

## Références

- **GCN** : Kipf & Welling, 2016
- **GAT** : Veličković et al., 2017
- **PyTorch Geometric** : [pytorch-geometric.readthedocs.io](https://pytorch-geometric.readthedocs.io/)
