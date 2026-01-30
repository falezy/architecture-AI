# Deep Time Series

Prévision via RNN/Transformers, etc.

## Idée clé

**Deep TS** : LSTM/Transformer pour séries temporelles. Apprend patterns complexes vs ARIMA linear.

## Exemples concrets

```python
import torch
import torch.nn as nn

class LSTMForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

model = LSTMForecaster(input_dim=1, hidden_dim=64, n_layers=2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training
for epoch in range(100):
    for X_batch, y_batch in dataloader:  # X: [batch, seq_len, features]
        pred = model(X_batch)
        loss = nn.MSELoss()(pred, y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Forecast
model.eval()
with torch.no_grad():
    future = model(last_sequence)
```

## Quand l'utiliser

- ✅ **Non-linear patterns** : Complex dynamics
- ✅ **Multivariate** : Multiple features
- ✅ **Long sequences** : History matters

**Quand NE PAS utiliser** : ❌ Small data → ARIMA, ❌ Interpretability critical

## Forces

✅ **Non-linear** : Complex patterns  
✅ **Multivariate** : Multiple series  
✅ **State-of-the-art** : Transformer-based

## Limites

❌ **Data hungry** : Needs много data  
❌ **Black box** : Less interpretable  
❌ **Overfitting risk**

## Variantes / liens

**N-BEATS** : FC architecture for TS  
**Temporal Fusion Transformer** : Attention-based  
**DeepAR** (Amazon) : Probabilistic forecasting

## Références

- **N-BEATS** : Oreshkin et al., 2019
- **TFT** : Lim et al., 2020
