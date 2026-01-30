# Prophet

Prévision avec saisonnalités/tendances (pratique).

## Idée clé

**Prophet** (Facebook) : Décomposition **additive** `y(t) = trend + seasonal + holidays + error`. Robuste, pratique, gère missing data.

## Exemples concrets

```python
from prophet import Prophet
import pandas as pd

# Data: 'ds' (date), 'y' (value)
df = pd.DataFrame({'ds': dates, 'y': values})

model = Prophet()
model.fit(df)

# Forecast
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)

# Plot
model.plot(forecast)
model.plot_components(forecast)  # Trend, weekly, yearly
```

## Quand l'utiliser

- ✅ **Business forecasting** : Sales, traffic
- ✅ **Strong seasonality** : Daily, weekly, yearly
- ✅ **Holidays** : Black Friday, etc.
- ✅ **Missing data** : Handles well

## Forces

✅ **Easy to use** : Minimal tuning  
✅ **Interpretable** : Decomposable  
✅ **Robust** : Outliers, missing data

## Limites

❌ **Univariate** : No covariates  
❌ **Additive assumption**

## Variantes / liens

**NeuralProphet** : Neural version

## Références

- **Prophet** : Taylor & Letham, 2017
- **Docs** : [facebook.github.io/prophet](https://facebook.github.io/prophet/)
