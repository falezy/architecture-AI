# ARIMA

Prévision séries temporelles (classique).

## Idée clé

**ARIMA(p,d,q)** : **AR**(p) + **I**(d) + **MA**(q). Modèle classique pour séries temporelles stationnaires.

```
AR (AutoRegressive): y_t = φ₁y_{t-1} + φ₂y_{t-2} + ... + ε_t
I (Integrated): Differencing pour stationnarité
MA (Moving Average): y_t = ε_t + θ₁ε_{t-1} + θ₂ε_{t-2} + ...

ARIMA(p,d,q): Combine les 3
```

## Exemples concrets

```python
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd

# Load data
data = pd.read_csv('sales.csv', index_col='date', parse_dates=True)

# Fit ARIMA(1,1,1)
model = ARIMA(data['sales'], order=(1, 1, 1))
fitted = model.fit()

print(fitted.summary())

# Forecast
forecast = fitted.forecast(steps=30)
print(forecast)

# Auto ARIMA (find best p,d,q)
from pmdarima import auto_arima
auto_model = auto_arima(data['sales'], seasonal=False, stepwise=True)
print(auto_model.summary())
```

## Quand l'utiliser

- ✅ **Univariate time series** : Sales, demand
- ✅ **Stationary/nearly stationary**  
- ✅ **Linear trends**
- ✅ **Baseline rapide**

**Quand NE PAS utiliser** : ❌ Non-linear → ML models, ❌ Multivariate → VAR, Prophet

## Forces

✅ **Classique** : Bien connu  
✅ **Interprétable** : Coefficients clairs  
✅ **Standard baseline**

## Limites

❌ **Stationnarité required**  
❌ **Linear only**  
❌ **Univariate** : Pas de covariates  
❌ **Manual tuning** : p, d, q

## Variantes / liens

**SARIMA** : Seasonal ARIMA  
**VAR** : Vector AR (multivariate)  
**Prophet** : Facebook's modern alternative

## Références

- **Box & Jenkins** : Time Series Analysis, 1970
- **statsmodels** : [statsmodels.org](https://www.statsmodels.org/)
