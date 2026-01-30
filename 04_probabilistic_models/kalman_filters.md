# Kalman Filters

Filtrage bayésien linéaire (estimation d'état).

## Idée clé

**Kalman Filter** : Optimal state estimator for **linear Gaussian** systems. **Predict-Update** cycle.

```
x_t = Ax_{t-1} + Bu_t + w  (dynamics, w ~ N(0,Q))
z_t = Hx_t + v              (observation, v ~ N(0,R))

1. Predict: x̂ = Ax̂ + Bu, P = APA' + Q
2. Update: K = PH'(HPH'+R)^-1, x̂ += K(z-Hx̂), P = (I-KH)P
```

## Exemples concrets

```python
from filterpy.kalman import KalmanFilter
import numpy as np

# 1D tracking
kf = KalmanFilter(dim_x=2, dim_z=1)
kf.x = np.array([0., 0.])  # Initial state [position, velocity]
kf.F = np.array([[1., 1.], [0., 1.]])  # State transition
kf.H = np.array([[1., 0.]])  # Measurement function
kf.P *= 10.  # Covariance
kf.R = 5  # Measurement noise
kf.Q = np.eye(2) * 0.1  # Process noise

measurements = [1., 2., 3., 4., 5.]
for z in measurements:
    kf.predict()
    kf.update(z)
    print(f"Position: {kf.x[0]:.2f}, Velocity: {kf.x[1]:.2f}")
```

## Quand l'utiliser

- ✅ **Object tracking** : Radar, GPS
- ✅ **Sensor fusion** : Combine noisy sensors
- ✅ **Linear systems** : Dynamics linear

**Quand NE PAS utiliser** : ❌ Non-linear → Extended/Unscented Kalman

## Forces

✅ **Optimal** : Linear Gaussian case  
✅ **Real-time** : O(n²)  
✅ **Uncertainty** : Covariance tracking

## Limites

❌ **Linear only** : Restrictive  
❌ **Gaussian** : Noise assumption

## Variantes / liens

**Extended KF** : Non-linear (linearization)  
**Unscented KF** : Better non-linear  
**Particle Filter** : Non-parametric

## Références

- **Kalman** : "A New Approach to Linear Filtering", 1960
- **filterpy** : [github.com/rlabbe/filterpy](https://github.com/rlabbe/filterpy)
