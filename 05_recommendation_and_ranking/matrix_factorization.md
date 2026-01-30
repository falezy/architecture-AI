# Matrix Factorization

Recommandation via facteurs latents (SVD-like).

## Idée clé

**MF** : Decompose user-item matrix `R ≈ U × V^T`. Learn latent factors for users & items.

`R_ui ≈ u_i^T v_j` (user i embedding × item j embedding)

## Exemples concrets

```python
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_df[['user', 'item', 'rating']], reader)

# SVD (Matrix Factorization)
algo = SVD(n_factors=20, n_epochs=20, lr_all=0.005, reg_all=0.02)
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5)

# Train
trainset = data.build_full_trainset()
algo.fit(trainset)

# Predict
pred = algo.predict(uid='user1', iid='item123')
print(f"Predicted rating: {pred.est:.2f}")

# Get user/item embeddings
user_embedding = algo.pu[trainset.to_inner_uid('user1')]
item_embedding = algo.qi[trainset.to_inner_iid('item123')]
```

## Quand l'utiliser

- ✅ **Sparse data** : Handles missing ratings well
- ✅ **Latent features** : Discover hidden patterns
- ✅ **Netflix Prize** : Industry standard

**Quand NE PAS utiliser** : ❌ Need explainability → CF

## Forces

✅ **Handles sparsity**  
✅ **Scalable** : O(k × (users + items))  
✅ **Embeddings** : Useful for downstream tasks

## Limites

❌ **Cold start** : New users/items  
❌ **Linear** : latent dot product

## Variantes / liens

**SVD** : Classic MF  
**SVD++** : Incorpore implicit feedback  
**NMF** : Non-negative MF  
**Neural CF** : Deep learning MF

## Références

- **Koren** : "Factorization Meets the Neighborhood", 2008
- **Surprise** : [surpriselib.com](http://surpriselib.com/)
