# Collaborative Filtering

Recommandation basée sur similarités utilisateurs/items.

## Idée clé

**CF** : Recommend items basé sur similar users/items. "Users who liked A also liked B".

**User-based** : Find similar users → recommend their items  
**Item-based** : Find similar items → recommend to user

## Exemples concrets

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# User-item matrix (ratings)
ratings = np.array([
    #    Item1, Item2, Item3
    [5, 3, 0],  # User1
    [4, 0, 0],  # User2
    [1, 1, 0],  # User3
    [1, 0, 5],  # User4
])

# Item-based CF
item_sim = cosine_similarity(ratings.T)  # Transpose for item similarity

# Recommend for User1 who rated Item1=5
user = 0
item_to_predict = 2  # Item3
similar_items = np.argsort(item_sim[item_to_predict])[::-1][1:3]  # Top-2 similar to Item3

prediction = np.dot(item_sim[item_to_predict, similar_items], ratings[user, similar_items]) / item_sim[item_to_predict, similar_items].sum()
print(f"Predicted rating: {prediction:.2f}")

#scikit-surprise
from surprise import KNNBasic, Dataset, Reader
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['user', 'item', 'rating']], reader)
algo = KNNBasic(sim_options={'name': 'cosine', 'user_based': False})  # Item-based
algo.fit(data.build_full_trainset())
prediction = algo.predict(uid='user1', iid='item3')
```

## Quand l'utiliser

- ✅ **Social proof** : "People like you..."
- ✅ **Implicit feedback** : Clicks, views
- ✅ **Cold start items** : New items get recommended via similar items

**Quand NE PAS utiliser** : ❌ Sparse data → Matrix Factorization better

## Forces

✅ **Simple** : Intuitive  
✅ **No content needed** : Pure behavior  
✅ **Serendipity** : Discover new things

## Limites

❌ **Sparsity** : Most ratings missing  
❌ **Cold start** : New users/items  
❌ **Scalabilité** : O(n²) similarity

## Variantes / liens

**User-based** : Similar users  
**Item-based** : Similar items (more scalable)  
**Matrix Factorization** : Latent factors

## Références

- **Koren et al** : "Matrix Factorization Techniques", 2009
- **Surprise** : [surpriselib.com](http://surpriselib.com/)
