# Learning to Rank

Optimise le classement (search/reco).

## Idée clé

**LTR** : Apprend à **rank** items optimally vs just classify/regress. Pointwise, Pairwise, Listwise.

**Pairwise** : Learn `score(A) > score(B)` if A > B in ground truth.

## Exemples concrets

```python
import xgboost as xgb

# Data: qid (query id), features, relevance labels
dtrain = xgb.DMatrix(X_train, label=y_train)
dtrain.set_group([5, 10, 15])  # Group sizes per query

# LambdaMART (pairwise ranking)
params = {
    'objective': 'rank:pairwise',
    'eval_metric': 'ndcg@10',
    'eta': 0.1,
    'max_depth': 6
}

model = xgb.train(params, dtrain, num_boost_round=100)

# Predict scores
dtest = xgb.DMatrix(X_test)
scores = model.predict(dtest)

# Rank items per query
ranked_indices = np.argsort(scores)[::-1]
```

## Quand l'utiliser

- ✅ **Search engines** : Google, Bing
- ✅ **Recommendation** : Ranking products
- ✅ **Ads** : CTR prediction + ranking

**Quand NE PAS utiliser** : ❌ Binary classification sufficient

## Forces

✅ **Ranking-optimized** : NDCG, MAP metrics  
✅ **Pairwise/Listwise** : Better than pointwise

## Limites

❌ **Complex** : Need query groups  
❌ **Data hungry**

## Variantes / liens

**Pointwise** : Regression on relevance  
**Pairwise** : RankNet, LambdaMART  
**Listwise** : ListNet, ListMLE

## Références

- **LambdaMART** : Burges, 2010
- **XGBoost** : [xgboost.readthedocs.io](https://xgboost.readthedocs.io/)
