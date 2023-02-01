from typing import Tuple, Union

import config
import constants
import datasets
import funk_svd
import implicit
import numpy as np
import pandas as pd
import recommender
import recommender_models
import scipy
import sklearn
import utils
from implicit import evaluation
from scipy import sparse
from sklearn.metrics import dcg_score
from sklearn.model_selection import train_test_split

# TODO correction: ALS -> ground truth MovieLens
#                  LMF -> ground truth LastFm
#                  SVD -> recommmenders both


def validate(
    labels_masked: np.ma.masked_array, estimates: np.ndarray, k=40, fill=False
) -> float:
    estimates_masked = np.ma.masked_array(estimates, labels_masked.mask)
    if fill:
        estimates_masked = estimates_masked.filled(labels_masked.fill_value)
        labels_masked = labels_masked.filled()
    return sklearn.metrics.dcg_score(labels_masked, estimates_masked, k=k)


lengthen_df = lambda df: df.melt(
    var_name="i_id", value_name="rating", ignore_index=False
).reset_index(names="u_id")


def train_svd(train, model):
    train_gt_df = lengthen_df(pd.DataFrame(train))
    model.fit(train_gt_df)
    return (model.pu_ @ model.qi_.T) + model.bu_[:, None] + model.bi_[None, :]


def train_implicit(train, model):
    train_gt_sparse = sparse.csr_array(train)
    model.fit(train_gt_sparse)
    return model.user_factors @ model.item_factors.T


seedgen = utils.SeedSequence(1)
rng = np.random.default_rng(constants.SEED)

model_makers = [
    lambda: funk_svd.SVD(lr=0.001, reg=0.005, n_epochs=20, n_factors=32, shuffle=False),
    lambda: implicit.als.AlternatingLeastSquares(
        factors=32, random_state=next(seedgen), iterations=20
    ),
    lambda: implicit.lmf.LogisticMatrixFactorization(
        factors=32, iterations=20, random_state=next(seedgen)
    ),
]


lastfm_gt = recommender_models.LMF.load(
    config.LASTFM_RECOMMENDER_DIR / "model_ground_truth.npz"
).preferences
train_gt, valid_gt, test_gt = utils.train_test_split(
    lastfm_gt, 0.7, seedgen, valid_prop=0.1
)

for fill in [True, False]:
    for trainer, model_fn in zip(
        [train_svd, train_implicit, train_implicit], model_makers
    ):
        model = model_fn()
        estimates = trainer(train_gt.filled(), model)
        score = validate(valid_gt, estimates, fill=fill)
        print(f"fill: {fill} model: {model.__class__} dcg: {score}")


# ------------------------------
# options:
# - ndcg with both, problem: works on ground truths (implicit), it doesn't with
#   svd (instance of model is not Implicit)
# - dcg with both, problems:
#   1. we do not know if just using preference values for the validation set is
#      the right use of dcg
#   2. how to subset Implicit models only at the validation sets? -> always use
#      our own train_test_split_mask


import time

from scipy.sparse.linalg import svds

movielens = datasets.get_movielens()
train_ml, valid_ml, test_ml = utils.train_test_split_mask(
    movielens.values, 0.7, rng, 0.1
)
# train_ml = train_ml.filled(0.0)

start = time.time()
# U, sigma, Vt = svds(train_gt.filled(0.0), k=32, random_state=rng)
U, sigma, Vt = svds(train_ml, k=32, random_state=rng)
print(f"svds time: {time.time() - start}")
# print(sklearn.metrics.mean_squared_error(lastfm_gt, U @ np.diag(sigma) @ Vt))
print(
    sklearn.metrics.mean_squared_error(
        movielens, U @ np.diag(sigma) @ Vt, squared=False
    )
)
print(sklearn.metrics.mean_squared_error(movielens, U @ Vt, squared=False))


fsvd = recommender_models.FSVD(32)
start = time.time()
# fsvd.train(lengthen_df(pd.DataFrame(train_gt)))
fsvd.train(lengthen_df(pd.DataFrame(train_ml)))
print(f"fsvd time: {time.time() - start}")
# print(sklearn.metrics.mean_squared_error(lastfm_gt, fsvd.preferences))
print(sklearn.metrics.mean_squared_error(movielens, fsvd.preferences, squared=False))


als = recommender_models.ALS(32)
start = time.time()
# als.train(sparse.csr_array(train_gt))
als.train(sparse.csr_array(train_ml))
print(f"als time: {time.time() - start}")
# print(sklearn.metrics.mean_squared_error(lastfm_gt, als.preferences))
print(sklearn.metrics.mean_squared_error(movielens, als.preferences, squared=False))


als = recommender_models.ALS(32)
train, _ = implicit.evaluation.train_test_split(
    sparse.csr_array(movielens.values), 0.7, random_state=next(seedgen)
)
start = time.time()
# als.train(sparse.csr_array(train_gt))
als.train(train)
print(f"als time: {time.time() - start}")
# print(sklearn.metrics.mean_squared_error(lastfm_gt, als.preferences))
print(sklearn.metrics.mean_squared_error(movielens, als.preferences, squared=False))


mat = rng.choice(10, size=(6, 6)).astype(np.float32)
mat
u, s, v = svds(mat, k=3, random_state=rng)
print(sklearn.metrics.mean_squared_error(mat, u @ np.diag(s) @ v, squared=False))

ss = np.sqrt(s)
assert np.allclose((u @ np.diag(ss)) @ (np.diag(ss) @ v), u @ np.diag(s) @ v)


start = time.time()
# U, sigma, Vt = svds(train_gt.filled(0.0), k=32, random_state=rng)
U, sigma, Vt = svds(train_gt, k=32, random_state=rng)
print(f"svds time: {time.time() - start}")
# print(sklearn.metrics.mean_squared_error(lastfm_gt, U @ np.diag(sigma) @ Vt))
reconstruction = U @ np.diag(sigma) @ Vt
print(sklearn.metrics.mean_squared_error(lastfm_gt, reconstruction, squared=False))
ss = np.diag(np.sqrt(sigma))
rec2 = (U @ ss) @ (ss @ Vt)
assert np.allclose(reconstruction, rec2, rtol=1e-5, atol=1e-5)
np.allclose(reconstruction, lastfm_gt)


import config
from implicit import evaluation
from sklearn.metrics import mean_squared_error as mse

seedgen = utils.SeedSequence(5)

lastfm_gt = LMF.load(config.LASTFM_RECOMMENDER_DIR / "model.npz").preferences
train, test = evaluation.train_test_split(
    sparse.csr_array(lastfm_gt), 0.7, next(seedgen)
)

svd = SVDS(8, seedgen)
svd.train(train)
print(f"rmse: {mse(lastfm_gt, svd.preferences, squared=False)}")
print(evaluation.ranking_metrics_at_k(svd, train, test, 40))

svd.save("pippo_svd")

new_svd = SVDS.load("pippo_svd")
print(evaluation.ranking_metrics_at_k(new_svd, train, test, 40))
