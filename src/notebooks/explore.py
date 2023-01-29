from typing import Tuple, Union

import config
import constants
import datasets
import funk_svd
import implicit
import numpy as np
import pandas as pd
import recommender
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


lastfm_gt = recommender.load_preferences(config.LASTFM_RECOMMENDER_DIR / "model.npz")
train_gt, valid_gt, test_gt = utils.train_test_split_mask(
    lastfm_gt, 0.7, rng, valid_prop=0.1
)

for fill in [True, False]:
    for trainer, model_fn in zip(
        [train_svd, train_implicit, train_implicit], model_makers
    ):
        model = model_fn()
        estimates = trainer(train_gt.filled(), model)
        score = validate(valid_gt, estimates, fill=fill)
        print(f"fill: {fill} model: {model.__class__} dcg: {score}")


# train_gt_df = lengthen_df(pd.DataFrame(train_gt.filled()))

# svd = funk_svd.SVD(lr=0.001, reg=0.005, n_epochs=20, n_factors=32, shuffle=False)
# svd.fit(train_gt_df)
# recommender_preferences = (svd.pu_ @ svd.qi_.T) + svd.bu_[:, None] + svd.bi_[None, :]

# print(validate(valid_gt, recommender_preferences))

# svd = funk_svd.SVD(lr=0.001, reg=0.005, n_epochs=20, n_factors=32, shuffle=False)
# svd.fit(train_gt_df)
# nan_prefs = (svd.pu_ @ svd.qi_.T) + svd.bu_[:, None] + svd.bi_[None, :]

# preds = svd.predict(valid_gt_df)
# print(sklearn.metrics.mean_absolute_error(valid_gt_df["rating"], preds))

# ------------------------------
# options:
# - ndcg with both, problem: works on ground truths (implicit), it doesn't with
#   svd (instance of model is not Implicit)
# - dcg with both, problems:
#   1. we do not know if just using preference values for the validation set is
#      the right use of dcg
#   2. how to subset Implicit models only at the validation sets? -> always use
#      our own train_test_split_mask


# train_gt_sparse = sparse.csr_array(train_gt.filled())

# model = implicit.als.AlternatingLeastSquares(
#     factors=32, random_state=next(seedgen), iterations=20
# )
# model.fit(train_gt_sparse)
# prefs = model.user_factors @ model.item_factors.T
# print(validate(valid_gt, prefs))


# model = implicit.lmf.LogisticMatrixFactorization(
#     factors=32, iterations=20, random_state=next(seedgen)
# )
# model.fit(train_gt_sparse)
# prefs = model.user_factors @ model.item_factors.T
# print(validate(valid_gt, prefs))
