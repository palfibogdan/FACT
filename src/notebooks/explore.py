import config
import constants
import datasets
import funk_svd
import implicit
import numpy as np
import pandas as pd
import recommender
import utils
from implicit import evaluation
from scipy import sparse
from sklearn.metrics import dcg_score
from sklearn.model_selection import train_test_split

# TODO correction: ALS -> ground truth MovieLens
#                  LMF -> ground truth LastFm
#                  SVD -> recommmenders both


# --------------------------- recommenders

seed_gen = utils.SeedSequence(constants.SEED)

lastfm_gt = recommender.load_preferences(config.LASTFM_RECOMMENDER_DIR / "model.npz")


# reshape ground truths into format expected by funk_svd.SVD.fit
# lastfm_gt_df_wide = pd.DataFrame(lastfm_gt)
# lastfm_gt_df = lastfm_gt_df_wide.melt(
#     var_name="i_id", value_name="rating", ignore_index=False
# ).reset_index(names="u_id")


lastfm_gt_df_wide_csr = sparse.csr_array(lastfm_gt)
train_csr, test_csr = implicit.evaluation.train_test_split(
    lastfm_gt_df_wide_csr, train_percentage=0.8, random_state=next(seed_gen)
)
lastfm_gt_train = train_csr.toarray()
lastfm_gt_test = test_csr.toarray()
train_df = (
    pd.DataFrame(lastfm_gt_train)
    .melt(var_name="i_id", value_name="rating", ignore_index=False)
    .reset_index(names="u_id")
)

svd = funk_svd.SVD(lr=0.001, reg=0.005, n_epochs=20, n_factors=32, shuffle=False)
svd.fit(train_df)


# sklearn.metrics.dcg_score()

# implicit.evaluation.ranking_metrics_at_k()


# gt_train = lastfm_gt_df.sample(frac=0.7, random_state=7)
# gt_val = lastfm_gt_df.drop(gt_train.index.tolist()).sample(frac=2 / 3, random_state=8)
# gt_test = lastfm_gt_df.drop(gt_train.index.tolist()).drop(gt_val.index.tolist())

# split into 70/10/20 train/validation/test set
gt_train, tmp = train_test_split(
    lastfm_gt_df,
    train_size=0.7,
    random_state=next(seed_gen)
    # lastfm_gt_df_wide,
    # train_size=0.7,
    # random_state=next(seed_gen),
)
gt_val, gt_test = train_test_split(tmp, train_size=2 / 3, random_state=next(seed_gen))

# train an SVD model to fit the ground truth preferences
svd = funk_svd.SVD(lr=0.001, reg=0.005, n_epochs=20, n_factors=32, shuffle=False)
svd.fit(gt_train)

# compute recommender system preferences
recommender_preferences = (svd.pu_ @ svd.qi_.T) + svd.bu_[:, None] + svd.bi_[None, :]


# ------------------------ ground truth lastfm
def train_test_split(ratings, train_percentage=0.8, random_state=None):
    ratings = ratings.tocoo()
    random_state = evaluation.check_random_state(random_state)
    random_index = random_state.random_sample(len(ratings.data))
    train_index = random_index < train_percentage
    test_index = random_index >= train_percentage

    train = sparse.csr_matrix(
        (
            ratings.data[train_index],
            (ratings.row[train_index], ratings.col[train_index]),
        ),
        shape=ratings.shape,
        dtype=ratings.dtype,
    )

    test = sparse.csr_matrix(
        (ratings.data[test_index], (ratings.row[test_index], ratings.col[test_index])),
        shape=ratings.shape,
        dtype=ratings.dtype,
    )

    test.data[test.data < 0] = 0
    test.eliminate_zeros()

    return (
        train,
        test,
        # np.dstack((ratings.row[train_index], ratings.col[train_index])).squeeze()
        train_index,
    )


lastfm = datasets.get_lastfm()
train, tmp, train_sampled_ids = train_test_split(
    sparse.csr_array(lastfm.values), train_percentage=0.7, random_state=next(seed_gen)
)
val, test, val_sampled_ids = train_test_split(
    tmp, train_percentage=2 / 3, random_state=next(seed_gen)
)
# train, val, test = recommender.csr_dataset_splits(lastfm.values, seed_gen)
# train = lastfm.sample(frac=0.7, random_state=next(seed_gen))
# val = lastfm.drop(train.index.tolist()).sample(frac=2 / 3, random_state=next(seed_gen))
# test = lastfm.drop(train.index.tolist()).drop(val.index.tolist())

model = implicit.lmf.LogisticMatrixFactorization(random_state=next(seed_gen))
# model.fit(sparse.csr_array(train.values))
model.fit(train)

estimated_prefs = recommender.get_preferences(model)

# val_ids = np.argwhere(val.toarray())
val_estimates = estimated_prefs[val_index[:, 0], val_index[:, 1]]
# true_estimates = train.toarray()[val_ids[:, 0], val_ids[:, 1]]

dcg_score(true_estimates, val_estimates)


to_generate = np.arange(val.shape[0], dtype="int32")
to_generate = to_generate[np.ediff1d(val.indptr) > 0]

batch_size = 1000
start_idx = 0

# while start_idx < len(to_generate):
batch = to_generate[start_idx : start_idx + batch_size]
ids, _ = model.recommend(batch, train[batch], N=10)
start_idx += batch_size
