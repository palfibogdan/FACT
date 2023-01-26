import config
import constants
import funk_svd
import numpy as np
import pandas as pd
import recommender
import utils
from sklearn.model_selection import train_test_split

# rng = np.random.default_rng(constants.SEED)
seed_gen = utils.SequenceGenerator(constants.SEED)

lastfm_gt = recommender.load_preferences(config.LASTFM_RECOMMENDER_DIR / "model.npz")

# reshape ground truths into format expected by funk_svd.SVD.fit
lastfm_gt_df = pd.DataFrame(lastfm_gt)
lastfm_gt_df = lastfm_gt_df.melt(var_name="i_id", value_name="rating").reset_index(
    names="u_id"
)

# split into 70/10/20 train/validation/test set
gt_train, tmp = train_test_split(
    lastfm_gt_df, train_size=0.7, random_state=next(seed_gen)
)
gt_val, gt_test = train_test_split(tmp, train_size=2 / 3, random_state=next(seed_gen))

# train an SVD model to fit the ground truth preferences
svd = funk_svd.SVD(lr=0.001, reg=0.005, n_epochs=100, n_factors=32, shuffle=False)
svd.fit(gt_train)

# compute recommender system preferences
recommender_preferences = (svd.pu_ @ svd.qi_.T) + svd.bu_ + svd.bi_


df = lastfm_gt_df
train = df.sample(frac=0.7, random_state=7)
val = df.drop(train.index.tolist()).sample(frac=2 / 3, random_state=8)
test = df.drop(train.index.tolist()).drop(val.index.tolist())
