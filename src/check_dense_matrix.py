import numpy as np
import implicit
from implicit import evaluation
import scipy
from scipy import sparse

import constants
import datasets
import recommender

seed = constants.SEED
rng = np.random.default_rng(seed)

lastfm = datasets.get_lastfm()

user_artist_csr = scipy.sparse.csr_matrix(lastfm.values)
train_csr, tmp_csr = evaluation.train_test_split(user_artist_csr, train_percentage=0.7, random_state=seed)
val_csr, test_csr = evaluation.train_test_split(tmp_csr, train_percentage=2/3, random_state=seed)

model = implicit.als.AlternatingLeastSquares(factors=32, alpha=10, regularization=10)
model.fit(train_csr)

ground_truth = recommender.get_preferences(model)

ground_truth_train_sample = ground_truth[rng.choice(ground_truth.shape[0], int(0.7 * ground_truth.shape[1]), replace=False), :]
print(ground_truth.shape, ground_truth_train_sample.shape)
ground_truth_train_csr = sparse.csr_array(ground_truth_train_sample)
model.fit(ground_truth_train_csr)