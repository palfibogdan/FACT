import itertools as it
import logging
import pprint
from copy import deepcopy
from typing import Dict, Sequence

import implicit
import numpy as np
from implicit import evaluation
from scipy import sparse

import config
import constants
import preprocessing as preproc
import utils

logger = logging.getLogger(__name__)


def recommender_grid_search(
    train_mat: sparse.csr_matrix,
    valid_mat: sparse.csr_matrix,
    best_model_paths: Sequence[str],
    metric: str = "map",
    **hyperparams: Dict[str, Sequence],
) -> implicit.als.AlternatingLeastSquares:
    """
    Fits a recommender system on a training set using matrix factorization as
    described in http://yifanhu.net/PUB/cf.pdf, and evaluates performance on
    a validation set according to the passed metric. This metric drives model
    selection in a hyperparameter grid search.

    Args:
        train_mat: User-item training set to fit the matrix factors,
                   in scipy.sparse.csr_matrix format.
        valid_mat: User-item validation set, evaluated according to `metric`,
                   in scipy.sparse.csr_matrix format.
        best_model_paths: List containing 2 paths. The best model is saved in
                          the first path, its hyperparameters in the second one.
        metric: The evaluation metric used to test the trained model, provided
                in the implicit/evaluation.pyx module. Supported values are:
                ['precision', map', 'ndcg', 'auc']. Defaults to 'map'.
        hyperparams: A mapping from valid implicit.als.AlternatingLeastSquares
                     keyword arguments to list of values. The cartesian product
                     of these values is used for grid search.

    Returns:
        The best matrix factorization model found according to `metric`, with
        factors stored in `model.user_factors` and `model.item_factors`.
    """

    logger.info("Hyperparameters in grid search:")
    logger.info(pprint.pformat(hyperparams))
    hyperparams_flat = list(it.product(*hyperparams.values()))

    best_model_path, best_hparams_path = best_model_paths
    best_score, best_model, best_hparams = -1.0, None, None

    for i, hparams in enumerate(hyperparams_flat):
        model = implicit.als.AlternatingLeastSquares(
            **dict(zip(hyperparams.keys(), hparams))
        )
        model.fit(train_mat)
        score = evaluation.ranking_metrics_at_k(model, train_mat, valid_mat)[metric]
        if score > best_score:
            logger.info(
                "%d: Best model found! Old %s: %f new %s: %f hparams: %s",
                i,
                metric,
                best_score,
                metric,
                score,
                hparams,
            )
            best_score = score
            best_model = deepcopy(model)
            best_hparams = hparams

    best_model.save(best_model_path)
    logger.debug("Saved best model to %s", best_model_path)
    with open(best_hparams_path, "w") as fd:
        fd.write("factor,regularizer,alpha,metric,score\n")
        fd.write(f"{','.join(list(map(str, best_hparams + (metric, best_score))))}\n")
    logger.debug("Saved best model hyperparams to %s", best_hparams_path)
    return best_model


def create_preferences(
    lastfm_csr: sparse.csr_matrix, seed: int, savepaths: Sequence[str], **kwargs
) -> implicit.als.AlternatingLeastSquares:
    # split into 0.7 train 0.2 val 0.1 test
    train_csr, tmp_csr = implicit.evaluation.train_test_split(
        lastfm_csr, train_percentage=0.7, random_state=seed
    )
    valid_csr, test_csr = implicit.evaluation.train_test_split(
        tmp_csr, train_percentage=2 / 3, random_state=seed
    )
    # create ground truth preferences
    model = recommender_grid_search(train_csr, valid_csr, savepaths, **kwargs)
    return model


if __name__ == "__main__":
    utils.setup_root_logging()

    seed = 42
    rng = np.random.default_rng(seed=seed)

    lastfm = preproc.get_lastfm()
    lastfm_csr = sparse.csr_matrix(lastfm.values)

    # make ground truths
    ground_truth_paths = [
        config.MODELS_DIR / "model_lastfm_ground_truth.npz",
        config.MODELS_DIR / "hparams_lastfm_ground_truth.txt",
    ]
    ground_truth_model = create_preferences(
        lastfm_csr, seed, ground_truth_paths, **constants.ground_truth_hparams
    )
    # matrix completion
    U, V = ground_truth_model.user_factors, ground_truth_model.item_factors
    # when run on GPUs, U and V are instances of implicit.gpu._cuda.Matrix, which
    # has no transpose attribute
    if implicit.gpu.HAS_CUDA:
        U, V = U.to_numpy(), V.to_numpy()
    ground_truth = U @ V.T
    # save the ground truths
    savepath = config.MODELS_DIR / "ground_truth_lastfm.npy"
    np.save(savepath, ground_truth)
    logger.debug("Saved ground truth preferences for lastfm at %s", savepath)

    # make recommender
    # TODO check that this is the correct interpretation & vectorize
    # we mask 80% of the ground truth data because in section 5.1 they say:
    # the simulated recommender system estimates relevance scores using low-rank
    # matrix completion (Bell and Sejnowski 1995) on a training sample of 20% of
    # the ground truth preferences
    indices = [
        (i, j)
        for i in range(ground_truth.shape[0])
        for j in range(ground_truth.shape[1])
    ]
    # pick some random preferences (20%) that will not be zeroed out
    kept_preferences = rng.choice(indices, size=int(0.2 * len(indices)), replace=False)
    ground_truth_masked = np.zeros_like(ground_truth)
    for i, j in kept_preferences:
        ground_truth_masked[i, j] = ground_truth[i, j]
    # estimate the true preferences with the actual recommender system
    ground_truth_masked_sparse = sparse.csr_matrix(ground_truth_masked)
    recommender_paths = [
        config.MODELS_DIR / "model_lastfm.npz",
        config.MODELS_DIR / "hparams_lastfm.txt",
    ]
    model = create_preferences(
        ground_truth_masked_sparse,
        seed,
        recommender_paths,
        **constants.recommender_hparams,
    )
