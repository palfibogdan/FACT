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
import datasets
import utils

logger = logging.getLogger(__name__)


def get_recommeder_model(filename: str) -> implicit.als.AlternatingLeastSquares:
    return implicit.cpu.als.AlternatingLeastSquares.load(filename)


def get_preferences(model: implicit.als.AlternatingLeastSquares) -> np.ndarray:
    return model.user_factors @ model.item_factors.T


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


# NOTE best_factors_path should contain the dataset in its name to avoid override
# NOTE does not save the best hyperparameters for each best model found
def best_model_each_factor(
    train_mat: sparse.csr_matrix,
    valid_mat: sparse.csr_matrix,
    best_factors_path: str,
    metric: str = "map",
    **hyperparams: Dict[str, Sequence],
):
    """
    This function trains a model for each factor and saves the best model for
    each factor.
    To be used with the recommender models.
    """

    factors = hyperparams["factors"]
    regularization = hyperparams["regularization"]
    alpha = hyperparams["alpha"]

    # save the best model for each factor
    for factor in factors:
        best_score, best_model = -1.0, None
        for reg in regularization:
            for a in alpha:
                hparams = (factor, reg, a)
                model = implicit.als.AlternatingLeastSquares(
                    factors=factor, regularization=reg, alpha=a
                )
                model.fit(train_mat)
                score = evaluation.ranking_metrics_at_k(model, train_mat, valid_mat)[
                    metric
                ]
                if score > best_score:
                    logger.info(
                        f"Best model found (for {factor} factors) ! Old {metric}: {best_score} new {metric}: {score} hparams: {hparams}"
                    )
                    best_score = score
                    best_model = deepcopy(model)

        name = f"model_{factor}_factors"

        path = best_factors_path / name
        best_model.save(path)


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
