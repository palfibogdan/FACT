import itertools as it
import logging
import pprint
from pathlib import Path
from typing import Dict, Sequence, Tuple

import implicit
import numpy as np
from implicit import evaluation
from scipy import sparse

import config
import constants
import datasets
import recommender
import recommender_models as recsys
import utils

logger = logging.getLogger(__name__)


def search_best_model(
    train_mat: sparse.csr_array,
    valid_mat: sparse.csr_array,
    hparams_names: Sequence[str],
    hparams_flat: Sequence[float],
    model_class,
    metric: str,
) -> Dict[str, dict]:
    best_metrics, best_model, best_hparams = {}, None, None

    for hparams in hparams_flat:
        model = model_class(**dict(zip(hparams_names, hparams)))
        model.train(train_mat)
        metrics = evaluation.ranking_metrics_at_k(model.model, train_mat, valid_mat)
        score, best_score = metrics[metric], best_metrics.get(metric, -1.0)
        if score > best_score:
            logger.info(
                "Best model found! Old %s: %f new %s: %f hparams: %s",
                metric,
                best_score,
                metric,
                score,
                hparams,
            )
            best_metrics = metrics
            best_model = model
            best_hparams = hparams

    return {
        "model": best_model,
        "hparams": dict(zip(hparams_names, best_hparams)),
        "metrics": best_metrics,
    }


def search_ground_truth(
    train_mat: sparse.csr_array,
    valid_mat: sparse.csr_array,
    base_save_path: Path,
    model_class,
    hyperparams: dict,
    metric: str = "ndcg",
    **_,
) -> implicit.als.AlternatingLeastSquares:

    # hyperparams = constants.ground_truth_hparams
    logger.info("Hyperparameters in grid search:")
    logger.info(pprint.pformat(hyperparams))

    best_dict = search_best_model(
        train_mat,
        valid_mat,
        hyperparams.keys(),
        list(it.product(*hyperparams.values())),
        model_class,
        metric,
    )

    best_dict["model"].save(base_save_path)
    logger.debug("Saved best model to %s", base_save_path)

    hparams_save_path = f"{base_save_path}_hparams.txt"
    recommender.save_hyperparams_and_metrics(
        hparams_save_path, best_dict["hparams"], best_dict["metrics"], metric
    )

    return best_dict["model"]


def search_recommender(
    train_mat: sparse.csr_array,
    valid_mat: sparse.csr_array,
    base_save_path: Path,
    metric: str = "ndcg",
    **_,
):
    """
    This function trains a model for each factor and saves the best model for
    each factor.
    To be used with the recommender models.
    """

    hyperparams = constants.recommender_hparams
    logger.info("Hyperparameters in recommender grid search:")
    logger.info(pprint.pformat(hyperparams))

    hyperparams_inner = hyperparams.copy()
    factors = hyperparams_inner.pop("n_factors")
    hyperparams_inner_flat = list(it.product(*hyperparams_inner.values()))

    # save the best model for each factor
    for factor in factors:
        # NOTE (factor, ) + hparams relies on the keys of hyperparams to be,
        # in order, factors,regularization,alpha; there is an easy fix for
        # this not implemented rn
        hparams = list((factor,) + hp for hp in hyperparams_inner_flat)
        best_dict = search_best_model(
            train_mat, valid_mat, hyperparams.keys(), hparams, recsys.FSVD
        )

        model_save_path = f"{base_save_path}_factors_{factor}.npz"
        best_dict["model"].save(model_save_path)
        logger.debug("Saved best model to %s", model_save_path)

        hparams_save_path = f"{base_save_path}_factors_{factor}_hparams.txt"
        recommender.save_hyperparams_and_metrics(
            hparams_save_path, best_dict["hparams"], best_dict["metrics"], metric
        )


# utils.setup_root_logging()
# seed_gen = utils.SeedSequence()
# lastfm = datasets.get_lastfm()
# train_lastfm, valid_lastfm, _ = recommender.csr_dataset_splits(lastfm.values, seed_gen)
# search_ground_truth(
#     train_lastfm,
#     valid_lastfm,
#     config.LASTFM_RECOMMENDER_DIR / "model_lastfm_LMF",
#     recsys.LMF,
#     constants.ground_truth_lastfm_hparams,
#     # constants.ground_truth_hparams,
# )
