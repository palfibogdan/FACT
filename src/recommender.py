import itertools as it
import logging
import pprint
from pathlib import Path
from typing import Dict, Sequence, Union

import numpy as np
import pandas as pd
from scipy import sparse

import constants
import datasets
import recommender_models as recsys
import utils

# from implicit import evaluation

# TODO global configuration object to pass around functions & access only the
# needed attributes instead of opaque **kwargs...
# TODO possibility to use other evaluation metrics, e.g. from
# implicit.evaluation? don't work out of the box with funk_svd.SVD though

logger = logging.getLogger(__name__)


def save_hyperparams_and_metrics(
    filename: Path, hparams: dict, metrics: dict, metric_used: str = "DCG@40"
):
    header = f"{','.join(list(hparams) + list(metrics))},used"
    content = list(hparams.values()) + list(metrics.values()) + [metric_used]
    content = ",".join(list(map(str, content)))
    with open(filename, "w") as fd:
        fd.write(header + "\n")
        fd.write(content + "\n")
    logger.debug("Saved best model hyperparams and metrics to %s", filename)


# NOTE each combination of hparams in hparams_flat should be ordered according
# to hparams_names
def search_best_model(
    # train_mat: sparse.csr_array,
    # valid_mat: sparse.csr_array,
    train_mat: Union[sparse.csr_array, pd.DataFrame],
    valid_mat: np.ma.masked_array,
    hparams_names: Sequence[str],
    hparams_flat: Sequence[float],
    model_class: recsys.AnyRecommender,
    # metric: str,
) -> Dict[str, dict]:
    # best_metrics, best_model, best_hparams = {}, None, None
    best_score = -1.0

    for hparams in hparams_flat:
        model = model_class(**dict(zip(hparams_names, hparams)))
        model.train(train_mat)
        # metrics = evaluation.ranking_metrics_at_k(model, train_mat, valid_mat)
        # score, best_score = metrics[metric], best_metrics.get(metric, -1.0)
        score = model.validate(valid_mat)
        if score > best_score:
            logger.info(
                # "Best model found! Old %s: %f new %s: %f hparams: %s",
                # metric,
                # best_score,
                # metric,
                # score,
                # hparams,
                "Best model found! Old DCG@40: %f new: %f hparams: %s",
                best_score,
                score,
                hparams,
            )
            # best_metrics = metrics
            best_score = score
            best_model = model
            best_hparams = hparams

    return {
        "model": best_model,
        "hparams": dict(zip(hparams_names, best_hparams)),
        # "metrics": best_metrics,
        "metrics": {"DCG@40": best_score},
    }


def search_ground_truth(
    train_mat: sparse.csr_array,
    valid_mat: np.ma.masked_array,
    base_save_path: Path,
    dataset: str,
    # metric: str = "ndcg",
    **_,
) -> recsys.Recommenders:
    hyperparams = constants.GROUND_TRUTH_HP[dataset]
    logger.info("Hyperparameters in grid search for dataset %s:", dataset)
    logger.info(pprint.pformat(hyperparams))

    best_dict = search_best_model(
        train_mat,
        valid_mat,
        hyperparams.keys(),
        list(it.product(*hyperparams.values())),
        recsys.GROUND_TRUTH_MODELS[dataset],
    )

    model_save_path = f"{base_save_path}.npz"
    best_dict["model"].save(model_save_path)
    logger.debug("Saved best model to %s", model_save_path)

    hparams_save_path = f"{base_save_path}_hparams.txt"
    save_hyperparams_and_metrics(
        hparams_save_path, best_dict["hparams"], best_dict["metrics"]
    )

    return best_dict["model"]


def search_recommender(
    train_mat: pd.DataFrame,
    valid_mat: np.ma.masked_array,
    base_save_path: Path,
    # metric: str = "ndcg",
    **_,
):
    hyperparams = constants.recommender_hparams
    logger.info("Hyperparameters in recommender grid search:")
    logger.info(pprint.pformat(hyperparams))

    hyperparams_inner = hyperparams.copy()
    factors = hyperparams_inner.pop("factors")
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
        save_hyperparams_and_metrics(
            hparams_save_path, best_dict["hparams"], best_dict["metrics"]
        )


def generate_ground_truth(
    dataset_name: str, base_save_path: Path, rng: np.random.Generator, **kwargs
) -> recsys.Recommenders:
    logger.info("Loading and preprocessing dataset %s...", dataset_name)
    dataset = datasets.get_dataset(dataset_name, **kwargs)
    train, valid, _ = utils.train_test_split_mask(dataset, 0.7, rng, valid_prop=0.1)
    return search_ground_truth(
        sparse.csr_array(train.filled()), valid, base_save_path, dataset_name, **kwargs
    )


def generate_recommenders(
    base_save_path: Path,
    rng: np.random.Generator,
    dataset_name: str = None,
    ground_truth_model_path: Path = None,
    downsampling_ratio: float = None,
    **kwargs,
):
    if ground_truth_model_path is not None and ground_truth_model_path.exists():
        logger.info("Loading ground truth preferences from %s", ground_truth_model_path)
        model_class = recsys.GROUND_TRUTH_MODELS[dataset_name]
        ground_truth_model = model_class.load(ground_truth_model_path)
    elif dataset_name is not None and dataset_name in datasets.DATASETS_RETRIEVE:
        logger.info("Generating ground truth preferences for dataset: %s", dataset_name)
        ground_truth_model = generate_ground_truth(
            dataset_name, base_save_path, rng, **kwargs
        )
    else:
        logger.error("No recommender system implemented for dataset %s", dataset_name)
        raise NotImplementedError

    train, valid, _ = utils.train_test_split_mask(
        ground_truth_model.preferences, 0.7, rng, valid_prop=0.1
    )
    train_df = (
        pd.DataFrame(train.filled())
        .melt(var_name="i_id", value_name="rating", ignore_index=False)
        .reset_index(names="u_id")
    )
    logger.info("Start fitting the recommender models.")
    search_recommender(train_df, valid, base_save_path, **kwargs)
