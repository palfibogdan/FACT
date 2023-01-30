import itertools as it
import logging
import multiprocessing as mp
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


def search_best_model(
    train_mat: Union[sparse.csr_array, pd.DataFrame],
    valid_mat: np.ma.masked_array,
    hparams_names: Sequence[str],
    hparams_flat: Sequence[float],
    model_class: recsys.AnyRecommender,
    parallel=False,
):
    # NOTE declared closure globally then delete it for parallelism
    # (https://stackoverflow.com/a/67050659)
    global search

    def search(hp, rest):
        global logger  # unnecessary but visible
        model_class, hparams_names, train = rest
        hp = dict(zip(hparams_names, hp))
        model = model_class(**hp)
        logger_ = getattr(model, "logger", logger)
        logger_.info("Grid search hparams: %s", hp)
        model.train(train)
        # metrics = evaluation.ranking_metrics_at_k(model, train_mat, valid_mat)
        # score, best_score = metrics[metric], best_metrics.get(metric, -1.0)
        return model.validate(valid_mat), hp, model

    args = list(
        zip(
            hparams_flat,
            [[model_class, list(hparams_names), train_mat]] * len(hparams_flat),
        )
    )

    if parallel:
        n_procs = min(len(hparams_flat), mp.cpu_count())
        logger.info("Spawning %d processes for grid search", n_procs)
        with mp.Pool(processes=n_procs) as pool:
            res = pool.starmap(search, args)
    else:
        res = [search(*arg) for arg in args]

    del search

    best_score, best_hparams, best_model = max(res, key=lambda el: el[0])
    return {
        "model": best_model,
        "hparams": best_hparams,
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

    ground_truth_model_class = recsys.GROUND_TRUTH_MODELS[dataset]
    logger.info("Ground truth for %s, model: %s", dataset, ground_truth_model_class)

    best_dict = search_best_model(
        train_mat,
        valid_mat,
        hyperparams.keys(),
        list(it.product(*hyperparams.values())),
        ground_truth_model_class,
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
    logger.info("%s, Hyperparameters in recommender grid search:", recsys.FSVD)
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
            train_mat,
            valid_mat,
            hyperparams.keys(),
            hparams,
            recsys.FSVD,
            parallel=True,
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
    # shape the dataframe in the format wanted by funk_svd.SVD
    train_df = (
        pd.DataFrame(train.filled())
        .melt(var_name="i_id", value_name="rating", ignore_index=False)
        .reset_index(names="u_id")
    )
    logger.info("Start fitting the recommender models.")
    search_recommender(train_df, valid, base_save_path, **kwargs)
