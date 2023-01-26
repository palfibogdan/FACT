import itertools as it
import logging
import pprint
from pathlib import Path
from typing import Dict, Sequence, Tuple

import implicit
import numpy as np
from implicit import evaluation
from scipy import sparse

import constants
import datasets
import utils

logger = logging.getLogger(__name__)


def save_hyperparams_and_metrics(
    filename: Path, hparams: dict, metrics: dict, metric_used: str
):
    header = f"{','.join(list(hparams) + list(metrics))},used"
    content = list(hparams.values()) + list(metrics.values()) + [metric_used]
    content = ",".join(list(map(str, content)))
    with open(filename, "w") as fd:
        fd.write(header + "\n")
        fd.write(content + "\n")
    logger.debug("Saved best model hyperparams and metrics to %s", filename)


def load_recommeder_model(filename: str) -> implicit.als.AlternatingLeastSquares:
    return implicit.cpu.als.AlternatingLeastSquares.load(filename)


def get_preferences(model: implicit.als.AlternatingLeastSquares) -> np.ndarray:
    user_factors, item_factors = model.user_factors, model.item_factors
    if implicit.gpu.HAS_CUDA:
        user_factors, item_factors = user_factors.to_numpy(), item_factors.to_numpy()
    return user_factors @ item_factors.T


def load_preferences(filename: str) -> np.ndarray:
    return get_preferences(load_recommeder_model(filename))


# NOTE each combination of hparams in hparams_flat should be ordered according
# to hparams_names
def search_best_model(
    train_mat: sparse.csr_array,
    valid_mat: sparse.csr_array,
    hparams_names: Sequence[str],
    hparams_flat: Sequence[float],
    metric: str,
) -> Dict[str, dict]:
    best_metrics, best_model, best_hparams = {}, None, None

    for hparams in hparams_flat:
        model = implicit.als.AlternatingLeastSquares(
            **dict(zip(hparams_names, hparams))
        )
        model.fit(train_mat)
        metrics = evaluation.ranking_metrics_at_k(model, train_mat, valid_mat)
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
    metric: str = "ndcg",
    **_,
) -> implicit.als.AlternatingLeastSquares:
    """
    Fits a recommender system on a training set using matrix factorization as
    described in http://yifanhu.net/PUB/cf.pdf, and evaluates performance on
    a validation set according to the passed metric. This metric drives model
    selection in a hyperparameter grid search.

    Args:
        train_mat: User-item training set to fit the matrix factors,
                   in scipy.sparse.csr_array format.
        valid_mat: User-item validation set, evaluated according to `metric`,
                   in scipy.sparse.csr_array format.
        base_save_path: The stem (complete filename without extension) of the
                        file the best model is saved to. The model is saved to
                        `base_save_path`.npy, and the hypeparameters and
                        evaluation metrics to  `base_save_path`_hparams.txt.
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

    hyperparams = constants.ground_truth_hparams
    logger.info("Hyperparameters in grid search:")
    logger.info(pprint.pformat(hyperparams))

    best_dict = search_best_model(
        train_mat,
        valid_mat,
        hyperparams.keys(),
        list(it.product(*hyperparams.values())),
        metric,
    )

    model_save_path = f"{base_save_path}.npz"
    best_dict["model"].save(model_save_path)
    logger.debug("Saved best model to %s", model_save_path)

    hparams_save_path = f"{base_save_path}_hparams.txt"
    save_hyperparams_and_metrics(
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
    factors = hyperparams_inner.pop("factors")
    hyperparams_inner_flat = list(it.product(*hyperparams_inner.values()))

    # save the best model for each factor
    for factor in factors:
        # NOTE (factor, ) + hparams relies on the keys of hyperparams to be,
        # in order, factors,regularization,alpha; there is an easy fix for
        # this not implemented rn
        hparams = list((factor,) + hp for hp in hyperparams_inner_flat)
        best_dict = search_best_model(
            train_mat, valid_mat, hyperparams.keys(), hparams, metric
        )

        model_save_path = f"{base_save_path}_factors_{factor}.npz"
        best_dict["model"].save(model_save_path)
        logger.debug("Saved best model to %s", model_save_path)

        hparams_save_path = f"{base_save_path}_factors_{factor}_hparams.txt"
        save_hyperparams_and_metrics(
            hparams_save_path, best_dict["hparams"], best_dict["metrics"], metric
        )


# NOTE increase the value of seed with next() so that calling the function twice
# yields different results; look at how an integer seed is always reinitialized
# in implicit.evaluation.train_test_split
def csr_dataset_splits(
    dataset: np.ndarray, seed_gen: utils.Seed
) -> Tuple[sparse.csr_array, ...]:
    dataset_csr = sparse.csr_array(dataset)
    logger.info("Splitting the dataset into 70/10/20% train/validation/test splits...")
    train_csr, tmp_csr = evaluation.train_test_split(
        dataset_csr, train_percentage=0.7, random_state=next(seed_gen)
    )
    valid_csr, test_csr = evaluation.train_test_split(
        tmp_csr, train_percentage=2 / 3, random_state=next(seed_gen)
    )
    return train_csr, valid_csr, test_csr


def recommender_input_data(
    ground_truth: np.ndarray,
    seed_gen: utils.Seed,
    rng: np.random.Generator,
    downsampling_ratio: float = None,
) -> Tuple[sparse.csr_array, ...]:

    if downsampling_ratio is None:
        logger.info("Take ground truth input for recommender without downsampling")
        return csr_dataset_splits(ground_truth, seed_gen)

    logger.info(
        "Downsampling the ground truth data to %d% to train the recommender",
        downsampling_ratio * 100,
    )
    # we mask 80% of the ground truth data because in section 5.1 they say:
    # the simulated recommender system estimates relevance scores using low-rank
    # matrix completion (Bell and Sejnowski 1995) on a training sample of 20% of
    # the ground truth preferences
    indices = [
        (i, j)
        for i in range(ground_truth.shape[0])
        for j in range(ground_truth.shape[1])
    ]
    sample = rng.choice(indices, size=int(0.2 * len(indices)), replace=False)
    ground_truth_masked = np.zeros_like(ground_truth)
    ground_truth_masked[sample[:, 0], sample[:, 1]] = ground_truth[
        sample[:, 0], sample[:, 1]
    ]
    return csr_dataset_splits(ground_truth_masked, seed_gen)


def generate_ground_truth(
    dataset_name: str, base_save_path: Path, seed_gen: utils.Seed, **kwargs
) -> implicit.als.AlternatingLeastSquares:
    logger.info("Loading and preprocessing dataset %s...", dataset_name)
    dataset = datasets.get_dataset(dataset_name, **kwargs)
    train_csr, valid_csr, _ = csr_dataset_splits(dataset.values, seed_gen)
    return search_ground_truth(train_csr, valid_csr, base_save_path, **kwargs)


def generate_recommenders(
    base_save_path: Path,
    seed_gen: utils.Seed,
    rng: np.random.Generator,
    dataset_name: str = None,
    ground_truth_model_path: Path = None,
    downsampling_ratio: float = None,
    **kwargs,
):
    if ground_truth_model_path is not None and ground_truth_model_path.exists():
        logger.info("Loading ground truth preferences from %s", ground_truth_model_path)
        ground_truth = load_preferences(ground_truth_model_path)
    elif dataset_name is not None and dataset_name in datasets.DATASETS_RETRIEVE:
        logger.info("Generating ground truth preferences for dataset: %s", dataset_name)
        ground_truth_model = generate_ground_truth(
            dataset_name,
            base_save_path,
            seed_gen,
            **kwargs,
        )
        ground_truth = get_preferences(ground_truth_model)
    else:
        logger.error("No recommender system implemented for dataset %s", dataset_name)
        raise NotImplementedError

    train_csr, valid_csr, test_csr = recommender_input_data(
        ground_truth, seed_gen, rng, downsampling_ratio=downsampling_ratio
    )
    logger.info("Start fitting the recommender models.")
    search_recommender(train_csr, valid_csr, base_save_path, **kwargs)
