import logging
from typing import Dict, Sequence, Tuple

import numpy as np

import config
import recommender
import recommender_models as recsys
import utils

logger = logging.getLogger(__name__)

# TODO try:
# - movielens with float ratings


def compute_utilities(
    reward_prob: np.ndarray, expected_reward: np.ndarray
) -> np.ndarray:
    # return a utility matrix where element i, j  represents
    # the utility of user i with the policies j
    # the utility of each user is on the diagonal of the returned matrix
    return expected_reward @ reward_prob.T


def delta_envy(utilities: np.ndarray) -> np.ndarray:
    """
    Computes the maximal envy degree of a set of users, given by
    .. math::
        \\max(\\underset{n\\in[M]}{\\max}(u^m(p^n)-u^m(p^m)),\\,0)

    Args:
        utilities: A 2D array of utility scores, of shape (#users, #users).

    Returns:
        The maximal envy degree of each user in `utilities`, 1D vector.
    """
    # get the maximum utility of each user
    max_envy_values = utilities.max(axis=1)
    # take the difference between the maximal utility of each user and the
    # current user's utility
    delta_envious = max_envy_values - utilities.diagonal()
    # set negative delta envy to 0 NOTE should be unnecessary
    delta_envious[delta_envious < 0] = 0.0
    return delta_envious


def get_envy_metrics(utilities: np.ndarray, eps: float) -> Tuple[float, float]:
    """
    Computes average envy and proportion of :math:`\\epsilon`-envy users.

    Args:
        utilities: 2D array of utilities of each user.
        eps: The :math:`\\epsilon` level for the proportion of
             :math:`\\epsilon`-envy users.

    Returns:
        A dictionary with the reported metrics.
    """
    deltas = delta_envy(utilities)
    return np.mean(deltas), np.mean(deltas > eps)


def experiment_5_1_1(
    recommender_filenames: Sequence[str],
    ground_truth: np.ndarray,
    eps: float,
    temperature: float,
    model_class: recsys.RecommenderType,
) -> Dict[str, Dict[int, np.ndarray]]:

    # ground_truth_probs = utils.softmax(ground_truth, temperature=temperature)
    # ground_truth_rescaled = ground_truth_probs
    # ground_truth_rescaled = utils.minmax_scale(ground_truth)
    ground_truth_rescaled = ground_truth

    # dictionary to store the metrics by factors, easy to use with pandas
    metrics_dict = {"mean_envy": {}, "prop_eps_envy": {}}

    for filename in recommender_filenames:

        recommender_model = model_class.load(filename)

        recommender_probs = utils.softmax(
            recommender_model.preferences, temperature=temperature
        )
        # recommender_probs = recommender_model.preferences

        utilities = compute_utilities(recommender_probs, ground_truth_rescaled)

        mean_envy, prop_envy_users = get_envy_metrics(utilities, eps)
        metrics_dict["mean_envy"][recommender_model.factors] = mean_envy
        metrics_dict["prop_eps_envy"][recommender_model.factors] = prop_envy_users

    return metrics_dict


def envy_from_misspecification(
    conf: config.Configuration,
) -> Dict[str, Dict[str, Dict[int, np.ndarray]]]:
    metrics = {}
    for dataset in conf.datasets:
        recommender.generate_recommenders(
            conf.ground_truth_files[dataset], dataset, conf.random_state, conf
        )
        models_dir = conf.recommender_dirs[dataset]
        recommenders_by_factors = models_dir.glob(
            f"{conf.model_base_name}_factors*.npz"
        )
        ground_truth_class = conf.ground_truth_models[dataset]
        ground_truth_model = ground_truth_class.load(conf.ground_truth_files[dataset])
        metrics[dataset] = experiment_5_1_1(
            recommenders_by_factors,
            ground_truth_model.preferences,
            conf.epsilon,
            conf.temperature,
            conf.recommender_models[dataset],
        )
    return metrics
