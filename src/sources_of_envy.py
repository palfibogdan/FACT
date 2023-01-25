import logging
from typing import Dict, Sequence, Tuple

import implicit
import numpy as np

import config
import constants
import recommender

import rl


# TODO ask if this is the right way of computing the utilities
def get_utilities(expected_rewards, recommendations):
    # return a utility matrix where element i, j  represents
    # the utility of user i with the policies j
    # utilities = ground_truth_preferences @ recommendations.T
    utilities = expected_rewards @ recommendations.T
    return utilities

logger = logging.getLogger(__name__)


def delta_envy(utilities: np.ndarray) -> np.ndarray:
    """
    Computes the maximal envy degree of a set of users, given by
    .. math::
        \\max(\\underset{n\\in[M]}{\\max}(u^m(p^n)-u^m(p^m)),\\,0)

    Args:
        utilities: A 2D array of utility scores, of shape (#users, #users).

    Returns:
        The maximal envy degree of each user in `utilities`.
    """
    # get the maximum utility of each user
    max_envy_values = utilities.max(axis=1)
    # take the difference between the maximal utility of each user and the
    # current user's utility
    delta_envious = max_envy_values - utilities.diagonal()
    # set negative delta envy to 0
    delta_envious[delta_envious < 0] = 0.0
    return delta_envious
    # """
    # M - set of users {m, n1, n2, ...}
    # Utilities - matrix - utilities for each user for each policy
    # """
    # for baseline_user in M:
    #     max_utility = np.max(utilities[baseline_user])
    #     deltas.append(max(max_utility - utilities[baseline_user], 0))
    # return np.array(deltas)


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
    # """
    # M - set of users {m, n1, n2, ...}
    # Utilities - matrix - utilities for each user for each policy
    # eps - scalar
    # """
    # deltas = get_deltas(M, utilities)


def run_experiment_5_1_1(
    filenames: Sequence[str],
    ground_truth: np.ndarray,
    eps: float,
    temperature: float,
    rng: np.random.Generator,
) -> Dict[str, Dict[int, np.ndarray]]:
    """
    Loads trained recommendation system models, gets the utility matrix given by
    these models and calculates the average envy and the proportion of
    eps-envious users.
    Return list of average envy and proportion of eps-envious users for each
    number of factors
    """

    # dictionary to store the metrics by factors, easy to use with pandas
    metrics_dict = {"mean_envy": {}, "prop_eps_envy": {}}

    for filename in filenames:

        # load recommender system model
        recommender = implicit.cpu.als.AlternatingLeastSquares.load(filename)

        # 2D: recommendation scores for each item per each user
        recommender_preferences = recommender.user_factors @ recommender.item_factors.T

        # get one-hot encoded recommendations
        recommendations = rl.user_policy_recommendation(
            recommender_preferences, temperature, rng
        )

        # get utilities of each user, shape: #users X #users
        utilities = get_utilities(ground_truth, recommendations)

        # compute the required metrics and store them in a dictionary for each
        # latent factor
        mean_envy, prop_envy_users = get_envy_metrics(utilities, eps)
        metrics_dict["mean_envy"][recommender.factors] = mean_envy
        metrics_dict["prop_eps_envy"][recommender.factors] = prop_envy_users

    return metrics_dict


def do_envy_from_mispecification(
    lastfm_data_dir=config.LASTFM_DATA_DIR,
    lastfm_models_dir=config.LASTFM_RECOMMENDER_DIR,
    lastfm_plots_dir=config.LASTFM_PLOTS_DIR,
    movielens_data_dir=config.MOVIELENS_DATA_DIR,
    movielens_models_dir=config.MOVIELENS_RECOMMENDER_DIR,
    movielens_plots_dir=config.MOVIELENS_PLOTS_DIR,
    rng: np.random.Generator = None,
    **_,
):
    # TODO finish this
    recommender.do_envy_from_mispecification(
        ...,
        ground_truth_hparams={"metric": "map", **constants.ground_truth_hparams},
        recommender_hparams={"metric": "map", **constants.recommender_hparams},
    )
