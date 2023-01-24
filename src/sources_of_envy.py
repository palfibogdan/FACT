from typing import Dict, Sequence, Tuple

import numpy as np

import recommender
import rl
import utils


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
    # set negative delta envy to 0, NOTE should be unnecessary
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

    # NOTE should we rescale the raw ground_truths or apply softmax first? the
    # min and max stay the same, overall scores are different but scaled equally
    # though. The paper does not mention softmax first, let's see what the
    # numerical results are
    ground_truth_probs = utils.softmax(ground_truth, temperature=temperature)
    ground_truth_rescaled = utils.minmax_scale(ground_truth_probs)

    # dictionary to store the metrics by factors, easy to use with pandas
    metrics_dict = {"mean_envy": {}, "prop_eps_envy": {}}

    for filename in filenames:
        # load recommender system model
        recommender_model = recommender.get_recommeder_model(filename)
        # 2D: recommendation scores for each item per each user
        recommender_preferences = recommender.get_preferences(recommender_model)
        # convert to probabilities
        recommender_probs = utils.softmax(
            recommender_preferences, temperature=temperature
        )
        # # one round of recommendations for each user
        # recommendations, recommendation_probs = rl.recommendation_policy(
        #     recommender_probs, temperature, rng
        # )
        # get (expected) rewards for recommendations
        # rewards = rl.get_reward(recommendations, ground_truth_rescaled)
        # get utilities: recommender prob of selected item * ground truth
        # expected reward for that item
        utilities = rl.compute_utilities(recommender_probs, ground_truth_rescaled)
        # compute the required metrics and store them in a dictionary for each
        # latent factor
        mean_envy, prop_envy_users = get_envy_metrics(utilities, eps)
        metrics_dict["mean_envy"][recommender_model.factors] = mean_envy
        metrics_dict["prop_eps_envy"][recommender_model.factors] = prop_envy_users

    return metrics_dict
