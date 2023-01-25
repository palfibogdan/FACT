from typing import Dict, Sequence, Tuple

import implicit
import numpy as np

import rl


# TODO ask if this is the right way of computing the utilities
def get_utilities(expected_rewards, recommendations):
    # return a utility matrix where element i, j  represents
    # the utility of user i with the policies j
    # utilities = ground_truth_preferences @ recommendations.T
    utilities = expected_rewards @ recommendations.T
    return utilities


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


# # Lists for storing the average envy and proportion of eps-envious users for each number of factors
#     avg_envy_list = []
#     prop_envy_users_list = []

#     temperature = 5
#     rng = np.random.default_rng(constants.SEED)
#     ground_truth_lastfm = np.load(config.MODELS_DIR / "ground_truth_lastfm.npy")

#     # set of users
#     M = np.arrange(num_users)

#     for filename in filenames:

#         # load recommender system model
#         recommender = np.load(config.MODELS_DIR / filename)

#         # 2D: recommendation scores for each item per each user
#         recommender_preferences = recommender.user_preferences @ recommender.item_preferences.T

#         # get indexes of recommended items given by the model
#         recommendations = rl.user_policy_recommendation(recommender_preferences, temperature, rng)

#         # 2D: for each user, what items are truly recommended
#         true_recommendations = rl.generate_true_recommendation(ground_truth_lastfm, temperature, rng)

#         #TODO fix this: expected rewards are 1D array, but true_recommendations is 2D
#         #TODO See how to get these two variables
#         expected_rewards = np.take_along_axis(true_recommendations, recommendations[:, None], axis=1).squeeze()
#         item_probabilities = recommendation_probs(recommender_preferences, temperature)

#         utility_matrix = get_utilities(expected_rewards, item_probabilities)
#         average_envy, prop_envy_users = get_average_proportion(M, utility_matrix, eps)

#         avg_envy_list.append(average_envy)
#         prop_envy_users_list.append(prop_envy_users)

#     return avg_envy_list, prop_envy_users_list
