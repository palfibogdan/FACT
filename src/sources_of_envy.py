import numpy as np
import config
import rl
import constants


def recommendation_probs(
    user_preferences: np.ndarray, temperature: float
) -> np.ndarray:
    return scipy.special.softmax(user_preferences / temperature, axis=1)


# NOTE works with both the full user-item preferences matrix and a single user's
# preferences, add a batch dimension in the latter case
def user_policy_recommendation(
    user_preferences: np.ndarray, temperature: float, rng: np.random.Generator
) -> np.ndarray:
    # apply softmax with temperature to turn preferences into probabilities
    probs = recommendation_probs(user_preferences, temperature)
    # retrieve the indexes of the most recommended item for each user according to the
    # softmax probabilities
    # NOTE the vectorized version that samples over the whole matrix returns the
    # same sample for each user :/
    recommended_items_indexes = np.array(
        [
            rng.choice(user_preferences.shape[1], replace=False, p=user_probs)
            for user_probs in recommendation_probs
        ]
    )
    return recommended_items_indexes


def get_utilities(expected_rewards, item_probabilities):
    # return a utility matrix where element i, j  represents
    # the utility of user i with the policies j
    utilities = item_probabilities @ expected_rewards.T
    return utilities


def get_deltas(M, utilities):
    """
    M - set of users {m, n1, n2, ...}
    Utilities - matrix - utilities for each user for each policy
    """
    deltas = []

    for baseline_user in M:
        max_utility = np.max(utilities[baseline_user])
        deltas.append(max(max_utility - utilities[baseline_user], 0))

    return np.array(deltas)


def get_average_proportion(M, utilities, eps):
    """
        M - set of users {m, n1, n2, ...}
        Utilities - matrix - utilities for each user for each policy
        eps - scalar
        """
    deltas = get_deltas(M, utilities)

    return np.mean(deltas), np.mean(deltas > eps)


def run_experiment_5_1_1(filenames, eps, num_users):
    """
    Loads trained recommendation system models, gets the utility matrix given by these models
    and calculates the average envy and the proportion of eps-envious users
    Return list of average envy and proportion of eps-envious users for each number of factors
    """
    
    # Lists for storing the average envy and proportion of eps-envious users for each number of factors
    avg_envy_list = []
    prop_envy_users_list = []

    temperature = 5
    rng = np.random.default_rng(constants.SEED)
    ground_truth_lastfm = np.load(config.MODELS_DIR / "ground_truth_lastfm.npy")

    # set of users
    M = np.arrange(num_users)

    for filename in filenames:

        # load recommender system model
        recommender = np.load(config.MODELS_DIR / filename)

        # 2D: recommendation scores for each item per each user
        recommender_preferences = recommender.user_preferences @ recommender.item_preferences.T

        # get indexes of recommended items given by the model
        recommendations = rl.user_policy_recommendation(recommender_preferences, temperature, rng)

        # 2D: for each user, what items are truly recommended
        true_recommendations = rl.generate_true_recommendation(ground_truth_lastfm, temperature, rng)

        #TODO fix this: expected rewards are 1D array, but true_recommendations is 2D
        #TODO See how to get these two variables
        expected_rewards = np.take_along_axis(true_recommendations, recommendations[:, None], axis=1).squeeze()
        item_probabilities = recommendation_probs(recommender_preferences, temperature)

        utility_matrix = get_utilities(expected_rewards, item_probabilities)
        average_envy, prop_envy_users = get_average_proportion(M, utility_matrix, eps)

        avg_envy_list.append(average_envy)
        prop_envy_users_list.append(prop_envy_users)

    return avg_envy_list, prop_envy_users_list
