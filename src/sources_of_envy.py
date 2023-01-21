import numpy as np

# TODO need to find way to get utilities from the data


def get_utilities():
    pass


def get_delta(m, M, utilities_m):

    max_utility = 0
    for user in M:
        if user != m and utilities_m[user] > max_utility:
            max_utility = utilities_m[user]

    return max(max_utility - utilities_m[m], 0)


def get_average_envy(M, utilities):
    avg_envy = 0
    for user in M:
        avg_envy += get_delta(user, M, utilities)

    return avg_envy / len(M)


def get_user_proportion(M, eps, utilities):
    user_proportion = 0

    for user in M:
        if get_delta(user, M, utilities) > eps:
            user_proportion += 1

    return user_proportion / len(M)


def degree_of_envy(user_recommendation: np.ndarray, user_reward: np.ndarray, artists_not_recommended: np.ndarray) -> np.ndarray:
    # get the index of the recommended item for this user
    recommended_item_idx = np.argmax(user_recommendation)
    degree_of_envy = max(np.ma.masked_array(
        user_reward, artists_not_recommended) - user_reward[recommended_item_idx])
    return degree_of_envy


def degrees_of_envy(rewards_mat: np.ndarray, recommendations_mat: np.ndarray) -> np.ndarray:
    # get items never recommended (so never envied by any user)
    artists_not_recommended = recommendations_mat.sum(axis=0) == 0
    degrees_of_envy = []
    for user_index in range(rewards_mat.shape[0]):
        user_envy = degree_of_envy(
            recommendations_mat[user_index], rewards_mat[user_index], artists_not_recommended)
        degrees_of_envy.append(user_envy)
    return np.array(degrees_of_envy)


# NEW:
def get_utilities(ground_truth_preferences, recommendations):
    # return a utility matrix where element i, j  represents
    # the utility of user i with the policies j
    # utility is calculated as the 
    utilities = ground_truth_preferences @ recommendations.T
    return utilities
    
# [u^m(pi^m), u^m(pi^n1), u^m(pi^n2), u^m(pi^n3), u^m(pi^n4), u^m(pi^n5)]


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

# row of utility matrix u^m [m: 0.1, n1: 0.2, n2: 0.3, n4: 0.4, n5: 0.5]