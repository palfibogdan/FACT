import numpy as np
import scipy


# NOTE works with both the full user-item preferences matrix and a single user's
# preferences, add a batch dimension in the latter case
def user_policy_recommendation(
    user_preferences: np.ndarray, temperature: float, rng: np.random.Generator
) -> np.ndarray:
    # apply softmax with temperature to turn preferences into probabilities
    recommendation_probs = scipy.special.softmax(user_preferences / temperature, axis=1)
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
    # recommended_items_indexes = np.argmax(recommendation_probs, axis=1, keepdims=True)
    # generate binary recommendations matrix
    recommended_items = np.zeros_like(user_preferences)
    np.put_along_axis(recommended_items, recommended_items_indexes[:, None], 1, axis=1)
    assert (np.where(recommended_items > 0)[1] == recommended_items_indexes).all()
    return recommended_items


# generate binary rewards using bernoulli distribution
# we will use p = corresponding ground truth value as per paper instructions
def generate_true_recommendation(
    true_preferences: np.ndarray, temperature: float, rng: np.random.Generator
) -> np.ndarray:
    # turn the true preferences into probabilities
    true_probs = scipy.special.softmax(true_preferences / temperature, axis=1)
    # binomial with n=1 == bernoulli distribution
    return rng.binomial(1, true_probs)


# def get_reward(
#     estimated_preferences: np.ndarray,
#     true_preferences: np.ndarray,
#     temperature: float,
#     rng: np.random.Generator,
# ) -> float:
#     recommendations = user_policy_recommendation(
#         estimated_preferences, temperature, rng
#     )
#     true_recommendations = generate_true_recommendation(
#         true_preferences, temperature, rng
#     )
#     ...
