from typing import Tuple

import numpy as np


# NOTE works with both the full user-item preferences matrix and a single user's
# preferences, add a batch dimension in the latter case
def recommendation_policy(
    user_preferences_prob: np.ndarray, temperature: float, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    # retrieve the indexes of the most recommended item for each user according
    # to the softmax probabilities
    # NOTE the vectorized version that samples over the whole matrix returns the
    # same sample for each user :/
    recommended_items_idx = np.array(
        [
            rng.choice(user_preferences_prob.shape[1], replace=False, p=user_probs)
            for user_probs in user_preferences_prob
        ]
    )
    # recommended_items_indexes = np.argmax(recommendation_probs, axis=1, keepdims=True)
    return recommended_items_idx, np.take_along_axis(
        user_preferences_prob, recommended_items_idx, 1
    )


def compute_reward(
    recommendations: np.ndarray,
    ground_truth_probs: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    if len(recommendations.shape) < 2:
        recommendations = recommendations[:, None]
    return np.take_along_axis(ground_truth_probs, recommendations, 1)
