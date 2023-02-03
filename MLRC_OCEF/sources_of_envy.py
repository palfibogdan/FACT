import logging
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import config
import recommender
import recommender_models as recsys
import utils

logger = logging.getLogger(__name__)


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

    # dictionary to store the metrics by factors, easy to use with pandas
    metrics_dict = {"mean_envy": {}, "prop_eps_envy": {}}

    for filename in recommender_filenames:
        recommender_model = model_class.load(filename)
        recommender_probs = utils.softmax(
            recommender_model.preferences, temperature=temperature
        )
        utilities = compute_utilities(recommender_probs, ground_truth)
        mean_envy, prop_envy_users = get_envy_metrics(utilities, eps)

        metrics_dict["mean_envy"][recommender_model.factors] = mean_envy
        metrics_dict["prop_eps_envy"][recommender_model.factors] = prop_envy_users

    return metrics_dict


def envy_from_misspecification(
    conf: config.Configuration,
) -> pd.DataFrame:

    metrics = {}

    for dataset in conf.datasets:
        recommender.generate_recommenders(
            conf.ground_truth_files[dataset], dataset, conf.random_state, conf
        )
        models_dir = conf.recommender_dirs[dataset]
        # recommenders_by_factors = models_dir.glob("model_factors*.npz")
        recommenders_by_factors = recommender.list_default_recommender_files(models_dir)
        ground_truth_class = conf.ground_truth_models[dataset]
        ground_truth_model = ground_truth_class.load(conf.ground_truth_files[dataset])
        metrics[dataset] = experiment_5_1_1(
            recommenders_by_factors,
            ground_truth_model.preferences,
            conf.epsilon,
            conf.temperature,
            conf.recommender_models[dataset],
        )

    return pd.concat(
        [
            pd.DataFrame(vals).assign(dataset=dataset)
            for dataset, vals in metrics.items()
        ]
    )


def persist_results(df: pd.DataFrame, conf: config.Configuration):
    font = {"size": 10}
    plt.rc("font", **font)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 5))

    sns.lineplot(data=df, x=df.index, y="mean_envy", hue="dataset", ax=ax0)
    ax0.set_xlim(-5, 120)
    ax0.set_xlabel("Factors")
    ax0.set_ylabel("Average envy")
    ax0.grid()

    sns.lineplot(data=df, x=df.index, y="prop_eps_envy", hue="dataset", ax=ax1)
    ax1.set_xlim(-5, 120)
    ax1.set_xlabel("Factors")
    ax1.set_ylabel("Prop. of \u03B5-envy users")
    ax1.grid()

    lastfm_gt = recsys.MODELS_MAP_REVERSE[conf.lastfm_ground_truth_model]
    lastfm_recommender = recsys.MODELS_MAP_REVERSE[conf.lastfm_recommender_model]
    movielens_gt = recsys.MODELS_MAP_REVERSE[conf.movielens_ground_truth_model]
    movielens_recommender = recsys.MODELS_MAP_REVERSE[conf.movielens_recommender_model]
    handles, _ = ax0.get_legend_handles_labels()
    labels = [
        f"Last.Fm-2k, {lastfm_gt}/{lastfm_recommender}",
        f"MovieLens-{conf.movielens_version}, {movielens_gt}/{movielens_recommender}",
    ]
    ax0.legend(handles, labels)
    ax1.legend(handles, labels)

    fig.suptitle(
        "Envy from model misspecification (ground truth model/recommender model)"
    )
    plt.tight_layout()

    utils.makedir(config.ENVY_DIR)
    plt.savefig(config.ENVY_DIR / f"{conf.envy_experiment_name}.png")
    df.to_csv(
        config.ENVY_DIR / f"{conf.envy_experiment_name}.csv", index_label="factors"
    )
    logger.info(
        "Saved plot and CSV dataframe to %s",
        config.ENVY_DIR / conf.envy_experiment_name,
    )

    plt.show(block=False)


def do_envy_from_misspecification(conf: config.Configuration):
    return persist_results(envy_from_misspecification(conf), conf)
