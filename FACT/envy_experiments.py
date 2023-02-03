import dataclasses
from pprint import pprint

import config
import utils
from sources_of_envy import do_envy_from_misspecification

# very first config with k=10, all als and movielens-1m
initial_all_als_conf = config.Configuration(
    envy_experiment_name="envy_all_als_initial",
    lastfm_ground_truth_file=config.ROOT_DIR
    / "lastfm"
    / "all_als_models"
    / "ground_truth.npz",
    lastfm_recommender_dir=config.ROOT_DIR / "lastfm" / "all_als_models",
    movielens_ground_truth_file=config.ROOT_DIR
    / "movielens-1m"
    / "all_als_models"
    / "ground_truth.npz",
    movielens_recommender_dir=config.ROOT_DIR / "movielens-1m" / "all_als_models",
    lastfm_ground_truth_model="ALS",
    movielens_ground_truth_model="ALS",
    lastfm_recommender_model="ALS",
    movielens_recommender_model="ALS",
)


# suggested by author: k=40, models as written in email, movielens-1m
# NOTE using scipy.svds instead of funk_svd.SVD since very slow and ndcg instead
# of dcg
paper_conf = config.Configuration(
    envy_experiment_name="paper_511_config",
    lastfm_ground_truth_file=config.ROOT_DIR
    / "lastfm"
    / "paper_models"
    / "ground_truth.npz",
    lastfm_recommender_dir=config.ROOT_DIR / "lastfm" / "paper_models",
    movielens_ground_truth_file=config.ROOT_DIR
    / "movielens-1m"
    / "paper_models"
    / "ground_truth.npz",
    movielens_recommender_dir=config.ROOT_DIR / "movielens-1m" / "paper_models",
)


# movielens-25m with half ratings, apply log on movielens too, ndcg@40, all als
best_conf_all_als = config.Configuration(
    envy_experiment_name="ml25_log_all_als",
    lastfm_ground_truth_file=config.ROOT_DIR
    / "lastfm"
    / "all_als_models"
    / "ground_truth.npz",
    lastfm_recommender_dir=config.ROOT_DIR / "lastfm" / "all_als_models",
    movielens_ground_truth_file=config.ROOT_DIR
    / "movielens-25m"
    / "all_als_log_models"
    / "ground_truth.npz",
    movielens_recommender_dir=config.ROOT_DIR / "movielens-25m" / "all_als_log_models",
    movielens_version="25m",
    movielens_do_log=True,
    lastfm_ground_truth_model="ALS",
    movielens_ground_truth_model="ALS",
    lastfm_recommender_model="ALS",
    movielens_recommender_model="ALS",
)


# movielens-25m with half ratings, apply log on movielens too, ndcg@40, models
# from email
best_conf_paper_models = config.Configuration(
    envy_experiment_name="ml25_log_paper_models",
    # don't regenerate lastfm
    lastfm_ground_truth_file=config.ROOT_DIR
    / "lastfm"
    / "paper_models"
    / "ground_truth.npz",
    lastfm_recommender_dir=config.ROOT_DIR / "lastfm" / "paper_models",
    movielens_ground_truth_file=config.ROOT_DIR
    / "movielens-25m"
    / "log_paper_models"
    / "ground_truth.npz",
    movielens_recommender_dir=config.ROOT_DIR / "movielens-25m" / "log_paper_models",
    movielens_version="25m",
    movielens_do_log=True,
)


# movielens-1m, apply log on movielens too, ndcg@40, models from email
best_conf_paper_models_1m = config.Configuration(
    envy_experiment_name="ml1m_log_paper_models",
    # don't regenerate for lastfm by giving same path
    lastfm_ground_truth_file=config.ROOT_DIR
    / "lastfm"
    / "paper_models"
    / "ground_truth.npz",
    lastfm_recommender_dir=config.ROOT_DIR / "lastfm" / "paper_models",
    movielens_ground_truth_file=config.ROOT_DIR
    / "movielens-1m"
    / "log_paper_models"
    / "ground_truth.npz",
    movielens_recommender_dir=config.ROOT_DIR / "movielens-1m" / "log_paper_models",
    movielens_do_log=True,
)

# movielens-1m, apply log on movielens too, ndcg@40, all ALS models
best_conf_1m_all_als = config.Configuration(
    envy_experiment_name="ml1m_log_all_als",
    # don't regenerate for lastfm by giving same path
    lastfm_ground_truth_file=config.ROOT_DIR
    / "lastfm"
    / "all_als_models"
    / "ground_truth.npz",
    lastfm_recommender_dir=config.ROOT_DIR / "lastfm" / "all_als_models",
    movielens_ground_truth_file=config.ROOT_DIR
    / "movielens-1m"
    / "all_als_log_models"
    / "ground_truth.npz",
    movielens_recommender_dir=config.ROOT_DIR / "movielens-1m" / "all_als_log_models",
    movielens_do_log=True,
    lastfm_ground_truth_model="ALS",
    movielens_ground_truth_model="ALS",
    lastfm_recommender_model="ALS",
    movielens_recommender_model="ALS",
)


all_confs = [
    paper_conf,
    initial_all_als_conf,
    best_conf_all_als,
    best_conf_paper_models,
    best_conf_1m_all_als,
]

if __name__ == "__main__":
    utils.setup_root_logging()

    for conf in all_confs:
        pprint(dataclasses.asdict(conf))
        do_envy_from_misspecification(conf)
