import argparse
import dataclasses
import logging
from pprint import pprint

import config
import constants
import recommender_models as recsys
import sources_of_envy
import utils

EXPERIMENT_NAMES = ["envy-misspecification", "bandit-synthetic"]
EXPERIMENT_FNS = dict(
    zip(EXPERIMENT_NAMES, [sources_of_envy.envy_from_misspecification, lambda _: None])
)


def parse_option() -> config.Configuration:
    parser = argparse.ArgumentParser(
        "OCEF by FACT-AI Group 20",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--log_level",
        type=int,
        default=logging.INFO,
        help=f"Console logging level, set to {logging.ERROR} to show only critical logs",
    )
    parser.add_argument(
        "--assets_root_dir",
        default=config.ROOT_DIR,
        help="Top-level folder where all lastfm and movielens files are stored.",
    )
    parser.add_argument(
        "--lastfm_recommender_dir",
        type=str,
        help="Folder where existing Last.fm-2k pretrained recommender models are stored.",
    )
    parser.add_argument(
        "--movielens_recommender_dir",
        type=str,
        help="Folder where existing MovieLens-1M pretrained recommender models are stored.",
    )
    parser.add_argument(
        "--lastfm_ground_truth_file",
        nargs="?",
        const="",
        help="Path to pretrained ground truth preferences for Last.Fm-2k.\nNOTE It will be loaded by the class specified in `lastfm_ground_truth_model`.",
    )
    parser.add_argument(
        "--movielens_ground_truth_file",
        nargs="?",
        const="",
        help="Path to pretrained ground truth preferences for MovieLens-1M.\nNOTE It will be loaded by the class specified in `movielens_ground_truth_model`.",
    )
    parser.add_argument(
        "--lastfm_ground_truth_model",
        nargs="?",
        default="LMF",
        choices=recsys.MODEL_NAMES,
        help="Model class to build ground truth preferences for Last.Fm-2k.",
    )
    parser.add_argument(
        "--lastfm_recommender_model",
        nargs="?",
        default="SVDS",
        choices=recsys.MODEL_NAMES,
        help="Model class to build estimated preferences for Last.Fm-2k.",
    )
    parser.add_argument(
        "--movielens_ground_truth_model",
        nargs="?",
        default="ALS",
        choices=recsys.MODEL_NAMES,
        help="Model class to build ground truth preferences for MovieLens-1M.",
    )
    parser.add_argument(
        "--movielens_recommender_model",
        nargs="?",
        default="SVDS",
        choices=recsys.MODEL_NAMES,
        help="Model class to build estimated preferences for MovieLens-1M.",
    )
    parser.add_argument(
        "--recommender_evaluation_metric",
        default="ndcg",
        choices=["ndcg", "map", "auc", "precision"],
        help="Evaluation metric for recommenders (validation).",
    )
    parser.add_argument(
        "--evaluation_k",
        type=int,
        default=40,
        help="ranking metric @k, used with `recommender_evaluation_metric`.",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Whether to run grid searches in parallel, useful with FSVD.",
    )

    # TODO various epsilon, temperature etc
    # parser.add_argument(
    #     "--plots_dir",
    #     type=str,
    #     default=config.PLOTS_DIR,
    #     help="Folder where the experiments plots are saved",
    # )
    parser.add_argument(
        "--datasets",
        default=["movielens", "lastfm"],
        nargs="+",
        choices=["movielens", "lastfm"],
        help="Datasets to use for envy-misspecification experiments, can be either one choice or both.",
    )
    parser.add_argument(
        "--model_base_name",
        type=str,
        default="model",
        help="Base file name used to save the ground truth models, recommenders and hyperparameters.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=constants.SEED,
        help="Starting seed used throughout the experiments",
    )
    parser.add_argument(
        "--experiment",
        default="envy-misspecification",
        nargs="?",
        const="envy-misspecification",
        choices=EXPERIMENT_NAMES,
        help="Name of the experiment to reproduce from the original paper.",
    )

    args = parser.parse_args()
    kwargs = vars(args)
    log_level = kwargs.pop("log_level")
    return config.Configuration(**kwargs), log_level


def main(conf: config.Configuration):
    pprint(dataclasses.asdict(conf))
    res = EXPERIMENT_FNS[conf.experiment](conf)
    pprint(res)
    return res


if __name__ == "__main__":
    conf, log_level = parse_option()
    utils.setup_root_logging(level=log_level)
    main(conf)
