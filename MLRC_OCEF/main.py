import argparse
import dataclasses
import logging
from pprint import pprint

import config
import constants
import ocef
import recommender_models as recsys
import sources_of_envy
import utils

EXPERIMENT_NAMES = ["envy-misspecification", "bandit-synthetic"]
EXPERIMENT_FNS = dict(
    zip(EXPERIMENT_NAMES, [sources_of_envy.do_envy_from_misspecification, ocef.main])
)


parsers_config = {
    "log_level": {
        "type": int,
        "default": logging.INFO,
        "help": f"Console logging level, set to {logging.ERROR} to show only critical logs",
    },
    "lastfm_recommender_dir": {
        "type": str,
        "help": "folder with pretrained last.fm-2k recommender models; if empty, new models are stored here. If not given, defaults to a unique folder with naming scheme `dataset``version`/unique slug.",
    },
    "movielens_recommender_dir": {
        "type": str,
        "help": "Folder with pretrained MovieLens-`version` recommender models; if empty, new models are stored here. If not given, defaults to a unique folder with naming scheme `dataset``version`/unique slug.",
    },
    "lastfm_ground_truth_file": {
        "nargs": "?",
        "const": "",
        "help": "Pretrained ground truth preferences file for Last.Fm-2k. If it exists, it is loaded by the given `lastfm_ground_truth_model` class, otherwise ground truth preferences are saved here. If not given, defaults to a unique file with naming scheme `dataset``version`/unique dirname/ground_truth.npz.",
    },
    "movielens_ground_truth_file": {
        "nargs": "?",
        "const": "",
        "help": "Pretrained ground truth preferences file for MovieLens-`version`. If it exists, it is loaded by the given `movielens_ground_truth_model` class, otherwise ground truth preferences are saved here. If not given, defaults to a unique file with naming scheme `dataset``version`/unique slug/ground_truth.npz.",
    },
    "lastfm_ground_truth_model": {
        "nargs": "?",
        "default": "LMF",
        "choices": recsys.MODEL_NAMES,
        "help": "Model class to build ground truth preferences for Last.Fm-2k.",
    },
    "lastfm_recommender_model": {
        "nargs": "?",
        "default": "SVDS",
        "choices": recsys.MODEL_NAMES,
        "help": "Model class to build estimated preferences for Last.Fm-2k.",
    },
    "movielens_ground_truth_model": {
        "nargs": "?",
        "default": "ALS",
        "choices": recsys.MODEL_NAMES,
        "help": "Model class to build ground truth preferences for MovieLens-`version`.",
    },
    "movielens_recommender_model": {
        "nargs": "?",
        "default": "SVDS",
        "choices": recsys.MODEL_NAMES,
        "help": "Model class to build estimated preferences for MovieLens-`version`.",
    },
    "recommender_evaluation_metric": {
        "default": "ndcg",
        "choices": ["ndcg", "map", "auc", "precision"],
        "help": "Evaluation metric for recommenders (validation.",
    },
    "evaluation_k": {
        "type": int,
        "default": 40,
        "help": "ranking metric @k, used with `recommender_evaluation_metric`.",
    },
    "parallel": {
        "action": "store_true",
        "help": "Whether to run grid searches in parallel.",
    },
    "datasets": {
        "default": ["movielens", "lastfm"],
        "nargs": "+",
        "choices": ["movielens", "lastfm"],
        "help": "Datasets to use for envy-misspecification experiments, can be either one choice or both.",
    },
    "movielens_version": {
        "default": "1m",
        "choices": ["1m", "25m"],
        "help": "MovieLens dataset version. 1M version contains integer ratings only, 25M also has half-star ratings.",
    },
    "seed": {
        "type": int,
        "default": constants.SEED,
        "help": "Starting seed used throughout the experiments",
    },
    "experiment": {
        "default": "envy-misspecification",
        "nargs": "?",
        "const": "envy-misspecification",
        "choices": EXPERIMENT_NAMES,
        "help": "Name of the experiment to reproduce from the original paper.",
    },
    "envy_experiment_name": {
        "type": str,
        "help": f"Base filename for envy-misspecification plot and pickled metrics. Defaults to {config.ENVY_DIR}/unique slug{{.png|.pkl}}.",
    },
    "movielens_do_log": {
        "action": "store_true",
        "help": "Whether to apply log-transform to MovieLens datasets.",
    },
}


def parse_option() -> config.Configuration:
    parser = argparse.ArgumentParser(
        "OCEF by FACT-AI Group 20",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    for flag, flag_conf in parsers_config.items():
        parser.add_argument(f"--{flag}", **flag_conf)
    kwargs = vars(parser.parse_args())
    log_level = kwargs.pop("log_level")
    return config.Configuration(**kwargs), log_level


def main(conf: config.Configuration):
    pprint(dataclasses.asdict(conf))
    exit(0)
    return EXPERIMENT_FNS[conf.experiment](conf)


if __name__ == "__main__":
    conf, log_level = parse_option()
    utils.setup_root_logging(level=log_level)
    main(conf)
