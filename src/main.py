import argparse
import logging
from pathlib import Path
from pprint import pprint

import numpy as np

import config
import constants
import sources_of_envy
import utils

EXPERIMENTS_FNS = {"envy-mispecification": sources_of_envy.do_envy_from_mispecification}


def parse_option() -> argparse.ArgumentParser:
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
        "--lastfm_dir",
        type=str,
        default=config.LASTFM_DIR,
        help="Root folder where the Last.fm datasets, models and plots are saved",
    )
    parser.add_argument(
        "--movielens_dir",
        type=str,
        default=config.MOVIELENS_DIR,
        help="Root folder where the MovieLens-1M datasets, models and plots are saved",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=constants.SEED,
        help="Starting seed used throughout the experiments",
    )
    parser.add_argument(
        "--experiment",
        default="envy-mispecification",
        const="envy-mispecification",
        nargs="?",
        choices=["envy-mispecification", "bandit-synthetic"],
        help="Name of the experiment to reproduce from the original paper",
    )

    args = parser.parse_args()

    # transform string paths into pathlib.Path objects and create folders
    for dataset in ["lastfm", "movielens"]:
        arg_name = f"{dataset}_dir"
        for destination in ["data", "models", "plots"]:
            folder = Path(getattr(args, arg_name)) / destination
            folder.mkdir(parents=True, exist_ok=True)
            setattr(args, f"{dataset}_{destination}_dir", folder)

    return args


def main(args: argparse.Namespace):
    kvargs = vars(args)
    pprint(kvargs)
    seed_seq = utils.SequenceGenerator(args.seed)
    rng = np.random.default_rng(args.seed)
    # return EXPERIMENTS_FNS[args.experiment](**kvargs)
    import recommender

    recommender.generate_recommenders(
        args.lastfm_models_dir / "model",
        seed_seq,
        rng,
        dataset_name="lastfm",
        **kvargs,
    )
    recommender.generate_recommenders(
        args.movielens_models_dir / "model",
        seed_seq,
        rng,
        dataset_name="movielens",
        **kvargs,
    )


if __name__ == "__main__":
    args = parse_option()

    utils.setup_root_logging(level=args.log_level)

    main(args)
