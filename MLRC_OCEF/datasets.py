import logging
from pathlib import Path

import numpy as np
import pandas as pd

import config
import utils

logger = logging.getLogger(__name__)


def download_dataset(dataset_url: str, dataset_dir: Path, ratings_fname: str):
    ratings_file = dataset_dir / ratings_fname
    if dataset_dir.is_dir() and ratings_file.is_file():
        logger.info("%s already exists, abort download", ratings_file)
    else:
        utils.makedir(dataset_dir)
        logger.info("Downloading ZIP dataset from %s", dataset_url)
        archive_target = dataset_dir / Path(dataset_url).name
        utils.tqdm_download(dataset_url, str(archive_target))
        utils.tqdm_extract(archive_target, dataset_dir)
        logger.info("Extracted files to %s", dataset_dir)


# 1. transform into full user-item preference matrix
# 2. keep only top-2500 most listened artists
# 3. pre-process raw counts with log transforms
def get_lastfm(conf: config.Configuration) -> pd.DataFrame:
    dataset_dir = config.LASTFM_DATA_DIR
    ratings_fname = "user_artists.dat"
    download_dataset(config.LASTFM_URL, dataset_dir, ratings_fname)
    user_item_df = pd.read_csv(dataset_dir / ratings_fname, sep="\t").rename(
        columns={"userID": "user", "artistID": "item", "weight": "score"}
    )
    # the table is in long format originally
    user_item_df = user_item_df.pivot(
        index="user", columns="item", values="score"
    ).fillna(0.0)
    # keep only top k artists in each row (user)
    top_k_artists = (
        user_item_df.sum(axis=0)
        .sort_values(ascending=False)
        .index[: conf.lastfm_topk_artists]
    )
    # drops only columns of user_item_df which do not appear in top_k_artists
    user_item_df = user_item_df[user_item_df.columns.intersection(top_k_artists)]
    assert (user_item_df.columns == top_k_artists.sort_values()).all()
    user_item_df = np.log(user_item_df, where=user_item_df > 0)  # log-transform
    return user_item_df


def remove_movielens_inter_dir(dataset_dir: Path, ml_version: str):
    # move files from dataset_dir/inter_dir folder to dataset_dir and delete
    # inter_dir
    inter_dir = dataset_dir / f"ml-{ml_version}"
    if inter_dir.exists():
        for f in inter_dir.iterdir():
            f.rename(dataset_dir / f.name)
        inter_dir.rmdir()


def read_movielens_1m() -> pd.DataFrame:
    dataset_dir = config.MOVIELENS_1M_DATA_DIR
    ratings_fname = "ratings.dat"
    download_dataset(config.MOVIELENS_1M_URL, dataset_dir, ratings_fname)
    remove_movielens_inter_dir(dataset_dir, "1m")
    # headers from movielens/data/README
    return pd.read_csv(
        dataset_dir / ratings_fname,
        names=["user", "item", "score", "timestamp"],
        sep="::",
        engine="python",
    )


def read_movielens_25m() -> pd.DataFrame:
    dataset_dir = config.MOVIELENS_25M_DATA_DIR
    ratings_fname = "ratings.csv"
    download_dataset(config.MOVIELENS_25M_URL, dataset_dir, ratings_fname)
    remove_movielens_inter_dir(dataset_dir, "25m")
    # headers from movielens/data/README.txt
    return pd.read_csv(dataset_dir / ratings_fname).rename(
        columns={"userId": "user", "movieId": "item", "rating": "score"}
    )


def get_movielens(conf: config.Configuration) -> pd.DataFrame:
    user_item_df = {"1m": read_movielens_1m, "25m": read_movielens_25m}[
        conf.movielens_version
    ]()
    # get rid of the timestamp column
    user_item_df = user_item_df.drop(columns="timestamp")
    filter_top_k = lambda group, select, topk: (
        user_item_df.groupby(group)[select]
        .count()
        .sort_values(ascending=False)
        .index[:topk]
    )
    topk_users = filter_top_k("user", "item", conf.movielens_topk_users)
    topk_movies = filter_top_k("item", "user", conf.movielens_topk_movies)
    # filter top 2500 movies with most ratings
    user_item_df = user_item_df[user_item_df["item"].isin(topk_movies)]
    # filter top 2000 users who gave most ratings
    user_item_df = user_item_df[user_item_df["user"].isin(topk_users)]
    # spread into user X item rating matrix
    user_item_df = user_item_df.pivot(
        index="user", columns="item", values="score"
    ).fillna(0.0)
    user_item_df[user_item_df < 3.0] = 0.0
    if conf.movielens_do_log:
        # log-transform
        user_item_df = np.log(user_item_df, where=user_item_df > 0.0)
        logger.info("Applied log-transform on MovieLens-%s", conf.movielens_version)
    return user_item_df


DATASETS_RETRIEVE_MAP = {"lastfm": get_lastfm, "movielens": get_movielens}


def get_dataset(dataset_name: str, conf: config.Configuration) -> pd.DataFrame:
    return DATASETS_RETRIEVE_MAP[dataset_name](conf)
