import functools as ft
import io
import logging
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import config

logger = logging.getLogger(__name__)


def download_dataset(dataset_url: str, dataset_dir: str):
    if os.path.exists(dataset_dir) and len(os.listdir(dataset_dir)) > 1:
        logger.info("%s already exists, abort download", dataset_dir)
        return
    logger.info("Downloading zipped dataset from %s", dataset_url)
    r = requests.get(dataset_url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(dataset_dir)
    logger.info("Extracted files to %s", dataset_dir)


# 1. transform into full user-item preference matrix
# 2. keep only top-2500 most listened artists
# 3. pre-process raw counts with log transforms
# NOTE
# The completed preference matrix has columns indexed from 0 to 2500, which
# should be mapped to top_k_artists.sort_values() if the actual item id is
# wanted, which should not be the case for our purposes
@ft.lru_cache()
def get_lastfm(
    topk_artists=2500, dataset_dir: Path = config.LASTFM_DATA_DIR, **_
) -> pd.DataFrame:
    download_dataset(config.LASTFM_URL, dataset_dir)
    user_item_df = pd.read_csv(dataset_dir / "user_artists.dat", sep="\t")
    user_item_df = user_item_df.rename(
        columns={"userID": "user", "artistID": "item", "weight": "score"}
    )
    # the table is in long format originally
    user_item_df = user_item_df.pivot(
        index="user", columns="item", values="score"
    ).fillna(0)
    # keep only top k artists in each row (user)
    top_k_artists = (
        user_item_df.sum(axis=0).sort_values(ascending=False).index[:topk_artists]
    )
    # drops only columns of user_item_df which do not appear in top_k_artists
    user_item_df = user_item_df[user_item_df.columns.intersection(top_k_artists)]
    assert (user_item_df.columns == top_k_artists.sort_values()).all()
    user_item_df = np.log(user_item_df, where=user_item_df > 0)  # log-transform
    return user_item_df


@ft.lru_cache()
def get_movielens(
    topk_users=2000,
    topk_movies=2500,
    dataset_dir: Path = config.MOVIELENS_DATA_DIR,
    **_
) -> pd.DataFrame:
    download_dataset(config.MOVIELENS_URL, dataset_dir)
    # move files from dataset_dir/ml-1M folder to dataset_dir and delete ml-1M
    ml_1m = dataset_dir / "ml-1m"
    if ml_1m.exists():
        for f in ml_1m.iterdir():
            f.rename(dataset_dir / f.name)
        ml_1m.rmdir()
    # headers from data/MovieLens/README
    column_names = ["user", "item", "score", "timestamp"]
    user_item_df = pd.read_csv(
        dataset_dir / "ratings.dat",
        names=column_names,
        sep="::",
        engine="python",
    )
    # get rid of the timestamp column
    user_item_df = user_item_df.drop(columns="timestamp")
    # spread into user X item rating matrix
    user_item_df = user_item_df.pivot(
        index="user", columns="item", values="score"
    ).fillna(0)
    # ratings are on a 1-5 scale, sum non-zero entries to get user/items with
    # most ratings
    bool_df = user_item_df.astype(bool)
    # filter top 2000 most rated users
    topk_users = bool_df.sum(axis=1).sort_values(ascending=False).index[:topk_users]
    user_item_df = user_item_df.loc[user_item_df.index.intersection(topk_users)]
    # filter top 2500 most rated items
    topk_items = bool_df.sum(axis=0).sort_values(ascending=False).index[:topk_movies]
    user_item_df = user_item_df[user_item_df.columns.intersection(topk_items)]
    # remap ratings {3, 4, 5} to a range of 5, and set ratings < 3 to 0
    # ratings_remap = dict(zip(range(1, 6), np.linspace(3, 5, 5)))
    # user_item_df = user_item_df.applymap(lambda rating: ratings_remap.get(rating, 0))
    user_item_df[user_item_df < 3] = 0
    return user_item_df


DATASETS_RETRIEVE = {"lastfm": get_lastfm, "movielens": get_movielens}


def get_dataset(dataset_name: str, **kwargs) -> pd.DataFrame:
    return DATASETS_RETRIEVE[dataset_name](**kwargs)
