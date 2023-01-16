import numpy as np
import pandas as pd

import config


# 1. keep only top-2500 most listened artists
# 2. pre-process raw counts with log transforms
# 3. transform into full user-item preference matrix
def get_lastfm(k=2500) -> pd.DataFrame:
    user_artist_df = pd.read_csv(config.LASTFM_DIR / "user_artists.dat", sep="\t")
    user_artist_df = user_artist_df.rename(
        columns={"userID": "user", "artistID": "item"}
    )
    top_k_artists = np.array(
        user_artist_df.groupby("item")["weight"]
        .sum()
        .sort_values(ascending=False)
        .index
    )[:k]
    user_artist_df = user_artist_df.loc[user_artist_df["item"].isin(top_k_artists)]
    assert set(user_artist_df["item"]) == set(top_k_artists)
    # log-transform
    user_artist_df = user_artist_df.copy()  # avoid SettingWithCopy warning
    user_artist_df.loc[:, "weight"] = np.log(user_artist_df["weight"])
    return user_artist_df.pivot(index="user", columns="item", values="weight").fillna(0)
