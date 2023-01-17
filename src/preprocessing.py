import numpy as np
import pandas as pd

import config


# 1. keep only top-2500 most listened artists
# 2. pre-process raw counts with log transforms
# 3. transform into full user-item preference matrix
def get_lastfm(k=2500) -> pd.DataFrame:
    # the table is in long format originally
    user_artist_df = pd.read_csv(config.LASTFM_DIR / "user_artists.dat", sep="\t")
    user_artist_df = user_artist_df.rename(
        columns={"userID": "user", "artistID": "item"}
    )
    # group by artist and sum on 'weight' to get each artist's total listening
    # counts, sort in descending order and only keep the top-k listened artists
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
    # spread into wide format, dimensions: #users X #artists. Fill the missing
    # preferences with 0
    return user_artist_df.pivot(index="user", columns="item", values="weight").fillna(0)
