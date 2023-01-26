from pathlib import Path

ROOT_DIR = Path("../")

assets_dir = ("data", "model")

CODE_DIR = ROOT_DIR.parent

PLOTS_DIR = ROOT_DIR / "plots"

LASTFM_DIR = ROOT_DIR / "lastfm"
MOVIELENS_DIR = ROOT_DIR / "movielens"

LASTFM_DATA_DIR = LASTFM_DIR / assets_dir[0]
MOVIELENS_DATA_DIR = MOVIELENS_DIR / assets_dir[0]

# Save best models for all factor hyperparmeter values
LASTFM_RECOMMENDER_DIR = LASTFM_DIR / assets_dir[1]
MOVIELENS_RECOMMENDER_DIR = MOVIELENS_DIR / assets_dir[1]

LASTFM_URL = "https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip"
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


# WANDB_DIR = ROOT_DIR
# DATA_DIR = ROOT_DIR / "data"
# LASTFM_DIR = DATA_DIR / "Lastfm"
# MOVIELENS_DIR = DATA_DIR / "MovieLens" / "ml-1m"
# MODELS_DIR = ROOT_DIR / "models"
