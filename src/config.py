from pathlib import Path

ROOT_DIR = Path("../")

CODE_DIR = ROOT_DIR.parent

DATA_DIR = ROOT_DIR / "data"
LASTFM_DIR = DATA_DIR / "Lastfm"
MOVIELENS_DIR = DATA_DIR / "MovieLens" / "ml-1m"

LASTFM_URL = "https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip"
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"

MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

WANDB_DIR = ROOT_DIR

# Save best models for all factor hyperparmeter values
MOVIELENS_RECOMMENDER_DIR = MODELS_DIR / "movielens_recommender"
MOVIELENS_RECOMMENDER_DIR.mkdir(exist_ok=True)
LASTFM_RECOMMENDER_DIR = MODELS_DIR / "lastfm_recommender"
LASTFM_RECOMMENDER_DIR.mkdir(exist_ok=True)