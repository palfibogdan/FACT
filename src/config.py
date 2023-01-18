from pathlib import Path

ROOT_DIR = Path("../")

CODE_DIR = ROOT_DIR.parent

DATA_DIR = ROOT_DIR / "data"
LASTFM_DIR = DATA_DIR / "Lastfm"
MOVIELENS_DIR = DATA_DIR / "MovieLens"

MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)
