from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import constants
import recommender_models as recsys
import utils

ROOT_DIR = Path("../")

PLOTS_DIR = ROOT_DIR / "plots"

LASTFM_DIR = ROOT_DIR / "lastfm"
MOVIELENS_DIR = ROOT_DIR / "movielens"

LASTFM_DATA_DIR = LASTFM_DIR / "data"
MOVIELENS_DATA_DIR = MOVIELENS_DIR / "data"

LASTFM_RECOMMENDER_DIR = LASTFM_DIR / "models"
MOVIELENS_RECOMMENDER_DIR = MOVIELENS_DIR / "models"


LASTFM_URL = "https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip"
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


OCEF_DIR = ROOT_DIR / "results_ocef"


@dataclass
class Configuration:
    assets_root_dir: Path = field(default_factory=lambda: ROOT_DIR)
    datasets: List[str] = field(default_factory=lambda: ["movielens", "lastfm"])
    experiment: str = None
    model_base_name: str = "model"
    seed: int = constants.SEED
    random_state: utils.SeedSequence = field(init=False)
    parallel: bool = False

    lastfm_ground_truth_model: recsys.RecommenderType = "LMF"
    lastfm_recommender_model: recsys.RecommenderType = "SVDS"
    movielens_ground_truth_model: recsys.RecommenderType = "ALS"
    movielens_recommender_model: recsys.RecommenderType = "SVDS"

    epsilon: float = 0.05
    temperature: float = 1 / 5.0
    train_size: float = 0.7
    validation_size: float = 0.1
    lastfm_topk_artists: int = 2500
    movielens_topk_users: int = 2000
    movielens_topk_movies: int = 2500
    recommender_evaluation_metric: str = "ndcg"
    evaluation_k: int = 40

    ocef_dir: Path = None
    lastfm_recommender_dir: Path = None
    movielens_recommender_dir: Path = None
    lastfm_ground_truth_file: Path = None
    movielens_ground_truth_file: Path = None

    # only for convenience
    ground_truth_files: Dict[str, Path] = field(init=False, default_factory=dict)
    recommender_dirs: Dict[str, Path] = field(init=False, default_factory=dict)
    ground_truth_models: Dict[str, recsys.RecommenderType] = field(
        init=False, default_factory=dict
    )
    recommender_models: Dict[str, recsys.RecommenderType] = field(
        init=False, default_factory=dict
    )

    def __post_init__(self):
        self.random_state = utils.SeedSequence(start=self.seed)
        self.assets_root_dir = Path(self.assets_root_dir)
        self.ocef_dir = self.assets_root_dir / "results_ocef"

        def make_folders(ds):
            for dir_ in ["data", "models"]:
                p = self.assets_root_dir / ds / dir_
                p.mkdir(parents=True, exist_ok=True)
            return p

        for dataset in self.datasets:
            # filenames and folders
            rec, gt = f"{dataset}_recommender_dir", f"{dataset}_ground_truth_file"
            rec_path, gt_path = getattr(self, rec), getattr(self, gt)

            if gt_path is None:
                p = make_folders(dataset)
                gt_path = p / f"{self.model_base_name}_ground_truth.npz"
            else:
                assert Path(
                    gt_path
                ).is_file(), f"{gt_path} is not a valid ground truth file"

            if rec_path is None:
                rec_path = make_folders(dataset)
            else:
                rec_p = Path(rec_path)
                assert rec_p.is_dir() and list(
                    rec_p.iterdir()
                ), f"{rec_path} is not a folder populated with recommender models"

            for attr_name, path, dict_ in zip(
                [gt, rec],
                [gt_path, rec_path],
                [self.ground_truth_files, self.recommender_dirs],
            ):
                p = Path(path)
                setattr(self, attr_name, p)
                dict_[dataset] = p

            # model classes
            for dest in ["ground_truth", "recommender"]:
                attr = f"{dataset}_{dest}_model"
                val = recsys.MODELS_MAP[getattr(self, attr)]
                setattr(self, attr, val)
                d = getattr(self, f"{dest}_models")
                d[dataset] = val
