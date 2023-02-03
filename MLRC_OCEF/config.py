from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from coolname import generate_slug

import constants
import recommender_models as recsys
import utils

ROOT_DIR = Path("../")

# NOTE fixed attributes not part of configuration
LASTFM_DATA_DIR = ROOT_DIR / "lastfm"
MOVIELENS_1M_DATA_DIR = ROOT_DIR / "movielens-1m"
MOVIELENS_25M_DATA_DIR = ROOT_DIR / "movielens-25m"
DATA_DIRS = {
    "lastfm": LASTFM_DATA_DIR,
    "movielens1m": MOVIELENS_1M_DATA_DIR,
    "movielens25m": MOVIELENS_25M_DATA_DIR,
}
OCEF_DIR = ROOT_DIR / "results_ocef"
ENVY_DIR = ROOT_DIR / "results_envy"

LASTFM_URL = "https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip"
MOVIELENS_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
MOVIELENS_25M_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"

RECOMMENDER_NAME = "model"
GROUND_TRUTH_NAME = "ground_truth"


# NOTE some defaults are set for ease of instantiation during development
@dataclass
class Configuration:
    datasets: List[str] = field(default_factory=lambda: ["movielens", "lastfm"])
    experiment: str = None
    seed: int = constants.SEED
    random_state: utils.SeedSequence = field(init=False)
    parallel: bool = False
    movielens_version: str = "1m"

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
    envy_experiment_name: str = None
    movielens_do_log: bool = False

    ocef_dir: Path = OCEF_DIR
    lastfm_recommender_dir: Path = None
    movielens_recommender_dir: Path = None
    lastfm_ground_truth_file: Path = None
    movielens_ground_truth_file: Path = None

    # only for convenience
    ground_truth_files: Dict[str, Path] = field(
        init=False, default_factory=dict, repr=False
    )
    recommender_dirs: Dict[str, Path] = field(
        init=False, default_factory=dict, repr=False
    )
    ground_truth_models: Dict[str, recsys.RecommenderType] = field(
        init=False, default_factory=dict, repr=False
    )
    recommender_models: Dict[str, recsys.RecommenderType] = field(
        init=False,
        default_factory=dict,
        repr=False,
    )

    def __post_init__(self):
        self.random_state = utils.SeedSequence(start=self.seed)
        if not isinstance(self.ocef_dir, Path):
            self.ocef_dir = Path(self.ocef_dir)
        if self.envy_experiment_name is None:
            self.envy_experiment_name = generate_slug()

        for dataset in self.datasets:
            # filenames and folders
            dataset_version = getattr(self, f"{dataset}_version", "")
            self.set_or_create_model_files(dataset, dataset_version)

            # model classes
            for model_scope in ["ground_truth", "recommender"]:
                attr = f"{dataset}_{model_scope}_model"
                model_scope_class = recsys.MODELS_MAP[getattr(self, attr)]
                setattr(self, attr, model_scope_class)
                model_scope_dict = getattr(self, f"{model_scope}_models")
                model_scope_dict[dataset] = model_scope_class

    def set_or_create_model_files(self, dataset_name: str, version: str = ""):
        versioned_dataset_name = f"{dataset_name}{version}"
        unique_dir_id = generate_slug()
        unique_dataset_dir = Path(DATA_DIRS[versioned_dataset_name] / unique_dir_id)

        # check ground truth file
        dataset_ground_truth_attr = f"{dataset_name}_ground_truth_file"
        dataset_ground_truth_file = getattr(self, dataset_ground_truth_attr)
        if dataset_ground_truth_file is None:
            # no filename given; create one in unique folder for this dataset
            utils.makedir(unique_dataset_dir)
            dataset_ground_truth_file = unique_dataset_dir / f"{GROUND_TRUTH_NAME}.npz"
        else:
            # create folder for given ground truth file if inexistent
            dataset_ground_truth_file = Path(dataset_ground_truth_file)
            utils.makedir(dataset_ground_truth_file.parent)
        # save in configuration
        setattr(self, dataset_ground_truth_attr, dataset_ground_truth_file)
        self.ground_truth_files[dataset_name] = dataset_ground_truth_file

        # check directory for recommender models
        dataset_recommender_dir_attr = f"{dataset_name}_recommender_dir"
        dataset_recommender_dir = getattr(self, dataset_recommender_dir_attr)
        if dataset_recommender_dir is None:
            # when e.g. movielens_recommender_dir is not given, defaults to
            # ../movielens-{version}/unique_dir_id
            dataset_recommender_dir = unique_dataset_dir
        else:
            dataset_recommender_dir = Path(dataset_recommender_dir)
        utils.makedir(dataset_recommender_dir)
        setattr(self, dataset_recommender_dir_attr, dataset_recommender_dir)
        self.recommender_dirs[dataset_name] = dataset_recommender_dir
