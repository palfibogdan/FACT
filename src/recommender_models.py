import logging
from pathlib import Path
from typing import Dict, Union

import implicit
import numpy as np
import pandas as pd
from funk_svd import SVD
from implicit import evaluation
from implicit.als import AlternatingLeastSquares as ALS_factory
from implicit.cpu.als import AlternatingLeastSquares
from implicit.cpu.lmf import LogisticMatrixFactorization
from implicit.cpu.matrix_factorization_base import MatrixFactorizationBase
from implicit.lmf import LogisticMatrixFactorization as LMF_factory
from scipy import sparse

import utils

# ALS -> ground truth MovieLens
# LMF -> ground truth LastFm
# SVD -> recommmenders both

RecommenderType = Union["SVDS", "FSVD", "ALS", "LMF"]


def check_extension(p: Path, ext: str = ".npz") -> Path:
    if not isinstance(p, Path):
        p = Path(p)
    if p.suffix != ".npz":
        p = p.parent / (p.name + ".npz")
    return p


# NOTE majority of functionality for Implicit models is here to avoid repetition
class Recommender:
    model: Union[AlternatingLeastSquares, LogisticMatrixFactorization, "SVDS"] = None
    factors: int = None
    logger: logging.Logger = None
    preferences: np.ndarray = None
    _model_class: type = None

    def __init__(self, factors: int):
        self.factors = factors
        self.logger = logging.getLogger(f"{__name__}:{type(self).__name__}")

    def train(self, train_mat: Union[np.ndarray, sparse.csr_array], show_progress=True):
        if isinstance(train_mat, np.ndarray):
            train_mat = sparse.csr_array(train_mat)
        self.model.fit(train_mat, show_progress=show_progress)
        self.set_preferences()

    def set_preferences(self):
        user_factors, item_factors = self.model.user_factors, self.model.item_factors
        if isinstance(user_factors, implicit.gpu._cuda.Matrix):
            # model was trained on the GPU
            user_factors, item_factors = (
                user_factors.to_numpy(),
                item_factors.to_numpy(),
            )
        self.preferences = user_factors @ item_factors.T

    def validate(
        self,
        train_mat: Union[np.ndarray, sparse.csr_matrix],
        test_mat: Union[np.ndarray, sparse.csr_matrix],
        k=40,
    ) -> Dict[str, float]:
        if not isinstance(train_mat, sparse.csr_matrix):
            train_mat = sparse.csr_matrix(train_mat)
        if not isinstance(test_mat, sparse.csr_matrix):
            train_mat = sparse.csr_matrix(test_mat)
        return evaluation.ranking_metrics_at_k(self.model, train_mat, test_mat, K=k)

    def save(self, savepath: Path):
        savepath = check_extension(savepath)
        self.model.save(savepath)
        self.logger.info("Saved best model to %s", savepath)

    @classmethod
    def load(cls, filename: Path) -> RecommenderType:
        ret = cls(32)
        ret.model = cls._model_class.load(filename)
        ret.factors = ret.model.factors
        ret.set_preferences()
        return ret

    def __repr__(self) -> str:
        return f"{type(self).__name__}:{self._model_class}"


class ALS(Recommender):
    _model_class = AlternatingLeastSquares

    def __init__(
        self,
        factors: int,
        random_state: int,
        regularization: float = None,
        alpha: float = None,
    ):
        super().__init__(factors)
        args = {"factors": factors, "iterations": 30, "random_state": random_state}
        # use defaults from implicit if regularization and alpha are not passed
        if regularization is not None:
            args["regularization"] = regularization
        if alpha is not None:
            args["alpha"] = alpha
        try:
            self.model = ALS_factory(**args, use_gpu=implicit.gpu.HAS_CUDA)
        except NotImplementedError:
            self.model = ALS_factory(**args)


class LMF(Recommender):
    _model_class = LogisticMatrixFactorization

    def __init__(
        self,
        factors: int,
        random_state: int,
        learning_rate: float = None,
        regularization: float = None,
    ):
        super().__init__(factors)
        args = {"factors": factors, "iterations": 30, "random_state": random_state}
        if learning_rate is not None:
            args["learning_rate"] = learning_rate
        if regularization is not None:
            args["regularization"] = regularization
        try:

            self.model = LMF_factory(**args, use_gpu=implicit.gpu.HAS_CUDA)
        except NotImplementedError:
            self.model = LMF_factory(**args)


class SVDS(Recommender, MatrixFactorizationBase):
    factors: int = None
    random_state: utils.SeedSequence = None
    num_threads: int = 0  # used by implicit.evaluation

    def __init__(
        self, factors: int, random_state: Union[utils.SeedSequence, int] = None
    ):
        super().__init__(factors)
        self._model_class = sparse.linalg.svds
        self._make_random_state(random_state)

    def _make_random_state(self, random_state):
        if isinstance(random_state, utils.SeedSequence):
            self.random_state = random_state
        elif isinstance(random_state, int):
            self.random_state = utils.SeedSequence(random_state)
        else:
            self.random_state = utils.SeedSequence()

    # to be able to use Recommender.set_preferences and other super methods
    # which refer to `model` attribute (here `model` is the instance itself)
    @property
    def model(self) -> "SVDS":
        return self

    def fit(self, user_items: Union[np.ndarray, sparse.csr_array], **_):
        U, sigma, Vt = self._model_class(
            user_items, k=self.factors, random_state=next(self.random_state)
        )
        sigma_sq = np.diag(np.sqrt(sigma))
        self.user_factors = U @ sigma_sq
        self.item_factors = (sigma_sq @ Vt).T

    def save(self, savepath: Path):
        savepath = check_extension(savepath)
        args = dict(
            user_factors=self.user_factors,
            item_factors=self.item_factors,
            factors=self.factors,
            num_threads=self.num_threads,
            random_state=self.random_state.val,
        )
        # filter out 'None' valued args, since we can't go np.load on
        # them without using pickle
        args = {k: v for k, v in args.items() if v is not None}
        np.savez(savepath, **args)
        self.logger.info("Saved best model to %s", savepath)

    @classmethod
    def load(cls, filename: Path) -> "SVDS":
        filename = check_extension(filename)
        ret = cls(32)
        with np.load(filename, allow_pickle=True) as data:
            for k, v in data.items():
                setattr(ret, k, v)
        # np.savez saves numeric attributes as np.ndarray, reset to correct
        # dtype; self.num_threads is left as is, we don't use it
        ret.factors = int(ret.factors)
        ret._make_random_state(int(ret.random_state))
        ret.set_preferences()
        return ret


# NOTE not working right now! would need to make the data into pandas long format
class FSVD(Recommender):
    _model_class = SVD

    def __init__(self, factors: int, lr: float = None, regularization: float = None):
        super().__init__(factors)
        args = {"n_factors": factors, "n_epochs": 30}
        for k, v in zip(["lr", "reg"], [lr, regularization]):
            if v is not None:
                args[k] = v
        self.model = SVD(**args)

    def set_preferences(self):
        self.preferences = (
            (self.model.pu_ @ self.model.qi_.T)
            + self.model.bu_[:, None]
            + self.model.bi_[None, :]
        )

    def train(self, train_df: pd.DataFrame):
        # NOTE assumes train_df is in long format and contains columns u_id,
        # i_id and rating
        self.model.fit(train_df)
        self.set_preferences()

    def save(self, savepath: Path):
        savepath = check_extension(savepath)
        # basically copy how implicit saves its models
        args = {k: v for k, v in self.model.__dict__.items() if v is not None}
        np.savez(savepath, **args)
        self.logger.info("Saved best model to %s", savepath)

    @classmethod
    def load(cls, filename: Path) -> "FSVD":
        filename = check_extension(filename)
        ret = cls(32)
        with np.load(filename, allow_pickle=True) as data:
            for k, v in data.items():
                setattr(ret.model, k, v)
        ret.set_preferences()
        return ret


MODEL_NAMES = ["SVDS", "FSVD", "ALS", "LMF"]
MODELS_MAP = dict(zip(MODEL_NAMES, [SVDS, FSVD, ALS, LMF]))
