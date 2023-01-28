import logging
from pathlib import Path
from typing import Union

import implicit
import numpy as np
import pandas as pd
from funk_svd import SVD
from implicit.als import AlternatingLeastSquares
from implicit.lmf import LogisticMatrixFactorization
from scipy import sparse

# TODO validate

AnyRecommender = Union[AlternatingLeastSquares, LogisticMatrixFactorization, SVD]


def check_extension(p: Path, ext: str = ".npz") -> Path:
    if not isinstance(p, Path):
        p = Path(p)
    if p.suffix != ".npz":
        p = p.parent / (p.name + ".npz")
    return p


class Recommender:
    model: AnyRecommender = None
    logger: logging.Logger = None
    preferences: np.ndarray = None

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}:{type(self).__name__}")

    def train(self, train_mat: Union[np.ndarray, pd.DataFrame, sparse.csr_array]):
        if isinstance(train_mat, np.ndarray):
            train_mat = sparse.csr_array(train_mat)
        self.model.fit(train_mat)
        # save preferences
        user_factors, item_factors = self.model.user_factors, self.model.item_factors
        if implicit.gpu.HAS_CUDA:
            user_factors, item_factors = (
                user_factors.to_numpy(),
                item_factors.to_numpy(),
            )
        self._preferences = user_factors @ item_factors.T

    def save(self, savepath: Path):
        savepath = check_extension(savepath)
        self.model.save(savepath)
        self.logger.info("Saved best model to %s", savepath)

    @classmethod
    def load(cls, filename: Path) -> AnyRecommender:
        ...


class ALS(Recommender):
    model: implicit.cpu.als.AlternatingLeastSquares = None

    def __init__(self, factors: int, regularization: float = None, alpha: float = None):
        super().__init__()
        args = {"factors": factors}
        # use defaults from implicit if regularization and alpha are not passed
        if regularization is not None:
            args["regularization"] = regularization
        if alpha is not None:
            args["alpha"] = alpha
        self.model = AlternatingLeastSquares(**args)

    @classmethod
    def load(cls, filename: Path) -> AlternatingLeastSquares:
        ret = cls(32)
        ret.model = implicit.cpu.als.AlternatingLeastSquares.load(filename)
        return ret


class LMF(Recommender):
    def __init__(
        self, factors: int, learning_rate: float = None, regularization: float = None
    ):
        super().__init__()
        args = {"factors": factors}
        if learning_rate is not None:
            args["learning_rate"] = learning_rate
        if regularization is not None:
            args["regularization"] = regularization
        self.model = LogisticMatrixFactorization(**args)

    @classmethod
    def load(cls, filename: Path) -> LogisticMatrixFactorization:
        ret = cls(32)
        ret.model = implicit.cpu.lmf.LogisticMatrixFactorization.load(filename)
        return ret


class FSVD(Recommender):
    def __init__(
        self,
        factors: int,
        lr: float = None,
        regularization: float = None,
        num_epochs: int = None,
    ):
        super().__init__()
        args = {"n_factors": factors}
        for k, v in zip(["lr", "reg", "n_epochs"], [lr, regularization, num_epochs]):
            if v is not None:
                args[k] = v
        self.model = SVD(**args)

    def train(self, train_df: pd.DataFrame):
        # NOTE assumes train_df is in long format and contains columns u_id,
        # i_id and rating
        self.model.fit(train_df)
        self.preferences = (
            (self.model.pu_ @ self.model.qi_.T)
            + self.model.bu_[:, None]
            + self.model.bi_[None, :]
        )

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
        return ret
