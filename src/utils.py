import logging

import numpy as np
import scipy


def setup_root_logging(level: int = logging.INFO):
    """
    Root entry point to set up logging config. Should be called once from the
    program's main entry point.

    Args:
        level: The minimum logging level of displayed messages, defaults to
               logging.INFO (everything below is suppressed).
    """

    logging.basicConfig(
        level=level,
        format="p%(process)s [%(asctime)s] [%(levelname)s]  %(message)s  (%(name)s:%(lineno)s)",
        datefmt="%y-%m-%d %H:%M",
    )


def one_hot(labels: np.ndarray, max_label: int) -> np.ndarray:
    """
    Makes a one-hot-encoded matrix from labels.

    Args:
        labels: 1 or 2D array of labels; if `labels` is 2D, the last dimension
                should be 1.
        max_label: The size of a one-hot-encoded vector (a row in the return
                   matrix)

    Returns:
        A one-hot-encoded 2D array with 1s at `labels`, 0s elsewhere.
    """
    assert (
        labels.max() <= max_label
    ), f"max_label is {max_label}, but labels containst values up to {labels.max()}"
    if len(labels.shape) < 2:
        labels = labels[:, None]
    one_hot_mat = np.zeros((labels.size, max_label), dtype=int)
    np.put_along_axis(one_hot_mat, labels, 1, axis=1)
    return one_hot_mat


def softmax(preferences: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Transforms preference scores into probabilities by applying softmax.

    Args:
        preferences: 1 or 2D array of unbounded scores.
        temperature: Optional temperature parameter for softmax, default: 1.0.

    Returns:
        An array of probabilities computed row-wise from `preferences`.
    """
    if len(preferences.shape) < 2:
        preferences = preferences[None, :]
    return scipy.special.softmax(preferences / temperature, axis=1).squeeze()


def minmax_scale(a: np.ndarray) -> np.ndarray:
    if len(a.shape) < 2:
        a = a[None, :]
    min_ = a.min(axis=1)[:, None]
    return np.squeeze((a - min_) / (a.max(axis=1)[:, None] - min_))


def remove_diag(a: np.ndarray) -> np.ndarray:
    return a[~np.eye(a.shape[0], dtype=bool)].reshape(a.shape[0], -1)
