import logging
from typing import Tuple, Union

import numpy as np
import scipy


class SeedSequence:
    val: int

    def __init__(self, start: int = 42):
        self.val = start

    def __next__(self) -> int:
        ret = self.val
        self.val += 1
        return ret

    def __repr__(self) -> str:
        return str(self.val)


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
        preferences: 2D array of unbounded scores.
        temperature: Optional temperature parameter for softmax, default: 1.0.

    Returns:
        A matrix of probabilities computed row-wise from `preferences`.
    """
    if len(preferences.shape) < 2:
        preferences = preferences[None, :]
    return scipy.special.softmax(preferences / temperature, axis=1).squeeze()


def minmax_scale(a: np.ndarray) -> np.ndarray:
    if len(a.shape) < 2:
        a = a[None, :]
    min_ = a.min(axis=1)[:, None]
    rescaled = np.squeeze((a - min_) / (a.max(axis=1)[:, None] - min_))
    # nan can appear in case of a 0 vector
    return np.nan_to_num(rescaled)


def make_mask(ids: np.ndarray, shape: Tuple[int]) -> np.ndarray:
    """
    Returns a binary mask of shape `shape` where the entries at indices `ids`
    indicate masking (invisible).
    NOTE ids should be a 2D np.ndarray of shape (#masked entries, 2).
    """
    assert len(ids.shape) == 2 and ids.shape[1] == 2
    mask = np.zeros(shape, dtype=np.int32)
    mask[ids[:, 0], ids[:, 1]] = 1
    return mask


def array_coords(shape: Tuple[int]) -> np.ndarray:
    """
    Returns the cartesian product of the indices of an array of size `shape` as
    a 2D np.ndarray with shape `(np.prod(shape), 2)`.
    """
    return np.dstack(np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))).reshape(
        -1, 2
    )


def sample_entries(
    arr: Union[np.ndarray, np.ma.masked_array],
    retain_prop: float,
    rng: np.random.Generator,
) -> np.ma.masked_array:
    """
    Returns a 2D np.ma.masked_array copy of `arr` downsampled uniformly to
    `retain_prop`, where retained entries are chosen from unmasked entries in
    `arr`. The `fill_value` of the returned masked array is 0.
    """
    # get proportion of elements to be masked
    drop_prop = 1.0 - retain_prop
    # if `arr` does not have a mask already, we can sample over every entry;
    # otherwise, we only sample where the existing mask is True (masked
    # entries), and we hide the previously visible entries through inversion
    if isinstance(arr, np.ma.masked_array):
        samplable_ids = np.argwhere(arr.mask)
        avail_mask = ~arr.mask
        arr = arr.data
    else:
        samplable_ids = array_coords(arr.shape)
        avail_mask = np.ma.nomask  # dummy initializer
    # sample indices of entries to be masked
    masked_ids = rng.choice(
        samplable_ids, size=int(drop_prop * len(samplable_ids)), replace=False
    )
    # print(len(masked_ids))
    # overlay the newly hidden entries on top of the existing ones, if any
    mask = avail_mask + make_mask(masked_ids, arr.shape)
    return np.ma.masked_array(arr, mask, fill_value=0.0)


def train_test_split_mask(
    data: np.ndarray,
    train_prop: float,
    rng: np.random.default_rng,
    valid_prop: float = None,
) -> Tuple[np.ma.masked_array, ...]:
    """
    Splits `data` into a train and a test set according to `train_prop`, where
    entries are sampled uniformly over the whole `data`. Returns the train and
    test partitions as np.ma.masked_array; if `valid_prop` is specified, a
    valdation set is also created and returned as the second element.
    """
    if valid_prop is not None and train_prop + valid_prop > 1.0:
        raise Exception("train_prop + valid_prop must be < 1")
    tot = np.prod(data.shape)
    train = sample_entries(data, train_prop, rng)
    # print(tot * train_prop)
    # print(len(np.argwhere(train.mask == 0)))
    # print(len(np.argwhere(train.mask)))

    # FIXME sometimes this assertion fails because the first operand has +1
    # element than the second! why?
    assert len(np.argwhere(train.mask == 0)) == (tot * train_prop)

    if valid_prop is not None:
        # rescale the valdation proportion to available data proportion after
        # train
        valid_prop_scaled = valid_prop / (1.0 - train_prop)
        valid = sample_entries(train, valid_prop_scaled, rng)
        assert len(np.argwhere(valid.mask == 0)) == (
            (tot - (tot * train_prop)) * valid_prop_scaled
        )
        # bitwise and between the train mask and validation mask compuounds the
        # masked entries, leaving only the test ones
        test = np.ma.masked_array(data, train.mask ^ valid.mask)
        rest = (valid, test)
    else:
        rest = (np.ma.masked_array(data, ~train.mask),)
    return (train,) + rest
