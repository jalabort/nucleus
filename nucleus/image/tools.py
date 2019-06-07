import numpy as np
import tensorflow as tf

from nucleus.box import tools as box_tools
from nucleus.utils import export, name_scope


@export
@name_scope
@tf.function
def hwc_to_chw(hwc: tf.Tensor) -> tf.Tensor:
    r"""

    Parameters
    ----------
    hwc

    Returns
    -------

    """
    n_dims = len(hwc.shape)
    if n_dims == 3:
        chw = tf.transpose(hwc, perm=[2, 0, 1])
    elif n_dims == 2:
        chw = hwc
    else:
        raise ValueError()

    return chw


@export
@name_scope
@tf.function
def chw_to_hwc(chw: tf.Tensor) -> tf.Tensor:
    r"""

    Parameters
    ----------
    chw

    Returns
    -------

    """
    n_dims = len(chw.shape)
    if n_dims == 3:
        hwc = tf.transpose(chw, perm=[1, 2, 0])
    elif n_dims == 2:
        hwc = chw
    else:
        raise ValueError()

    return hwc


@export
@name_scope
@tf.function
def crop_chw(chw: np.ndarray, ijhw: np.array) -> np.ndarray:
    r"""

    Parameters
    ----------
    chw
    ijhw

    Returns
    -------

    """
    ijhw = box_tools.scale_coords(ijhw, chw.shape[-2:])
    ijkl = box_tools.ijhw_to_ijkl(ijhw)
    ijkl = tf.cast(ijkl, dtype=tf.int32)
    return chw[..., ijkl[0]:ijkl[2], ijkl[1]:ijkl[3]]
