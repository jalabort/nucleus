import tensorflow as tf
from public import public

from nucleus.box import scale_coords, ijhw_to_ijkl
from nucleus.utils import name_scope


@public
@name_scope
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


@public
@name_scope
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


@public
@name_scope
def crop_chw(chw: tf.Tensor, ijhw: tf.Tensor) -> tf.Tensor:
    r"""

    Parameters
    ----------
    chw
    ijhw

    Returns
    -------

    """
    ijhw = scale_coords(ijhw, chw.shape[-2:])
    ijkl = ijhw_to_ijkl(ijhw)
    ijkl = tf.cast(ijkl, dtype=tf.int32)
    return chw[..., ijkl[0]:ijkl[2], ijkl[1]:ijkl[3]]
