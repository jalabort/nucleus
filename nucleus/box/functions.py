from typing import Union, Tuple

import tensorflow as tf


def ijhw_to_yx(ijhw: tf.Tensor) -> tf.Tensor:
    return ijhw[..., :2] + ijhw[..., 2:4] / 2


def ijhw_to_yxhw(ijhw: tf.Tensor) -> tf.Tensor:
    return tf.concat([ijhw_to_yx(ijhw), ijhw[..., 2:]], axis=-1)


def ijhw_to_kl(ijhw: tf.Tensor) -> tf.Tensor:
    return ijhw[..., :2] + ijhw[..., 2:4]


def ijhw_to_ijkl(ijhw: tf.Tensor) -> tf.Tensor:
    return tf.concat([ijhw[..., :2], ijhw_to_kl(ijhw), ijhw[..., 4:]], axis=-1)


def yxhw_to_ij(yxhw: tf.Tensor) -> tf.Tensor:
    return yxhw[..., :2] - yxhw[..., 2:4] / 2


def yxhw_to_ijhw(yxhw: tf.Tensor) -> tf.Tensor:
    return tf.concat([yxhw_to_ij(yxhw), yxhw[..., 2:]], axis=-1)


def yxhw_to_hw(yxhw: tf.Tensor) -> tf.Tensor:
    return yxhw[..., :2] + yxhw[..., 2:4] / 2


def yxhw_to_ijkl(yxhw: tf.Tensor) -> tf.Tensor:
    return tf.concat(
        [yxhw_to_ij(yxhw), yxhw_to_hw(yxhw), yxhw[..., 4:]],
        axis=-1
    )


def ijkl_to_hw(ijkl: tf.Tensor) -> tf.Tensor:
    return ijkl[..., :2] - ijkl[..., 2:4]


def ijkl_to_ijhw(ijkl: tf.Tensor) -> tf.Tensor:
    return tf.concat([ijkl[..., :2], ijkl_to_hw(ijkl), ijkl[..., 4:]], axis=-1)


def ijkl_to_xy(ijkl: tf.Tensor) -> tf.Tensor:
    return ijkl[..., :2] + ijkl_to_hw(ijkl) / 2


def ijkl_to_yxhw(ijkl: tf.Tensor) -> tf.Tensor:
    return tf.concat(
        [ijkl_to_hw(ijkl), ijkl_to_hw(ijkl), ijkl[..., 4:]],
        axis=-1
    )


def swap_axes_order(coords: tf.Tensor) -> tf.Tensor:
    coord_indices = [1, 0, 3, 2]
    other_indices = list(range(len(coord_indices), coords.shape[-1]))
    indices = coord_indices + other_indices
    return coords[..., indices]


def scale_coords(
        coords: tf.Tensor,
        resolution: Union[int, Tuple[int, int]]
) -> tf.Tensor:
    if isinstance(resolution, int):
        resolution = (resolution, resolution)
    ext_resolution = tf.cast(
        tf.concat([resolution, resolution], axis=-1),
        dtype=tf.float32
    )
    coord = coords[..., :4] * ext_resolution
    return tf.concat([coord, coords[..., 4:]], axis=-1)
