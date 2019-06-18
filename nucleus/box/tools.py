from typing import Union, Tuple

import tensorflow as tf
from public import public

from nucleus.utils import name_scope, tf_get_shape


@public
@name_scope
def ijhw_to_yx(ijhw: tf.Tensor) -> tf.Tensor:
    return ijhw[..., :2] + ijhw[..., 2:4] / 2


@public
@name_scope
def ijhw_to_yxhw(ijhw: tf.Tensor) -> tf.Tensor:
    return tf.concat([ijhw_to_yx(ijhw), ijhw[..., 2:]], axis=-1)


@public
@name_scope
def ijhw_to_kl(ijhw: tf.Tensor) -> tf.Tensor:
    return ijhw[..., :2] + ijhw[..., 2:4]


@public
@name_scope
def ijhw_to_ijkl(ijhw: tf.Tensor) -> tf.Tensor:
    return tf.concat([ijhw[..., :2], ijhw_to_kl(ijhw), ijhw[..., 4:]], axis=-1)


@public
@name_scope
def yxhw_to_ij(yxhw: tf.Tensor) -> tf.Tensor:
    return yxhw[..., :2] - yxhw[..., 2:4] / 2


@public
@name_scope
def yxhw_to_ijhw(yxhw: tf.Tensor) -> tf.Tensor:
    return tf.concat([yxhw_to_ij(yxhw), yxhw[..., 2:]], axis=-1)


@public
@name_scope
def yxhw_to_hw(yxhw: tf.Tensor) -> tf.Tensor:
    return yxhw[..., :2] + yxhw[..., 2:4] / 2


@public
@name_scope
def yxhw_to_ijkl(yxhw: tf.Tensor) -> tf.Tensor:
    return tf.concat(
        [yxhw_to_ij(yxhw), yxhw_to_hw(yxhw), yxhw[..., 4:]],
        axis=-1
    )


@public
@name_scope
def ijkl_to_hw(ijkl: tf.Tensor) -> tf.Tensor:
    return ijkl[..., :2] - ijkl[..., 2:4]


@public
@name_scope
def ijkl_to_ijhw(ijkl: tf.Tensor) -> tf.Tensor:
    return tf.concat([ijkl[..., :2], ijkl_to_hw(ijkl), ijkl[..., 4:]], axis=-1)


@public
@name_scope
def ijkl_to_xy(ijkl: tf.Tensor) -> tf.Tensor:
    return ijkl[..., :2] + ijkl_to_hw(ijkl) / 2


@public
@name_scope
def ijkl_to_xy(ijkl: tf.Tensor) -> tf.Tensor:
    return tf.concat(
        [ijkl_to_hw(ijkl), ijkl_to_hw(ijkl), ijkl[..., 4:]],
        axis=-1
    )


@public
@name_scope
def swap_axes_order(coords: tf.Tensor) -> tf.Tensor:
    coord_indices = [1, 0, 3, 2]
    other_indices = list(range(len(coord_indices), tf_get_shape(coords)[-1]))
    indices = coord_indices + other_indices
    return coords[..., indices]


@public
@name_scope
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


@public
@name_scope
def match_up_tensors(
        tensor_a: tf.Tensor,
        tensor_b: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    r"""
    Extend two tensors with the same dimensions but the second to last one to
    match all pairs element-wise.
    For example, if `tensor_a` is (5, 3, 4) and `tensor_b` is (5, 2, 4),
    then the output will be a tensor with dimensions (5, 3, 2, 4) containing
    every pair of vectors from the last dimension of `tensor_a` and `tensor_b`.

    Parameters
    ----------
    tensor_a
        ``(..., num_a, m)`` tensor.
    tensor_b
        ``(..., num_b, m)`` tensor.

    Returns
    -------
    matched
        ``(..., num_a, num_b, m)`` containing every pair of vectors from the
        last dimension of `tensor_a` and `tensor_b`.
    """
    tensor_b_shape = tf_get_shape(tensor_b)
    num_b, _ = tensor_b_shape[-2:]
    remainder_shape = tuple([d for d in tensor_b_shape[:-2]])

    repeated_a = tf.tile(tensor_a, (num_b,) + (1,) * (len(remainder_shape) + 1))

    reshaped_a = tf.reshape(
        repeated_a,
        [num_b] + tf_get_shape(tensor_a)
    )

    dims = [d + 1 for d in range(len(remainder_shape) + 1)]

    transposed_a = tf.transpose(
        reshaped_a,
        dims + [0, len(remainder_shape) + 2]
    )

    tensor_a_shape = tf_get_shape(tensor_a)
    num_a, _ = tensor_a_shape[-2:]

    repeated_b = tf.tile(tensor_b, (num_a,) + (1,) * (len(remainder_shape) + 1))

    reshaped_b = tf.reshape(
        repeated_b,
        [num_a] + tf_get_shape(tensor_b)
    )

    dims = [d + 1 for d in range(len(remainder_shape))]

    transposed_b = tf.transpose(
        reshaped_b,
        dims + [0, len(remainder_shape) + 1, len(remainder_shape) + 2]
    )

    return transposed_a, transposed_b


@public
@name_scope
def calculate_intersections(
        ijhw_a: tf.Tensor,
        ijhw_b: tf.Tensor
) -> tf.Tensor:
    r"""
    Calculate the intersection between pairs of bounding boxes.

    Parameters
    ----------
    ijhw_a
        ``(..., n_dims)`` tensor containing bounding boxes.
    ijhw_b
        ``(..., n_dims)`` tensor containing bounding boxes.

    Return
    ------
    intersections
        ``(...)`` Tensor representing the intersection between all pairs of
        boxes.
    """
    a_ij, a_hw = ijhw_a[..., :2], ijhw_a[..., 2:4]
    b_ij, b_hw = ijhw_b[..., :2], ijhw_b[..., 2:4]

    maxed_ij = tf.maximum(a_ij, b_ij)
    intersection_hw = tf.maximum(
        0.0,
        tf.minimum(a_ij + a_hw, b_ij + b_hw) - maxed_ij
    )

    return tf.reduce_prod(intersection_hw, axis=-1)


@public
@name_scope
def calculate_unions(
        ijhw_a: tf.Tensor,
        ijhw_b: tf.Tensor
) -> tf.Tensor:
    r"""
    Calculate the union between pairs of bounding boxes.

    Parameters
    ----------
    ijhw_a
        ``(..., n_dims)`` tensor containing bounding boxes.
    ijhw_b
        ``(..., n_dims)`` tensor containing bounding boxes.

    Return
    ------
    intersections
        ``(...)`` Tensor representing the union between all pairs of boxes.
    """
    intersections = calculate_intersections(ijhw_a=ijhw_a, ijhw_b=ijhw_b)
    a_areas = tf.reduce_prod(ijhw_a[..., 2:4], axis=-1)
    b_areas = tf.reduce_prod(ijhw_b[..., 2:4], axis=-1)

    return a_areas + b_areas - intersections


@public
@name_scope
def calculate_ious(
        ijhw_a: tf.Tensor,
        ijhw_b: tf.Tensor
) -> tf.Tensor:
    r"""
    Calculate the Intersection over Union (IoU) between pairs of bounding boxes.

    Parameters
    ----------
    ijhw_a
        ``(..., n_dims)`` tensor containing bounding boxes.
    ijhw_b
        ``(..., n_dims)`` tensor containing bounding boxes.

    Return
    ------
    ious
        ``(...)`` Tensor representing the IoUs between all pairs of boxes.
    """
    intersections = calculate_intersections(ijhw_a=ijhw_a, ijhw_b=ijhw_b)
    a_areas = tf.reduce_prod(ijhw_a[..., 2:4], axis=-1)
    b_areas = tf.reduce_prod(ijhw_b[..., 2:4], axis=-1)

    return intersections / (a_areas + b_areas - intersections)


@public
@name_scope
def pad_tensor(
        tensor: tf.Tensor,
        max_length: int,
        padding_value: float = -1
) -> tf.Tensor:
    r"""

    Parameters
    ----------
    tensor
    max_length
    padding_value

    Returns
    -------
    """
    tensor_shape = tf_get_shape(tensor)

    tf.assert_equal(tf.less_equal(tensor_shape[0], max_length), True)

    padding = padding_value * tf.ones(
        shape=[max_length - tensor_shape[0]] + tensor_shape[1:],
        dtype=tensor.dtype
    )

    return tf.concat([tensor, padding], axis=0)


@public
@name_scope
def unpad_tensor(
        tensor: tf.Tensor,
        padding_value: float = -1,
        boolean_fn=tf.equal
) -> tf.Tensor:
    r"""

    Parameters
    ----------
    tensor
    padding_value
    boolean_fn

    Returns
    -------

    """
    padding_sum = tf.reduce_sum(
        padding_value * tf.ones(tf_get_shape(tensor)[-1:])
    )
    return tensor[
        tf.logical_not(
            boolean_fn(
                tf.reduce_sum(tensor, axis=-1),
                padding_sum
            )
        )
    ]


@public
@name_scope
def fix_tensor_length(
        tensor: tf.Tensor,
        max_length: int,
        padding_value: float = -1
) -> tf.Tensor:
    r"""

    Parameters
    ----------
    tensor
    max_length
    padding_value

    Returns
    -------
    """
    tensor_shape = tf_get_shape(tensor)

    if tf.less(tensor_shape[0], max_length):
        tensor = pad_tensor(
            tensor=tensor,
            max_length=max_length,
            padding_value=padding_value
        )
    if tf.greater(tensor_shape[0], max_length):
        tensor = tensor[:max_length]

    return tensor


@public
@name_scope
def filter_boxes(
        boxes: tf.Tensor,
        pad: bool = True
) -> tf.Tensor:
    r"""
    Filters boxes that are completely outside of the image.

    Parameters
    ----------
    boxes
        The boxes to be filtered.
    pad

    Returns
    -------
    The filtered boxes.
    """
    max_boxes, n_dims = tf_get_shape(boxes)

    ijhw_tensor = boxes[..., :4]

    ijkl_tensor = ijhw_to_ijkl(ijhw_tensor)
    indices = tf.where(
        tf.logical_and(
            tf.logical_and(
                    tf.less_equal(ijkl_tensor[..., 0], 1),
                    tf.less_equal(ijkl_tensor[..., 1], 1)
            ),
            tf.logical_and(
                tf.greater(ijkl_tensor[..., 2], 0),
                tf.greater(ijkl_tensor[..., 3], 0)
            )
        )
    )
    boxes = tf.gather(boxes, tf.squeeze(indices))
    n_boxes = tf_get_shape(indices)[0]

    boxes = tf.cond(
        tf.equal(n_boxes, 1),
        lambda: boxes[None],
        lambda: boxes
    )
    boxes = tf.cond(
        tf.logical_and(tf.less(n_boxes, max_boxes), tf.equal(pad, True)),
        lambda: tf.reshape(tf.concat(
            [
                boxes,
                -1 * tf.ones((max_boxes - n_boxes, n_dims))
            ],
            axis=0
        ), [max_boxes, n_dims]),
        lambda: tf.reshape(boxes, [n_boxes, n_dims])
    )

    return boxes


@public
@name_scope
def flip_boxes_left_right(
        boxes: tf.Tensor
) -> tf.Tensor:
    r"""
    Horizontally flips bounding boxes defined in an image.

    Parameters
    ----------
    boxes
        The boxes to be horizontally flipped.
    image_width
        The width of the image where the bounding boxes are defined.

    Returns
    -------
    The horizontally flipped bounding boxes.
    """
    x = 1.0 - boxes[..., 1] - boxes[..., 3]
    boxes = tf.concat([boxes[..., 0:1], x[..., None], boxes[..., 2:]], axis=-1)
    return boxes
