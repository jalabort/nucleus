from typing import Union, Tuple, Sequence

import tensorflow as tf

from nucleus.box.tools import (
    match_up_tensors, calculate_ious, ijhw_to_yxhw, ijhw_to_yx
)
from nucleus.utils import export


# TODO: Document me!
@export
@tf.function
def create_anchors(
        scales: Union[float, Sequence[float]],
        ratios: Union[float, Sequence[float]],
        grid_height: float,
        grid_width: float,
        flatten: bool = False
) -> tf.Tensor:
    r"""
    Creates anchor boxes.

    Notes
    -----
    Anchor boxes are naturally defined on a 2-dimensional grid of the same
    height and width as the feature map that is used to make bounding box
    predictions.

    Parameters
    ----------
    scales

    ratios

    grid_height
        The vertical number of cells in the grid i.e the number of columns in
        the grid.
    grid_width
        The horizontal number of cells in the grid i.e. the number of rows in
        the grid.

    flatten
        Whether to flatten the anchor boxes or not. If `False`, the dimensions
        of the anchors will be ``(height, width, n_anchors, 4)``. If `True`,
        they will be ``(height * width * n_anchors, 4)``.

    Returns
    -------
    anchors
        ``(height, width, n_anchors, 4)`` or ``(height * width * n_anchors, 4)``
        tensor representing the anchor boxes.
    """
    # Convert scales and ratios to tensors if necessary
    if isinstance(scales, (int, float)):
        scales = [scales]
    if isinstance(ratios, (int, float)):
        ratios = [ratios]
    if not isinstance(scales, tf.Tensor):
        scales = tf.convert_to_tensor(scales, dtype=tf.float32)
    if not isinstance(ratios, tf.Tensor):
        ratios = tf.convert_to_tensor(ratios, dtype=tf.float32)

    # Determine the total number of anchors
    n_anchors = len(scales) * len(ratios)

    # Compute the centers of the grid cells
    cell_h = 1 / grid_height
    cell_w = 1 / grid_width

    cell_x, cell_y = tf.meshgrid(
        tf.range(start=0.5 * cell_w, limit=1, delta=cell_w),
        tf.range(start=0.5 * cell_h, limit=1, delta=cell_h)
    )

    cell_y = tf.tile(cell_y[..., None], [1, 1, n_anchors])
    cell_x = tf.tile(cell_x[..., None], [1, 1, n_anchors])

    # Compute the height and width of the anchors
    gridded_scales, gridded_ratios = tf.meshgrid(scales, ratios)

    anchors_h = cell_h * gridded_scales / tf.sqrt(gridded_ratios)
    anchors_w = cell_w * gridded_scales * tf.sqrt(gridded_ratios)

    anchors_h = tf.reshape(anchors_h, shape=(-1,))
    anchors_w = tf.reshape(anchors_w, shape=(-1,))

    anchors_h = tf.tile(anchors_h[None, None, :], [grid_height, grid_width, 1])
    anchors_w = tf.tile(anchors_w[None, None, :], [grid_height, grid_width, 1])

    # Compute the top left coordinates of the anchors
    anchors_i = cell_y - 0.5 * anchors_h
    anchors_j = cell_x - 0.5 * anchors_w

    # Create the grid of anchors
    anchors = tf.stack([anchors_i, anchors_j, anchors_h, anchors_w], axis=-1)

    if flatten:
        anchors = tf.reshape(anchors, shape=(-1, 4))

    return anchors


# TODO: Add max_box_size as an argument
@export
@tf.function
def match_boxes_iou(
        boxes: tf.Tensor,
        anchors: tf.Tensor,
        iou_threshold: float
) -> Tuple[tf.Tensor, tf.Tensor]:
    r"""
    Associates ground truth bounding boxes with anchor boxes.

    Parameters
    ----------
    boxes
        ``(n_boxes, n_dims)`` tensor representing the ground truth bounding
        boxes.
    anchors
        ``(height, width, n_anchors, 4)`` tensor representing the anchor boxes.
    iou_threshold
        The iou threshold above which ground truth bounding boxes are
        associated with anchor boxes.

    Returns
    -------
    matched_boxes
        ``(height, width, n_anchors, n_dims)`` tensor representing the ground
        truth bounding boxes that have been successfully associated with the
        anchor boxes.
    matched_anchors
        ``(height, width, n_anchors, 4)`` tensor representing the anchor boxes
        that have been successfully associated with the ground truth bounding
        boxes.
    """
    n_boxes, n_dims = boxes.shape
    grid_height, grid_width, n_anchors, _ = anchors.shape

    # Flatten the anchors
    flat_anchors = tf.reshape(
        anchors,
        shape=(grid_height * grid_width * n_anchors, 4)
    )

    # Calculate iou between the flat anchors and the boxes
    matched_anchors, matched_boxes = match_up_tensors(flat_anchors, boxes)
    ious = calculate_ious(matched_anchors, matched_boxes)

    # Find the index of the maximum iou for every box
    condition = tf.equal(ious, 0)
    ious = tf.tensor_scatter_nd_update(
        tensor=ious,
        indices=tf.where(condition),
        updates=-1 * tf.ones_like(ious[condition])
    )
    max_indices_0 = tf.argmax(ious, axis=0, output_type=tf.int32)
    max_indices = [[max_indices_0[i], i] for i in range(len(max_indices_0))]

    # Add 1 to the maximum iou for every box. This makes it more likely that
    # every box gets matched with an anchor.
    ious = tf.tensor_scatter_nd_update(
        tensor=ious,
        indices=max_indices,
        updates=tf.stack([ious[i, j] for i, j in max_indices]) + 1
    )

    # Reshape the ious, boxes and anchors to the original anchor shape
    ious = tf.reshape(
        ious,
        shape=[grid_height, grid_width, n_anchors, n_boxes]
    )
    matched_boxes = tf.reshape(
        matched_boxes,
        shape=[grid_height, grid_width, n_anchors, n_boxes, n_dims]
    )
    matched_anchors = tf.reshape(
        matched_anchors,
        shape=[grid_height, grid_width, n_anchors, n_boxes, 4]
    )

    # Only keep the ious, boxes and anchors with iou above the threshold
    condition = tf.less_equal(ious, iou_threshold)

    ious = tf.tensor_scatter_nd_update(
        tensor=ious,
        indices=tf.where(condition),
        updates=tf.zeros_like(ious[condition])
    )
    matched_boxes = tf.tensor_scatter_nd_update(
        tensor=matched_boxes,
        indices=tf.where(condition),
        updates=tf.zeros_like(matched_boxes[condition])
    )
    matched_anchors = tf.tensor_scatter_nd_update(
        tensor=matched_anchors,
        indices=tf.where(condition),
        updates=tf.zeros_like(matched_anchors[condition])
    )

    # Get indices that sort the ious by descending order. We want to
    # prioritize those anchors and boxes with the highest ious.
    indices = tf.argsort(ious, direction='DESCENDING')

    # TF cannot use the previous indices and unravel them to directly index
    # into a tensor, as one would do in numpy. Therefore, we need to do a
    # little work to convert the indices into a form that we can use to
    # index the tensor
    indices = tf.reshape(tf.transpose(indices, perm=[3, 0, 1, 2]), (-1,))
    indices *= grid_height * grid_width * n_anchors
    base_indices = tf.range(0, grid_height * grid_width * n_anchors)
    base_indices = tf.tile(base_indices, [n_boxes])
    final_indices = base_indices + indices

    # TODO: Can this be vectorized further?
    # Sort the boxes and anchors by descending iou
    matched_boxes = tf.transpose(matched_boxes, perm=[4, 3, 0, 1, 2])
    matched_anchors = tf.transpose(matched_anchors, perm=[4, 3, 0, 1, 2])

    boxes_list = []
    anchors_list = []
    for i in range(n_dims):
        if i < 4:
            # Sort matched_anchors; anchors only have 4 dimensions
            anchors_i = tf.gather(
                tf.reshape(matched_anchors[i], (-1,)),
                final_indices
            )
            anchors_list.append(
                tf.reshape(
                    anchors_i,
                    [n_boxes, grid_height, grid_width, n_anchors]
                )
            )
        # Sort matched_boxes
        boxes_i = tf.gather(
            tf.reshape(matched_boxes[i], (-1,)),
            final_indices
        )
        boxes_list.append(
            tf.reshape(
                boxes_i,
                [n_boxes, grid_height, grid_width, n_anchors]
            )
        )

    # Reshape boxes and anchors to their original shapes
    matched_boxes = tf.transpose(tf.stack(boxes_list), perm=[2, 3, 4, 1, 0])
    matched_anchors = tf.transpose(tf.stack(anchors_list), perm=[2, 3, 4, 1, 0])

    # Assign only one box per anchor
    matched_boxes = matched_boxes[..., 0, :]
    matched_anchors = matched_anchors[..., 0, :]

    return ijhw_to_yxhw(matched_boxes), ijhw_to_yxhw(matched_anchors)


@export
@tf.function
def match_boxes_distance(
        boxes: tf.Tensor,
        anchors: tf.Tensor,
        max_distance: Union[float, Tuple[float, float]],
        max_box_size: Union[float, Tuple[float, float]]
) -> Tuple[tf.Tensor, tf.Tensor]:
    r"""
    Associates ground truth bounding boxes with anchor boxes.

    Parameters
    ----------
    boxes
        ``(n_boxes, 4)`` tensor representing the ground truth bounding boxes.
    anchors
        ``(height, width, n_anchors, 4)`` tensor representing the anchor boxes.
    max_distance
        The maximum distance between the center of ground truth bounding box
        and the center of an anchor boxes below which the two can be
        associated together.
    max_box_size
        Determines the maximum size that a bounding box can have in order to
        be associated to an anchor.

    Returns
    -------
    matched_boxes
        ``(height, width, n_anchors, n_dims)`` tensor representing the ground
        truth bounding boxes that have been successfully associated with the
        anchor boxes.
    matched_anchors
        `(height, width, n_anchors, 4)`` tensor representing the anchor boxes
        that have been successfully associated with the ground truth bounding
        boxes.
    """
    if isinstance(max_distance, (int, float)):
        max_distance = [max_distance, max_distance]
    if isinstance(max_box_size, (int, float)):
        max_box_size = [max_box_size, max_box_size]

    n_boxes, n_dims = boxes.shape
    grid_height, grid_width, n_anchors, _ = anchors.shape

    # Flatten the anchors
    flat_anchors = tf.reshape(
        anchors,
        shape=(grid_height * grid_width * n_anchors, 4)
    )

    # Calculate iou between the flat anchors and the boxes
    matched_anchors, matched_boxes = match_up_tensors(flat_anchors, boxes)
    ious = calculate_ious(matched_anchors, matched_boxes)

    # Calculate the euclidean distance between boxes and the anchors centers
    matched_boxes_yx = ijhw_to_yx(matched_boxes)
    matched_anchors_yx = ijhw_to_yx(matched_anchors)
    differences = matched_boxes_yx - matched_anchors_yx
    distances = tf.sqrt(tf.reduce_sum(differences ** 2, axis=-1))

    # Find the index of the minimum euclidean distance for every box
    condition = tf.logical_or(
        tf.equal(ious, 0),
        tf.logical_or(
            tf.logical_or(
                tf.greater(tf.abs(differences[..., 0]), max_distance[0]),
                tf.greater(tf.abs(differences[..., 1]), max_distance[1])
            ),
            tf.logical_or(
                tf.greater(matched_boxes[..., 2], max_box_size[0]),
                tf.greater(matched_boxes[..., 3], max_box_size[1])
            )
        )
    )
    distances = tf.tensor_scatter_nd_update(
        tensor=distances,
        indices=tf.where(condition),
        updates=1e6 * tf.ones_like(distances[condition])
    )
    min_indices_0 = tf.argmin(distances, axis=0, output_type=tf.int32)
    min_indices = [[min_indices_0[i], i] for i in range(len(min_indices_0))]

    # Set the minimum euclidean distance for every box to 0. This makes it more
    # likely that every box gets matched with an anchor.
    distances = tf.tensor_scatter_nd_update(
        tensor=distances,
        indices=min_indices,
        updates=tf.stack([0.0 for _, _ in min_indices])
    )

    # Only keep the distances, boxes and anchors that ...
    matched_boxes = tf.tensor_scatter_nd_update(
        tensor=matched_boxes,
        indices=tf.where(condition),
        updates=tf.zeros_like(matched_boxes[condition])
    )
    matched_anchors = tf.tensor_scatter_nd_update(
        tensor=matched_anchors,
        indices=tf.where(condition),
        updates=tf.zeros_like(matched_anchors[condition])
    )

    # Reshape the distances, boxes and anchors to the original anchor shape
    distances = tf.reshape(
        distances,
        shape=[grid_height, grid_width, n_anchors, n_boxes]
    )
    matched_boxes = tf.reshape(
        matched_boxes,
        shape=[grid_height, grid_width, n_anchors, n_boxes, n_dims]
    )
    matched_anchors = tf.reshape(
        matched_anchors,
        shape=[grid_height, grid_width, n_anchors, n_boxes, 4]
    )

    # Get indices that sort the euclidean distances by ascending order. We
    # want to prioritize those anchors and boxes with the smallest distance.
    indices = tf.argsort(distances, direction='ASCENDING')

    # TF cannot use the previous indices and unravel them to directly index
    # into a tensor, as one would do in numpy. Therefore, we need to do a
    # little work to convert the indices into a form that we can use to
    # index the tensor
    indices = tf.reshape(tf.transpose(indices, perm=[3, 0, 1, 2]), (-1,))
    indices *= grid_height * grid_width * n_anchors
    base_indices = tf.range(0, grid_height * grid_width * n_anchors)
    base_indices = tf.tile(base_indices, [n_boxes])
    final_indices = base_indices + indices

    # TODO: Can this be vectorized further?
    # Sort the boxes and anchors by descending iou
    matched_boxes = tf.transpose(matched_boxes, perm=[4, 3, 0, 1, 2])
    matched_anchors = tf.transpose(matched_anchors, perm=[4, 3, 0, 1, 2])

    boxes_list = []
    anchors_list = []
    for i in range(n_dims):
        if i < 4:
            # Sort matched_anchors; anchors only have 4 dimensions
            anchors_i = tf.gather(
                tf.reshape(matched_anchors[i], (-1,)),
                final_indices
            )
            anchors_list.append(
                tf.reshape(
                    anchors_i,
                    [n_boxes, grid_height, grid_width, n_anchors]
                )
            )
        # Sort matched_boxes
        boxes_i = tf.gather(
            tf.reshape(matched_boxes[i], (-1,)),
            final_indices
        )
        boxes_list.append(
            tf.reshape(
                boxes_i,
                [n_boxes, grid_height, grid_width, n_anchors]
            )
        )

    # Reshape boxes and anchors to their original shapes
    matched_boxes = tf.transpose(tf.stack(boxes_list), perm=[2, 3, 4, 1, 0])
    matched_anchors = tf.transpose(tf.stack(anchors_list), perm=[2, 3, 4, 1, 0])

    # Assign only one box per anchor
    matched_boxes = matched_boxes[..., 0, :]
    matched_anchors = matched_anchors[..., 0, :]

    return ijhw_to_yxhw(matched_boxes), ijhw_to_yxhw(matched_anchors)


# TODO: Currently not used
@export
@tf.function
def match_all_boxes(
        all_boxes: tf.Tensor,
        anchors: tf.Tensor,
        match_boxes_fn: callable = match_boxes_iou,
) -> Tuple[tf.Tensor, tf.Tensor]:
    r"""
    Associates a batch of ground truth bounding boxes with anchor boxes.

    Parameters
    ----------
    all_boxes
        ``(batch_size, n_boxes, 4)`` tensor representing a batch of ground
        truth bounding boxes.
    anchors
        ``(height, width, n_anchors, 4)`` tensor representing the anchor boxes.
    match_boxes_fn
        The specific matching function used to associate every batch of ground
        truth bounding boxes with the anchor boxes.

    Returns
    -------
    matched_all_boxes
        ``(batch_size, height, width, n_anchors, 4)`` tensor representing the
        ground truth bounding boxes that have been successfully associated
        with the anchor boxes.
    matched_all_anchors
        ``(batch_size, height, width, n_anchors, 4)`` tensor representing the
        anchor boxes that have been successfully associated with the ground
        truth bounding boxes.
    """
    batch_size = all_boxes.shape[0]

    matched_all_boxes = []
    matched_all_anchors = []

    for i in range(batch_size):
        matched_boxes, matched_anchors = match_boxes_fn(
            boxes=all_boxes[i],
            anchors=anchors
        )

        matched_all_boxes.append(matched_boxes)
        matched_all_anchors.append(matched_anchors)

    return tf.stack(matched_all_boxes), tf.stack(matched_all_anchors)


@export
@tf.function
def combine_boxes_ssd(
        matched_boxes: tf.Tensor,
        matched_anchors: tf.Tensor,
) -> Tuple[tf.Tensor]:
    r"""
    Combines matched ground truth bounding boxes with matched anchor boxes to
    create ground truth bounding boxes as defined by the SSD paper.

    Notes
    -----
    This ground truth bounding boxes parametrization was first proposed in
    the Faster RCNN paper.

    References
    ----------
    .. [1] Wei Liu, et. al, "SSD: Single Shot MultiBox Detector", ECCV 2016,
           https://arxiv.org/abs/1512.02325.
    .. [2] Ren, S., et.al, "Faster R-CNN: Towards real-time object detection
           with region proposal networks", NIPS 2015,
           https://arxiv.org/abs/1506.01497

    Parameters
    ----------
    matched_boxes
        ``(..., n_dims)`` tensor representing the ground truth bounding boxes.
    matched_anchors
        ``(..., 4)`` tensor representing the anchor boxes.

    Returns
    -------
    ssd_boxes
        ``(..., n_dims)`` tensor representing ground truth bounding boxes as
        defined by the SSD paper.
    """
    return combine_boxes(
        matched_boxes=matched_boxes,
        matched_anchors=matched_anchors,
        combine_yx_fn=combine_yx_ssd,
        combine_hw_fn=combine_hw_sdd
    )


@export
@tf.function
def combine_boxes_yolo(
        matched_boxes: tf.Tensor,
        matched_anchors: tf.Tensor,
) -> Tuple[tf.Tensor]:
    r"""
    Combines matched ground truth bounding boxes with matched anchor boxes to
    create ground truth bounding boxes as defined by the Yolo paper.

    References
    ----------
    .. [1] Joseph Redmon, et. al, "YOLO9000: Better, Faster, Stronger",
           CVPR 2017, https://arxiv.org/abs/1612.08242.

    Parameters
    ----------
    matched_boxes
        ``(..., n_dims)`` tensor representing the ground truth bounding boxes.
    matched_anchors
        ``(..., 4)`` tensor representing the anchor boxes.

    Returns
    -------
    yolo_boxes
        ``(..., n_dims)`` tensor representing ground truth bounding boxes as
        defined by the Yolo paper.
    """
    return combine_boxes(
        matched_boxes=matched_boxes,
        matched_anchors=matched_anchors,
        combine_yx_fn=combine_yx_yolo,
        combine_hw_fn=combine_hw_sdd
    )


@export
@tf.function
def combine_boxes_reinspect(
        matched_boxes: tf.Tensor,
        matched_anchors: tf.Tensor,
) -> Tuple[tf.Tensor]:
    r"""
    Combines matched ground truth bounding boxes with matched anchor boxes to
    create ground truth bounding boxes as defined by the Yolo paper.

    References
    ----------
    .. [1] Joseph Redmon, et. al, "YOLO9000: Better, Faster, Stronger",
           CVPR 2017, https://arxiv.org/abs/1612.08242.

    Parameters
    ----------
    matched_boxes
        ``(..., n_dims)`` tensor representing the ground truth bounding boxes.
    matched_anchors
        ``(..., 4)`` tensor representing the anchor boxes.

    Returns
    -------
    yolo_boxes
        ``(..., n_dims)`` tensor representing ground truth bounding boxes as
        defined by the Yolo paper.
    """
    return combine_boxes(
        matched_boxes=matched_boxes,
        matched_anchors=matched_anchors,
        combine_yx_fn=combine_yx_reinspect,
        combine_hw_fn=combine_hw_reinspect
    )


# TODO: Rewrite docs
@export
@tf.function
def combine_boxes(
        matched_boxes: tf.Tensor,
        matched_anchors: tf.Tensor,
        combine_yx_fn: callable,
        combine_hw_fn: callable
) -> Tuple[tf.Tensor]:
    r"""
    Combines matched ground truth bounding boxes with matched anchor boxes to
    create ground truth bounding boxes.

    Parameters
    ----------
    matched_boxes
        ``(..., n_dims)`` tensor representing the ground truth bounding boxes.
    matched_anchors
        ``(..., 4)`` tensor representing the anchor boxes.
    combine_yx_fn

    combine_hw_fn

    Returns
    -------
    combines_boxes
        ``(..., n_dims)`` tensor representing ground truth bounding boxes.
    """
    condition = tf.greater(tf.reduce_sum(matched_boxes, axis=-1), 0)

    boxes = matched_boxes[condition]
    matched_anchors = matched_anchors[condition]

    # Compute the yx parametrization
    boxes_yx = combine_yx_fn(boxes, matched_anchors)
    # Compute the hw parametrization
    boxes_hw = combine_hw_fn(boxes, matched_anchors)

    updates = tf.concat([boxes_yx, boxes_hw, boxes[..., 4:]], axis=-1)

    return tf.tensor_scatter_nd_update(
        tensor=matched_boxes,
        indices=tf.where(condition),
        updates=updates
    )


@export
@tf.function
def combine_yx_ssd(boxes: tf.Tensor, anchors: tf.Tensor) -> tf.Tensor:
    r"""
    Combines matched yx coordinates of the ground truth bounding boxes with
    matched anchor boxes to create yx coordinates ground truth bounding boxes
    as defined by the SSD paper.

    Notes
    -----
    This yx coordinate parametrization was first proposed in the Faster RCNN
    paper.

    References
    ----------
    .. [1] Wei Liu, et. al, "SSD: Single Shot MultiBox Detector", ECCV 2016,
           https://arxiv.org/abs/1512.02325.

    Parameters
    ----------
    boxes
        ``(..., n_dims)`` tensor representing the ground truth bounding boxes.
    anchors
        ``(..., 4)`` tensor representing the anchor boxes.

    Returns
    -------
    ssd_boxes_yx
        ``(..., 2)`` tensor representing ground the yx coordinates of the truth
        bounding boxes as defined by the SSD paper.
    """
    return (boxes[..., :2] - anchors[..., :2]) / anchors[..., 2:4]


@export
@tf.function
def combine_yx_yolo(boxes: tf.Tensor, anchors: tf.Tensor) -> tf.Tensor:
    r"""
    Combines matched yx coordinates of the ground truth bounding boxes with
    matched anchor boxes to create yx coordinates ground truth bounding boxes
    as defined by the Yolo paper.

    References
    ----------
    .. [1] Joseph Redmon, et. al, "YOLO9000: Better, Faster, Stronger",
           CVPR 2017, https://arxiv.org/abs/1612.08242.

    Parameters
    ----------
    boxes
        ``(..., n_dims)`` tensor representing the ground truth bounding boxes.
    anchors
        ``(..., 4)`` tensor representing the anchor boxes.

    Returns
    -------
    ssd_boxes_yx
        ``(..., 2)`` tensor representing ground the yx coordinates of the truth
        bounding boxes as defined by the SSD paper.
    """
    return (boxes[..., :2] - anchors[..., :2]) / anchors[..., 2:4] + 0.5


@export
@tf.function
def combine_yx_reinspect(boxes: tf.Tensor, anchors: tf.Tensor) -> tf.Tensor:
    r"""
    Combines matched yx coordinates of the ground truth bounding boxes with
    matched anchor boxes to create yx coordinates ground truth bounding boxes
    as defined by the ReInspect paper.

    References
    ----------
    .. [1] Russell Stewart and Mykhaylo Andriluka, "End-to-end people detection
        in crowded scenes", NIPS 2015. http://arxiv.org/abs/1506.04878.

    Parameters
    ----------
    boxes
        ``(..., n_dims)`` tensor representing the ground truth bounding boxes.
    anchors
        ``(..., 4)`` tensor representing the anchor boxes.

    Returns
    -------
    ssd_boxes_yx
        ``(..., 2)`` tensor representing ground the yx coordinates of the truth
        bounding boxes as defined by the ReInspect paper.
    """
    return boxes[..., :2] - anchors[..., :2]


@export
@tf.function
def combine_hw_sdd(boxes: tf.Tensor, anchors: tf.Tensor) -> tf.Tensor:
    r"""
    Combines matched hw coordinates of the ground truth bounding boxes with
    matched anchor boxes to create hw coordinates ground truth bounding boxes
    as defined by the SSD paper.

    Notes
    -----
    This is the same hw coordinate parametrization used by the Yolo paper.
    This yx coordinate parametrization was first proposed in the Faster RCNN
    paper.

    References
    ----------
    .. [1] Wei Liu, et. al, "SSD: Single Shot MultiBox Detector", ECCV 2016,
        https://arxiv.org/abs/1512.02325.
    .. [2] Joseph Redmon, et. al, "YOLO9000: Better, Faster, Stronger",
        CVPR 2017, https://arxiv.org/abs/1612.08242.
    .. [3] Ren, S., et.al, "Faster R-CNN: Towards real-time object detection
        with region proposal networks", NIPS 2015,
        https://arxiv.org/abs/1506.01497

    Parameters
    ----------
    boxes
        ``(..., n_dims)`` tensor representing the ground truth bounding boxes.
    anchors
        ``(..., 4)`` tensor representing the anchor boxes.

    Returns
    -------
    ssd_boxes_hw
        ``(..., 2)`` tensor representing ground the hw coordinates of the truth
        bounding boxes as defined by the SSD paper.
    """
    return tf.math.log(boxes[..., 2:4] / anchors[..., 2:4])


@export
@tf.function
def combine_hw_reinspect(boxes: tf.Tensor, anchors: tf.Tensor) -> tf.Tensor:
    r"""
    Combines matched hw coordinates of the ground truth bounding boxes with
    matched anchor boxes to create hw coordinates ground truth bounding boxes
    as defined by the SSD paper.

    References
    ----------
    .. [1] Russell Stewart and Mykhaylo Andriluka, "End-to-end people detection
        in crowded scenes", NIPS 2015. http://arxiv.org/abs/1506.04878.

    Parameters
    ----------
    boxes
        ``(..., n_dims)`` tensor representing the ground truth bounding boxes.
    anchors
        ``(..., 4)`` tensor representing the anchor boxes. Note that, this
        parameter is not used in this function and that is only present here
        to conform with the expected interface.

    Returns
    -------
    ssd_boxes_hw
        ``(..., 2)`` tensor representing ground the hw coordinates of the truth
        bounding boxes as defined by the SSD paper.
    """
    return boxes[..., 2:4]


# TODO: Rewrite docs
@export
@tf.function
def extract_boxes_ssd(
        matched_boxes: tf.Tensor,
        matched_anchors: tf.Tensor,
) -> Tuple[tf.Tensor]:
    r"""
    Combines matched ground truth bounding boxes with matched anchor boxes to
    create ground truth bounding boxes as defined by the SSD paper.

    Notes
    -----
    This ground truth bounding boxes parametrization was first proposed in
    the Faster RCNN paper.

    References
    ----------
    .. [1] Wei Liu, et. al, "SSD: Single Shot MultiBox Detector", ECCV 2016,
           https://arxiv.org/abs/1512.02325.
    .. [2] Ren, S., et.al, "Faster R-CNN: Towards real-time object detection
           with region proposal networks", NIPS 2015,
           https://arxiv.org/abs/1506.01497

    Parameters
    ----------
    matched_boxes
        ``(..., n_dims)`` tensor representing the ground truth bounding boxes.
    matched_anchors
        ``(..., 4)`` tensor representing the anchor boxes.

    Returns
    -------
    ssd_boxes
        ``(..., n_dims)`` tensor representing ground truth bounding boxes as
        defined by the SSD paper.
    """
    return extract_boxes(
        matched_boxes=matched_boxes,
        matched_anchors=matched_anchors,
        extract_yx_fn=extract_yx_ssd,
        extract_hw_fn=extract_hw_sdd
    )


# TODO: Rewrite docs
@export
@tf.function
def extract_boxes_yolo(
        matched_boxes: tf.Tensor,
        matched_anchors: tf.Tensor,
) -> Tuple[tf.Tensor]:
    r"""
    Combines matched ground truth bounding boxes with matched anchor boxes to
    create ground truth bounding boxes as defined by the Yolo paper.

    References
    ----------
    .. [1] Joseph Redmon, et. al, "YOLO9000: Better, Faster, Stronger",
           CVPR 2017, https://arxiv.org/abs/1612.08242.

    Parameters
    ----------
    matched_boxes
        ``(..., n_dims)`` tensor representing the ground truth bounding boxes.
    matched_anchors
        ``(..., 4)`` tensor representing the anchor boxes.

    Returns
    -------
    yolo_boxes
        ``(..., n_dims)`` tensor representing ground truth bounding boxes as
        defined by the Yolo paper.
    """
    return extract_boxes(
        matched_boxes=matched_boxes,
        matched_anchors=matched_anchors,
        extract_yx_fn=extract_yx_yolo,
        extract_hw_fn=extract_hw_sdd
    )


# TODO: Rewrite docs
@export
@tf.function
def extract_boxes_reinspect(
        matched_boxes: tf.Tensor,
        matched_anchors: tf.Tensor,
) -> Tuple[tf.Tensor]:
    r"""
    Combines matched ground truth bounding boxes with matched anchor boxes to
    create ground truth bounding boxes as defined by the Yolo paper.

    References
    ----------
    .. [1] Joseph Redmon, et. al, "YOLO9000: Better, Faster, Stronger",
           CVPR 2017, https://arxiv.org/abs/1612.08242.

    Parameters
    ----------
    matched_boxes
        ``(..., n_dims)`` tensor representing the ground truth bounding boxes.
    matched_anchors
        ``(..., 4)`` tensor representing the anchor boxes.

    Returns
    -------
    yolo_boxes
        ``(..., n_dims)`` tensor representing ground truth bounding boxes as
        defined by the Yolo paper.
    """
    return extract_boxes(
        matched_boxes=matched_boxes,
        matched_anchors=matched_anchors,
        extract_yx_fn=extract_yx_reinspect,
        extract_hw_fn=extract_hw_reinspect
    )


# TODO: Rewrite docs
@export
@tf.function
def extract_boxes(
        matched_boxes: tf.Tensor,
        matched_anchors: tf.Tensor,
        extract_yx_fn: callable,
        extract_hw_fn: callable
) -> Tuple[tf.Tensor]:
    r"""
    Combines matched ground truth bounding boxes with matched anchor boxes to
    create ground truth bounding boxes.

    Parameters
    ----------
    matched_boxes
        ``(..., n_dims)`` tensor representing the ground truth bounding boxes.
    matched_anchors
        ``(..., 4)`` tensor representing the anchor boxes.
    extract_yx_fn

    extract_hw_fn

    Returns
    -------
    combines_boxes
        ``(..., n_dims)`` tensor representing ground truth bounding boxes.
    """
    condition = tf.greater(tf.reduce_sum(matched_boxes, axis=-1), 0)

    boxes = matched_boxes[condition]
    matched_anchors = matched_anchors[condition]

    # Compute the yx parametrization
    boxes_yx = extract_yx_fn(boxes, matched_anchors)
    # Compute the hw parametrization
    boxes_hw = extract_hw_fn(boxes, matched_anchors)

    return tf.concat([boxes_yx, boxes_hw, boxes[..., 4:]], axis=-1)


# TODO: Rewrite docs
@export
@tf.function
def extract_yx_ssd(boxes: tf.Tensor, anchors: tf.Tensor) -> tf.Tensor:
    r"""
    Combines matched yx coordinates of the ground truth bounding boxes with
    matched anchor boxes to create yx coordinates ground truth bounding boxes
    as defined by the SSD paper.

    Notes
    -----
    This yx coordinate parametrization was first proposed in the Faster RCNN
    paper.

    References
    ----------
    .. [1] Wei Liu, et. al, "SSD: Single Shot MultiBox Detector", ECCV 2016,
           https://arxiv.org/abs/1512.02325.

    Parameters
    ----------
    boxes
        ``(..., n_dims)`` tensor representing the ground truth bounding boxes.
    anchors
        ``(..., 4)`` tensor representing the anchor boxes.

    Returns
    -------
    ssd_boxes_yx
        ``(..., 2)`` tensor representing ground the yx coordinates of the truth
        bounding boxes as defined by the SSD paper.
    """
    return boxes[..., :2] * anchors[..., 2:4] + anchors[..., :2]


# TODO: Rewrite docs
@export
@tf.function
def extract_yx_yolo(boxes: tf.Tensor, anchors: tf.Tensor) -> tf.Tensor:
    r"""
    Combines matched yx coordinates of the ground truth bounding boxes with
    matched anchor boxes to create yx coordinates ground truth bounding boxes
    as defined by the Yolo paper.

    References
    ----------
    .. [1] Joseph Redmon, et. al, "YOLO9000: Better, Faster, Stronger",
           CVPR 2017, https://arxiv.org/abs/1612.08242.

    Parameters
    ----------
    boxes
        ``(..., n_dims)`` tensor representing the ground truth bounding boxes.
    anchors
        ``(..., 4)`` tensor representing the anchor boxes.

    Returns
    -------
    ssd_boxes_yx
        ``(..., 2)`` tensor representing ground the yx coordinates of the truth
        bounding boxes as defined by the SSD paper.
    """
    return (boxes[..., :2] - 0.5) * anchors[..., 2:4] + anchors[..., :2]

# TODO: Rewrite docs
@export
@tf.function
def extract_yx_reinspect(
        boxes: tf.Tensor,
        anchors: tf.Tensor
) -> tf.Tensor:
    r"""
    Combines matched yx coordinates of the ground truth bounding boxes with
    matched anchor boxes to create yx coordinates ground truth bounding boxes
    as defined by the ReInspect paper.

    References
    ----------
    .. [1] Russell Stewart and Mykhaylo Andriluka, "End-to-end people detection
        in crowded scenes", NIPS 2015. http://arxiv.org/abs/1506.04878.

    Parameters
    ----------
    boxes
        ``(..., n_dims)`` tensor representing the ground truth bounding boxes.
    anchors
        ``(..., 4)`` tensor representing the anchor boxes.

    Returns
    -------
    ssd_boxes_yx
        ``(..., 2)`` tensor representing ground the yx coordinates of the truth
        bounding boxes as defined by the ReInspect paper.
    """
    return boxes[..., :2] + anchors[..., :2]


# TODO: Rewrite docs
@export
@tf.function
def extract_hw_sdd(boxes: tf.Tensor, anchors: tf.Tensor) -> tf.Tensor:
    r"""
    Combines matched hw coordinates of the ground truth bounding boxes with
    matched anchor boxes to create hw coordinates ground truth bounding boxes
    as defined by the SSD paper.

    Notes
    -----
    This is the same hw coordinate parametrization used by the Yolo paper.
    This yx coordinate parametrization was first proposed in the Faster RCNN
    paper.

    References
    ----------
    .. [1] Wei Liu, et. al, "SSD: Single Shot MultiBox Detector", ECCV 2016,
        https://arxiv.org/abs/1512.02325.
    .. [2] Joseph Redmon, et. al, "YOLO9000: Better, Faster, Stronger",
        CVPR 2017, https://arxiv.org/abs/1612.08242.
    .. [3] Ren, S., et.al, "Faster R-CNN: Towards real-time object detection
        with region proposal networks", NIPS 2015,
        https://arxiv.org/abs/1506.01497

    Parameters
    ----------
    boxes
        ``(..., n_dims)`` tensor representing the ground truth bounding boxes.
    anchors
        ``(..., 4)`` tensor representing the anchor boxes.

    Returns
    -------
    ssd_boxes_hw
        ``(..., 2)`` tensor representing ground the hw coordinates of the truth
        bounding boxes as defined by the SSD paper.
    """
    return tf.math.exp(boxes[..., 2:4]) * anchors[..., 2:4]


# TODO: Rewrite docs
@export
@tf.function
def extract_hw_reinspect(
        boxes: tf.Tensor,
        anchors: tf.Tensor
) -> tf.Tensor:
    r"""
    Combines matched hw coordinates of the ground truth bounding boxes with
    matched anchor boxes to create hw coordinates ground truth bounding boxes
    as defined by the SSD paper.

    References
    ----------
    .. [1] Russell Stewart and Mykhaylo Andriluka, "End-to-end people detection
        in crowded scenes", NIPS 2015. http://arxiv.org/abs/1506.04878.

    Parameters
    ----------
    boxes
        ``(..., n_dims)`` tensor representing the ground truth bounding boxes.
    anchors
        ``(..., 4)`` tensor representing the anchor boxes. Note that, this
        parameter is not used in this function and that is only present here
        to conform with the expected interface.

    Returns
    -------
    ssd_boxes_hw
        ``(..., 2)`` tensor representing ground the hw coordinates of the truth
        bounding boxes as defined by the SSD paper.
    """
    return boxes[..., 2:4]


# TODO: Improve docs
@export
def unmatch_boxes(matched_boxes: tf.Tensor) -> tf.Tensor:
    r"""
    Extracts real valid bounding boxes from matched boxes.

    Parameters
    ----------
    matched_boxes
        ``(height, width, n_anchors, n_dims)`` tensor representing bounding
        boxes that have been successfully matched with anchor boxes.

    Returns
    -------
    matched_boxes
        ``(..., n_dims)`` tensor containing the original bounding boxes that
        were successfully matched with the anchor boxes.
    """
    return matched_boxes[matched_boxes[..., 4] > 0]

