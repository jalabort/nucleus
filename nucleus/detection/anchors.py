from typing import Union, Sequence, Dict

import tensorflow as tf
from public import public


@public
class AnchorParameters:
    """
    The parameters that define how anchor boxes are generated.

    Parameters
    ----------
    scales
        Scales to be used at each location in the feature map grid. Scales
        are relative to the size of cells of the feature map.
    ratios
        Ratios to be used at each location in the feature map grid. Ratios
        are relative to the size of cells of the feature map.
    """

    def __init__(
            self,
            scales: Union[float, Sequence[float], tf.Tensor],
            ratios: Union[float, Sequence[float], tf.Tensor]
    ) -> None:
        if isinstance(scales, (int, float)):
            scales = [scales]
        if isinstance(ratios, (int, float)):
            ratios = [ratios]
        self.scales = scales
        self.ratios = ratios

    @property
    def n_anchors(self):
        return len(self.scales) * len(self.ratios)


yolo_anchor_parameters = AnchorParameters(scales=[1, 3, 4], ratios=[.5, 1, 2])
public(yolo_anchor_parameters)


# TODO: Document me!
@public
def create_anchors(
        # anchor_parameters: AnchorParameters,
        scales: Union[float, Sequence[float], tf.Tensor],
        ratios: Union[float, Sequence[float], tf.Tensor],
        n_anchors: int,
        grid_height: Union[float, tf.Tensor],
        grid_width: Union[float, tf.Tensor],
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
        The parameters that define how anchor boxes are generated.
    ratios

    n_anchors

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
        ``(height, width, n_anchors, 6)`` or ``(height * width * n_anchors, 4)``
        tensor representing the anchor boxes.
    """
    if not isinstance(grid_height, tf.Tensor):
        grid_height = tf.convert_to_tensor(grid_height)
    if not isinstance(grid_width, tf.Tensor):
        grid_width = tf.convert_to_tensor(grid_width)
    grid_height = tf.cast(grid_height, dtype=tf.float32)
    grid_width = tf.cast(grid_width, dtype=tf.float32)

    # scales = anchor_parameters.scales
    # ratios = anchor_parameters.ratios
    # n_anchors = anchor_parameters.n_anchors
    if not isinstance(scales, tf.Tensor):
        scales = tf.convert_to_tensor(scales)
    if not isinstance(ratios, tf.Tensor):
        ratios = tf.convert_to_tensor(ratios)
    scales = tf.cast(scales, dtype=tf.float32)
    ratios = tf.cast(ratios, dtype=tf.float32)

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

    cells_h = cell_h * tf.ones_like(anchors_i)
    cells_w = cell_w * tf.ones_like(anchors_i)

    # Create the grid of anchors
    anchors = tf.stack([
        anchors_i, anchors_j, anchors_h, anchors_w, cells_h, cells_w
    ], axis=-1)

    if flatten:
        anchors = tf.reshape(anchors, shape=(-1, 4))

    return anchors
