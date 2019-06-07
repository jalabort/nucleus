from typing import Union, Tuple

import tensorflow as tf
from functools import partial

from nucleus.box import yxhw_to_ijhw, ijhw_to_yxhw
from nucleus.utils import export, name_scope

from .tools import (
    match_boxes_iou, match_boxes_distance, combine_boxes_ssd,
    combine_boxes_yolo, combine_boxes_reinspect,
    extract_boxes_ssd, extract_boxes_yolo,
    extract_boxes_reinspect,
)


# TODO: docstrings
@export
class AnchorMatcher:
    r"""

    Parameters
    ----------
    anchors
        ``(height, width, n_anchors, 4)`` tensor representing the anchor boxes.
    match_boxes_fn
    combine_boxes_fn
    extract_boxes_fn
    """
    def __init__(
            self,
            anchors: tf.Tensor,
            match_boxes_fn: callable,
            combine_boxes_fn: callable,
            extract_boxes_fn: callable
    ) -> None:
        self.anchors = anchors
        self.match_boxes_fn = match_boxes_fn
        self.combine_boxes_fn = combine_boxes_fn
        self.extract_boxes_fn = extract_boxes_fn

    # TODO: docstrings
    @name_scope
    @tf.function
    def _unmatch(
            self,
            image: tf.Tensor,
            boxes: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        r"""

        Parameters
        ----------
        image
        boxes

        Returns
        -------

        """
        boxes = self.extract_boxes_fn(
            matched_boxes=boxes,
            matched_anchors=ijhw_to_yxhw(self.anchors)
        )
        return image, yxhw_to_ijhw(boxes)

    # TODO: docstrings
    @name_scope
    @tf.function
    def __call__(
            self,
            image: tf.Tensor,
            boxes: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        r"""

        Parameters
        ----------
        image
        boxes

        Returns
        -------

        """
        matched_boxes, matched_anchors = self.match_boxes_fn(
            boxes=boxes,
            anchors=self.anchors
        )
        return image, self.combine_boxes_fn(matched_boxes, matched_anchors)


# TODO: Add max_box_size as an argument
@export
class SsdAnchorMatcher(AnchorMatcher):
    r"""

    Parameters
    ----------
    anchors
        ``(height, width, n_anchors, 4)`` tensor representing the anchor boxes.
    iou_threshold
        The iou threshold above which ground truth bounding boxes are
        associated with anchor boxes.
    """
    def __init__(
            self,
            anchors: tf.Tensor,
            iou_threshold: float = 0.75
    ) -> None:
        super().__init__(
            anchors=anchors,
            match_boxes_fn=partial(
                match_boxes_iou,
                iou_threshold=iou_threshold
            ),
            combine_boxes_fn=combine_boxes_ssd,
            extract_boxes_fn=extract_boxes_ssd
        )


# TODO: Add max_box_size as an argument
@export
class YoloAnchorMatcher(AnchorMatcher):
    r"""

    Parameters
    ----------
    anchors
        `(height, width, n_anchors, 4)`` tensor representing the anchor boxes.
    iou_threshold
        The iou threshold above which ground truth bounding boxes are
        associated with anchor boxes.
    """
    def __init__(
            self,
            anchors: tf.Tensor,
            iou_threshold: float = 0.75
    ) -> None:
        super().__init__(
            anchors=anchors,
            match_boxes_fn=partial(
                match_boxes_iou,
                iou_threshold=iou_threshold
            ),
            combine_boxes_fn=combine_boxes_yolo,
            extract_boxes_fn=extract_boxes_yolo
        )


@export
class ReInspectAnchorMatcher(AnchorMatcher):
    r"""

    Parameters
    ----------
    anchors
        `(height, width, n_anchors, 4)`` tensor representing the anchor boxes.
    focus_factor
        Determines the area of the original images that is assigned to each
        grid cell. If >1 there is overlap between cells.
    max_box_size
        Determines the maximum size that a bounding box can have in order to
        be associated to a grid cell.
    """
    def __init__(
            self,
            anchors: tf.Tensor,
            focus_factor: Union[float, Tuple[float, float]] = 1.0,
            max_box_size: Union[float, Tuple[float, float]] = 0.2
    ) -> None:
        super().__init__(
            anchors=anchors,
            match_boxes_fn=partial(
                match_boxes_distance,
                max_distance=tf.cast(
                    0.5 * focus_factor * (1 / tf.shape(anchors)[:2]),
                    dtype=tf.float32
                ),
                max_box_size=max_box_size
            ),
            combine_boxes_fn=combine_boxes_reinspect,
            extract_boxes_fn=extract_boxes_reinspect
        )

