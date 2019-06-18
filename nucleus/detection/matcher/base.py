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
class Matcher:
    r"""

    Parameters
    ----------
    match_boxes_fn
    combine_boxes_fn
    extract_boxes_fn
    """
    def __init__(
            self,
            match_boxes_fn: callable,
            combine_boxes_fn: callable,
            extract_boxes_fn: callable
    ) -> None:
        self.match_boxes_fn = match_boxes_fn
        self.combine_boxes_fn = combine_boxes_fn
        self.extract_boxes_fn = extract_boxes_fn

    # TODO: docstrings
    @name_scope
    # @tf.function
    def unmatch(
            self,
            matched_boxes: tf.Tensor,
            anchors: tf.Tensor
    ) -> tf.Tensor:
        r"""

        Parameters
        ----------
        matched_boxes
            ``(height, width, n_anchors, n_dims)`` tensor representing the
            matched bounding boxes.
        anchors
            ``(height, width, n_anchors, 4)`` tensor representing the
            anchor boxes.

        Returns
        -------
        ``(n_boxes, n_dims)`` tensor representing the bounding boxes.
        """
        boxes = self.extract_boxes_fn(
            matched_boxes=matched_boxes,
            matched_anchors=ijhw_to_yxhw(anchors)
        )
        return yxhw_to_ijhw(boxes)

    # TODO: docstrings
    @name_scope
    # @tf.function
    def match(self, boxes: tf.Tensor, anchors: tf.Tensor) -> tf.Tensor:
        r"""

        Parameters
        ----------
        boxes
            ``(n_boxes, n_dims)`` tensor representing the bounding boxes.
        anchors
            ``(height, width, n_anchors, 4)`` tensor representing the
            anchor boxes.

        Returns
        -------
        matched_boxes
            ``(height, width, n_anchors, n_dims)`` tensor representing the
            matched bounding boxes.
        """
        matched_boxes, matched_anchors = self.match_boxes_fn(
            boxes=boxes,
            anchors=anchors
        )
        return self.combine_boxes_fn(matched_boxes, matched_anchors)


# TODO: Add max_box_size as an argument
@export
class SsdMatcher(Matcher):
    r"""

    Parameters
    ----------
    iou_threshold
        The iou threshold above which ground truth bounding boxes are
        associated with anchor boxes.
    """
    def __init__(self, iou_threshold: float = 0.75) -> None:
        super().__init__(
            match_boxes_fn=partial(
                match_boxes_iou,
                iou_threshold=iou_threshold
            ),
            combine_boxes_fn=combine_boxes_ssd,
            extract_boxes_fn=extract_boxes_ssd
        )


# TODO: Add max_box_size as an argument
@export
class YoloMatcher(Matcher):
    r"""

    Parameters
    ----------
    iou_threshold
        The iou threshold above which ground truth bounding boxes are
        associated with anchor boxes.
    """
    def __init__(self, iou_threshold: float = 0.75) -> None:
        super().__init__(
            match_boxes_fn=partial(
                match_boxes_iou,
                iou_threshold=iou_threshold
            ),
            combine_boxes_fn=combine_boxes_yolo,
            extract_boxes_fn=extract_boxes_yolo
        )


@export
class ReInspectMatcher(Matcher):
    r"""

    Parameters
    ----------
    focus_factor
        Determines the area of the original images that is assigned to each
        grid cell. If >1 there is overlap between cells.
    max_box_size
        Determines the maximum size that a bounding box can have in order to
        be associated to a grid cell.
    """
    def __init__(
            self,
            focus_factor: Union[float, Tuple[float, float]] = 1.0,
            max_box_size: Union[float, Tuple[float, float]] = 0.2
    ) -> None:
        def _match_boxes_fn(boxes, anchors):
            return match_boxes_distance(
                boxes=boxes,
                anchors=anchors,
                max_distance=tf.cast(
                    0.5 * focus_factor * (1 / tf.shape(anchors)[:2]),
                    dtype=tf.float32
                ),
                max_box_size=max_box_size
            )
        super().__init__(
            match_boxes_fn=_match_boxes_fn,
            combine_boxes_fn=combine_boxes_reinspect,
            extract_boxes_fn=extract_boxes_reinspect
        )

