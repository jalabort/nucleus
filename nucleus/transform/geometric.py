from typing import Union, Tuple, Sequence

import tensorflow as tf

from nucleus.box import (
    pad_tensor, unpad_tensor, filter_boxes, flip_boxes_left_right,
    ijhw_to_yxhw, scale_coords
)
from nucleus.utils import export, tf_get_shape

from .base import DeterministicTransform, RandomTransform


@export
class HorizontalFlip(DeterministicTransform):
    r"""
    Callable class for horizontally flipping images and bounding boxes.
    """
    n_factors = 0

    def _operation(
            self,
            image: tf.Tensor,
            boxes: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        r"""
        Horizontally flips the given image and, if provided, its bounding boxes.

        Parameters
        ----------
        image
            The image to be horizontally flipped.
        boxes
            The boxes to be horizontally flipped.

        Returns
        -------
        flipped_image
            The horizontally flipped image.
        flipped_boxes
            The horizontally flipped boxes.
        """
        max_boxes = tf_get_shape(boxes)[0]
        boxes = unpad_tensor(boxes)
        image = tf.image.flip_left_right(image)
        boxes = flip_boxes_left_right(boxes)
        boxes = pad_tensor(boxes, max_length=max_boxes)
        return image, boxes


@export
class Zoom(DeterministicTransform):
    r"""
    Callable class for zooming, in and out, images and bounding boxes.
    """
    n_factors = 2

    def __init__(self, pad: bool = True) -> None:
        self.pad = pad

    def _operation(
            self,
            image: tf.Tensor,
            boxes: tf.Tensor,
            zoom_factor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        r"""
        Zooms in or out the given image and/or bounding boxes by the given
        factor.

        Parameters
        ----------
        image
            The image to be zoomed.
        boxes
            The boxes to be zoomed.
        zoom_factor


        Returns
        -------
        zoomed_image
            The zoomed image.
        zoomed_boxes
            The zoomed boxes.
        """
        max_boxes = tf_get_shape(boxes)[0]
        boxes = unpad_tensor(boxes)

        image_shape = tf_get_shape(image)
        height, width, _ = image_shape

        new_height = tf.cast(height, tf.float32) * zoom_factor[0]
        new_width = tf.cast(width, tf.float32) * zoom_factor[1]

        image = tf.image.resize(image, size=(new_height, new_width))
        image = tf.image.resize_with_crop_or_pad(image, height, width)

        ijhw_tensor = boxes[..., :4] * tf.convert_to_tensor(
            [zoom_factor[0], zoom_factor[1]] * 2
        )
        offsets = tf.stack(
            [(1.0 - zoom_factor[0]) / 2, (1.0 - zoom_factor[1]) / 2]
        )
        boxes = tf.concat(
            [
                ijhw_tensor[..., :2] + offsets,
                ijhw_tensor[..., 2:],
                boxes[..., 4:]
            ],
            axis=-1
        )

        boxes = filter_boxes(boxes, pad=self.pad)

        boxes = tf.cond(
            tf.equal(self.pad, True),
            lambda: pad_tensor(boxes, max_length=max_boxes),
            lambda: boxes
        )

        return image, boxes


@export
class RandomZoom(RandomTransform):
    r"""
    Callable class for randomly zooming, in and out, images and bounding boxes.

    Parameters
    ----------
    pad

    min_factor

    max_factor

    """
    def __init__(
            self,
            pad: bool = True,
            min_factor: float = 0.25,
            max_factor: float = 1.25
    ) -> None:
        super().__init__(
            transform=Zoom(pad=pad),
            min_factor=min_factor,
            max_factor=max_factor
        )


@export
class JitterBoxes(DeterministicTransform):
    r"""
    Callable class for jittering the bounding boxes associated to an images.
    """
    n_factors = -1

    def _operation(
            self,
            image: tf.Tensor,
            boxes: tf.Tensor,
            jitter_factors: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        r"""
        Jitters the given  bounding boxes by a the given factors.

        Parameters
        ----------
        image
            The image to which the bounding boxes are associated.
        boxes
            The bounding boxes to be jittered.
        jitter_factors

        Returns
        -------
        image
            The image to which the bounding boxes are associated unchanged.
        jittered_boxes
            The jittered bounding boxes.
        """
        max_boxes = tf_get_shape(boxes)[0]

        jitter = tf.concat(
            [boxes[..., 2:4], boxes[..., 2:4]], axis=-1
        ) * jitter_factors
        ijhw_tensor = boxes[..., :4] + jitter
        boxes = tf.concat([ijhw_tensor, boxes[..., 4:]], axis=-1)

        boxes = unpad_tensor(boxes, padding_value=0, boolean_fn=tf.less)
        boxes = pad_tensor(boxes, max_length=max_boxes)

        return image, boxes


@export
class RandomJitterBoxes(RandomTransform):
    r"""
    Callable class for randomly jittering the bounding boxes associated to
    an images.

    Parameters
    ----------
    min_factor

    max_factor

    """
    def __init__(
            self,
            min_factor: float = -0.05,
            max_factor: float = 0.05
    ) -> None:
        super().__init__(
            transform=JitterBoxes(),
            min_factor=min_factor,
            max_factor=max_factor
        )


# TODO: Update to use TF2.0 control flow operations
@export
class CropAroundBox(DeterministicTransform):
    r"""
    Callable class for ...

    Parameters
    ----------
    size

    pad

    """
    n_factors = -3

    def __init__(
            self,
            size: Union[int, Sequence[int], tf.Tensor],
            pad: bool = True
    ) -> None:
        if isinstance(size, int):
            size = [size, size]
        if not isinstance(size, tf.Tensor):
            size = tf.convert_to_tensor(size, dtype=tf.float32)
        tf.assert_equal(tf.equal(tf_get_shape(size)[0], 2), True)

        self.size = size
        self.pad = pad

    def _operation(
            self,
            image: tf.Tensor,
            boxes: tf.Tensor,
            box_index: int
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        r"""


        Parameters
        ----------
        image

        boxes

        box_index

        Returns
        -------
        image

        boxes

        """
        if not isinstance(box_index, tf.Tensor):
            box_index = tf.convert_to_tensor(box_index)
        box_index = tf.round(box_index)
        box_index = tf.cast(box_index, dtype=tf.int32)

        max_boxes = tf_get_shape(boxes)[0]
        boxes = unpad_tensor(boxes)

        height, width, _ = tf_get_shape(image)
        resolution = tf.convert_to_tensor([height, width], dtype=tf.float32)

        ijhw_tensor = scale_coords(boxes[..., :4], resolution)
        yxhw = ijhw_to_yxhw(ijhw_tensor[box_index])

        offset_size = yxhw - tf.concat([self.size, self.size], axis=-1) / 2
        target_size = tf.concat([self.size, self.size], axis=-1)

        offset_height, target_height = tf.cond(
            tf.less(offset_size[0], 0),
            lambda: (
                tf.constant(0, dtype=tf.float32),
                target_size[0]
            ),
            lambda: (offset_size[0], target_size[0])
        )
        offset_height = tf.cast(offset_height, tf.float32)
        target_height = tf.cast(target_height, tf.float32)

        offset_width, target_width = tf.cond(
            tf.less(offset_size[1], 0),
            lambda: (
                tf.constant(0, dtype=tf.float32),
                target_size[1]
            ),
            lambda: (offset_size[1], target_size[1])
        )
        offset_width = tf.cast(offset_width, tf.float32)
        target_width = tf.cast(target_width, tf.float32)

        offset_height, target_height = tf.cond(
            tf.greater_equal(
                offset_height + target_height,
                tf.cast(height, dtype=tf.float32)
            ),
            lambda: (
                offset_height - (tf.cast(height, dtype=tf.float32) - target_height),
                target_size[0]
            ),
            lambda: (offset_height, target_height)
        )

        offset_width, target_width = tf.cond(
            tf.greater_equal(
                offset_width + target_width,
                tf.cast(width, dtype=tf.float32)
            ),
            lambda: (
                offset_width - (tf.cast(width, dtype=tf.float32) - target_width),
                target_size[1]
            ),
            lambda: (offset_width, target_width)
        )

        image = tf.image.crop_to_bounding_box(
            image,
            tf.cast(offset_height, dtype=tf.int32),
            tf.cast(offset_width, dtype=tf.int32),
            tf.cast(target_height, dtype=tf.int32),
            tf.cast(target_width, dtype=tf.int32)
        )

        offset = tf.convert_to_tensor(
            [offset_height, offset_width],
            dtype=tf.float32
        )
        ij_tensor = ijhw_tensor[..., :2] - offset

        ijhw_tensor = tf.concat(
            [ij_tensor, ijhw_tensor[..., 2:4]], axis=-1
        )

        scale = tf.convert_to_tensor(
            [target_height, target_width],
            dtype=tf.float32
        )
        ijhw_tensor = ijhw_tensor / tf.concat([scale, scale], axis=-1)

        boxes = tf.concat([ijhw_tensor, boxes[..., 4:]], axis=-1)

        boxes = filter_boxes(boxes, pad=self.pad)

        boxes = tf.cond(
            tf.equal(self.pad, True),
            lambda: pad_tensor(boxes, max_length=max_boxes),
            lambda: boxes
        )

        return image, boxes


@export
class RandomCropAroundBox(RandomTransform):
    r"""
    Callable class for randomly ...

    Parameters
    ----------
    size

    pad
    """
    def __init__(
            self,
            size: Union[int, Sequence[int], tf.Tensor],
            pad: bool = True
    ) -> None:
        super().__init__(
            transform=CropAroundBox(size=size, pad=pad),
            min_factor=0,
            max_factor=None
        )
