from typing import Tuple

import tensorflow as tf
import tensorflow_addons as tfa
from math import pi

from nucleus.box import filter_boxes, ijhw_to_ijkl, ijkl_to_ijhw, scale_coords
from nucleus.utils import export, tf_get_shape

from .base import DeterministicTransform, RandomTransform


@export
class Pan(DeterministicTransform):
    r"""
    Callable class for panning images and bounding boxes.
    """
    n_factors = 2

    def __init__(self, pad: bool = True) -> None:
        self.pad = pad

    def _operation(
            self,
            image: tf.Tensor,
            boxes: tf.Tensor,
            pan_factor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        r"""
        Pans the given image and/or bounding boxes by a random amount.

        Parameters
        ----------
        image
            The image to be panned.
        boxes
            The boxes to be panned.
        pan_factor


        Returns
        -------
        panned_image
            The panned image.
        panned_boxes
            The panned boxes.
        """
        height, width, _ = tf_get_shape(image)
        dy = tf.cast(height, dtype=tf.float32) * pan_factor[0]
        dx = tf.cast(width, dtype=tf.float32) * pan_factor[1]
        image = tfa.image.transform(
            image,
            [1, 0, -dx, 0, 1, -dy, 0, 0],
            interpolation='BILINEAR',
        )

        if self.valid_boxes(boxes=boxes):
            offsets = tf.stack([pan_factor[0], pan_factor[1]])
            boxes = tf.concat(
                [
                    boxes[..., :2] + offsets,
                    boxes[..., 2:4],
                    boxes[..., 4:]
                ],
                axis=-1
            )
            boxes = filter_boxes(boxes, pad=self.pad)

        return image, boxes


@export
class RandomPan(RandomTransform):
    r"""
    Callable class for randomly panning images and bounding boxes.

    Parameters
    ----------
    min_factor

    max_factor

    pad

    """
    def __init__(
            self,
            min_factor: float = -0.1,
            max_factor: float = 0.1,
            pad: bool = True
    ) -> None:
        super().__init__(
            transform=Pan(pad=pad),
            min_factor=min_factor,
            max_factor=max_factor
        )


@export
class Rotate(DeterministicTransform):
    r"""
    Callable class for rotating images and bounding boxes.
    """
    n_factors = 1

    def __init__(self, pad: bool = True) -> None:
        self.pad = pad

    def _operation(
            self,
            image: tf.Tensor,
            boxes: tf.Tensor,
            angle_factor: float
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        r"""
        Rotates the given image and/or bounding boxes by a random amount.

        Parameters
        ----------
        image
            The image to be rotated.
        boxes
            The boxes to be rotated.
        angle_factor

        Returns
        -------
        rotated_image
            The rotated image.
        rotated_boxes
            The rotated boxes.
        """
        angle = (pi / 180) * angle_factor
        image = tfa.image.rotate(image, angle, interpolation='BILINEAR')

        if self.valid_boxes(boxes=boxes):
            height, width, _ = tf_get_shape(image)
            image_shape = tf.stack([height, width])

            image_center = tf.cast(image_shape, dtype=tf.float32) / 2
            image_centers = tf.tile(image_center, [2])

            centered_coords = ijhw_to_ijkl(
                scale_coords(boxes[..., :4], image_shape)
            ) - image_centers

            tl = centered_coords[..., 0:2]
            tr = tf.stack([
                centered_coords[..., 0], centered_coords[..., 3]
            ], axis=-1)
            bl = tf.stack([
                centered_coords[..., 2], centered_coords[..., 1]
            ], axis=-1)
            br = centered_coords[..., 2:4]

            rotated_tl = tf.stack([
                tf.cos(angle) * tl[..., 0] - tf.sin(angle) * tl[..., 1],
                tf.sin(angle) * tl[..., 0] + tf.cos(angle) * tl[..., 1]
            ], axis=-1)
            rotated_tr = tf.stack([
                tf.cos(angle) * tr[..., 0] - tf.sin(angle) * tr[..., 1],
                tf.sin(angle) * tr[..., 0] + tf.cos(angle) * tr[..., 1]
            ], axis=-1)
            rotated_bl = tf.stack([
                tf.cos(angle) * bl[..., 0] - tf.sin(angle) * bl[..., 1],
                tf.sin(angle) * bl[..., 0] + tf.cos(angle) * bl[..., 1]
            ], axis=-1)
            rotated_br = tf.stack([
                tf.cos(angle) * br[..., 0] - tf.sin(angle) * br[..., 1],
                tf.sin(angle) * br[..., 0] + tf.cos(angle) * br[..., 1]
            ], axis=-1)

            rotated_boxes = tf.concat([
                rotated_tl,
                rotated_tr,
                rotated_bl,
                rotated_br
            ], axis=-1)

            ys = rotated_boxes[..., 0::2]
            xs = rotated_boxes[..., 1::2]

            aligned_tl = tf.stack([
                tf.reduce_min(ys, axis=-1),
                tf.reduce_min(xs, axis=-1)
            ], axis=-1)
            aligned_br = tf.stack([
                tf.reduce_max(ys, axis=-1),
                tf.reduce_max(xs, axis=-1)
            ], axis=-1)

            aligned_coords = tf.concat([aligned_tl, aligned_br], axis=-1)

            boxes = tf.concat([
                ijkl_to_ijhw(aligned_coords + image_centers),
                boxes[..., 4:]
            ], axis=-1)
            boxes = scale_coords(boxes, 1 / image_shape)
            boxes = filter_boxes(boxes, pad=self.pad)

        return image, boxes


@export
class RandomRotate(RandomTransform):
    r"""
    Callable class for randomly rotating images and bounding boxes.

    Parameters
    ----------
    max_angle

    pad
    """
    def __init__(
            self,
            max_angle: float = 10,
            pad: bool = True
    ) -> None:
        super().__init__(
            transform=Rotate(pad=pad),
            min_factor=-max_angle,
            max_factor=max_angle
        )
