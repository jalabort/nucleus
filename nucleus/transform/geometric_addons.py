from typing import Tuple

import tensorflow as tf
import tensorflow_addons as tfa
from math import pi
from public import public

from nucleus.box import filter_boxes, ijhw_to_ijkl, ijkl_to_ijhw
from nucleus.utils import tf_get_shape

from .base import DeterministicTransform, RandomTransform


@public
class Pan(DeterministicTransform):
    r"""
    Callable class for panning images and bounding boxes.
    """
    n_factors = 2

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
            [1, 0, dx, 0, 1, dy, 0, 0],
            interpolation='BILINEAR',
        )

        offsets = tf.stack([dy, dx])
        boxes = tf.concat(
            [
                boxes[..., :2] + offsets,
                boxes[..., 2:4],
                boxes[..., 4:]
            ],
            axis=-1
        )
        boxes = filter_boxes(boxes)

        return image, boxes


@public
class RandomPan(RandomTransform):
    r"""
    Callable class for randomly panning images and bounding boxes.

    Parameters
    ----------
    min_factor

    max_factor

    """
    def __init__(
            self,
            min_factor: float = -0.1,
            max_factor: float = 0.1
    ) -> None:
        super().__init__(
            transform=Pan(),
            min_factor=min_factor,
            max_factor=max_factor
        )


@public
class Rotate(DeterministicTransform):
    r"""
    Callable class for rotating images and bounding boxes.
    """
    n_factors = 1

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
        height, width, _ = tf_get_shape(image)

        angle = (pi / 4) * angle_factor

        image = tfa.image.rotate(image, -angle, interpolation='BILINEAR')

        image_center = tf.cast(tf.stack([height, width]), dtype=tf.float32) / 2
        image_centers = tf.tile(image_center, [2])

        centered_boxes = ijhw_to_ijkl(boxes) - image_centers

        tl = centered_boxes[..., 0:2]
        tr = tf.stack([
            centered_boxes[..., 0], centered_boxes[..., 3]
        ], axis=-1)
        bl = tf.stack([
            centered_boxes[..., 2], centered_boxes[..., 0]
        ], axis=-1)
        br = centered_boxes[..., 2:4]

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

        aligned_boxes = tf.concat([aligned_tl, aligned_br], axis=-1)

        boxes = ijkl_to_ijhw(aligned_boxes + image_centers)
        boxes = filter_boxes(boxes, width, height)

        return image, boxes


@public
class RandomRotate(RandomTransform):
    r"""
    Callable class for randomly rotating images and bounding boxes.

    Parameters
    ----------
    min_factor

    max_factor

    """
    def __init__(
            self,
            min_factor: float = -0.1,
            max_factor: float = 0.1
    ) -> None:
        super().__init__(
            transform=Rotate(),
            min_factor=min_factor,
            max_factor=max_factor
        )
