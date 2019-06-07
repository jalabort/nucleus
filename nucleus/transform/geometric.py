from typing import Tuple

from math import pi
import tensorflow as tf

from nucleus.box import tools as box_tools
from nucleus.utils import name_scope, tf_get_shape

from .base import RandomTransform


__all__ = [
    'RandomHorizontalFlip',
    'RandomZoom',
    'RandomJitterBoxes',
    'RandomCropAroundBox'
]


class RandomHorizontalFlip(RandomTransform):
    r"""
    Callable class for horizontally flipping images and bounding boxes.

    Parameters
    ----------
    op_rate
        The probability at which the flipping operation will be performed.
    """

    def __init__(self, op_rate: float = 0.5) -> None:
        super(RandomHorizontalFlip, self).__init__(op_rate)

    @name_scope
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
        boxes = box_tools.unpad_tensor(boxes)
        image = tf.image.flip_left_right(image)
        boxes = box_tools.flip_boxes_left_right(boxes)
        boxes = box_tools.pad_tensor(boxes, max_length=max_boxes)
        return image, boxes


class RandomZoom(RandomTransform):
    r"""
    Callable class for zooming, in and out, images and bounding boxes.

    Parameters
    ----------
    op_rate
        The probability at which the zooming operation will be performed.
    min_factor
        Defines the maximum zoom out factor to be applied to the image and
        bounding boxes. Given as a percentage, which is multiplied by the
        image width and height, respectively, to determine the maximum
        horizontal and vertical zooming out factors.
    max_factor
        Defines the maximum zoom in factor to be applied to the image and
        bounding boxes. Given as a percentage, which is multiplied by the
        image width and height, respectively, to determine the maximum
        horizontal and vertical zooming in factors.
    """

    def __init__(
            self,
            op_rate: float = 0.5,
            min_factor: float = 0.9,
            max_factor: float = 1.1
    ) -> None:
        super(RandomZoom, self).__init__(op_rate)
        self._max_factor = max_factor
        self._min_factor = min_factor

    @name_scope
    def _operation(
            self,
            image: tf.Tensor,
            boxes: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        r"""
        Zooms in or out the given image and/or bounding boxes by a random
        amount.

        Parameters
        ----------
        image
            The image to be zoomed.
        boxes
            The boxes to be zoomed.

        Returns
        -------
        zoomed_image
            The zoomed image.
        zoomed_boxes
            The zoomed boxes.
        """
        max_boxes = tf_get_shape(boxes)[0]
        boxes = box_tools.unpad_tensor(boxes)

        image_shape = tf_get_shape(image)
        height, width, _ = image_shape

        factors = tf.random.uniform(
            shape=(2,),
            minval=self._min_factor,
            maxval=self._max_factor,
        )

        new_height = tf.cast(height, tf.float32) * factors[0]
        new_width = tf.cast(width, tf.float32) * factors[1]

        image = tf.image.resize(image, size=(new_height, new_width))
        image = tf.image.resize_image_with_crop_or_pad(image, height, width)

        ijhw_tensor = boxes[..., :4] * tf.convert_to_tensor(
            [factors[0], factors[1]] * 2
        )
        offsets = tf.stack([(1.0 - factors[0]) / 2, (1.0 - factors[1]) / 2])
        boxes = tf.concat(
            [
                ijhw_tensor[..., :2] + offsets,
                ijhw_tensor[..., 2:4],
                boxes[..., 4:]
            ],
            axis=-1
        )

        boxes = box_tools.filter_boxes(boxes)

        boxes = box_tools.pad_tensor(boxes, max_length=max_boxes)

        return image, boxes


class RandomJitterBoxes(RandomTransform):
    r"""
    Callable class for jittering the bounding boxes associated to an images.

    Parameters
    ----------
    op_rate
        The probability at which the bounding box jitter operation will be
        performed.
    max_factor
        Defines the maximum distance by which the bounding boxes will be
        jitter. Given as a percentage, which is multiplied by the bounding
        boxes width and height, respectively, to determine the maximum
        horizontal and vertical jitter for each bonding box.
    """

    def __init__(
            self,
            op_rate: float = 0.5,
            max_factor: float = 0.05
    ) -> None:
        super(RandomJitterBoxes, self).__init__(op_rate)
        self._max_factor = max_factor

    @name_scope
    def _operation(
            self,
            image: tf.Tensor,
            boxes: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        r"""
        Adjust the brightness of the given image by a random amount.

        Parameters
        ----------
        image
            The image to which the bounding boxes are associated.
        boxes
            The bounding boxes to be jittered.

        Returns
        -------
        image
            The image to which the bounding boxes are associated unchanged.
        jittered_boxes
            The jittered bounding boxes.
        """
        max_boxes = tf_get_shape(boxes)[0]
        boxes = box_tools.unpad_tensor(boxes)

        factors = tf.random.uniform(
            shape=tf_get_shape(boxes[..., :4]),
            minval=-self._max_factor,
            maxval=self._max_factor,
        )

        jitter = tf.concat(
            [boxes[..., 2:4], boxes[..., 2:4]], axis=-1
        ) * factors
        ijhw_tensor = boxes[..., :4] + jitter
        boxes = tf.concat([ijhw_tensor, boxes[..., 4:]], axis=-1)

        boxes = box_tools.pad_tensor(boxes, max_length=max_boxes)

        return image, boxes


class RandomCropAroundBox(RandomTransform):

    def __init__(
            self,
            op_rate: float = 0.5,
            size: Tuple[int, int] = (224, 224)
    ) -> None:
        super(RandomCropAroundBox, self).__init__(op_rate)
        self._size = tf.convert_to_tensor(size, dtype=tf.float32)

    @name_scope
    def _operation(
            self,
            image: tf.Tensor,
            boxes: tf.Tensor,
            pad: bool = True
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        r"""


        Parameters
        ----------
        image
            The image to which the bounding boxes are associated.
        boxes
            The bounding boxes to be jittered.

        Returns
        -------
        image

        boxes

        """
        max_boxes = tf_get_shape(boxes)[0]
        boxes = box_tools.unpad_tensor(boxes)

        index = tf.random.uniform(
            shape=(),
            minval=0,
            maxval=tf_get_shape(boxes)[0],
            dtype=tf.int32
        )

        height, width, _ = tf_get_shape(image)
        resolution = tf.convert_to_tensor([height, width], dtype=tf.float32)

        ijhw_tensor = box_tools.scale_coords(boxes[..., :4], resolution)
        yxhw = box_tools.ijhw_to_yxhw(ijhw_tensor[index])

        offset_size = yxhw - tf.concat([self._size, self._size], axis=-1) / 2
        target_size = tf.concat([self._size, self._size], axis=-1)

        offset_height, target_height = tf.cond(
            tf.less(offset_size[0], 0),
            lambda: (
                tf.constant(0, dtype=tf.float32),
                target_size[0]
            ),
            lambda: (offset_size[0], target_size[0])
        )

        offset_width, target_width = tf.cond(
            tf.less(offset_size[1], 0),
            lambda: (
                tf.constant(0, dtype=tf.float32),
                target_size[1]
            ),
            lambda: (offset_size[1], target_size[1])
        )

        offset_height, target_height = tf.cond(
            tf.greater_equal(offset_height + target_height, height),
            lambda: (
                offset_height - (height - target_height),
                target_size[0]
            ),
            lambda: (offset_height, target_height)
        )

        offset_width, target_width = tf.cond(
            tf.greater_equal(offset_width + target_width, width),
            lambda: (
                offset_width - (width - target_width),
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

        boxes = box_tools.filter_boxes(boxes, pad=pad)

        boxes = tf.cond(
            tf.equal(pad, True),
            lambda: box_tools.pad_tensor(boxes, max_length=max_boxes),
            lambda: boxes
        )

        return image, boxes

# class Pan(DataAugmentationOp):
#     r"""
#     Callable class for panning images and bounding boxes.
#
#     Parameters
#     ----------
#     op_rate
#         The probability at which the panning operation will be performed.
#     max_factor
#         Defines the maximum panning distance to be applied to the image and
#         bounding boxes. Given as a percentage, which is multiplied by the
#         image width and height, respectively, to determine the maximum
#         horizontal and vertical panning distance.
#     """
#
#     def __init__(self, op_rate: float = 0.5, max_factor: float = 0.05) -> None:
#         super(Pan, self).__init__(op_rate)
#         self._max_factor = max_factor
#
#     @name_scope
#     def _operation(
#             self,
#             image: tf.Tensor,
#             boxes: tf.Tensor,
#     ) -> Tuple[tf.Tensor, tf.Tensor]:
#         r"""
#         Pans the given image and/or bounding boxes by a random amount.
#
#         Parameters
#         ----------
#         image
#             The image to be panned.
#         boxes
#             The boxes to be panned.
#
#         Returns
#         -------
#         panned_image
#             The panned image.
#         panned_boxes
#             The panned boxes.
#         """
#         height, width, _ = tf_get_shape(image)
#         factors = tf.random_uniform(
#             shape=(2,),
#             minval=-self._max_factor,
#             maxval=self._max_factor,
#         )
#         dx = tf.to_float(width) * factors[1]
#         dy = tf.to_float(height) * factors[0]
#         image = tf.contrib.image.translate(
#             image,
#             [dx, dy],
#             interpolation='BILINEAR',
#         )
#
#         offsets = tf.stack([dx, dy])
#         boxes = tf.concat(
#             [
#                 boxes[..., :2] + offsets,
#                 boxes[..., 2:4]
#             ],
#             axis=-1
#         )
#         boxes = box_tools.filter_boxes(boxes)
#
#         return image, boxes
#
#
# class Rotate(DataAugmentationOp):
#     r"""
#     Callable class for rotating images and bounding boxes.
#
#     Parameters
#     ----------
#     op_rate
#         The probability at which the rotation operation will be performed.
#     max_factor
#         Defines the maximum rotation angle to be applied to the image and
#         bounding boxes. Given as a percentage, which is multiplied by 45
#         degrees, to determine the maximum rotation angle.
#     """
#     def __init__(self, op_rate: float = 0.5, max_factor: float = 0.1) -> None:
#         super(Rotate, self).__init__(op_rate)
#         self._max_factor = max_factor
#
#     @name_scope
#     def _operation(
#             self,
#             image: tf.Tensor,
#             boxes: tf.Tensor,
#     ) -> Tuple[tf.Tensor, tf.Tensor]:
#         r"""
#         Rotates the given image and/or bounding boxes by a random amount.
#
#         Parameters
#         ----------
#         image
#             The image to be rotated.
#         boxes
#             The boxes to be rotated.
#
#         Returns
#         -------
#         rotated_image
#             The rotated image.
#         rotated_boxes
#             The rotated boxes.
#         """
#         height, width, _ = tf_get_shape(image)
#
#         factor = tf.random.uniform(
#             shape=(1,),
#             minval=-self._max_factor,
#             maxval=self._max_factor,
#         )
#
#         angle = (pi / 4) * factor
#
#         image = tf.image.r(
#             image,
#             -angle,
#             interpolation='BILINEAR',
#         )
#
#         image_center = tf.to_float(tf.stack([width, height])) / 2
#         image_centers = tf.tile(image_center, [2])
#
#         centered_boxes = box_tools.ijhw_to_ijkl(boxes) - image_centers
#
#         tl = centered_boxes[..., 0:2]
#         tr = tf.stack([
#             centered_boxes[..., 2], centered_boxes[..., 1]
#         ], axis=-1)
#         bl = tf.stack([
#             centered_boxes[..., 0], centered_boxes[..., 3]
#         ], axis=-1)
#         br = centered_boxes[..., 2:4]
#
#         rotated_tl = tf.stack([
#             tf.cos(angle) * tl[..., 0] - tf.sin(angle) * tl[..., 1],
#             tf.sin(angle) * tl[..., 0] + tf.cos(angle) * tl[..., 1]
#         ], axis=-1)
#         rotated_tr = tf.stack([
#             tf.cos(angle) * tr[..., 0] - tf.sin(angle) * tr[..., 1],
#             tf.sin(angle) * tr[..., 0] + tf.cos(angle) * tr[..., 1]
#         ], axis=-1)
#         rotated_bl = tf.stack([
#             tf.cos(angle) * bl[..., 0] - tf.sin(angle) * bl[..., 1],
#             tf.sin(angle) * bl[..., 0] + tf.cos(angle) * bl[..., 1]
#         ], axis=-1)
#         rotated_br = tf.stack([
#             tf.cos(angle) * br[..., 0] - tf.sin(angle) * br[..., 1],
#             tf.sin(angle) * br[..., 0] + tf.cos(angle) * br[..., 1]
#         ], axis=-1)
#
#         rotated_boxes = tf.concat([
#             rotated_tl,
#             rotated_tr,
#             rotated_bl,
#             rotated_br
#         ], axis=-1)
#
#         xs = rotated_boxes[..., 0::2]
#         ys = rotated_boxes[..., 1::2]
#
#         aligned_tl = tf.stack([
#             tf.reduce_min(xs, axis=-1),
#             tf.reduce_min(ys, axis=-1)
#         ], axis=-1)
#         aligned_br = tf.stack([
#             tf.reduce_max(xs, axis=-1),
#             tf.reduce_max(ys, axis=-1)
#         ], axis=-1)
#
#         aligned_boxes = tf.concat([aligned_tl, aligned_br], axis=-1)
#
#         boxes = box_tools.ijkl_to_ijhw(aligned_boxes + image_centers)
#         boxes = box_tools.filter_boxes(boxes, width, height)
#
#         return image, boxes
