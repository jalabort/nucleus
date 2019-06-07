from typing import Tuple

import tensorflow as tf

from nucleus.utils import export, name_scope

from .base import RandomTransform, image_compatible


@export
class TransformChainer:
    r"""
    Callable class for chaining data augmentation operations together.

    Parameters
    ----------
    transforms
        The transform operations to be chained together.
    """
    def __init__(self, transforms: Tuple[RandomTransform]) -> None:
        self._transforms = transforms

    @tf.function
    @name_scope
    @image_compatible
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
        for op in self._transforms:
            image, boxes = op(image, boxes)

        return image, boxes
