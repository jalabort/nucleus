from typing import Tuple, Sequence

import tensorflow as tf

from nucleus.utils import export, name_scope

from .base import Transform


@export
class TransformChainer(Transform):
    r"""
    Callable class for chaining data augmentation operations together.

    Notes
    -----
    A `TransformChainer` is also a `Transform`.

    Parameters
    ----------
    transforms
        The transform operations to be chained together.
    """
    def __init__(self, transforms: Sequence[Transform]) -> None:
        self._transforms = transforms

    @name_scope
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
