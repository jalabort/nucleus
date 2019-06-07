from typing import Tuple

import abc
import functools
import tensorflow as tf

from copy import deepcopy
from tensorflow.python.util import tf_decorator

from nucleus.image import Image
from nucleus.box import BoxCollection
from nucleus.utils import export, name_scope


@export
def image_compatible(arg):
    r"""
    Decorator that makes it possible to pass a nucleus image to a function
    or class method that expects two tensors representing image pixels and
    bounding boxes.

    Notes
    -----
    This decorator is extensively used on the transform module.

    Parameters
    ----------
    arg : callable | str, optional
        If explicitly specified, `arg` is the name of the new name scope.
        If not, `arg` is the function to be decorated. In that case,
        the name of the new name scope is the decorated function's name.
    """
    if callable(arg):
        @functools.wraps(arg)
        def wrapper(*args, **kwargs):
            if isinstance(args[1], Image):
                img: Image = args[1]
                hwc, box_tensor = arg(
                    args[0],
                    img.hwc,
                    img.box_collection.as_tensor(),
                    *args[2:],
                    **kwargs
                )
                new_img = deepcopy(img)
                new_img.hwc = hwc
                new_img.box_collection = BoxCollection.from_tensor(
                    box_tensor, unique_labels=img.box_collection.unique_labels
                )
                return new_img
            else:
                return arg(*args, **kwargs)

        return tf_decorator.make_decorator(arg, wrapper)
    else:
        def decorator(func):
            @functools.wraps(func)
            def inner_wrapper(*args, **kwargs):
                if isinstance(args[0], Image):
                    img: Image = args[0]
                    hwc, box_tensor = arg(
                        img.hwc,
                        img.box_collection.as_tensor(),
                        *args[1:],
                        **kwargs
                    )
                    img = deepcopy(img)
                    img.hwc = hwc
                    img.box_collection.from_tensor(
                        box_tensor,
                        unique_labels=img.box_collection.unique_labels
                    )
                    return img
                else:
                    return func(*args, **kwargs)

            return tf_decorator.make_decorator(arg, inner_wrapper)

        return decorator


class Transform(abc.ABC):
    r"""
    Abstract base class for defining transform classes that act on image pixels
    and bounding boxes.
    """

    # TODO: Figure out tf.function for transforms
    # @name_scope
    @abc.abstractmethod
    @image_compatible
    def __call__(
            self,
            image: tf.Tensor,
            boxes: tf.Tensor,
            **kwargs
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        r"""
        Performs a transform operation on the given image pixels and bounding
        boxes.

        Parameters
        ----------
        image
            The image on which to apply the transform operation.
        boxes
            The boxes on which to apply the transform operation.

        Returns
        -------
        transformed_image
            The image after applying the transform operation.
        transformed_boxes
            The boxes after applying the transform operation.
        """


@export
class RandomTransform(Transform):
    r"""
    Abstract base class for defining data augmentation callable classes that
    act on images and bounding boxes.

    Parameters
    ----------
    transform
        The transform.
    op_rate
        The probability at which the transform operation will be performed.
    """

    def __init__(self, transform, op_rate: float = 0.5) -> None:
        self.transform = transform
        self._op_rate = op_rate

    @property
    def op_rate(self) -> float:
        r"""
        The probability at which the transform operation will be performed.
        """
        return self._op_rate

    # TODO: Figure out tf.function for transforms
    # @name_scope
    @image_compatible
    def __call__(
            self,
            image: tf.Tensor,
            boxes: tf.Tensor,
            **kwargs
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        r"""
        Randomly performs a transform operation on the given image and 
        bounding boxes with the probability defined by the `op_rate`
        instance property.

        Parameters
        ----------
        image
            The image on which to apply the transform operation.
        boxes
            The boxes on which to apply the transform operation.

        Returns
        -------
        transformed_image
            The image after applying the transform operation.
        transformed_boxes
            The boxes after applying the transform operation.
        """
        do_op = tf.less_equal(tf.random.uniform(shape=()), self._op_rate)

        image, boxes = tf.cond(
            do_op,
            lambda: self.transform(image, boxes, **kwargs),
            lambda: (image, boxes)
        )
        return image, boxes
