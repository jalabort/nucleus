from typing import Tuple

import tensorflow as tf
from abc import ABC, abstractmethod

from nucleus.box import unpad_tensor
from nucleus.utils import export, name_scope, tf_get_shape


# TODO: Transforms seem to be faster without @tf.function?
class Transform(ABC):
    r"""
    Abstract base class for defining transform classes that act on image pixels
    and bounding boxes.
    """

    @abstractmethod
    def __call__(
            self,
            image: tf.Tensor,
            boxes: tf.Tensor,
            *args
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
        args
            Additional arguments.

        Returns
        -------
        transformed_image
            The image after applying the transform operation.
        transformed_boxes
            The boxes after applying the transform operation.
        """


class DeterministicTransform(Transform):
    r"""
    Abstract base class for defining transform classes that act upon image
    pixels and bounding boxes.
    """
    n_factors: int

    @abstractmethod
    def _operation(
            self,
            image: tf.Tensor,
            boxes: tf.Tensor,
            *args
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        r"""
        Performs a deterministic transform operation on the given image pixels
        and bounding boxes.

        Parameters
        ----------
        image
            The image on which to apply the transform operation.
        boxes
            The boxes on which to apply the transform operation.
        args
            Additional arguments passed to the transform.

        Returns
        -------
        transformed_image
            The image after applying the transform operation.
        transformed_boxes
            The boxes after applying the transform operation.
        """

    @name_scope
    def __call__(
            self,
            image: tf.Tensor,
            boxes: tf.Tensor,
            *args
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        r"""
        Performs a deterministic transform operation on the given image pixels
        and bounding boxes.

        Parameters
        ----------
        image
            The image on which to apply the transform operation.
        boxes
            The boxes on which to apply the transform operation.
        args
            Additional arguments.

        Returns
        -------
        transformed_image
            The image after applying the transform operation.
        transformed_boxes
            The boxes after applying the transform operation.
        """
        return self._operation(image, boxes, *args)


# TODO: Better docstring!
class RandomTransform(Transform):
    r"""
    Abstract base class for defining random data augmentation callable classes
    that act upon images and bounding boxes.

    Parameters
    ----------
    transform
        The deterministic transform that this random transform is based upon.
    min_factor

    max_factor

    """

    def __init__(
            self,
            transform: DeterministicTransform,
            min_factor: float,
            max_factor: float
    ) -> None:
        if transform.n_factors == 0:
            raise ValueError(
                f'An instance of  {transform.__class__.__name__} cannot be '
                f'used to create an instance of {self.__class__.__name__}.'
            )
        self.transform = transform
        self.max_factor = max_factor
        self.min_factor = min_factor

    @name_scope
    def __call__(
            self,
            image: tf.Tensor,
            boxes: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        r"""
        Performs a random transform operation on the given image pixels and
        bounding boxes.

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
        # TODO: Some of these n_factors checks seem a bit hacky... Can we do
        #  better?
        if self.transform.n_factors == -1:
            factors_shape = tf_get_shape(boxes[..., :4])
        elif self.transform.n_factors == -2:
            factors_shape = tf_get_shape(image)
        elif self.transform.n_factors == -3:
            factors_shape = ()
            self.min_factor = 0
            self.max_factor = tf.cast(
                tf_get_shape(unpad_tensor(boxes))[0] - 1,
                dtype=tf.float32
            )
        else:
            factors_shape = (self.transform.n_factors,)

        factors = tf.random.uniform(
            shape=factors_shape,
            minval=self.min_factor,
            maxval=self.max_factor,
        )

        return self.transform(image, boxes, factors)


# TODO: Better docstring!
@export
class RandomApplyTransform(Transform):
    r"""
    Abstract base class for defining random data augmentation callable classes
    that are randomly applied to images and bounding boxes.

    Parameters
    ----------
    transform
       The random transform that this random apply transform is based upon.
    op_rate
        The probability at which the random transform operation will be
        performed.
    """

    def __init__(
            self,
            transform: Transform, op_rate: float = 0.5
    ) -> None:
        self.random_transform = transform
        self.op_rate = op_rate

    @name_scope
    def __call__(
            self,
            image: tf.Tensor,
            boxes: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        r"""
        Randomly applies a transform on the given image and bounding boxes
        with the probability defined by the `op_rate` property.

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
        do_op = tf.less_equal(tf.random.uniform(shape=()), self.op_rate)

        image, boxes = tf.cond(
            do_op,
            lambda: self.random_transform(image, boxes),
            lambda: (image, boxes)
        )
        return image, boxes
