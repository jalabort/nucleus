from typing import Tuple

import tensorflow as tf

from nucleus.utils import export, name_scope, tf_get_shape

from .base import RandomTransform


@export
class RandomPixelValueScale(RandomTransform):
    r"""
    Callable class for scaling the value of images pixels by a random amount.

    Parameters
    ----------
    op_rate
        The probability at which the pixel value scaling operation will be
        performed.
    min_factor
        Defines the minimum factor by which pixel values will be scaled.
    max_factor
        Defines the maximum factor by which pixel values will be scaled.
    """

    def __init__(
            self,
            op_rate: float = 0.5,
            min_factor: float = 0.8,
            max_factor: float = 1.2
    ) -> None:
        super(RandomPixelValueScale, self).__init__(op_rate)
        self._max_factor = max_factor
        self._min_factor = min_factor

    @name_scope
    def _operation(
            self,
            image: tf.Tensor,
            boxes: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        r"""
        Scales the value of every pixel in the given image by a random amount.

        Parameters
        ----------
        image
            The image whose pixel values will be scaled.
        boxes
            The bounding boxes associated to the previous image.

        Returns
        -------
        pixel_scaled_image
            The pixel scaled image.
        boxes
            The bounding boxes associated to the original image unchanged.
        """
        factors = tf.random.uniform(
            shape=tf_get_shape(image),
            minval=self._min_factor,
            maxval=self._max_factor,
        )
        image = image * factors
        image = tf.clip_by_value(image, 0.0, 255.0)

        return image, boxes


@export
class RandomAdjustBrightness(RandomTransform):
    r"""
    Callable class for adjusting the brightness of images.

    Parameters
    ----------
    op_rate
        The probability at which the adjusting brightness operation will be
        performed.
    max_factor
        Defines the maximum change in brightness allowed.
    """

    def __init__(
            self,
            op_rate: float = 0.5,
            max_factor: float = 0.5
    ) -> None:
        super(RandomAdjustBrightness, self).__init__(op_rate)
        self._max_factor = max_factor

    @name_scope
    def _operation(
            self,
            image: tf.Tensor,
            boxes: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        r"""
        Adjusts the brightness of the given image by a random amount.

        Parameters
        ----------
        image
            The image which brightness we want to adjust.
        boxes
            The bounding boxes associated to the previous image.

        Returns
        -------
        adjusted_image
            The brightness adjusted image.
        boxes
            The bounding boxes associated to the original image unchanged.
        """
        image = _random_adjust_image(
            tf.image.adjust_brightness,
            image,
            0,
            self._max_factor
        )
        return image, boxes


@export
class RandomAdjustContrast(RandomTransform):
    r"""
    Callable class for adjusting the contrast of images.

    Parameters
    ----------
    op_rate
        The probability at which the adjusting contrast operation will be
        performed.
    min_factor
        Defines the minimum contrast factor.
    max_factor
        Defines the maximum contrast factor.
    """

    def __init__(
            self,
            op_rate: float = 0.5,
            min_factor: float = 0.8,
            max_factor: float = 1.2
    ) -> None:
        super(RandomAdjustContrast, self).__init__(op_rate)
        self._min_factor = min_factor
        self._max_factor = max_factor

    @name_scope
    def _operation(
            self,
            image: tf.Tensor,
            boxes: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        r"""
        Adjusts the contrast of the given image by a random amount.

        Parameters
        ----------
        image
            The image which contrast we want to adjust.
        boxes
            The bounding boxes associated to the previous image.

        Returns
        -------
        adjusted_image
            The contrast adjusted image.
        boxes
            The bounding boxes associated to the original image unchanged.
        """
        image = _random_adjust_image(
            tf.image.adjust_contrast, image, self._min_factor, self._max_factor
        )
        return image, boxes


@export
class RandomAdjustHue(RandomTransform):
    r"""
    Callable class for adjusting the hue of images.

    Parameters
    ----------
    op_rate
        The probability at which the adjusting hue operation will be
        performed.
    max_factor
        Defines the maximum change in hue allowed.
    """

    def __init__(
            self,
            op_rate: float = 0.5,
            max_factor: float = 0.02
    ) -> None:
        super(RandomAdjustHue, self).__init__(op_rate)
        self._max_factor = max_factor

    @name_scope
    def _operation(
            self,
            image: tf.Tensor,
            boxes: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        r"""
        Adjusts the hue of the given image by a random amount.

        Parameters
        ----------
        image
            The image which hue we want to adjust.
        boxes
            The bounding boxes associated to the previous image.

        Returns
        -------
        adjusted_image
            The hue adjusted image.
        boxes
            The bounding boxes associated to the original image unchanged.
        """
        image = _random_adjust_image(
            tf.image.adjust_hue,
            image,
            -self._max_factor,
            self._max_factor
        )
        return image, boxes


@export
class RandomAdjustSaturation(RandomTransform):
    r"""
    Callable class for adjusting the saturation of images.

    Parameters
    ----------
    op_rate
        The probability at which the adjusting saturation operation will be
        performed.
    min_factor
        Defines the minimum saturation factor.
    max_factor
        Defines the maximum saturation factor.
    """

    def __init__(
            self,
            op_rate: float = 0.5,
            min_factor: float = 0.8,
            max_factor: float = 1.2
    ) -> None:
        super(RandomAdjustSaturation, self).__init__(op_rate)
        self._min_factor = min_factor
        self._max_factor = max_factor

    @name_scope
    def _operation(
            self,
            image: tf.Tensor,
            boxes: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        r"""
        Adjusts the saturation of the given image by a random amount.

        Parameters
        ----------
        image
            The image which saturation we want to adjust.
        boxes
            The bounding boxes associated to the previous image.

        Returns
        -------
        adjusted_image
            The saturation adjusted image.
        boxes
            The bounding boxes associated to the original image unchanged.
        """
        image = _random_adjust_image(
            tf.image.adjust_saturation,
            image,
            self._min_factor,
            self._max_factor
        )
        return image, boxes


@export
class RandomDistortColor(RandomTransform):
    r"""
    Callable class for adjusting the saturation of images.

    Parameters
    ----------
    op_rate
        The probability at which the adjusting saturation operation will be
        performed.
    color_ordering
        Defines the order in which the adjusting operations will be
        performed. If `0` brightness, saturation, hue and contrast. If `1`
        brightness, contrast, saturation and hue.
    adjust_brightness
        An adjust brightness object.
    adjust_saturation
        An adjust saturation object.
    adjust_hue
        An adjust hue object.
    adjust_contrast
        An adjust contrast object.
    """

    def __init__(
            self,
            op_rate: float = 0.5,
            color_ordering: int = 0,
            adjust_brightness: RandomAdjustBrightness = RandomAdjustBrightness(),
            adjust_saturation: RandomAdjustSaturation = RandomAdjustSaturation(),
            adjust_hue: RandomAdjustHue = RandomAdjustHue(),
            adjust_contrast: RandomAdjustContrast = RandomAdjustContrast(),
    ) -> None:
        super(RandomDistortColor, self).__init__(op_rate)
        self._color_ordering = color_ordering
        self._adjust_brightness = adjust_brightness
        self._adjust_saturation = adjust_saturation
        self._adjust_hue = adjust_hue
        self._adjust_contrast = adjust_contrast

    @name_scope
    def _operation(
            self,
            image: tf.Tensor,
            boxes: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        r"""
        Distort the color of the given image by a random amount.

        Parameters
        ----------
        image
            The image whose color we want to distort.
        boxes
            The bounding boxes associated to the previous image.

        Returns
        -------
        adjusted_image
            The color distorted image.
        boxes
            The bounding boxes associated to the original image unchanged.
        """
        if self._color_ordering == 0:
            image, _ = self._adjust_brightness(image, boxes)
            image, _ = self._adjust_saturation(image, boxes)
            image, _ = self._adjust_hue(image, boxes)
            image, _ = self._adjust_contrast(image, boxes)
        elif self._color_ordering == 1:
            image, _ = self._adjust_brightness(image, boxes)
            image, _ = self._adjust_contrast(image, boxes)
            image, _ = self._adjust_saturation(image, boxes)
            image, _ = self._adjust_hue(image, boxes)
        return image, boxes


@name_scope
def _random_adjust_image(
        adjusting_function: callable,
        image: tf.Tensor,
        min_factor: float,
        max_factor: float,
) -> tf.Tensor:
    r"""
    Randomly adjust an image color property.

    Parameters
    ----------
    adjusting_function
        A function for adjusting and image color property.
        Examples: `tf.image.adjust_brightness`, `tf.image.adjust_contrast`,
        tf.image.adjust_hue`, tf.image.adjust_saturation`.
    image
        The image whose color property we want to adjust
    min_factor
        The minimum factor by which the color property will be modified.
    max_factor
        The maximum factor by which the color property will be modified.

    Returns
    -------
    The color property adjusted image.
    """
    factors = tf.random.uniform(
        shape=(),
        minval=min_factor,
        maxval=max_factor,
    )
    image = adjusting_function(image / 255, factors) * 255
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=255.0)

    return image
