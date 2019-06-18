from typing import Tuple

import tensorflow as tf
from public import public

from .base import DeterministicTransform, RandomTransform


@public
class PixelValueScale(DeterministicTransform):
    r"""
    Callable class for scaling the value of images pixels.
    """
    n_factors = -2

    def _operation(
            self,
            image: tf.Tensor,
            boxes: tf.Tensor,
            pixel_value_factors: float,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        r"""
        Scales the value of every pixel in the given image by the given factors.

        Parameters
        ----------
        image
            The image whose pixel values will be scaled.
        boxes
            The bounding boxes associated to the previous image.
        pixel_value_factors

        Returns
        -------
        pixel_scaled_image
            The pixel scaled image.
        boxes
            The bounding boxes associated to the original image unchanged.
        """
        image = image * pixel_value_factors
        image = tf.clip_by_value(image, 0.0, 255.0)

        return image, boxes


@public
class RandomPixelValueScale(RandomTransform):
    r"""
    Callable class for randomly scaling the value of images pixels.

    Parameters
    ----------
    min_factor

    max_factor

    """
    def __init__(
            self,
            min_factor: float = 0.8,
            max_factor: float = 1.2
    ) -> None:
        super().__init__(
            transform=PixelValueScale(),
            min_factor=min_factor,
            max_factor=max_factor
        )


@public
class AdjustBrightness(DeterministicTransform):
    r"""
    Callable class for adjusting the brightness of images.
    """
    n_factors = 1

    def _operation(
            self,
            image: tf.Tensor,
            boxes: tf.Tensor,
            brightness_factor: float
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        r"""
        Adjusts the brightness of the given image by the given factor.

        Parameters
        ----------
        image
            The image which brightness we want to adjust.
        boxes
            The bounding boxes associated to the previous image.
        brightness_factor

        Returns
        -------
        adjusted_image
            The brightness adjusted image.
        boxes
            The bounding boxes associated to the original image unchanged.
        """
        image = _adjust_image(
            adjusting_function=tf.image.adjust_brightness,
            image=image,
            factor=brightness_factor
        )
        return image, boxes


@public
class RandomAdjustBrightness(RandomTransform):
    r"""
    Callable class for randomly adjusting the brightness of images.

    Parameters
    ----------
    min_factor

    max_factor

    """
    def __init__(
            self,
            min_factor: float = 0.0,
            max_factor: float = 0.5
    ) -> None:
        super().__init__(
            transform=AdjustBrightness(),
            min_factor=min_factor,
            max_factor=max_factor
        )


@public
class AdjustContrast(DeterministicTransform):
    r"""
    Callable class for adjusting the contrast of images.
    """
    n_factors = 1

    def _operation(
            self,
            image: tf.Tensor,
            boxes: tf.Tensor,
            contrast_factor: float
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        r"""
        Adjusts the contrast of the given image by the given factor.

        Parameters
        ----------
        image
            The image which contrast we want to adjust.
        boxes
            The bounding boxes associated to the previous image.
        contrast_factor

        Returns
        -------
        adjusted_image
            The contrast adjusted image.
        boxes
            The bounding boxes associated to the original image unchanged.
        """
        image = _adjust_image(
            adjusting_function=tf.image.adjust_contrast,
            image=image,
            factor=contrast_factor
        )
        return image, boxes


@public
class RandomAdjustContrast(RandomTransform):
    r"""
    Callable class for randomly adjusting the contrast of images.

    Parameters
    ----------
    min_factor

    max_factor

    """
    def __init__(
            self,
            min_factor: float = 0.8,
            max_factor: float = 1.2
    ) -> None:
        super().__init__(
            transform=AdjustContrast(),
            min_factor=min_factor,
            max_factor=max_factor
        )


@public
class AdjustHue(DeterministicTransform):
    r"""
    Callable class for adjusting the hue of images.
    """
    n_factors = 1

    def _operation(
            self,
            image: tf.Tensor,
            boxes: tf.Tensor,
            hue_factor: float
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        r"""
        Adjusts the hue of the given image by the given factor.

        Parameters
        ----------
        image
            The image which hue we want to adjust.
        boxes
            The bounding boxes associated to the previous image.
        hue_factor

        Returns
        -------
        adjusted_image
            The hue adjusted image.
        boxes
            The bounding boxes associated to the original image unchanged.
        """
        image = _adjust_image(
            adjusting_function=tf.image.adjust_hue,
            image=image,
            factor=hue_factor
        )
        return image, boxes


@public
class RandomAdjustHue(RandomTransform):
    r"""
    Callable class for randomly adjusting the hue of images.

    Parameters
    ----------
    min_factor

    max_factor

    """
    def __init__(
            self,
            min_factor: float = -0.02,
            max_factor: float = 0.02
    ) -> None:
        super().__init__(
            transform=AdjustHue(),
            min_factor=min_factor,
            max_factor=max_factor
        )


@public
class AdjustSaturation(DeterministicTransform):
    r"""
    Callable class for adjusting the saturation of images.
    """
    n_factors = 1

    def _operation(
            self,
            image: tf.Tensor,
            boxes: tf.Tensor,
            saturation_factor: float
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        r"""
        Adjusts the saturation of the given image by the given factor.

        Parameters
        ----------
        image
            The image which saturation we want to adjust.
        boxes
            The bounding boxes associated to the previous image.
        saturation_factor

        Returns
        -------
        adjusted_image
            The saturation adjusted image.
        boxes
            The bounding boxes associated to the original image unchanged.
        """
        image = _adjust_image(
            adjusting_function=tf.image.adjust_saturation,
            image=image,
            factor=saturation_factor
        )
        return image, boxes


@public
class RandomAdjustSaturation(RandomTransform):
    r"""
    Callable class for randomly adjusting the saturation of images.

    Parameters
    ----------
    min_factor

    max_factor

    """
    def __init__(
            self,
            min_factor: float = 0.8,
            max_factor: float = 1.2
    ) -> None:
        super().__init__(
            transform=AdjustSaturation(),
            min_factor=min_factor,
            max_factor=max_factor
        )


def _adjust_image(
        adjusting_function: callable,
        image: tf.Tensor,
        factor: float
) -> tf.Tensor:
    r"""
    Auxiliary function used to adjusts several image color properties.

    Parameters
    ----------
    adjusting_function
        A function for adjusting an image color property.
        Examples: `tf.image.adjust_brightness`, `tf.image.adjust_contrast`,
        tf.image.adjust_hue`, tf.image.adjust_saturation`.
    image
        The image whose color property we want to adjust.
    factor
        Factor by which the color property will be modified.

    Returns
    -------
    The color property adjusted image.
    """
    image = adjusting_function(image / 255, factor[0]) * 255
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=255.0)
    return image
