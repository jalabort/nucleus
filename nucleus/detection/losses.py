from typing import Optional

import tensorflow as tf
from stringcase import snakecase

from nucleus.utils import export


class BaseLoss(tf.keras.losses.Loss):
    r"""


    Parameters
    ----------
    reduction
    name
    """
    def __init__(
            self,
            reduction: str = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
            name: Optional[str] = None
    ) -> None:
        if name is None:
            name = snakecase(self.__class__.__name__)
        super().__init__(reduction=reduction, name=name)


@export
class SsdLoss(BaseLoss):
    r"""
    Single Shot Detection (SSD) loss function.

    References
    ----------
    .. [1] Wei Liu, et. al, "SSD: Single Shot MultiBox Detector", ECCV 2016.
       https://arxiv.org/abs/1512.02325.

    Parameters
    ----------
    n_classes
    delta
    reduction
    """
    def __init__(
            self,
            n_classes: int,
            delta: float = 0.5,
            coords_weight: float = 5.0,
            no_obj_weight: float = 1e-1,
            reduction: str = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
            name: Optional[str] = None
    ) -> None:
        super().__init__(reduction=reduction, name=name)
        self.n_classes = n_classes
        self.delta = delta
        self.no_obj_weight = no_obj_weight

        self.ssd_coords_loss = SsdCoordsLoss(
            delta=delta,
            coords_weight=coords_weight,
            reduction=reduction
        )
        self.ssd_scores_loss = SsdScoresLoss(
            no_obj_weight=no_obj_weight,
            reduction=reduction
        )
        self.ssd_labels_loss = SsdLabelsLoss(
            n_classes=n_classes,
            no_obj_weight=no_obj_weight,
            reduction=reduction
        )

    @tf.function
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        r"""

        Parameters
        ----------
        y_true
        y_pred

        Returns
        -------

        """
        coords_term = self.ssd_coords_loss(y_true=y_true, y_pred=y_pred)
        scores_term = self.ssd_scores_loss(y_true=y_true, y_pred=y_pred)
        labels_term = self.ssd_labels_loss(y_true=y_true, y_pred=y_pred)
        return coords_term + scores_term + labels_term


@export
class SsdCoordsLoss(BaseLoss):
    r"""
    Single Shot Detection (SSD) coordinates loss function.

    References
    ----------
    .. [1] Wei Liu, et. al, "SSD: Single Shot MultiBox Detector", ECCV 2016.
       https://arxiv.org/abs/1512.02325.

    Parameters
    ----------
    delta

    reduction

    """
    def __init__(
            self,
            delta: float = 0.5,
            coords_weight: float = 5.0,
            reduction: str = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
            name: Optional[str] = None
    ) -> None:
        super().__init__(reduction=reduction, name=name)
        self.delta = delta
        self.coords_weight = coords_weight
        self.huber_loss = tf.keras.losses.Huber(
            delta=delta,
            reduction=reduction
        )

    @tf.function
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        r"""

        Parameters
        ----------
        y_true
        y_pred

        Returns
        -------

        """
        coords_true = y_true[..., :4]
        coords_pred = y_pred[..., :4]

        mask = tf.cast(
            tf.greater(y_true[..., 4], 0), dtype=tf.float32
        )[..., None]

        return self.coords_weight * self.huber_loss(
            y_true=coords_true * mask,
            y_pred=coords_pred * mask
        )


@export
class SsdScoresLoss(BaseLoss):
    r"""
    Single Shot Detection (SSD) labels loss function.

    References
    ----------
    .. [1] Wei Liu, et. al, "SSD: Single Shot MultiBox Detector", ECCV 2016.
       https://arxiv.org/abs/1512.02325.

    Parameters
    ----------
    reduction
    """
    def __init__(
            self,
            no_obj_weight: float = 1e-1,
            reduction: str = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
            name: Optional[str] = None
    ) -> None:
        super().__init__(reduction=reduction, name=name)
        self.no_obj_weight = no_obj_weight

    @tf.function
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        r"""

        Parameters
        ----------
        y_true
        y_pred

        Returns
        -------

        """
        labels_true = tf.one_hot(
            tf.cast(tf.greater(y_true[..., 4], 0), dtype=tf.int32),
            depth=2,
            dtype=tf.float32
        )
        labels_pred = y_pred[..., 4:6]

        no_obj_mask = tf.cast(tf.equal(labels_true[..., 0], 1), tf.float32)
        obj_mask = 1 - no_obj_mask
        sample_weights = self.no_obj_weight * no_obj_mask + obj_mask

        return tf.keras.losses.BinaryCrossentropy()(
            y_true=labels_true,
            y_pred=labels_pred,
            sample_weight=sample_weights
        )


@export
class SsdLabelsLoss(BaseLoss):
    r"""
    Single Shot Detection (SSD) labels loss function.

    References
    ----------
    .. [1] Wei Liu, et. al, "SSD: Single Shot MultiBox Detector", ECCV 2016.
       https://arxiv.org/abs/1512.02325.

    Parameters
    ----------
    n_classes
    reduction
    """
    def __init__(
            self,
            n_classes: int,
            no_obj_weight: float = 1e-1,
            reduction: str = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
            name: Optional[str] = None
    ) -> None:
        super().__init__(reduction=reduction, name=name)
        self.n_classes = n_classes + 1
        self.no_obj_weight = no_obj_weight

    @tf.function
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        r"""

        Parameters
        ----------
        y_true
        y_pred

        Returns
        -------

        """
        # labels_true = tf.one_hot(
        #     indices=tf.cast(y_true[..., 4], dtype=tf.int32),
        #     depth=self.n_classes,
        #     dtype=tf.float32
        # )
        # labels_pred = y_pred[..., 4:]
        #
        # mask = tf.cast(
        #     tf.greater(y_true[..., 4], 0), dtype=tf.float32
        # )[..., None]
        #
        # return tf.keras.losses.categorical_crossentropy(
        #     y_true=labels_true * mask,
        #     y_pred=labels_pred * mask
        # )

        labels_true = tf.one_hot(
            tf.cast(y_true[..., 4], dtype=tf.int32),
            depth=self.n_classes,
            dtype=tf.float32
        )
        labels_pred = y_pred[..., 6:]

        no_obj_mask = tf.cast(tf.equal(labels_true[..., 0], 1), tf.float32)
        obj_mask = 1 - no_obj_mask
        sample_weights = self.no_obj_weight * no_obj_mask + obj_mask

        return tf.keras.losses.CategoricalCrossentropy()(
            y_true=labels_true,
            y_pred=labels_pred,
            sample_weight=sample_weights
        )
