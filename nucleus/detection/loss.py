import tensorflow as tf

from nucleus.utils import export, name_scope


@export
class SsdLoss(tf.keras.losses.Loss):
    r"""
    Single Shot Detection (SSD) loss function.

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
            reduction: str = 'SUM_OVER_BATCH_SIZE'
    ) -> None:
        super().__init__(reduction=reduction, name='ssd_loss')
        self.ssd_coords_loss = SsdCoordsLoss(delta=delta, reduction=reduction)
        self.ssd_labels_loss = SsdLabelsLoss(reduction=reduction)

    @name_scope
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
        labels_term = self.ssd_labels_loss(y_true=y_true, y_pred=y_pred)
        return coords_term + labels_term


@export
class SsdCoordsLoss(tf.keras.losses.Loss):
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
            reduction: str = 'SUM_OVER_BATCH_SIZE'
    ) -> None:
        super().__init__(reduction=reduction, name='ssd_coords_loss')
        self.delta = delta

    @name_scope
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
        coords_true = get_box_coords(y_true)
        coords_pred = get_box_coords(y_pred)

        mask = tf.greater(tf.reduce_sum(coords_true, axis=-1), 0)

        return tf.keras.losses.huber_loss(
            y_true=coords_true[mask],
            y_pred=coords_pred[mask],
            delta=self.delta
        )


@export
class SsdLabelsLoss(tf.keras.losses.Loss):
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
            reduction: str = 'SUM_OVER_BATCH_SIZE'
    ) -> None:
        super().__init__(reduction=reduction, name='ssd_labels_loss')

    @name_scope
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
        labels_true = get_box_labels(y_true)
        labels_pred = get_box_labels(y_pred)

        return tf.keras.losses.categorical_crossentropy(
            y_true=labels_true,
            y_pred=labels_pred
        )


@name_scope
def get_box_coords(boxes: tf.Tensor) -> tf.Tensor:
    r"""
    Returns the coordinates of the given bounding boxes.

    Parameters
    ----------
    boxes
        ``(..., 4 + n_labels + 1)`` tensor representing bounding boxes.

    Returns
    -------
    boxes_coords
        ``(..., 4)`` tensor containing the coordinates of the bounding boxes.
    """
    return boxes[..., :4]


@name_scope
def get_box_labels(boxes: tf.Tensor) -> tf.Tensor:
    r"""
    Returns the one hot encoded labels of the given bounding boxes.

    Parameters
    ----------
    boxes
        ``(..., 4 + n_labels + 1)`` tensor representing bounding boxes.

    Returns
    -------
    boxes_labels
        ``(..., n_labels + 1)`` tensor containing the one hot encoded labels
        of the bounding boxes.
    """
    return boxes[..., 4:]
