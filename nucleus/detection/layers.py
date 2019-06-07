from typing import Optional, Union, Tuple

import tensorflow as tf

from nucleus.utils import export, name_scope


@export
class BnConv2D(tf.keras.layers.Layer):
    r"""

    Parameters
    ----------
    filters
    kernel_size
    strides
    padding
    data_format
    activation
    **kwargs
    """

    def __init__(
            self,
            filters: int,
            kernel_size: Union[int, Tuple[int, int]],
            strides: Union[int, Tuple[int, int]] = 1,
            padding: str = 'valid',
            data_format: Optional[str] = None,
            activation: Optional[str, callable] = None,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.batch_norm = tf.python.keras.layers.BatchNormalizationV2(
            axis=-1 if data_format is 'channels_last' else 0
        )
        self.conv = tf.python.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activation,
            use_bias=False
        )

    @name_scope
    @tf.function
    def call(self, inputs):
        x = self.conv(inputs)
        return self.batch_norm(x)


@export
class DarkNetConv(BnConv2D):
    r"""

    Parameters
    ----------
    filters
    kernel_size
    stride
    data_format
    **kwargs
    """

    def __init__(
            self,
            filters: int,
            kernel_size: Union[int, Tuple[int, int]],
            stride: int = 1,
            data_format: Optional[str] = None,
            **kwargs
    ) -> None:
        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=stride,
            padding='valid',
            data_format=data_format,
            activation=tf.keras.activations.relu(alpha=0.1),
            **kwargs
        )


@export
class DarkNetBlock(tf.keras.layers.Layer):
    r"""

    Parameters
    ----------
    filters
    kernel_size
    padding
    data_format
    kwargs
    """

    def __init__(
            self,
            filters: int,
            data_format: Optional[str] = None,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.conv1 = DarkNetConv(
            filters=filters // 2,
            kernel_size=1,
            data_format=data_format
        )
        self.conv2 = DarkNetConv(
            filters=filters,
            kernel_size=3,
            data_format=data_format
        )

    @name_scope
    @tf.function
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        return inputs + x


# TODO: Missing NMS
@export
class DetectionLayer(tf.keras.layers.Layer):
    r"""

    Parameters
    ----------
    filters
    kernel_size
    padding
    data_format
    kwargs
    """
    def __init__(self, anchors, score_threshold=0.5, **kwargs):
        super().__init__(**kwargs)
        self.anchors = anchors
        self.score_threshold = score_threshold

    @name_scope
    @tf.function
    def call(self, inputs):
        condition = tf.greater(inputs[..., 5], self.score_threshold)

        pred = inputs[condition]
        anchors = self.anchors[condition]

        # Compute the yx box parametrization
        pred_yx = pred[..., :2] * anchors[..., 2:4] + anchors[..., :2]

        # Compute the hw box parametrization
        pred_hw = tf.math.exp(pred[..., 2:4]) * anchors[..., 2:4]

        # Ensemble detections
        return tf.concat([pred_yx, pred_hw, pred[..., 4:]], axis=-1)
