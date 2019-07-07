from typing import Optional, Union, Tuple, Sequence

import stringcase
import tensorflow as tf
from tensorflow.python.keras.utils import conv_utils

from nucleus.box import fix_tensor_length, unpad_tensor, ijhw_to_ijkl
from nucleus.utils import export, name_scope

from .matcher import YoloMatcher
from .anchors import create_anchors
from .data_format import get_prediction_tensor_shape


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
            activation: Optional[Union[str, callable]] = None,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.batch_norm = tf.keras.layers.BatchNormalization(
            axis=-1 if data_format is 'channels_last' else 0
        )
        self.conv = tf.keras.layers.Conv2D(
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


# TODO: Port name logic to a BaseLayer class similar to BaseLoss
# TODO: Document me!
class DetectorInferenceLayer(tf.keras.layers.Layer):
    r"""

    Parameters
    ----------
    matcher

    anchors_parameters

    max_boxes

    non_maximum_suppression

    data_format

    **kwargs

    """
    def __init__(
            self,
            scales: Union[float, Sequence[float], tf.Tensor],
            ratios: Union[float, Sequence[float], tf.Tensor],
            n_anchors: int,
            max_detections: int = 50,
            score_threshold: float = 0.5,
            nms_iou_threshold: Optional[float] = 0.75,
            data_format: Optional[str] = None,
            name: Optional[str] = None,
            **kwargs
    ) -> None:
        if name is not None:
            name = stringcase.snakecase(self.__class__.__name__)
        super().__init__(name=name, **kwargs)

        self.scales = scales
        self.ratios = ratios
        self.n_anchors = n_anchors
        self.max_detections = max_detections
        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.data_format = conv_utils.normalize_data_format(data_format)

    @tf.function
    def call(self, inputs):
        r"""

        Parameters
        ----------
        inputs

        Returns
        -------

        """
        # Get the height and width of the predicted tensor
        _, grid_height, grid_width, _, _ = get_prediction_tensor_shape(
            feature_map=inputs,
            data_format=self.data_format
        )

        # Create the anchors
        anchors = create_anchors(
            scales=self.scales,
            ratios=self.ratios,
            n_anchors=self.n_anchors,
            grid_height=grid_height,
            grid_width=grid_width
        )

        def detector_inference(tensor: tf.Tensor) -> tf.Tensor:
            r"""

            Parameters
            ----------
            tensor

            Returns
            -------

            """
            # Unmatch the prediction tensor
            detections = self.unmatch_fn(
                matched_boxes=tensor,
                anchors=anchors
            )

            detections = unpad_tensor(detections)

            # Apply non maximum suppression
            if self.nms_iou_threshold is not None:
                boxes = ijhw_to_ijkl(detections[..., :4])
                scores = detections[..., 5]

                indices = tf.image.non_max_suppression(
                    boxes,
                    scores,
                    self.max_detections,
                    iou_threshold=self.nms_iou_threshold,
                    score_threshold=self.score_threshold
                )

                detections = tf.gather(detections, indices)
            else:
                detections = detections[
                    tf.greater_equal(detections[..., 5], self.score_threshold)
                ]

            coords = detections[..., :4]
            scores = detections[..., 5:6]
            labels = tf.cast(
                tf.argmax(detections[..., 6:], axis=-1),
                dtype=tf.float32
            )
            if tf.not_equal(len(labels.shape), 2):
                labels = labels[..., None]
            detections = tf.concat([coords, labels, scores], axis=-1)

            # Make sure we output the maximum number of allowed detections
            detections = fix_tensor_length(
                detections,
                max_length=self.max_detections
            )

            return detections

        return tf.map_fn(fn=detector_inference, elems=inputs)

    def get_config(self):
        config = super().get_config()
        config.update({
            'scales': self.scales,
            'ratios': self.ratios,
            'n_anchors': self.n_anchors,
            'max_detections': self.max_detections,
            'nms': self.nms,
            'nms_iou_threshold': self.nms_iou_threshold,
            'nms_score_threshold': self.nms_score_threshold,
            'data_format': self.data_format
        })
        return config


# TODO: Document me!
@export
class YoloInferenceLayer(DetectorInferenceLayer):
    r"""

    Parameters
    ----------
    matcher

    anchors_parameters

    max_boxes

    non_maximum_suppression

    data_format

    **kwargs

    """
    def __init__(
            self,
            iou_threshold: float,
            scales: Union[float, Sequence[float], tf.Tensor],
            ratios: Union[float, Sequence[float], tf.Tensor],
            n_anchors: int,
            max_detections: int = 50,
            score_threshold: float = 0.5,
            nms_iou_threshold: Optional[float] = 0.75,
            data_format: Optional[str] = None,
            name: Optional[str] = None,
            **kwargs
    ) -> None:
        super().__init__(
            scales=scales,
            ratios=ratios,
            n_anchors=n_anchors,
            max_detections=max_detections,
            score_threshold=score_threshold,
            nms_iou_threshold=nms_iou_threshold,
            data_format=data_format,
            name=name,
            **kwargs
        )
        self.iou_threshold = iou_threshold
        self.unmatch_fn = YoloMatcher(iou_threshold=iou_threshold).unmatch

    def get_config(self):
        config = super().get_config()
        config.update({
            'iou_threshold': self.iou_threshold
        })
        return config
