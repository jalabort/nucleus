from typing import Optional, Union, Sequence

import tensorflow as tf
from tensorflow.python.keras.utils import conv_utils

from nucleus.utils import export

from .data_format import feature_map_shape, get_prediction_tensor_shape


# TODO: Define
@export
def create_ssd_head(): pass


# TODO: Should this be a subclass of BaseModel?
@export
def create_yolo_head(
        feature_map: tf.Tensor,
        n_classes: int,
        n_anchors: int,
        n_layers: int,
        n_features: Union[int, Sequence[int]],
        cls_activation: Union[str, callable],
        data_format: Optional[str] = None,
) -> tf.keras.Model:
    r"""
    Creates the YOLO head as described in the original paper.

    References
    ----------
    .. [1] Joseph Redmon, et. al, "YOLO9000: Better, Faster, Stronger",
           CVPR 2017, https://arxiv.org/abs/1612.08242.

    Parameters
    ----------
    feature_map
        The feature map on top of which this detection head will be created.
    n_classes
        The number of classes to predict.
    n_anchors
        The number of anchors attached to each cell of the feature map that
        this subnet acts upon.
    n_layers
        The number of intermediate feature layers.
    n_features
        The number of features produced by the intermediate feature layers.
    cls_activation
        The activation function used for the classes prediction. Typically,
        this is set to ``sigmoid`` or ``softmax``.
    data_format
        The position of the image channels in the input dimension. Either
        ``channels_last`` (or ``None``) or ``channels_first``.

    Returns
    -------
    The RetinaNet head model as described in the original paper.
    """
    # Handle data_format appropriately
    data_format = conv_utils.normalize_data_format(data_format)

    n_classes += 1

    # Determine the expected inputs and outputs shape
    if data_format == 'channels_last':
        batch_size, grid_height, grid_width, n_channels = feature_map.shape
        input_shape = grid_height, grid_width, n_channels
    else:
        batch_size, n_channels, grid_height, grid_width, = feature_map.shape
        input_shape = n_channels, grid_height, grid_width

    # Create the input layer
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Determine the number of features in each layer
    if isinstance(n_features, int):
        n_features = [n_features for _ in range(n_layers)]
    elif isinstance(n_features, Sequence):
        n_features = list(n_features)
        assert len(n_features) == n_layers

    # Define intermediate feature layers
    x = inputs
    for n_f in n_features:
        x = tf.keras.layers.Conv2D(
            filters=n_f,
            kernel_size=3,
            padding='same',
            activation='relu',
            kernel_initializer=tf.keras.initializers.he_normal(),
            data_format=data_format,
        )(x)

    # Define final prediction layers
    yxhw_output = tf.keras.layers.Conv2D(
        filters=4 * n_anchors,
        kernel_size=3,
        padding='same',
        data_format=data_format,
    )(x)

    obj_output = tf.keras.layers.Conv2D(
        filters=2 * n_anchors,
        padding='same',
        kernel_size=3,
        data_format=data_format,
    )(x)

    cls_output = tf.keras.layers.Conv2D(
        filters=n_classes * n_anchors,
        padding='same',
        kernel_size=3,
        data_format=data_format,
    )(x)

    # Split last dimension of outputs predictions in n_anchors x n_predictions
    def _split(tensor: tf.Tensor):
        return tf.stack([
            t for t in tf.split(tensor, num_or_size_splits=n_anchors, axis=-1)
        ], axis=-2)
    yxhw_output = tf.keras.layers.Lambda(function=_split)(yxhw_output)
    if data_format == 'channels_last':
        yx_output = tf.keras.layers.Lambda(
            function=lambda t: t[..., :2]
        )(yxhw_output)
        hw_output = tf.keras.layers.Lambda(
            function=lambda t: t[..., 2:4]
        )(yxhw_output)
    else:
        yx_output = tf.keras.layers.Lambda(
            function=lambda t: t[:, :2]
        )(yxhw_output)
        hw_output = tf.keras.layers.Lambda(
            function=lambda t: t[:, 2:4]
        )(yxhw_output)
    yx_output = tf.keras.layers.Activation(activation='sigmoid')(yx_output)

    obj_output = tf.keras.layers.Lambda(function=_split)(obj_output)
    obj_output = tf.keras.layers.Softmax()(obj_output)

    cls_output = tf.keras.layers.Lambda(function=_split)(cls_output)
    cls_output = tf.keras.layers.Activation(
        activation=cls_activation
    )(cls_output)

    # Concatenate all outputs
    outputs = tf.keras.layers.Concatenate(axis=-1)(
        [yx_output, hw_output, obj_output, cls_output]
    )

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='yolo_head')


@export
def create_retina_net_head(
        n_classes: int,
        n_anchors: int,
        n_features: int = 256,
        n_layers: int = 4,
        cls_activation: Union[str, callable] = 'sigmoid',
        data_format: Optional[str] = None,
) -> tf.keras.Model:
    r"""
    Creates the RetinaNet head as described in the original paper.

    References
    ----------
    .. [1] Tsung-Yi Lin, et. al, "Focal Loss for Dense Object Detection",
           ICCV 2017, https://arxiv.org/abs/1708.02002.

    Parameters
    ----------
    n_classes
        The number of classes to predict.
    n_anchors
        The number of anchors attached to each cell of the feature map that
        this subnet acts upon.
    n_features
        The number of channels produced by the intermediate feature layers.
    n_layers
        The number of intermediate feature layers.
    cls_activation
        The activation function used for the classes prediction. Typically,
        this is set to ``sigmoid`` or ``softmax``.
    data_format
        The position of the image channels in the input dimension. Either
        ``channels_last`` (or ``None``) or ``channels_first``.

    Returns
    -------
    The RetinaNet head model as described in the original paper.
    """
    # Handle data_format appropriately
    data_format = conv_utils.normalize_data_format(data_format)

    # Determine the expected inputs and outputs shape
    if data_format == 'channels_last':
        input_shape = (None, None, n_features)
        reg_output_shape = (None, None, n_anchors, 4)
        cls_output_shape = (None, None, n_anchors, n_classes)
    else:
        input_shape = (n_features, None, None)
        reg_output_shape = (n_anchors, 4, None, None)
        cls_output_shape = (n_anchors, n_classes, None, None)

    # Create the regression subnet
    reg_subnet = _create_retina_net_reg_subnet(
        n_anchors=n_anchors,
        n_features=n_features,
        n_layers=n_layers,
        data_format=data_format
    )
    # Create the classification subnet
    cls_subnet = _create_retina_net_cls_subnet(
        n_classes=n_classes,
        n_anchors=n_anchors,
        n_features=n_features,
        n_layers=n_layers,
        cls_activation=cls_activation,
        data_format=data_format
    )

    # Create the retina net head
    inputs = tf.keras.layers.Input(shape=input_shape)
    reg_output = reg_subnet(inputs)
    cls_output = cls_subnet(inputs)

    # Reshape and concatenate outputs
    reg_output = tf.keras.layers.Reshape(
        target_shape=reg_output_shape
    )(reg_output)

    cls_output = tf.keras.layers.Reshape(
        target_shape=cls_output_shape
    )(cls_output)

    axis = -1 if data_format == 'channels_last' else 0
    outputs = tf.keras.layers.Concatenate(axis=axis, name='outputs')([
        reg_output,
        cls_output
    ])

    # Create the retina net head model
    return tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name='retina_net_head'
    )


def _create_retina_net_reg_subnet(
        n_anchors: int,
        n_features: int = 256,
        n_layers: int = 4,
        data_format: Optional[str] = None,
) -> tf.keras.Sequential:
    r"""
    Creates the RetinaNet regression subnet as described in the original paper.

    References
    ----------
    .. [1] Tsung-Yi Lin, et. al, "Focal Loss for Dense Object Detection",
           ICCV 2017, https://arxiv.org/abs/1708.02002.

    Parameters
    ----------
    n_anchors
        The number of anchors attached to each cell of the feature map that
        this subnet acts upon.
    n_features
        The number of channels produced by the intermediate feature layers.
    n_layers
        The number of intermediate feature layers.
    data_format
        The position of the image channels in the input dimension. Either
        ``channels_last`` (or ``None``) or ``channels_first``.

    Returns
    -------
    The RetinaNet regression subnet as described in the original paper.
    """
    # Create retina net regression subnet
    return _create_subnet(
        name='retina_net_reg_subnet',
        activation='linear',
        n_outputs=4 * n_anchors,
        n_features=n_features,
        n_layers=n_layers,
        data_format=data_format
    )


def _create_retina_net_cls_subnet(
        n_classes: int,
        n_anchors: int,
        n_features: int = 256,
        n_layers: int = 4,
        cls_activation: Union[str, callable] = 'sigmoid',
        data_format: Optional[str] = None,
) -> tf.keras.Sequential:
    r"""
    Creates the RetinaNet classification subnet as described in the original
    paper.

    References
    ----------
    .. [1] Tsung-Yi Lin, et. al, "Focal Loss for Dense Object Detection",
           ICCV 2017, https://arxiv.org/abs/1708.02002.

    Parameters
    ----------
    n_classes
        The number of classes to predict.
    n_anchors
        The number of anchors attached to each cell of the feature map that
        this subnet acts upon.
    n_features
        The number of channels produced by the intermediate feature layers.
    n_layers
        The number of intermediate feature layers.
    cls_activation
        The activation function used for the classes prediction. Typically,
        this is set to ``sigmoid`` or ``softmax``.
    data_format
        The position of the image channels in the input dimension. Either
        ``channels_last`` (or ``None``) or ``channels_first``.

    Returns
    -------
    The RetinaNet classification subnet as described in the original paper.
    """
    # Create retina net classification subnet
    return _create_subnet(
        name='retina_net_cls_subnet',
        activation=cls_activation,
        n_outputs=n_classes * n_anchors,
        n_features=n_features,
        n_layers=n_layers,
        data_format=data_format
    )


def _create_subnet(
        name: str,
        activation: Optional[str],
        n_outputs: int,
        n_features: int,
        n_layers: int,
        data_format: Optional[str] = None
) -> tf.keras.Sequential:
    r"""
    Auxiliary function used to create the regression and classification subnets.

    Parameters
    ----------
    name
        The name of the subnet.
    activation
        The activation function used on the final prediction layer.
    n_outputs
        The number of outputs predicted by the final prediction layer.
    n_features
        The number of channels produced by the intermediate feature layers.
    n_layers
        The number of intermediate feature layers.
    data_format
        The position of the image channels in the input dimension. Either
        ``channels_last`` (or ``None``) or ``channels_first``.

    Returns
    -------
    The subnet.
    """
    # Handle data_format appropriately
    data_format = conv_utils.normalize_data_format(data_format)

    # Determine the expected input shape
    if data_format == 'channels_last':
        input_shape = (None, None, n_features)
    else:
        input_shape = (n_features, None, None)

    # Create retina net subnet
    return tf.keras.Sequential([
        # Input layer
        tf.keras.layers.Input(shape=input_shape),

        # Feature layers
        *[tf.keras.layers.Conv2D(
            filters=n_features,
            kernel_size=3,
            activation='relu',
            data_format=data_format
        ) for _ in range(n_layers)],

        # Prediction layer
        tf.keras.layers.Conv2D(
            filters=n_outputs,
            kernel_size=3,
            activation=activation,
            data_format=data_format
        ),
    ], name=name)
