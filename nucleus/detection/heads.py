from typing import Optional, Union

from tensorflow.python.keras import Model, Sequential
from tensorflow.python.keras.layers import (
    Input, Conv2D, Reshape, Concatenate, Activation
)


def create_yolo_head(
        n_classes: int,
        n_anchors: int,
        n_features: int = 1024,
        n_layers: int = 3,
        cls_activation: Union[str, callable] = 'sigmoid',
        data_format: Optional[str] = None,
) -> Model:
    r"""
    Creates the YOLO head as described in the original paper.

    References
    ----------
    .. [1] Joseph Redmon, et. al, "YOLO9000: Better, Faster, Stronger",
           CVPR 2017, https://arxiv.org/abs/1612.08242.

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
    # Compute the number of predictions
    n_predictions = 4 + 1 + n_classes

    # Determine the expected inputs and outputs shape
    if data_format is 'channels_last':
        input_shape = (None, None, n_features)
        output_shape = (None, None, n_anchors, n_predictions)
    else:
        input_shape = (n_features, None, None)
        output_shape = (n_anchors, n_predictions, None, None)

    # Create the yolo subnet
    yolo_subnet = _create_subnet(
        name='yolo_subnet',
        activation=None,
        n_outputs=n_predictions * n_anchors,
        n_features=n_features,
        n_layers=n_layers,
        data_format=data_format
    )

    # Define the inputs and outputs of the model
    inputs = Input(shape=input_shape)
    outputs = yolo_subnet(inputs)

    # Reshape and concatenate outputs
    outputs = Reshape(target_shape=output_shape)(outputs)
    if data_format is 'channels_last':
        yx_output = outputs[..., :2]
        hw_output = outputs[..., 2:4]
        obj_output = outputs[..., 4:5]
        cls_output = outputs[..., 5:]
    else:
        yx_output = outputs[:2, ...]
        hw_output = outputs[2:4, ...]
        obj_output = outputs[4:5, ...]
        cls_output = outputs[5:, ...]
    yx_output = Activation(activation='sigmoid')(yx_output)
    obj_output = Activation(activation='sigmoid')(obj_output)
    cls_output = Activation(activation=cls_activation)(cls_output)
    axis = -1 if data_format is 'channels_last' else 0
    outputs = Concatenate(axis=axis, name='outputs')([
        yx_output,
        hw_output,
        obj_output,
        cls_output
    ])

    # Create the yolo head model
    return Model(inputs=inputs, outputs=outputs, name='yolo_head')


def create_retina_net_head(
        n_classes: int,
        n_anchors: int,
        n_features: int = 256,
        n_layers: int = 4,
        cls_activation: Union[str, callable] = 'sigmoid',
        data_format: Optional[str] = None,
) -> Model:
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
    # Determine the expected inputs and outputs shape
    if data_format is 'channels_last':
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
    inputs = Input(shape=input_shape)
    reg_output = reg_subnet(inputs)
    cls_output = cls_subnet(inputs)

    # Reshape and concatenate outputs
    reg_output = Reshape(target_shape=reg_output_shape)(reg_output)
    cls_output = Reshape(target_shape=cls_output_shape)(cls_output)
    axis = -1 if data_format is 'channels_last' else 0
    outputs = Concatenate(axis=axis, name='outputs')([reg_output, cls_output])

    # Create the retina net head model
    return Model(inputs=inputs, outputs=outputs, name='retina_net_head')


def _create_retina_net_reg_subnet(
        n_anchors: int,
        n_features: int = 256,
        n_layers: int = 4,
        data_format: Optional[str] = None,
) -> Sequential:
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
) -> Sequential:
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
) -> Sequential:
    r"""
    Auxiliary function used to create the regression and classification
    RetinaNet subnets.

    References
    ----------
    .. [1] Tsung-Yi Lin, et. al, "Focal Loss for Dense Object Detection",
           ICCV 2017, https://arxiv.org/abs/1708.02002.

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
    The RetinaNet subnet.
    """
    # Determine the expected input shape
    if data_format is 'channels_last':
        input_shape = (None, None, n_features)
    else:
        input_shape = (n_features, None, None)

    # Create retina net subnet
    return Sequential([
        # Input layer
        Input(shape=input_shape),

        # Feature layers
        *[Conv2D(
            filters=n_features,
            kernel_size=3,
            activation='relu',
            data_format=data_format
        ) for _ in range(n_layers)],

        # Prediction layer
        Conv2D(
            filters=n_outputs,
            kernel_size=3,
            activation=activation,
            data_format=data_format
        ),
    ], name=name)
