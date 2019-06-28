from typing import Tuple

import tensorflow as tf

from nucleus.utils import export, tf_get_shape


# TODO: Docstrings
@export
def feature_map_shape(
        feature_map: tf.Tensor,
        data_format: str
) -> Tuple[int, int, int, int]:
    r"""

    Parameters
    ----------
    feature_map
    data_format

    Returns
    -------

    """
    if data_format == 'channels_last':
        batch_size, grid_height, grid_width, n_channels = tf_get_shape(
            feature_map
        )
    elif data_format == 'channels_first':
        batch_size, n_channels, grid_height, grid_width = tf_get_shape(
            feature_map
        )
    else:
        raise ValueError(
            f'{data_format} is not a valid data format.'
        )
    return batch_size, grid_height, grid_width, n_channels


@export
def get_prediction_tensor_shape(
        feature_map: tf.Tensor,
        data_format: str
) -> Tuple[int, int, int, int, int]:
    r"""

    Parameters
    ----------
    feature_map
    data_format

    Returns
    -------

    """
    if data_format == 'channels_last':
        (batch_size,
         grid_height,
         grid_width,
         n_anchors,
         n_predictions) = tf_get_shape(feature_map)
    elif data_format == 'channels_first':
        (batch_size,
         n_anchors,
         n_predictions,
         grid_height,
         grid_width) = tf_get_shape(feature_map)
    else:
        raise ValueError(
            f'{data_format} is not a valid data format.'
        )
    return batch_size, grid_height, grid_width, n_anchors, n_predictions
