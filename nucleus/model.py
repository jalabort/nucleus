from typing import Tuple, List

import numpy as np
import tensorflow.python as tf


def compute_layers_receptive_field(
        layers: List[tf.keras.layers.Layer],
) -> List[Tuple[int, int]]:
    r"""

    Parameters
    ----------
    layers
    input_shape

    Returns
    -------

    """
    receptive_field = np.array((0, 0))
    receptive_fields = []
    for layer in layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            layer: tf.keras.layers.Conv2D

            receptive_field += layer.kernel_size
        if isinstance(layer, (tf.keras.layers.MaxPool2D,
                              tf.keras.layers.AvgPool2D)):
            layer: tf.keras.layers.MaxPool2D

            receptive_field *= layer.pool_size
        receptive_fields.append(tuple(receptive_field))

    return receptive_fields
