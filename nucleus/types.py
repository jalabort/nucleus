from typing import Union, List, Dict

import numpy as np
import tensorflow as tf


__all__ = [
    'Num',
    'Coords', 'CoordsTensor',
    'ParsedBox', 'ParsedBoxCollection', 'ParsedImage', 'ParsedDataset'
]


Num = Union[int, float]

Coords = Union[tf.Tensor, np.ndarray, List[Num]]
CoordsTensor = Union[tf.Tensor, np.ndarray, List[Coords]]
ParsedBox = Dict[str, Union[Num, str]]
ParsedBoxCollection = List[Union[str, ParsedBox]]

# TODO[jalabort]: Double check these 2
ParsedImage = Dict[str, Union[str, ParsedBoxCollection]]
ParsedDataset = Dict[str, Union[str, ParsedBoxCollection]]
