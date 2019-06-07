from typing import Union, List, Dict

import numpy as np
import tensorflow as tf


__all__ = [
    'Num',
    'Coords',
    'CoordsTensor',
    'Parsed',
    'ParsedBox',
    'ParsedBoxCollection',
    'ParsedImage',
    'ParsedDataset'
]


Num = Union[int, float]

Coords = Union[tf.Tensor, np.ndarray, List[Num]]
CoordsTensor = Union[tf.Tensor, np.ndarray, List[Coords]]

Parsed = Union[List, Dict]
ParsedBox = Dict[str, Union[Num, str]]
ParsedBoxCollection = List[Union[str, ParsedBox]]

# TODO: Double check differences between these 2. Is something wrong?
ParsedImage = Dict[str, Union[str, ParsedBoxCollection]]
ParsedDataset = Dict[str, Union[str, ParsedBoxCollection]]
