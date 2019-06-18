from enum import Enum
from public import public


@public
class DatasetKeys(Enum):
    NAME = 'name'
    CACHE = 'cache'
    PATH = 'path'
    LABELS = 'labels'
    BOXES = 'boxes'
    BOXES_LABELS = 'boxes_labels'
    N_BOXES = 'n_boxes'


@public
class DatasetListKeys(Enum):
    LABELS = 'labels'
    BOXES = 'boxes'
    BOXES_LABELS = 'boxes_labels'


@public
class DatasetSplitKeys(Enum):
    RANDOM = 'split_random'


@public
class DatasetPartitionKeys(Enum):
    TEST = 'test'
    VAL = 'val'
    TRAIN = 'train'
