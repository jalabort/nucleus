from enum import Enum

from nucleus.utils import export


@export
class DatasetKeys(Enum):
    NAME = 'name'
    CACHE = 'cache'
    PATH = 'path'
    LABELS = 'labels'
    BOXES = 'boxes'
    BOXES_LABELS = 'boxes_labels'
    N_BOXES = 'n_boxes'


@export
class DatasetListKeys(Enum):
    LABELS = 'labels'
    BOXES = 'boxes'
    BOXES_LABELS = 'boxes_labels'


@export
class DatasetSplitKeys(Enum):
    RANDOM = 'split_random'


@export
class DatasetPartitionKeys(Enum):
    TEST = 'test'
    VAL = 'val'
    TRAIN = 'train'
