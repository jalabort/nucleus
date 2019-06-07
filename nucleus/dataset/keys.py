import enum

from nucleus.utils import export


@export
class DatasetKeys(enum.Enum):
    NAME = 'name'
    CACHE = 'cache'
    PATH = 'path'
    LABELS = 'labels'
    BOXES = 'boxes'
    BOXES_LABELS = 'boxes_labels'
    N_BOXES = 'n_boxes'


@export
class DatasetListKeys(enum.Enum):
    LABELS = 'labels'
    BOXES = 'boxes'
    BOXES_LABELS = 'boxes_labels'


@export
class DatasetSplitKeys(enum.Enum):
    RANDOM = 'split_random'


@export
class DatasetPartitionKeys(enum.Enum):
    TEST = 'test'
    VAL = 'val'
    TRAIN = 'train'
