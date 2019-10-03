from enum import Enum

from nucleus.utils import export


@export
class DatasetKeys(Enum):
    NAME = 'name'
    CACHE = 'cache'
    PATH = 'path'
    ATTRS = 'attrs'
    BOXES = 'bbxs'
    LABELS = 'labels'
    N_BOXES = 'n_bbxs'


@export
class DatasetListKeys(Enum):
    ATTRS = 'attrs'
    BOXES = 'bbxs'
    LABELS = 'labels'


@export
class DatasetSplitKeys(Enum):
    RANDOM = 'split_random'


@export
class DatasetPartitionKeys(Enum):
    TEST = 'test'
    DEV = 'dev'
    TRAIN = 'train'
