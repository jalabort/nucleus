import enum


__all__ = ['DatasetKeys', 'DatasetListKeys']


class DatasetKeys(enum.Enum):
    NAME = 'name'
    PATH = 'path'
    LABELS = 'labels'
    BOXES = 'boxes'
    BOXES_LABELS = 'boxes_labels'
    N_BOXES = 'n_boxes'


class DatasetListKeys(enum.Enum):
    LABELS = 'labels'
    BOXES = 'boxes'
    BOXES_LABELS = 'boxes_labels'
