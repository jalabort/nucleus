import enum


__all__ = ['DatasetKeys']


class DatasetKeys(enum.Enum):
    NAME = 'name'
    PATH = 'path'
    LABELS = 'labels'
    BOXES = 'boxes'
    BOXES_LABELS = 'boxes_labels'
    N_BOXES = 'n_boxes'
