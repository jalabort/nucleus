import tensorflow.python.keras as keras

from nucleus.model import BaseModel

from .backbones import create_dark_net_19
from .heads import create_yolo_head


class YoloDetectionModel(BaseModel):

    def __init__(self,):
        backbone = create_dark_net_19()
        head = create_yolo_head()
        model = keras.Sequential(
            [backbone, head],
            name='yolo_detection_model'
        )
        super().__init__(model=model)


def create_detection_model(
        backbone: Model,
        detection_head: Model,
        name: str = 'detection_model'
) -> Sequential:
    r"""

    Parameters
    ----------
    backbone
    detection_head
    name

    Returns
    -------
    The detection model.
    """
    return Sequential([backbone, detection_head], name=name)


def create_ssd_detection_model() -> Sequential:
    r"""


    Returns
    -------
    The ssd detection model as defined in the paper.
    """
    return Sequential([backbone, detection_head], name=name)