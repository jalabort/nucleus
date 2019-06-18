from .managers import (
    MobileNetManager, MobileNetV2Manager, NasNetMobileManager,
    DenseNet121Manager, DenseNet169Manager, DenseNet201Manager,
    XceptionManager, InceptionV3Manager, ResNet50Manager
)

__all__ = [*managers.__all__]
