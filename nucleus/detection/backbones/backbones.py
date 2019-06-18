from typing import Optional

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import MaxPool2D

from detection.layers import DarkNetConv, DarkNetBlock


# TODO: Make modules Sequential models
def create_dark_net_19(
        data_format: Optional[str] = None,
) -> Sequential:
    r"""
    Creates the DarkNet19 backbone as described in the original paper.

    References
    ----------
    .. [1] Joseph Redmon, et. al, "YOLO9000: Better, Faster, Stronger",
           CVPR 2017, https://arxiv.org/abs/1612.08242.

    Parameters
    ----------
    data_format
        The position of the image channels in the input dimension. Either
        ``channels_last`` (or ``None``) or ``channels_first``.

    Returns
    -------
    The DarkNet19 backbone model as described in the original paper.
    """
    return Sequential([
        # 1st block
        DarkNetConv(filters=32, kernel_size=3, data_format=data_format),

        # 1st downsample
        MaxPool2D(pool_size=2, strides=2, data_format=data_format),

        # 2nd block
        DarkNetConv(filters=64, kernel_size=3, data_format=data_format),

        # 2nd downsample
        MaxPool2D(pool_size=2, strides=2, data_format=data_format),

        # 3rd block
        DarkNetConv(filters=128, kernel_size=3, data_format=data_format),
        DarkNetConv(filters=64, kernel_size=1, data_format=data_format),
        DarkNetConv(filters=128, kernel_size=3, data_format=data_format),

        # 3rd downsample
        MaxPool2D(pool_size=2, strides=2, data_format=data_format),

        # 4th block
        DarkNetConv(filters=256, kernel_size=3, data_format=data_format),
        DarkNetConv(filters=128, kernel_size=1, data_format=data_format),
        DarkNetConv(filters=256, kernel_size=3, data_format=data_format),

        # 4th downsample
        MaxPool2D(pool_size=2, strides=2, data_format=data_format),

        # 5th block
        DarkNetConv(filters=512, kernel_size=3, data_format=data_format),
        DarkNetConv(filters=256, kernel_size=1, data_format=data_format),
        DarkNetConv(filters=512, kernel_size=3, data_format=data_format),
        DarkNetConv(filters=256, kernel_size=1, data_format=data_format),
        DarkNetConv(filters=512, kernel_size=3, data_format=data_format),

        # 5th downsample
        MaxPool2D(pool_size=2, strides=2, data_format=data_format),

        # 6th block
        DarkNetConv(filters=1024, kernel_size=3, data_format=data_format),
        DarkNetConv(filters=512, kernel_size=1, data_format=data_format),
        DarkNetConv(filters=1024, kernel_size=3, data_format=data_format),
        DarkNetConv(filters=512, kernel_size=1, data_format=data_format),
        DarkNetConv(filters=1024, kernel_size=3, data_format=data_format),
    ], name='dark_net_19')


# TODO: Make modules Sequential models
def create_dark_net_53(
        data_format: Optional[str] = None,
) -> Sequential:
    r"""
    Creates the DarkNet53 backbone as described in the original paper.

    References
    ----------
    .. [1] Joseph Redmon, et. al, "YOLOv3: An Incremental Improvement",
           ArXiv 2018, https://arxiv.org/abs/1804.02767.

    Parameters
    ----------
    data_format
        The position of the image channels in the input dimension. Either
        ``channels_last`` (or ``None``) or ``channels_first``.

    Returns
    -------
    The DarkNet53 backbone model as described in the original paper.
    """
    return Sequential([
        # 1st block
        DarkNetConv(filters=32, kernel_size=3, data_format=data_format),

        # 1st downsample
        DarkNetConv(filters=64, kernel_size=3, stride=2,
                    data_format=data_format),

        # 2nd block
        DarkNetBlock(filters=64, data_format=data_format),

        # 2nd downsample
        DarkNetConv(filters=128, kernel_size=3, stride=2,
                    data_format=data_format),

        # 3rd block
        DarkNetBlock(filters=128, data_format=data_format),
        DarkNetBlock(filters=128, data_format=data_format),

        # 3rd downsample
        DarkNetConv(filters=256, kernel_size=3, stride=2,
                    data_format=data_format),

        # 4th block
        DarkNetBlock(filters=256, data_format=data_format),
        DarkNetBlock(filters=256, data_format=data_format),
        DarkNetBlock(filters=256, data_format=data_format),
        DarkNetBlock(filters=256, data_format=data_format),
        DarkNetBlock(filters=256, data_format=data_format),
        DarkNetBlock(filters=256, data_format=data_format),
        DarkNetBlock(filters=256, data_format=data_format),
        DarkNetBlock(filters=256, data_format=data_format),

        # 4th downsample
        DarkNetConv(filters=512, kernel_size=3, stride=2,
                    data_format=data_format),

        # 5th block
        DarkNetBlock(filters=512, data_format=data_format),
        DarkNetBlock(filters=512, data_format=data_format),
        DarkNetBlock(filters=512, data_format=data_format),
        DarkNetBlock(filters=512, data_format=data_format),
        DarkNetBlock(filters=512, data_format=data_format),
        DarkNetBlock(filters=512, data_format=data_format),
        DarkNetBlock(filters=512, data_format=data_format),
        DarkNetBlock(filters=512, data_format=data_format),

        # 5th downsample
        DarkNetConv(filters=1024, kernel_size=3, stride=2,
                    data_format=data_format),

        # 6th block
        DarkNetBlock(filters=1024, data_format=data_format),
        DarkNetBlock(filters=1024, data_format=data_format),
        DarkNetBlock(filters=1024, data_format=data_format),
        DarkNetBlock(filters=1024, data_format=data_format),
    ], name='dark_net_53')
