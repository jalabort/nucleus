from typing import Union, Optional, Tuple, List, Dict

import os
import io
import pathlib
from urllib.request import urlopen

import numpy as np
import tensorflow as tf
import PIL.Image as PilImage

from nucleus.box import Box, BoxCollection

from nucleus.base import Serializable
from nucleus.dataset import DatasetKeys
from nucleus.visualize import ImageViewer
# TODO: Use python-aws
from nucleus.s3 import is_s3_path, get_signed_s3_url
from nucleus.types import ParsedImage

from .functions import chw_to_hwc, hwc_to_chw, crop_chw


class Image(Serializable):
    r"""

    Parameters
    ----------
    chw
    labels
    box_collection

    Attributes
    ----------
    chw
    box_collection
    """
    def __init__(
            self,
            chw: Union[tf.Tensor, np.ndarray],
            labels: Optional[Union[List[str], str]] = None,
            box_collection: Optional[Union[List[Box], BoxCollection]] = None,
    ) -> None:

        if chw.ndim < 2:
            raise ValueError(
                f'Pixel array has to be 2D (implicitly 1 channel, '
                f'2D shape) or 3D (n_channels, 2D shape) '
                f' - a {chw.ndim}D array was provided'
            )

        if not isinstance(chw, tf.Tensor):
            chw = tf.convert_to_tensor(chw)
        if chw.ndim == 2:
            chw = chw[None]

        self.chw = chw
        self.labels = labels if not isinstance(labels, str) else [labels]

        if (box_collection is not None
                and isinstance(box_collection, list)):
            box_collection = BoxCollection.from_boxes(boxes=box_collection)

        self.box_collection = box_collection

    @classmethod
    def deserialize(cls, parsed: ParsedImage) -> 'Image':
        r"""

        Parameters
        ----------
        parsed

        Returns
        -------

        """
        box_collection = parsed.get(DatasetKeys.BOXES.value)
        if box_collection is not None:
            box_collection = BoxCollection.deserialize(parsed=box_collection)

        return cls.from_path(
            path=parsed['path'],
            labels=parsed['labels'],
            box_collection=box_collection
        )

    def serialize(self, path: Union[str, pathlib.Path]) -> ParsedImage:
        r"""

        Returns
        -------

        """
        parsed = dict(
            path=str(pathlib.Path(path).absolute()),
            labels=self.labels
        )

        if self.box_collection is not None:
            parsed['boxes'] = self.box_collection.serialize()

        return parsed

    def save(
            self,
            path: Union[str, pathlib.Path],
            compress: bool = False,
            image_format: str = 'png',
            rewrite: bool = False
    ) -> None:
        r"""

        Parameters
        ----------
        path
        compress
        image_format
        rewrite

        Returns
        -------

        """
        path = pathlib.Path(path)

        stem = path.stem
        suffix = path.suffix[1:]
        if suffix is not None:
            image_format = suffix
        name = '.'.join([stem, image_format])

        path = pathlib.Path(path.parent / name)

        if not path.exists() or rewrite:
            parsed = self.serialize(path=path)
            # TODO: Use tensorflow here?
            pil_image = PilImage.fromarray(self.np_hwc())
            pil_image.save(path)

            self._save(
                parsed=parsed,
                path=path.parent / '.'.join([stem, 'json']),
                compress=compress
            )

    @classmethod
    def from_path(
            cls,
            path: Union[str, pathlib.Path],
            labels: List[str] = None,
            box_collection: Optional[Union[List[Box], BoxCollection]] = None
    ) -> 'Image':
        r"""

        Parameters
        ----------
        path
        labels
        box_collection

        Returns
        -------

        """
        if os.path.exists(path):
            contents = tf.io.read_file(path)
        else:
            if is_s3_path(path):
                path = get_signed_s3_url(path)
            contents = urlopen(path).read()

        return cls.from_hwc(
            hwc=tf.image.decode_image(contents=contents),
            labels=labels,
            box_collection=box_collection
        )

    @classmethod
    def from_hwc(
            cls,
            hwc: Union[tf.Tensor, np.ndarray],
            labels: List[str] = None,
            box_collection: Optional[Union[List[Box], BoxCollection]] = None
    )-> 'Image':
        r"""

        Parameters
        ----------
        hwc
        labels
        box_collection

        Returns
        -------

        """
        return cls(
            chw=hwc_to_chw(hwc),
            labels=labels,
            box_collection=box_collection
        )

    @property
    def n_channels(self) -> int:
        r"""

        Returns
        -------

        """
        return self.chw.shape[0]

    @property
    def width(self) -> int:
        r"""

        Returns
        -------

        """
        return self.chw.shape[-1]

    @property
    def height(self) -> int:
        r"""

        Returns
        -------

        """
        return self.chw.shape[-2]

    @property
    def n_dims(self) -> int:
        r"""

        Returns
        -------

        """
        return len(self.shape)

    @property
    def resolution(self) -> Tuple[int, int]:
        r"""

        Returns
        -------

        """
        return self.chw.shape[1:]

    @property
    def shape(self) -> Tuple[int, int, int]:
        r"""

        Returns
        -------

        """
        return self.chw.shape

    @property
    def n_elements(self) -> int:
        r"""

        Returns
        -------

        """
        return self.chw.size

    @property
    def n_pixels(self) -> int:
        r"""

        Returns
        -------

        """
        return self.chw[0, ...].size

    @property
    def hwc(self) -> tf.Tensor:
        r"""

        Returns
        -------

        """
        return chw_to_hwc(self.chw)

    def np_chw(self) -> tf.Tensor:
        r"""

        Returns
        -------

        """
        return self.chw.numpy()

    def np_hwc(self) -> tf.Tensor:
        r"""

        Returns
        -------

        """
        return self.hwc.numpy()

    def bytes(self, image_format: str = 'png') -> bytes:
        r"""

        Returns
        -------

        """
        pil_image = PilImage.fromarray(self.np_hwc())
        byte_array = io.BytesIO()
        pil_image.save(byte_array, format=image_format)
        return byte_array.getvalue()

    def images_from_box_collection(
            self,
            skip_labels: Union[str, List[str]] = None
    ) -> List['Image']:
        r"""

        Parameters
        ----------
        skip_labels

        Returns
        -------

        """
        if skip_labels is str:
            skip_labels = [skip_labels]

        return [
            Image(chw=crop_chw(self.chw, box.ijhw), labels=box.labels)
            for box in self.box_collection.boxes()
            if skip_labels is None or (
                all(label not in skip_labels for label in box.labels)
                and
                len(box.labels) > 0
            )
        ]

    def __str__(self) -> str:
        r"""

        Returns
        -------

        """
        return (
            f'{self.width}H x {self.height}W Image with '
            f'{self.n_channels} channel{"s" * (self.n_channels > 1)}'
        )

    def view(
            self,
            figure_id=None,
            new_figure=False,
            box_args: Union[Dict] = None,
            **kwargs,
    ) -> None:
        ImageViewer(
            figure_id=figure_id,
            new_figure=new_figure,
            pixels=self.hwc,
            labels=self.labels
        ).render(**kwargs)

        if box_args is None:
            box_args = {}

        box_args['resolution'] = self.resolution
        if self.box_collection:
            self.box_collection.view(
                figure_id=figure_id,
                new_figure=new_figure,
                **box_args
            )
