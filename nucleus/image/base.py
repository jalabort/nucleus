from typing import Union, Optional, Tuple, List

import os
import numpy as np
import tensorflow as tf
from pathlib import Path
from warnings import warn
from urllib.request import urlopen
from PIL import Image as PilImage

from nucleus.box import Box, BoxCollection
from nucleus.base import Serializable
from nucleus.visualize import ImageViewer
# TODO: Switch to python-aws whenever possible
from nucleus.s3 import is_s3_path, get_signed_s3_url
from nucleus.types import ParsedImage
from nucleus.utils import export

from . import tools as img_tools


@export
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
            dtype: Optional[tf.DType] = tf.float32
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
        if dtype is not None:
            chw = tf.cast(chw, dtype=dtype)

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
        box_collection = parsed.get('boxes')
        if box_collection is not None:
            box_collection = BoxCollection.deserialize(parsed=box_collection)

        return cls.from_path(
            path=parsed['path'],
            labels=parsed['labels'],
            box_collection=box_collection
        )

    def serialize(self, path: Union[str, Path]) -> ParsedImage:
        r"""

        Returns
        -------

        """
        parsed = dict(
            path=str(Path(path).absolute()),
            labels=self.labels
        )

        if self.box_collection is not None:
            parsed['boxes'] = self.box_collection.serialize()

        return parsed

    def save(
            self,
            path: Union[str, Path],
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
        path = Path(path)

        stem = path.stem
        suffix = path.suffix[1:]
        if suffix is not None:
            image_format = suffix
        name = '.'.join([stem, image_format])

        path = Path(path.parent / name)

        if not path.exists() or rewrite:
            parsed = self.serialize(path=path)
            # TODO: Use tensorflow here?
            pil_image = PilImage.fromarray(tf.cast(self.hwc, tf.uint8).numpy())
            pil_image.save(path)

            self._save(
                parsed=parsed,
                path=path.parent / '.'.join([stem, 'json']),
                compress=compress
            )

    @classmethod
    def from_path(
            cls,
            path: Union[str, Path],
            labels: List[str] = None,
            box_collection: Optional[Union[List[Box], BoxCollection]] = None
    ) -> Optional['Image']:
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

        try:
            image = cls.from_hwc(
                hwc=tf.image.decode_image(contents=contents),
                labels=labels,
                box_collection=box_collection
            )
        except tf.errors.InvalidArgumentError as e:
            warn(
                f'Unable to create image from path: {path}.'
            )
            image = None

        return image

    @classmethod
    def from_hwc(
            cls,
            hwc: Union[tf.Tensor, np.ndarray],
            labels: List[str] = None,
            box_collection: Optional[Union[List[Box], BoxCollection]] = None
    ) -> 'Image':
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
            chw=img_tools.hwc_to_chw(hwc),
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
    def resolution(self) -> tf.TensorShape:
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
        return img_tools.chw_to_hwc(self.chw)

    @hwc.setter
    def hwc(self, hwc: tf.Tensor) -> None:
        r"""


        Parameters
        ----------
        hwc

        """
        self.chw = img_tools.hwc_to_chw(hwc)

    def bytes(self, image_format: str = 'png', **kwargs) -> bytes:
        r"""

        Returns
        -------

        """
        if image_format == 'png':
            encoded = tf.image.encode_png(
                tf.cast(self.hwc, dtype=tf.uint8), **kwargs
            )
        elif image_format == 'jpeg':
            encoded = tf.image.encode_jpeg(
                tf.cast(self.hwc, dtype=tf.uint8), **kwargs
            )
        else:
            raise ValueError()

        return encoded.numpy()

    @property
    def labels_tensor(self) -> tf.Tensor:
        return tf.convert_to_tensor(self.labels)

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
            Image(
                chw=img_tools.crop_chw(self.chw, box.ijhw),
                labels=box.labels
            )
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
            view_boxes: bool = True,
            figure_id: int = None,
            new_figure: bool = False,
            **kwargs,
    ) -> None:
        r"""

        Parameters
        ----------
        view_boxes
        figure_id
        new_figure
        kwargs

        Returns
        -------

        """
        if kwargs.get('box_args'):
            box_args = kwargs.pop('box_args')
        else:
            box_args = {}

        ImageViewer(
            figure_id=figure_id,
            new_figure=new_figure,
            pixels=tf.cast(self.hwc, dtype=tf.uint8),
            labels=self.labels
        ).render(**kwargs)

        if view_boxes and self.box_collection:
            box_args['resolution'] = tuple(self.resolution)
            self.box_collection.view(
                figure_id=figure_id,
                new_figure=new_figure,
                **box_args
            )

    def view_with_grid(
            self,
            grid_shape: Tuple[int, int],
            view_boxes: bool = False,
            mask: tf.Tensor = None,
            grid_edge_color: str = 'blue',
            grid_face_color: str = 'red',
            figure_id=None,
            new_figure=False,
            **kwargs,
    ) -> None:
        r"""

        Parameters
        ----------
        grid_shape
        view_boxes
        mask
        grid_edge_color
        grid_face_color
        figure_id
        new_figure
        kwargs

        Returns
        -------

        """
        self.view(
            view_boxes=view_boxes,
            figure_id=figure_id,
            new_figure=new_figure,
            **kwargs
        )

        cells = self._create_cells(grid_shape=grid_shape).numpy()

        # TODO: This code belongs in visualization
        import matplotlib.pyplot as plt
        from matplotlib import patches
        ax = plt.gca()

        if mask is None:
            for cell in cells:
                rect = patches.Rectangle(
                    cell[:2][::-1],
                    *cell[-2:][::-1],
                    fill=False,
                    alpha=0.5,
                    edgecolor=grid_edge_color,
                    linewidth=2
                )
                ax.add_patch(rect)
        else:
            mask = tf.reshape(tf.transpose(mask, [1, 0]), (-1,))
            for cell, m in zip(cells, mask):
                rect = patches.Rectangle(
                    cell[:2][::-1],
                    *cell[-2:][::-1],
                    fill=True if m else False,
                    alpha=0.5,
                    facecolor=grid_face_color,
                    edgecolor=grid_edge_color,
                    linewidth=2
                )
                ax.add_patch(rect)

    def _create_cells(self, grid_shape: Tuple[int, int]) -> tf.Tensor:
        grid_h, grid_w = grid_shape
        img_h, img_w = self.resolution

        cell_h = img_h / grid_h
        cell_w = img_w / grid_w

        ys, xs = tf.meshgrid(tf.range(grid_h + 0.), tf.range(grid_w + 0.))
        ys *= cell_h
        xs *= cell_w

        hs = cell_h * tf.ones_like(input=ys)
        ws = cell_w * tf.ones_like(input=xs)

        cells = tf.stack([ys, xs, hs, ws], axis=-1)
        return tf.reshape(cells, shape=(-1, 4))
