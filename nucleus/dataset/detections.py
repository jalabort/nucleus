from typing import Optional, Union, List, Dict

import pathlib
import tensorflow as tf

from nucleus.image import Image
from nucleus.box import tools as box_tools
from nucleus.types import ParsedDataset
from nucleus.utils import export

from .base import QuiltDataset
from .keys import DatasetKeys
from .encode import _bytes_feature, _int64_feature


# TODO: Rethink this class
@export
class BasketballDetectionsDataset(QuiltDataset):
    r"""

    Parameters
    ----------
    hash_key
    force
    cache
    """
    user = 'hudlrd'
    package = 'basketball_detections'

    def __init__(
            self,
            hash_key: Optional[str] = None,
            force: Optional[bool] = True,
            cache: Union[str, pathlib.Path] = './dataset_cache',
            max_serialized_boxes: int = 50
    ) -> None:
        super().__init__(
            user=self.user,
            package=self.package,
            hash_key=hash_key,
            force=force,
            cache=cache
        )
        self.max_serialized_boxes = max_serialized_boxes

    @classmethod
    def deserialize(
            cls,
            parsed: ParsedDataset
    ) -> 'BasketballDetectionsDataset':
        r"""

        Parameters
        ----------
        parsed

        Returns
        -------

        """
        max_serialized_boxes = parsed.pop('max_serialized_boxes')
        ds: cls = super().deserialize(parsed=parsed)
        ds.max_serialized_boxes = max_serialized_boxes
        return ds

    def serialize(self) -> dict:
        r"""

        Returns
        -------

        """
        parsed = super().serialize()
        parsed['max_serialized_boxes'] = self.max_serialized_boxes
        return parsed

    def _serialize_example(
            self,
            image: Image,
            image_format: str = 'png'
    ) -> bytes:
        r"""

        Parameters
        ----------
        image
        image_format

        Returns
        -------

        """
        encoded_image = image.bytes(image_format=image_format)

        # encoded_labels = image.labels_tensor.numpy().tostring()

        encoded_boxes = image.box_collection.ijhw_tensor.numpy().tostring()

        boxes_labels_tensor = image.box_collection.labels_tensor(
            unique_labels=self.unique_boxes_labels
        )
        encoded_boxes_labels = boxes_labels_tensor.numpy().tostring()

        return tf.train.Example(features=tf.train.Features(
            feature={
                'image/encoded': _bytes_feature(encoded_image),
                'image/height': _int64_feature(image.height),
                'image/width': _int64_feature(image.width),
                'image/channels': _int64_feature(image.n_channels),
                # 'labels/encoded': _bytes_feature(encoded_labels),
                # 'labels/length': _int64_feature(len(image.labels)),
                'boxes/encoded': _bytes_feature(encoded_boxes),
                'boxes/n_boxes': _int64_feature(len(image.box_collection)),
                'boxes_labels/encoded': _bytes_feature(encoded_boxes_labels),
                'boxes_labels/length': _int64_feature(
                    boxes_labels_tensor.shape[-1]
                ),
            }
        )).SerializeToString()

    def _parse_example(self, example_proto):
        r"""

        Parameters
        ----------
        example_proto

        Returns
        -------

        """
        # TODO: Add default values
        example_description = {
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
            'image/height': tf.io.FixedLenFeature([], tf.int64),
            'image/width': tf.io.FixedLenFeature([], tf.int64),
            'image/channels': tf.io.FixedLenFeature([], tf.int64),
            # 'labels/encoded': tf.io.FixedLenFeature([], tf.string),
            # 'labels/length': tf.io.FixedLenFeature([], tf.int64),
            'boxes/encoded': tf.io.FixedLenFeature([], tf.string),
            'boxes/n_boxes': tf.io.FixedLenFeature([], tf.int64),
            'boxes_labels/encoded': tf.io.FixedLenFeature([], tf.string),
            'boxes_labels/length': tf.io.FixedLenFeature([], tf.int64)
        }

        example = tf.io.parse_single_example(example_proto, example_description)

        height = tf.cast(example['image/height'], tf.int32)
        width = tf.cast(example['image/width'], tf.int32)
        channels = tf.cast(example['image/channels'], tf.int32)
        image = tf.image.decode_image(example['image/encoded'])
        image = tf.reshape(
            tf.cast(image, dtype=tf.float32),
            shape=[height, width, channels]
        )

        if height != 1080:
            image = tf.image.resize(image, size=(1080, 1920))

        # labels_length = tf.cast(example['labels/length'], tf.int32)
        # labels = tf.io.decode_raw(example['labels/encoded'], tf.float32)
        # labels = tf.reshape(labels, tf.stack([labels_length]))

        n_boxes = tf.cast(example['boxes/n_boxes'], tf.int32)
        boxes = tf.io.decode_raw(example['boxes/encoded'], tf.float32)
        boxes = tf.reshape(
            tf.cast(boxes, dtype=tf.float32),
            shape=[n_boxes, 4]
        )
        boxes = box_tools.pad_tensor(
            boxes, max_length=self.max_serialized_boxes
        )
        boxes = tf.reshape(boxes, [self.max_serialized_boxes, 4])

        boxes_labels_length = tf.cast(example['boxes_labels/length'], tf.int32)
        boxes_labels = tf.io.decode_raw(
            example['boxes_labels/encoded'], tf.int32
        )
        boxes_labels = tf.reshape(
            tf.cast(boxes_labels, dtype=tf.float32),
            shape=tf.stack([n_boxes, boxes_labels_length])
        )
        boxes_labels = box_tools.pad_tensor(
            boxes_labels, max_length=self.max_serialized_boxes
        )
        boxes_labels = tf.reshape(
            boxes_labels,
            shape=[self.max_serialized_boxes, boxes_labels_length]
        )

        boxes = tf.concat([boxes, boxes_labels], axis=-1)
        boxes.set_shape([self.max_serialized_boxes, 5])

        return image, boxes

    @property
    def unique_boxes_labels(self) -> List[List[str]]:
        r"""

        Returns
        -------

        """
        return self.unique_elements_from_list_column(
            column=DatasetKeys.BOXES_LABELS.value
        )

    def view_row(self, index: int, image_args: Dict = None):
        if image_args is None:
            from nucleus.visualize import BasketballDetectionsLabelColorMap
            image_args = dict(
                box_args=dict(
                    label_color_map=BasketballDetectionsLabelColorMap
                )
            )
        super().view_row(index=index, image_args=image_args)
