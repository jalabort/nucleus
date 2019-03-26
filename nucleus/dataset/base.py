from typing import Optional, Union, Tuple, List, Dict

import math
import pathlib
import numpy as np
import pandas as pd
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor

from hudl_aws.s3 import write_to_s3, ContentType, S3Location

from nucleus.base import Serializable, LazyList
from nucleus.image import Image
from nucleus.box import Box
from nucleus.types import ParsedDataset
from nucleus.utils import progress_bar

from .keys import DatasetKeys, DatasetListKeys
from .encode import _bytes_feature, _float_feature, _int64_feature
from .tools import vq as vq_tools, watson as watson_tools, quilt as quilt_tools


__all__ = ['BaseDataset', 'VqDataset',  'WatsonDataset', 'QuiltDataset']


class BaseDataset(Serializable):
    r"""
    Dataset base class.

    Parameters
    ----------
    name
    df
    cache

    Attributes
    ----------
    name
    df
    cache
    images
    """

    def __init__(
            self,
            name: str,
            df: pd.DataFrame,
            cache: Union[str, pathlib.Path] = './dataset_cache'
    ) -> None:
        self.name = name
        self.df = df
        self.cache = cache

    @property
    def df(self):
        r"""
        """
        return self._df

    @df.setter
    def df(self, df: pd.DataFrame) -> None:
        self.images = LazyList.from_index_callable(
            f=self._create_image_from_row_index,
            n_elements=len(df)
        )
        self._df = df

    @property
    def cache(self):
        r"""
        """
        return self._cache

    @cache.setter
    def cache(self, cache: Union[str, pathlib.Path]):
        self._cache = pathlib.Path(cache).absolute()
        self._cache_path = self.cache / self.name
        self.cache_path.mkdir(parents=True, exist_ok=True)

    @property
    def cache_path(self):
        r"""
        """
        return self._cache_path

    @classmethod
    def deserialize(cls, parsed: ParsedDataset) -> 'BaseDataset':
        r"""

        Parameters
        ----------
        parsed

        Returns
        -------

        """
        name = parsed.pop(DatasetKeys.NAME.value)
        df = pd.DataFrame.from_dict(parsed)
        df.index = df.index.astype(int)
        return cls(name=name, df=df)

    def serialize(self) -> dict:
        r"""

        Returns
        -------

        """
        parsed = self.df.to_dict()
        parsed[DatasetKeys.NAME.value] = self.name
        return parsed

    def save(
            self,
            parallel: bool = True,
            compress: bool = True,
            image_format: str = 'png',
            rewrite: bool = False
    ) -> None:
        r"""

        Parameters
        ----------
        parallel
        compress
        image_format
        rewrite

        Returns
        -------

        """
        if parallel:
            def _save_row_from_index(index):
                return self._save_row_from_index(
                    index=index,
                    compress=compress,
                    image_format=image_format,
                    rewrite=rewrite
                )

            with ThreadPoolExecutor() as executor:
                list(
                    progress_bar(
                        executor.map(
                            _save_row_from_index,
                            range(len(self))
                        ),
                        total=len(self)
                    )
                )
        else:
            for i in progress_bar(range(len(self)), total=len(self)):
                self._save_row_from_index(
                    index=i,
                    compress=compress,
                    image_format=image_format,
                    rewrite=rewrite
                )

        self._save(
            parsed=self.serialize(),
            path=self.cache_path / f'{self.name}.json',
            compress=compress
        )

    def _save_row_from_index(
            self,
            index: int,
            compress: bool = False,
            image_format: str = 'png',
            rewrite: bool = False
    ) -> None:
        r"""

        Parameters
        ----------
        index
        compress
        image_format
        rewrite

        Returns
        -------

        """
        row = self._get_row_from_index(index=index)

        path = row[DatasetKeys.PATH.value]
        stem = (path.rsplit('/', 1)[-1]).rsplit('.', 1)[0]
        local_path = self.cache_path / f'{stem}.{image_format}'

        if not local_path.exists() or rewrite:
            image = self._create_image_from_row(row=row)
            image.save(
                path=local_path,
                compress=compress,
                image_format=image_format,
                rewrite=True
            )
        self.df.at[index, DatasetKeys.PATH.value] = str(local_path.absolute())

    def _get_row_from_index(
            self,
            index: int
    ) -> pd.Series:
        r"""

        Parameters
        ----------
        index

        Returns
        -------

        """
        return self.df.iloc[index]

    @staticmethod
    def _create_image_from_row(row: pd.Series) -> Image:
        r"""

        Parameters
        ----------
        row

        Returns
        -------

        """
        path = row[DatasetKeys.PATH.value]
        labels = row.get(DatasetKeys.LABELS.value)
        boxes = row.get(DatasetKeys.BOXES.value)
        boxes_labels = row.get(DatasetKeys.BOXES_LABELS.value)

        if boxes is not None:
            box_list = [
                Box(ijhw=ijhw, labels=labels)
                for ijhw, labels in zip(boxes, boxes_labels)
                if None not in ijhw and all([c > 0 for c in ijhw[:2]])
            ]
        else:
            box_list = None

        return Image.from_path(
            path=path,
            labels=labels,
            box_collection=box_list
        )

    def _create_image_from_row_index(self, index: int) -> Image:
        return self._create_image_from_row(
            self._get_row_from_index(index=index)
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int) -> Tuple[pd.Series, Image]:
        row = self._get_row_from_index(index=index)
        return row, self._create_image_from_row(row=row)

    def __iter__(self) -> Tuple[pd.Series, Image]:
        for index in range(len(self)):
            yield self[index]

    # TODO: Allow images to be gzip compressed?
    def upload_images_to_s3(
            self,
            bucket: str,
            key: str,
            parallel: bool = True,
            image_format: str = 'png',
            update_df=True,
    ) -> None:
        r"""

        Parameters
        ----------
        bucket
        key
        parallel
        image_format
        update_df

        Returns
        -------

        """
        if parallel:
            def _upload_image_to_s3(index):
                return self._upload_image_from_row_index(
                    index=index,
                    bucket=bucket,
                    key=key,
                    image_format=image_format,
                    update_df=update_df
                )

            with ThreadPoolExecutor() as executor:
                list(
                    progress_bar(
                        executor.map(
                            _upload_image_to_s3,
                            range(len(self))
                        ),
                        total=len(self)
                    )
                )
        else:
            for i in progress_bar(range(len(self)), total=len(self)):
                self._upload_image_from_row_index(
                    index=i,
                    bucket=bucket,
                    key=key,
                    image_format=image_format,
                    update_df=update_df
                )

    # TODO: Check if image already exist and add rewrite flag?
    def _upload_image_from_row_index(
            self,
            index: int,
            bucket: str,
            key: str,
            image_format: str = 'png',
            update_df: bool = True
    ) -> None:
        r"""

        Parameters
        ----------
        index
        bucket
        key
        image_format

        Returns
        -------

        """
        row = self._get_row_from_index(index=index)
        image = self._create_image_from_row(row=row)

        stem = pathlib.Path(row[DatasetKeys.PATH.value]).stem
        full_key = f'{key}/{stem}.{image_format}'
        write_to_s3(
            data=image.bytes(image_format=image_format),
            bucket=bucket,
            key=full_key,
            content_type=ContentType[image_format]
        )
        if update_df:
            self.df.at[index, DatasetKeys.PATH.value] = S3Location(
                bucket, full_key
            ).path

    def create_quilt_dataset(
            self,
            user: str,
            package: str,
            readme: Optional[str] = None,
            hash_key=None
    ):
        r"""

        Parameters
        ----------
        user
        package
        readme
        hash_key

        Returns
        -------

        """
        readme = self.create_default_readme() if readme is None else readme

        quilt_tools.update_pkg(
            self.df,
            user=user,
            package=package,
            readme=readme,
            hash_key=hash_key
        )

    def create_default_readme(self) -> str:
        r"""

        Parameters
        ----------
        df

        Returns
        -------

        """
        return f'''
        # Description

        This is a `hudlrd quilt dataset` containing {len(self)} data examples. 
        The typical structure of these datasets is as follows:

        ```
             --- Columns ---

        path            object -- The image path
        labels          object -- The image labels
        boxes           object -- The image boxes
        boxes_labels    object -- The boxes labels
        n_boxes          int64 -- The number of boxes
        ```
        '''

    def create_tfrecords(self) -> None:

        def generator():
            for image in self.images:
                yield self._serialize_image(image)

    def _serialize_image(self, image: Image, image_format: str = 'png') -> Dict:

        encoded_image = image.bytes(image_format=image_format)

        return {
            'image/encoded': _bytes_feature(encoded_image),
            'image/height': _int64_feature(image.height),
            'image/width': _int64_feature(image.width),
            'image/channels': _int64_feature(image.n_channels),
            'image/labels/encoded': _bytes_feature(
                image.labels_tensor
            ),
            'image/labels/length': _int64_feature(
                len(image.labels)
            ),
            'image/boxes/encoded': _bytes_feature(
                image.box_collection.ijhw_tensor
            ),
            'boxes/boxes/length': _int64_feature(
                len(image.box_collection)
            ),
            'image/boxes/labels': _bytes_feature(
                image.box_collection.labels_list_as_int_tensor(
                    self.unique_boxes_labels
                )
            ),
        }

    def serialize_example(feature0, feature1, feature2, feature3):
        """
        Creates a tf.Example message ready to be written to a file.
        """
        # Create a dictionary mapping the feature name to the tf.Example-compatible
        # data type.


    def _to_example(
            df_row: pd.Series,
            image_encoding,
            image_shape,
            max_boxes,
    ) -> Union[tf.train.Example, None]:
        r"""
        Convert image and data to tf.Example
        Parameters
        ----------
        df_row
            Row of info in the decoded pandas dataframe restored from the quilt
            dataset.
        Returns
        -------
        example
            A serialized tensorflow protobuffer example.
        """
        source_id = df_row['source_id']
        path = df_row['path']
        image_height = df_row['image_height']
        image_width = df_row['image_width']
        n_boxes = np.int64(df_row['n_bbxs'])
        boxes = df_row['bbxs'].copy()
        proj_matrix = df_row['pmat']

        assert image_height == image_shape[0]
        assert image_width == image_shape[1]

        raw_image, image_shape = _serialize_image(path, image_encoding,
                                                  image_shape)

        if n_boxes > max_boxes:
            logger.warning(
                f'Frame {path} has {n_boxes} boxes while max_boxes is set to '
                f'{max_boxes} and will be ignored. You can set '
                f'max_boxes to a higher number.'
            )
        elif n_boxes <= 0:
            logger.warning(
                f'Frame {path} has {n_boxes} boxes and will be ignored.'
            )

        else:
            # Convert from relative to absolute coordinates
            boxes[:, [0, 2]] *= image_width
            boxes[:, [1, 3]] *= image_height

            # Convert multi-dim arrays to strings
            raw_boxes = _pad_boxes(boxes, max_boxes).tostring()
            raw_proj_matrix = proj_matrix.astype(np.float64).tostring()

            return tf.train.Example(features=tf.train.Features(
                feature={
                    'image/height': tf_int_feature(image_shape[0]),
                    'image/width': tf_int_feature(image_shape[1]),
                    'image/channels': tf_int_feature(image_shape[2]),
                    'image/encoded': tf_bytes_feature(raw_image),
                    'boxes/n_boxes': tf_int_feature(n_boxes),
                    'boxes/encoded': tf_bytes_feature(raw_boxes),
                    'frame/id': tf_int_feature(np.int64(source_id)),
                    'proj_matrix/encoded': tf_bytes_feature(raw_proj_matrix)
                }
            ))

    # TODO: Check given column contains a list
    def expand_list_column(self, column: str):
        r"""

        Parameters
        ----------
        column

        Returns
        -------

        """
        df = self.df

        length = len(self.unique_elements_from_list_column(column=column))

        df[[f'{column}_{i}' for i in range(length)]] = pd.DataFrame(
            df[column].values.tolist(),
            index=df.index
        )

        self.df = df

    # TODO: Check given column contains a list
    def unique_elements_from_list_column(
            self,
            column: str,
            # label_position: Optional[Union[int, List[int]]] = None
    ) -> List[List[str]]:
        r"""

        Parameters
        ----------
        column
        label_position

        Returns
        -------

        """
        try:
            is_list = not isinstance(self.df[column][0][0], list)
        except IndexError:
            is_list = True

        if is_list:
            # if label_position is None:
            label_position = range(
                np.max([len(labels) for labels in self.df[column]])
            )

            uniques = [[] for _ in label_position]
            for labels in self.df[column]:
                for i, label in enumerate(labels):
                    if label is None:
                        continue
                    uniques[i].append(label)

            return [sorted(list(set(unique))) for unique in uniques]
        else:
            # if label_position is None:
            label_position = range(len(self.df[column][0][0]))

            uniques = [[] for _ in label_position]
            for labels in self.df[column]:
                if not isinstance(labels, list):
                    continue
                for label in labels:
                    for i, l in enumerate(np.asanyarray(label)):
                        if l is None:
                            continue
                        uniques[i].append(l)

            return [sorted(list(set(unique))) for unique in uniques]

    def view_row(self, index: int, image_args: Dict = None):
        _, image = self[index]

        if image_args is None:
            image_args = {}

        image.view(**image_args)


class VqDataset(BaseDataset):
    r"""
    """
    # TODO: Allow instantiation via s3 path
    def __init__(
            self,
            name: str,
            bucket: str,
            key: Union[List[str], str],
            pattern: str,
            n_jobs: Optional[int] = None,
            parallel: bool = True,
            show_progress: bool = True,
            cache: Union[str, pathlib.Path] = './dataset_cache'
    ) -> None:
        self.name = name
        self.bucket = bucket
        self.key = [key] if isinstance(key, str) else key
        self.pattern = pattern
        self.n_jobs = n_jobs

        df = self._create_df_from_s3(
            bucket=self.bucket,
            key=self.key,
            pattern=self.pattern,
            n_jobs=self.n_jobs,
            parallel=parallel,
            show_progress=show_progress
        )

        super(VqDataset, self).__init__(name=name, df=df, cache=cache)

    @staticmethod
    def _create_df_from_s3(
            bucket: str,
            key: Union[List[str], str],
            pattern: str,
            n_jobs: Optional[int] = None,
            parallel: bool = True,
            show_progress: bool = True
    ) -> pd.DataFrame:
        r"""

        Parameters
        ----------
        bucket
        key
        pattern
        n_jobs
        parallel
        show_progress

        Returns
        -------

        """
        return vq_tools.create_df_from_s3(
            bucket=bucket,
            key=key,
            pattern=pattern,
            n_jobs=n_jobs,
            parallel=parallel,
            show_progress=show_progress
        )

    @classmethod
    def deserialize(cls, parsed: ParsedDataset) -> 'VqDataset':
        r"""

        Parameters
        ----------
        parsed

        Returns
        -------

        """
        name = parsed.pop('name')
        bucket = parsed.pop('bucket')
        key = parsed.pop('key')
        n_jobs = parsed.pop('n_jobs')

        df = pd.DataFrame.from_dict(parsed)
        df = df.reset_index(drop=True)
        df.index = df.index.astype(int)

        ds: VqDataset = BaseDataset(name=name, df=df)
        ds.__class__ = VqDataset
        ds.bucket = bucket
        ds.key = key
        ds.n_jobs = n_jobs

        return ds

    def serialize(self) -> dict:
        r"""

        Returns
        -------

        """
        parsed = self.df.to_dict()
        parsed['name'] = self.name
        parsed['bucket'] = self.bucket
        parsed['key'] = self.key
        parsed['n_jobs'] = self.n_jobs
        return parsed

    def reload_df_from_s3(
            self,
            n_jobs=None,
            show_progress=True
    ) -> None:
        r"""

        Parameters
        ----------
        n_jobs
        show_progress
        """
        if n_jobs is not None:
            self.n_jobs = n_jobs

        self.df = self._create_df_from_s3(
            bucket=self.bucket,
            key=self.key,
            n_jobs=self.n_jobs,
            show_progress=show_progress
        )


class WatsonDataset(VqDataset):
    r"""
    """
    # TODO: Allow instantiation via s3 path
    def __init__(
            self,
            name: str,
            bucket: str,
            key: Union[List[str], str],
            pattern: str,
            n_jobs: Optional[int] = None,
            parallel: bool = True,
            show_progress: bool = True,
            cache: Union[str, pathlib.Path] = './dataset_cache'
    ) -> None:
        self.name = name
        self.bucket = bucket
        self.key = [key] if isinstance(key, str) else key
        self.pattern = pattern
        self.n_jobs = n_jobs

        df = self._create_df_from_s3(
            bucket=self.bucket,
            key=self.key,
            pattern=self.pattern,
            n_jobs=self.n_jobs,
            parallel=parallel,
            show_progress=show_progress
        )

        super(VqDataset, self).__init__(name=name, df=df, cache=cache)

    @staticmethod
    def _create_df_from_s3(
            bucket: str,
            key: Union[List[str], str],
            pattern: str,
            n_jobs: Optional[int] = None,
            parallel: bool = True,
            show_progress: bool = True
    ) -> pd.DataFrame:
        r"""

        Parameters
        ----------
        bucket
        key
        pattern
        n_jobs
        parallel
        show_progress

        Returns
        -------

        """
        return watson_tools.create_df_from_s3(
            bucket=bucket,
            key=key,
            pattern=pattern,
            n_jobs=n_jobs,
            parallel=parallel,
            show_progress=show_progress
        )


class QuiltDataset(BaseDataset):
    r"""
    """
    # TODO: Allow instantiation via quilt path
    def __init__(
            self,
            user: str,
            package: str,
            hash_key: str = None,
            force: bool = True,
            cache: Union[str, pathlib.Path] = './dataset_cache'
    ) -> None:
        self.user = user
        self.package = package
        self.hash_key = hash_key

        df = self._create_df_from_quilt(
            user=user,
            package=package,
            hash_key=hash_key,
            force=force
        )

        super(QuiltDataset, self).__init__(name=package, df=df, cache=cache)

    @classmethod
    def _create_df_from_quilt(
            cls,
            user: str,
            package: str,
            hash_key: str = None,
            force: bool = True,
    ) -> pd.DataFrame:
        r"""

        Parameters
        ----------
        user
        package
        hash_key
        force

        Returns
        -------

        """
        return quilt_tools.get_df(
            user=user,
            package=package,
            hash_key=hash_key,
            force=force,
            column_keys=[key.value for key in DatasetListKeys]
        )

    @classmethod
    def deserialize(cls, parsed: ParsedDataset) -> 'QuiltDataset':
        user = parsed.pop('user')
        package = parsed.pop('package')
        hash_key = parsed.pop('hash_key')

        if user in cls.__dict__ and parsed.pop('user') != cls.user:
            raise RuntimeError()
        if package in cls.__dict__ and parsed.pop('package') != cls.package:
            raise RuntimeError()

        df = pd.DataFrame.from_dict(parsed)
        df = df.reset_index(drop=True)
        df.index = df.index.astype(int)

        ds = BaseDataset(name=package, df=df)
        ds.__class__ = cls
        ds: cls
        ds.hash_key = hash_key
        return ds

    def serialize(self) -> dict:
        r"""

        Returns
        -------

        """
        parsed = self.df.to_dict()
        parsed['user'] = self.user
        parsed['package'] = self.package
        parsed['hash_key'] = self.hash_key
        return parsed

    def reload_df_from_quilt(
            self,
            hash_key=None,
            force=True,
    ) -> None:
        r"""

        Parameters
        ----------
        hash_key
        force
        """
        if hash_key is not None:
            self.hash_key = hash_key

        self.df = quilt_tools.get_df(
            user=self.user,
            package=self.package,
            hash_key=self.hash_key,
            force=force,
            column_keys=[key.value for key in DatasetListKeys]
        )
