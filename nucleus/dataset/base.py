from typing import Optional, Union, Tuple, List, Dict

import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from warnings import warn
from stringcase import snakecase
from concurrent.futures import ThreadPoolExecutor

from hudl_aws.s3 import write_to_s3, ContentType, S3Location

from nucleus.base import Serializable, LazyList
from nucleus.image import Image
from nucleus.box import Box
from nucleus.types import ParsedDataset
from nucleus.utils import export, progress_bar

from .keys import DatasetKeys, DatasetListKeys, DatasetSplitKeys, DatasetPartitionKeys
from .tools import vq as vq_tools, watson as watson_tools, quilt as quilt_tools


@export
class BaseDataset(Serializable):
    r"""
    Dataset base class.

    Parameters
    ----------
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
            cache: Union[str, Path] = Path.home() / '.hudlrd' / 'dataset_cache'
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
    def cache(self, cache: Union[str, Path]):
        self._cache = Path(cache).absolute()
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
        cache = parsed.pop(DatasetKeys.CACHE.value)
        df = pd.DataFrame.from_dict(parsed)
        df = df.reset_index(drop=True)
        df.index = df.index.astype(int)
        return BaseDataset(name=name, df=df, cache=cache)

    def serialize(self) -> dict:
        r"""

        Returns
        -------

        """
        parsed = self.df.to_dict()
        parsed[DatasetKeys.NAME.value] = self.name
        parsed[DatasetKeys.CACHE.value] = str(self.cache)
        return parsed

    def save(
            self,
            parallel: bool = True,
            compress: bool = False,
            image_format: str = 'png',
            rewrite: bool = False,
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

    def n_partition_examples(
            self,
            partition: Union[DatasetPartitionKeys, str],
            split_column: Union[DatasetSplitKeys, str] = None,
    ) -> int:
        r"""
        Returns the number of examples of a particular partition key of a
        particular split column.

        Parameters
        ----------
        partition
            The partition key of the dataset split for which we want to
            count the number of examples.
        split_column
            The column containing the dataset split.
        """
        if split_column is None:
            split_column = DatasetSplitKeys.RANDOM
        else:
            split_column = DatasetSplitKeys(split_column)

        if self.df.get(split_column.value) is None:
            raise ValueError()

        partition = DatasetPartitionKeys(partition)

        return len(self.df[self.df[split_column.value] == partition.value])

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

        stem = Path(row[DatasetKeys.PATH.value]).stem
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

    def create_random_split(
            self,
            val_prop: float = 0.1,
            test_prop: float = 0.1
    ) -> None:
        r"""

        Parameters
        ----------
        val_prop
        test_prop

        Returns
        -------

        """
        self.df[DatasetSplitKeys.RANDOM.value] = DatasetPartitionKeys.TRAIN.value

        for i, row in progress_bar(self.df.iterrows(),
                                   total=len(self),
                                   desc='Randomizing'):
            sample = tf.random.uniform((), minval=0, maxval=1)
            if sample <= val_prop:
                self.df.at[i, DatasetSplitKeys.RANDOM.value] = (
                    DatasetPartitionKeys.VAL.value
                )
            if val_prop < sample <= test_prop + val_prop:
                self.df.at[i, DatasetSplitKeys.RANDOM.value] = (
                    DatasetPartitionKeys.TEST.value
                )

    def convert(
            self,
            split_column: Union[str] = None,
            image_format: str = 'png',
            parallel: bool = True,
            rewrite: bool = False
    ) -> None:
        r"""
        Converts the dataset to the tfrecord binary format.

        Parameters
        ----------
        split_column
            The column containing the dataset split.
        image_format
            The format in which the images will be encoded.
        parallel
            Whether to use multiple processes to generate the tfrecord
            binary file or not.
        rewrite
        """
        if split_column is None:
            split_column = DatasetSplitKeys.RANDOM
        else:
            split_column = DatasetSplitKeys(split_column)

        if self.df.get(split_column.value) is None:
            raise ValueError()

        if parallel:
            def _convert_partition(part):
                return self.convert_partition(
                    partition=part,
                    split_column=split_column,
                    image_format=image_format,
                    parallel=parallel,
                    rewrite=rewrite
                )

            with ThreadPoolExecutor() as executor:
                list(
                    progress_bar(
                        executor.map(
                            _convert_partition,
                            DatasetPartitionKeys
                        ),
                        total=len(DatasetPartitionKeys),
                        desc='Converting'
                    )
                )
        else:
            for partition in progress_bar(DatasetPartitionKeys,
                                          total=len(DatasetPartitionKeys),
                                          desc='Converting'):
                self.convert_partition(
                    partition=partition,
                    split_column=split_column,
                    image_format=image_format,
                    parallel=parallel,
                    rewrite=rewrite
                )

    def convert_partition(
            self,
            partition: Union[DatasetPartitionKeys, str],
            split_column: Union[DatasetSplitKeys, str] = None,
            image_format: str = 'png',
            parallel: bool = True,
            rewrite: bool = False
    ) -> None:
        r"""

        Parameters
        ----------
        partition
            The partition key to be converted.
        split_column
            The column containing the dataset split.
        image_format
            The format in which the images will be encoded.
        parallel
            Whether to use multiple processes to generate the tfrecord
            binary file or not.
        rewrite

        """
        if split_column is None:
            split_column = DatasetSplitKeys.RANDOM
        else:
            split_column = DatasetSplitKeys(split_column)

        partition = DatasetPartitionKeys(partition)

        indices = self.df[self.df[split_column.value] == partition.value].index

        file_name = f'{partition.value}.tfrecord'
        file_path = self.cache_path / split_column.value / file_name
        if file_path.exists() and not rewrite:
            warn(
                f'{file_path} already exist. Set the rewrite argument to True '
                f'in order to rewrite it.'
            )
            return

        file_path.parent.mkdir(parents=True, exist_ok=True)

        with tf.io.TFRecordWriter(str(file_path)) as writer:
            if parallel:
                def _write_image(img):
                    return self._write_example(
                        writer=writer,
                        image=img,
                        image_format=image_format
                    )

                with ThreadPoolExecutor() as executor:
                    list(
                        progress_bar(
                            executor.map(
                                _write_image,
                                self.images[indices]
                            ),
                            total=len(indices),
                            desc=partition.name
                        )
                    )
            else:
                for image in progress_bar(self.images[indices],
                                          desc=partition.name):
                    self._write_example(
                        writer=writer,
                        image=image,
                        image_format=image_format,
                    )

    def _write_example(
            self,
            writer: tf.io.TFRecordWriter,
            image: Image,
            image_format: str = 'png'
    ) -> None:
        r"""

        Parameters
        ----------
        writer
        image
        image_format

        Returns
        -------

        """
        if image is not None:
            example = self._serialize_example(image, image_format)
            writer.write(example)

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
        raise NotImplemented()

    def get_ds(
            self,
            partition: Union[DatasetPartitionKeys, str],
            split_column: Optional[Union[DatasetPartitionKeys, str]] = None,
            n_examples: Optional[int] = None,
            shuffle: Optional[int] = 100,
            repeat: Optional[int] = 1,
            batch: Optional[int] = 1,
            prefetch: Optional[int] = tf.data.experimental.AUTOTUNE,
            transform_fn: callable = lambda *args: args,
    ) -> tf.data.Dataset:
        r"""

        Parameters
        ----------
        partition
        split_column
        n_examples
        shuffle
        repeat
        batch
        prefetch
        transform_fn

        Returns
        -------

        """
        if split_column is None:
            split_column = DatasetSplitKeys.RANDOM
        else:
            split_column = DatasetSplitKeys(split_column)

        partition = DatasetPartitionKeys(partition)

        file_name = f'{partition.value}.tfrecord'
        file_path = self.cache_path / split_column.value / file_name
        if not file_path.exists():
            raise ValueError()

        files = tf.data.Dataset.list_files(str(file_path))
        ds = files.interleave(
            tf.data.TFRecordDataset,
            cycle_length=tf.data.experimental.AUTOTUNE,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        if n_examples is not None:
            ds = ds.take(n_examples)

        if shuffle is not None:
            ds = ds.shuffle(buffer_size=shuffle)

        if repeat is not None:
            ds = ds.repeat(count=repeat)

        ds = ds.map(
            lambda x: transform_fn(*self._parse_example(x)),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        if batch is not None:
            ds = ds.batch(batch_size=batch)

        if prefetch is not None:
            ds = ds.prefetch(buffer_size=prefetch)

        return ds

    def _parse_example(self, example_proto):
        r"""

        Parameters
        ----------
        example_proto

        Returns
        -------

        """
        raise NotImplemented()

    # TODO: Check given column contains a list
    def expand_list_column(self, column: str) -> None:
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


@export
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
            cache: Union[str, Path] = Path.home() / '.hudlrd' / 'dataset_cache'
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

        super().__init__(name=name, df=df, cache=cache)

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
        bucket = parsed.pop('bucket')
        key = parsed.pop('key')
        n_jobs = parsed.pop('n_jobs')

        df = pd.DataFrame.from_dict(parsed)
        df = df.reset_index(drop=True)
        df.index = df.index.astype(int)

        ds: VqDataset = BaseDataset(df=df)
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


@export
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
            cache: Union[str, Path] = Path.home() / '.hudlrd' / 'dataset_cache'
    ) -> None:
        super().__init__(
            name=name,
            bucket=bucket,
            key=key,
            pattern=pattern,
            n_jobs=n_jobs,
            parallel=parallel,
            show_progress=show_progress,
            cache=cache
        )

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


@export
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
            cache: Union[str, Path] = Path.home() / '.hudlrd' / 'dataset_cache'
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

        super().__init__(name=package, df=df, cache=cache)

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

        dataset = super().deserialize(parsed)
        dataset.__class__ = cls
        dataset: cls
        dataset.hash_key = hash_key
        return dataset

    def serialize(self) -> dict:
        r"""

        Returns
        -------

        """
        parsed = super().serialize()
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
