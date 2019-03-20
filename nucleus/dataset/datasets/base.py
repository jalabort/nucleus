from typing import Optional, Union, Tuple, Dict

import pathlib
import pandas as pd

from concurrent.futures import ThreadPoolExecutor

from hudl_aws.s3 import write_to_s3, ContentType, S3Location

from nucleus.base import Serializable, LazyList
from nucleus.dataset import DatasetKeys, DatasetListKeys, quilt_tools, vq_tools
from nucleus.image import Image
from nucleus.box import Box, BoxCollection
from nucleus.types import ParsedImage, ParsedDataset
from nucleus.utils import progress_bar


__all__ = ['BaseDataset', 'QuiltDataset', 'VqDataset']


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

    def view_row(self, index: int, image_args: Dict = None):
        _, image = self[index]

        if image_args is None:
            from nucleus.visualize import BasketballJerseyLabelColorMap
            image_args = dict(
                box_args=dict(
                    label_color_map=BasketballJerseyLabelColorMap
                )
            )

        image.view(**image_args)


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

    # TODO: Rethink serializable interface loading method for this class
    @classmethod
    def deserialize(cls, parsed: ParsedDataset) -> 'QuiltDataset':
        r"""

        Parameters
        ----------
        parsed

        Returns
        -------

        """
        user = parsed.pop('user')
        package = parsed.pop('package')
        hash_key = parsed.pop('hash_key')

        df = pd.DataFrame.from_dict(parsed)
        df.index = df.index.astype(int)

        return cls(user=user, package=package, hash_key=hash_key)

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


class VqDataset(BaseDataset):
    r"""
    """
    # TODO: Allow instantiation via s3 path
    def __init__(
            self,
            name: str,
            bucket: str,
            key: str,
            n_jobs: Optional[int] = None,
            parallel: bool = True,
            show_progress: bool = True,
            cache: Union[str, pathlib.Path] = './dataset_cache'
    ) -> None:
        self.name = name
        self.bucket = bucket
        self.key = key
        self.n_jobs = n_jobs

        df = self._create_df_from_s3(
            bucket=bucket,
            key=key,
            n_jobs=n_jobs,
            parallel=parallel,
            show_progress=show_progress
        )

        super(VqDataset, self).__init__(name=name, df=df, cache=cache)

    @classmethod
    def _create_df_from_s3(
            cls,
            bucket: str,
            key: str,
            n_jobs: Optional[int] = None,
            parallel: bool = True,
            show_progress: bool = True
    ) -> pd.DataFrame:
        r"""

        Parameters
        ----------
        bucket
        key
        n_jobs
        parallel
        show_progress

        Returns
        -------

        """
        return vq_tools.create_df_from_s3(
            bucket=bucket,
            key=key,
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
