from typing import Optional, Union, Tuple, List, Dict

import pathlib
import pandas as pd
import concurrent.futures

from hudl_aws.s3 import write_to_s3, ContentType, S3Location

from nucleus.base import Serializable, LazyList
from nucleus.image import Image
from nucleus.box import Box, BoxCollection
from nucleus.types import ParsedDataset
from nucleus.utils import progress_bar

from .keys import DatasetKeys
from .tools import quilt_tools, vq_tools


__all__ = ['VqDataset']


class VqDataset(Serializable):
    r"""

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
    images_lazy
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
        self.images_lazy = LazyList.from_index_callable(
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
        cache = pathlib.Path(cache) / self.name
        cache.mkdir(parents=True, exist_ok=True)
        self._cache = cache.absolute()

    @classmethod
    def deserialize(cls, parsed: ParsedDataset) -> 'VqDataset':
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
        d = self.df.to_dict()
        d[DatasetKeys.NAME.value] = self.name
        return d

    # TODO: Implement parallel saving
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
            with concurrent.futures.ThreadPoolExecutor() as executor:
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
            path=self.cache / 'dataset.json',
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
        image = self._create_image_from_row(row=row)

        image_path = row[DatasetKeys.PATH.value]
        image_name = (image_path.rsplit('/', 1)[-1])
        local_path = self.cache / f'{image_name}'

        image.save(
            path=local_path,
            compress=compress,
            image_format=image_format,
            rewrite=rewrite
        )
        self.df.at[index, DatasetKeys.PATH.value] = str(
            local_path.parent.absolute()
            / '.'.join([local_path.stem, image_format])
        )

    @classmethod
    def from_quilt(
            cls,
            name: str,
            user: str,
            pkg: str,
            hash_key=None,
            force=True,
            cache: Union[str, pathlib.Path] = './dataset_cache'
    ) -> 'VqDataset':
        r"""

        Parameters
        ----------
        name
        user
        pkg
        hash_key
        force
        cache

        Returns
        -------

        """
        df = cls._from_quilt(
            user=user,
            pkg=pkg,
            hash_key=hash_key,
            force=force,
        )
        return cls(name=name, df=df, cache=cache)

    @classmethod
    def _from_quilt(
            cls,
            user: str,
            pkg: str,
            hash_key=None,
            force=True
    ) -> pd.DataFrame:
        return quilt_tools.get_df(
            user=user,
            pkg=pkg,
            hash_key=hash_key,
            force=force
        )

    @classmethod
    def from_s3(
            cls,
            name: str,
            bucket: str,
            key: str,
            n_jobs: Optional[int] = None,
            cache: Union[str, pathlib.Path] = './dataset_cache',
            show_progress: bool = True
    ) -> 'VqDataset':
        r"""

        Parameters
        ----------
        name
        bucket
        key
        n_jobs
        cache
        show_progress

        Returns
        -------

        """
        df = cls._from_s3(
            bucket=bucket,
            key=key,
            n_jobs=n_jobs,
            show_progress=show_progress
        )
        return cls(name=name, df=df, cache=cache)

    @classmethod
    def _from_s3(
            cls,
            bucket: str,
            key: str,
            n_jobs: Optional[int] = None,
            show_progress: bool = True
    ) -> pd.DataFrame:
        r"""

        Parameters
        ----------
        bucket
        key
        n_jobs
        show_progress

        Returns
        -------

        """
        return vq_tools.create_df_from_s3(
            bucket=bucket,
            key=key,
            n_jobs=n_jobs,
            show_progress=show_progress
        )

    @classmethod
    def from_folder(
            cls,
            name: str,
            path: Union[str, pathlib.Path],
            cache: Union[str, pathlib.Path] = './dataset_cache',
            show_progress: bool = True
    ) -> 'VqDataset':
        r"""

        Parameters
        ----------
        name
        path
        cache
        show_progress

        Returns
        -------

        """
        df = cls._from_folder(path=path, show_progress=show_progress)
        return cls(name=name, df=df, cache=cache)

    @classmethod
    def _from_folder(
            cls,
            path: Union[str, pathlib.Path],
            show_progress: bool = True
    ) -> pd.DataFrame:
        r"""

        Parameters
        ----------
        path
        show_progress

        Returns
        -------

        """
        raise NotImplemented

    def __len__(self):
        return len(self.df)

    # TODO: Change df structure so that this method works!
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
    def _get_image_args_from_row(
            row: pd.Series
    ) -> Tuple[str, List[str], List[List[float]], List[List[str]]]:
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
        return path, labels, boxes, boxes_labels

    # TODO: Should this live directly on Image?
    @staticmethod
    def _create_image_from_args(
            path: str,
            labels: List[str],
            boxes: List[List[float]],
            boxes_labels: List[List[str]]
    ) -> Image:
        r"""

        Parameters
        ----------
        path
        labels
        boxes
        boxes_labels

        Returns
        -------

        """
        box_collection = None
        if boxes is not None:
            box_collection = BoxCollection.from_boxes(
                [Box(ijhw=ijhw, labels=labels)
                 for ijhw, labels in zip(boxes, boxes_labels)]
            )
        return Image.from_path(
            path=path,
            labels=labels,
            box_collection=box_collection
        )

    def _create_image_from_row(self, row: pd.Series) -> Image:
        return self._create_image_from_args(
            *self._get_image_args_from_row(row)
        )

    def _create_image_from_row_index(self, index: int) -> Image:
        return self._create_image_from_row(
            self._get_row_from_index(index=index)
        )

    def update_quilt_df(
            self,
            user: str,
            pkg: str,
            readme: Optional[str] = None,
            hash_key=None
    ):
        r"""

        Parameters
        ----------
        user
        pkg
        readme
        hash_key

        Returns
        -------

        """
        if readme is None:
            readme = self.create_default_readme(self.df)
        quilt_tools.update_df(
            self.df,
            user=user,
            pkg=pkg,
            readme=readme,
            hash_key=hash_key
        )

    @classmethod
    def create_default_readme(cls, df: pd.DataFrame) -> str:
        r"""

        Parameters
        ----------
        df

        Returns
        -------

        """
        return '''
        # Description

        This is a `hudlrd quilt dataset`. The typical structure of these 
        datasets is as follows:
        
        ```
             --- Columns ---

        path            object -- The image path
        labels          object -- The image labels
        boxes           object -- The image boxes
        boxes_labels    object -- The boxes labels
        n_boxes          int64 -- The number of boxes
        ```
        '''

    def view_row(self, index: int = 0, image_args: Dict = None):
        from nucleus.visualize.color_maps import BasketballJerseyLabelColorMap

        image = self.images_lazy[index]

        if image_args is None:
            image_args = dict(
                box_args=dict(
                    label_color_map=BasketballJerseyLabelColorMap,
                )
            )

        image.view(**image_args)

    def __iter__(self) -> Tuple[pd.Series, Image]:
        for index in range(len(self)):
            row = self._get_row_from_index(index=index)
            yield row, self._create_image_from_row(row=row)

    def __getitem__(self, index: int) -> Tuple[pd.Series, Image]:
        row = self._get_row_from_index(index=index)
        return row, self._create_image_from_row(row=row)

    # TODO: Allow compressed?
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
            with concurrent.futures.ThreadPoolExecutor() as executor:
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
            for row, image in progress_bar(self):
                stem = pathlib.Path(row[DatasetKeys.PATH.value]).stem
                full_key = f'{key}/{stem}.{image_format}'
                write_to_s3(
                    data=image.bytes(image_format=image_format),
                    bucket=bucket,
                    key=full_key,
                    content_type=ContentType[image_format]
                )
                if update_df:
                    row[DatasetKeys.PATH.value] = S3Location(bucket, full_key).path

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
            row[DatasetKeys.PATH.value] = S3Location(bucket, full_key).path
