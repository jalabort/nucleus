from typing import Optional, Union, Tuple, List, Dict

import pathlib
import pandas as pd

from nucleus.base import Serializable, LazyList
from nucleus.image import Image
from nucleus.box import Box, BoxCollection
from nucleus.types import ParsedDataset
from nucleus.utils import progress_bar

from .keys import DatasetKeys
from .tools import quilt_tools, watson_tools


__all__ = ['Dataset']


class Dataset(Serializable):
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
            f=self._create_image_from_row,
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
        self._cache = cache

    # TODO: Change df structure so that this method works!
    def _get_row(
            self,
            index: int
    ) -> Tuple[str, List[str], List[List[float]], List[List[str]]]:
        r"""

        Parameters
        ----------
        index

        Returns
        -------

        """
        path = self.df.iloc[index][DatasetKeys.PATH.value]
        labels = self.df.iloc[index].get(DatasetKeys.LABELS.value)
        boxes = self.df.iloc[index].get(DatasetKeys.BOXES.value)
        boxes_labels = self.df.iloc[index].get(DatasetKeys.BOXES_LABELS.value)
        return path, labels, boxes, boxes_labels

    # TODO[jalabort]: Should this live directly on Image?
    @staticmethod
    def _create_image(
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

    def _create_image_from_row(self, index) -> Image:
        return self._create_image(*self._get_row(index))

    @classmethod
    def deserialize(cls, parsed: ParsedDataset) -> 'Dataset':
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
            compress: bool = False,
            image_format: str = 'png',
            rewrite: bool = False
    ) -> None:
        r"""

        Parameters
        ----------
        compress
        image_format
        rewrite

        Returns
        -------

        """
        for i, row in progress_bar(self.df.iterrows(), total=len(self)):
            image = self._create_image_from_row(index=i)

            image_path = row['path']
            image_name = (image_path.rsplit('/', 1)[-1])
            local_path = self.cache / f'{image_name}'

            image.save(
                path=local_path,
                compress=compress,
                image_format=image_format,
                rewrite=rewrite
            )
            self.df.at[i, 'path'] = str(
                local_path.parent.absolute()
                / '.'.join([local_path.stem, image_format])
            )

        self._save(
            parsed=self.serialize(),
            path=self.cache / '.'.join([self.name, 'json']),
            compress=compress
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
    ) -> 'Dataset':
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
            pattern: str,
            n_jobs: Optional[int] = None,
            cache: Union[str, pathlib.Path] = './dataset_cache',
            show_progress: bool = True
    ) -> 'Dataset':
        r"""

        Parameters
        ----------
        name
        bucket
        key
        pattern
        n_jobs
        cache
        show_progress

        Returns
        -------

        """
        df = cls._from_s3(
            bucket=bucket,
            key=key,
            pattern=pattern,
            n_jobs=n_jobs,
            show_progress=show_progress
        )
        return cls(name=name, df=df, cache=cache)

    @classmethod
    def _from_s3(
            cls,
            bucket: str,
            key: str,
            pattern: str,
            n_jobs: Optional[int] = None,
            show_progress: bool = True
    ) -> pd.DataFrame:
        r"""

        Parameters
        ----------
        bucket
        key
        pattern
        n_jobs
        show_progress

        Returns
        -------

        """
        return watson_tools.create_df_from_s3(
            bucket=bucket,
            key=key,
            pattern=pattern,
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
    ) -> 'Dataset':
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
        raise NotImplemented

    def view_row(self, index: int = 0, image_args: Dict = None):
        image = self.images_lazy[index]

        if image_args is None:
            image_args = {}

        image.view(**image_args)
