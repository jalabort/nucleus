from typing import Optional, Union

import pathlib
import pandas as pd

from nucleus.dataset.base import Dataset


__all__ = ['BasketballDataset']


S3_BUCKET = 'hudlrd-experiments'
S3_KEY = 'annotation/bball-tactical-source'
S3_PATTERN = '.*/output/'
QUILT_USER = 'hudlrd'
QUILT_PKG = 'basketball'


# TODO: Rethink this class
class BasketballDataset(Dataset):
    r"""

    Attributes
    ----------
    name
    df
    cache
    images_lazy
    """
    s3_bucket = S3_BUCKET
    s3_key = S3_KEY
    s3_pattern = S3_PATTERN
    quilt_user = QUILT_USER
    quilt_pkg = QUILT_PKG

    def __init__(
            self,
            df: pd.DataFrame,
            cache: Union[str, pathlib.Path] = './dataset_cache'
    ) -> None:
        super(BasketballDataset, self).__init__(
            name=self.__class__.__name__, df=df, cache=cache
        )

    @classmethod
    def from_quilt(
            cls,
            hash_key=None,
            force=True,
            cache: Union[str, pathlib.Path] = './dataset_cache'
    ) -> 'Dataset':
        r"""

        Parameters
        ----------
        hash_key
        force
        cache

        Returns
        -------

        """
        df = cls._from_quilt(
            user=cls.quilt_user,
            pkg=cls.quilt_pkg,
            hash_key=hash_key,
            force=force,
        )
        return cls(df=df, cache=cache)

    @classmethod
    def from_s3(
            cls,
            n_jobs: Optional[int] = None,
            cache: Union[str, pathlib.Path] = './dataset_cache',
            show_progress: bool = True,
    ) -> 'BasketballDataset':
        r"""

        Parameters
        ----------
        n_jobs
        cache
        show_progress

        Returns
        -------

        """
        df = cls._from_s3(
            bucket=cls.s3_bucket,
            key=cls.s3_key,
            pattern=cls.s3_pattern,
            n_jobs=n_jobs,
            show_progress=show_progress
        )
        return cls(df=df, cache=cache)

    def update_quilt_df(
            self,
            user=QUILT_USER,
            pkg=QUILT_PKG,
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
        super(BasketballDataset, self).update_quilt_df(
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
