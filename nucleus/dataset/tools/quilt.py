from typing import Optional, Iterable

import quilt
import numpy as np
import pandas as pd
from public import public
from tempfile import NamedTemporaryFile


@public
def get_pkg(
        user: str,
        package: str,
        hash_key=None,
        force=True
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
    pkg_path = f'{user}/{package}'
    quilt.install(pkg_path, hash=hash_key, force=force)
    return quilt.load(pkg_path)


@public
def decode_df_columns(
        df: pd.DataFrame,
        column_keys: Iterable
) -> pd.DataFrame:
    r"""

    Parameters
    ----------
    df
    column_keys

    Returns
    -------

    """
    def _decode_array_column(array: np.ndarray):
        if not isinstance(array, (pd.Series, np.ndarray)):
            return array
        else:
            return [_decode_array_column(a) for a in array]

    for key in column_keys:
        if df.get(key) is not None:
            df[key] = _decode_array_column(df[key])

    return df


@public
def get_df(
        user: str,
        package: str,
        hash_key=None,
        force: bool = True,
        column_keys: Optional[Iterable] = None
) -> pd.DataFrame:
    r"""

    Parameters
    ----------
    user
    package
    hash_key
    force
    column_keys

    Returns
    -------

    """
    package = get_pkg(user=user, package=package, hash_key=hash_key, force=force)
    df = package.df()
    if column_keys is not None:
        df = decode_df_columns(df=df, column_keys=column_keys)
    return df


@public
def update_pkg(
        df: pd.DataFrame,
        user: str,
        package: str,
        readme: Optional[str] = None,
        hash_key=None
):
    r"""

    Parameters
    ----------
    df
    user
    package
    readme
    hash_key

    Returns
    -------

    """
    pkg_path = f'{user}/{package}'

    quilt.build(
        pkg_path,
        quilt.nodes.GroupNode(dict(
            author='@hudlrd'
        ))
    )

    quilt.build(
        f'{pkg_path}/df',
        quilt.nodes.DataNode(None, None, df, {})
    )

    # TODO: warn the user if readme if not provided
    if readme is not None:
        with NamedTemporaryFile() as tmp:
            tmp.write(readme.encode('UTF-8'))
            tmp.flush()
            quilt.build(f'{pkg_path}/README', tmp.model_name)

    quilt.login()
    quilt.push(pkg_path, is_public=True, hash=hash_key)