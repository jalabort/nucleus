from typing import Optional

import quilt
import tempfile
import pandas as pd


__all__ = ['get_df', 'update_df']


def get_df(
        user: str,
        pkg: str,
        hash_key=None,
        force=True
) -> pd.DataFrame:
    r"""

    Parameters
    ----------
    user
    pkg
    hash_key
    force

    Returns
    -------

    """
    pkg_path = f'{user}/{pkg}'
    quilt.install(pkg_path, hash=hash_key, force=force)
    pkg = quilt.load(pkg_path)
    return pkg.df


def update_df(
        df: pd.DataFrame,
        user: str,
        pkg: str,
        readme: Optional[str] = None,
        hash_key=None
):
    r"""

    Parameters
    ----------
    df
    user
    pkg
    readme
    hash_key

    Returns
    -------

    """
    pkg_path = f'{user}/{pkg}'

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
        with tempfile.NamedTemporaryFile() as tmp:
            tmp.write(readme.encode('UTF-8'))
            tmp.flush()
            quilt.build(f"{pkg_path}/README", tmp.name)

    quilt.login()
    quilt.push(pkg_path, is_public=True, hash=hash_key)