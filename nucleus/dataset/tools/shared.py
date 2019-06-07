from typing import Union, Iterable, Dict

import pandas as pd

from nucleus.types import Num
from nucleus.utils import export

from ..keys import DatasetKeys


@export
def create_df_from_examples(
        examples: Iterable[Dict[str, Union[Num, str]]]
) -> pd.DataFrame:
    r"""

    Parameters
    ----------
    examples

    Returns
    -------

    """
    df = pd.DataFrame(examples)
    if df.empty:
        raise RuntimeError(
            'The `examples` iterator is empty. If you are creating the '
            'iterator automatically using other `watson_tools` functions '
            'make sure your `s3` parameters are correct.'
        )
    df = df[df[DatasetKeys.N_BOXES.value] > 0]
    df = df.reset_index(drop=True)
    return df
