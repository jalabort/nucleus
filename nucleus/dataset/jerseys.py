from typing import Optional, Union, List, Dict

import numpy as np
import pandas as pd
from pathlib import Path
from public import public

from .base import QuiltDataset
from .keys import DatasetKeys


@public
class BasketballJerseysDataset(QuiltDataset):
    r"""

    Parameters
    ----------
    hash_key
    force
    cache
    """
    user = 'hudlrd'
    package = 'basketball_jerseys'

    def __init__(
            self,
            hash_key: Optional[str] = None,
            force: Optional[bool] = True,
            cache: Union[str, Path] = './dataset_cache',
    ) -> None:
        super(BasketballJerseysDataset, self).__init__(
            user=self.user,
            package=self.package,
            hash_key=hash_key,
            force=force,
            cache=cache
        )

    def unique_labels(
            self,
            label_position: Optional[Union[int, List[int]]] = None
    ) -> List[List[str]]:
        r"""

        Parameters
        ----------
        label_position

        Returns
        -------

        """
        if label_position is None:
            label_position = range(len(self.df[DatasetKeys.LABELS.value][0]))

        uniques = [[] for _ in label_position]
        for labels in self.df[DatasetKeys.LABELS.value]:
            for i, label in enumerate(np.asanyarray(labels)[label_position]):
                if label is None:
                    continue
                uniques[i].append(label)

        return [sorted(list(set(unique))) for unique in uniques]

    def create_label_count_df(
            self,
            df: Optional[pd.DataFrame] = None,
            label_position: Optional[Union[int, List[int]]] = None
    ) -> List[pd.DataFrame]:
        r"""

        Parameters
        ----------
        df
        label_position

        Returns
        -------

        """

        if df is None:
            df = self.df

        if label_position is None:
            label_position = range(len(df[DatasetKeys.LABELS.value][0]))
        elif isinstance(label_position, int):
            label_position = [label_position]

        labels_dict = {}
        for labels in df[DatasetKeys.LABELS.value]:
            for i, label in enumerate(np.asanyarray(labels)[label_position]):
                if label is None:
                    continue
                if labels_dict.get(i) is None:
                    labels_dict[i] = {}
                    labels_dict[i][label] = 1
                else:
                    if labels_dict[i].get(label) is None:
                        labels_dict[i][label] = 1
                    else:
                        labels_dict[i][label] += 1

        dfs = []
        for i in range(len(label_position)):
            data = sorted(labels_dict[i].items())
            df = pd.DataFrame.from_records(
                data=data,
                columns=['label', 'count']
            )
            dfs.append(df)

        return dfs

    # TODO: Move viewing code to visualization
    def view_labels_distributions(
            self,
            df: Optional[pd.DataFrame] = None,
            label_position: Optional[Union[int, List[int]]] = None,
            vertical: bool = False,
            return_charts: bool = False
    ) -> Optional[List[object]]:
        r"""

        Parameters
        ----------
        df
        label_position
        vertical
        return_charts

        Returns
        -------

        """
        import altair as alt
        alt.renderers.enable('notebook')

        x, y = 'label', 'count'
        if vertical:
            x, y = y, x

        charts = []
        for df in self.create_label_count_df(df=df, label_position=label_position):
            chart = alt.Chart(df).mark_bar().encode(x=x, y=y)
            chart.display()
            charts.append(chart)

        if return_charts:
            return charts

    def view_row(self, index: int, image_args: Dict = None):
        if image_args is None:
            from nucleus.visualize import BasketballJerseysLabelColorMap
            image_args = dict(
                box_args=dict(
                    label_color_map=BasketballJerseysLabelColorMap
                )
            )
        super(QuiltDataset, self).view_row(index=index, image_args=image_args)
