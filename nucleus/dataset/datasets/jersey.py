from typing import Optional, Union, List


import pathlib
import numpy as np
import pandas as pd

from nucleus.dataset import DatasetKeys

from .base import QuiltDataset


class BasketballJerseyDataset(QuiltDataset):
    r"""
    """
    user = 'hudlrd'
    package = 'basketball_jerseys'

    def __init__(
            self,
            hash_key: Optional[str] = None,
            force: Optional[bool] = True,
            cache: Union[str, pathlib.Path] = './dataset_cache',
    ) -> None:
        super(BasketballJerseyDataset, self).__init__(
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
            label_position: Optional[Union[int, List[int]]] = None
    ) -> List[pd.DataFrame]:
        r"""

        Returns
        -------

        """
        if label_position is None:
            label_position = range(len(self.df[DatasetKeys.LABELS.value][0]))

        labels_dict = {}
        for labels in self.df[DatasetKeys.LABELS.value]:
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
        for i in label_position:
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
            label_position: Optional[Union[int, List[int]]] = None,
            vertical: bool = False,
            return_charts: bool = False
    ) -> Optional[List[object]]:
        r"""

        Returns
        -------

        """
        import altair as alt
        alt.renderers.enable('notebook')

        x, y = 'label', 'count'
        if vertical:
            x, y = y, x

        charts = []
        for df in self.create_label_count_df(label_position=label_position):
            chart = alt.Chart(df).mark_bar().encode(x=x, y=y)
            chart.display()
            charts.append(chart)

        if return_charts:
            return charts
