from typing import Optional, Union, List, Dict

import pathlib
import numpy as np

from .base import QuiltDataset
from .keys import DatasetKeys


__all__ = ['BasketballDetectionsDataset']


# TODO: Rethink this class
class BasketballDetectionsDataset(QuiltDataset):
    r"""

    Parameters
    ----------
    hash_key
    force
    cache
    """
    user = 'hudlrd'
    package = 'basketball_detections'

    def __init__(
            self,
            hash_key: Optional[str] = None,
            force: Optional[bool] = True,
            cache: Union[str, pathlib.Path] = './dataset_cache',
    ) -> None:
        super(BasketballDetectionsDataset, self).__init__(
            user=self.user,
            package=self.package,
            hash_key=hash_key,
            force=force,
            cache=cache
        )

    def unique_boxes_labels(
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
            label_position = range(
                len(self.df[DatasetKeys.BOXES_LABELS.value][0])
            )

        uniques = [[] for _ in label_position]
        for boxes_labels in self.df[DatasetKeys.BOXES_LABELS.value]:
            for i, label in enumerate(
                    np.asanyarray(boxes_labels)[label_position]):
                if label is None:
                    continue
                uniques[i].append(label)

        return [sorted(list(set(unique))) for unique in uniques]

    def view_row(self, index: int, image_args: Dict = None):
        if image_args is None:
            from nucleus.visualize import BasketballDetectionsLabelColorMap
            image_args = dict(
                box_args=dict(
                    label_color_map=BasketballDetectionsLabelColorMap
                )
            )
        super(QuiltDataset, self).view_row(index=index, image_args=image_args)
