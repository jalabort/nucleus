from typing import Union, Optional, List, Iterable

import warnings
import numpy as np
import tensorflow as tf
from public import public

from nucleus.base import Serializable
from nucleus.visualize import BoxViewer
from nucleus.types import (
    Num, ParsedBox, Coords, CoordsTensor, ParsedBoxCollection
)

from . import tools as box_tools


# TODO[jalabort]: There are score and label_scores
@public
class Box(Serializable):
    r"""

    Parameters
    ----------
    ijhw
    labels
    scores

    Attributes
    ----------
    ijhw
    labels
    scores
    """
    def __init__(
            self,
            ijhw: Coords,
            labels: Optional[Union[List[str], str]] = None,
            scores: Optional[Union[List[float], float]] = None
    ) -> None:
        self.ijhw = tf.cast(tf.convert_to_tensor(ijhw), dtype=tf.float32)
        self.labels = labels if not isinstance(labels, str) else [labels]
        self.scores = scores if not isinstance(scores, str) else [scores]

    @classmethod
    def deserialize(cls, parsed: ParsedBox) -> 'Box':
        r"""

        Parameters
        ----------
        parsed

        Returns
        -------

        """
        ijhw = [parsed['i'], parsed['j'], parsed['h'], parsed['w']]
        labels = parsed.get('labels', None)
        scores = parsed.get('scores', None)
        return cls(ijhw=ijhw, labels=labels, scores=scores)

    def serialize(self) -> ParsedBox:
        r"""

        Returns
        -------

        """
        return dict(
            i=float(self.i.numpy()),
            j=float(self.j.numpy()),
            h=float(self.h.numpy()),
            w=float(self.w.numpy()),
            labels=self.labels,
            score=self.scores
        )

    @classmethod
    def from_yxhw(
            cls,
            yxhw: tf.Tensor,
            labels: Optional[Union[List[str], str]] = None,
            scores: Optional[Union[List[float], float]] = None
    ) -> 'Box':
        r"""

        Parameters
        ----------
        yxhw
        labels
        scores

        Returns
        -------

        """
        return cls(
            ijhw=box_tools.yxhw_to_ijhw(yxhw),
            labels=labels,
            scores=scores
        )

    @classmethod
    def from_ijkl(
            cls,
            ijkl: tf.Tensor,
            labels: Optional[Union[List[str], str]] = None,
            scores: Optional[Union[List[float], float]] = None
    ) -> 'Box':
        r"""

        Parameters
        ----------
        ijkl
        labels
        scores

        Returns
        -------

        """
        return cls(
            ijhw=box_tools.ijkl_to_ijhw(ijkl),
            labels=labels,
            scores=scores
        )

    @classmethod
    def from_xywh(
            cls,
            xywh: tf.Tensor,
            labels: Optional[Union[List[str], str]] = None,
            scores: Optional[Union[List[float], float]] = None
    ) -> 'Box':
        r"""

        Parameters
        ----------
        xywh
        labels
        scores

        Returns
        -------

        """
        return cls(
            ijhw=box_tools.swap_axes_order(xywh),
            labels=labels,
            scores=scores
        )

    @property
    def i(self) -> tf.Tensor:
        r"""

        Returns
        -------

        """
        return self.ijhw[0]

    @property
    def j(self) -> tf.Tensor:
        r"""

        Returns
        -------

        """
        return self.ijhw[1]

    @property
    def h(self) -> tf.Tensor:
        r"""

        Returns
        -------

        """
        return self.ijhw[2]

    @property
    def w(self) -> tf.Tensor:
        r"""

        Returns
        -------

        """
        return self.ijhw[3]

    @property
    def ij(self) -> tf.Tensor:
        r"""

        Returns
        -------

        """
        return self.ijhw[:2]

    @property
    def hw(self) -> tf.Tensor:
        r"""

        Returns
        -------

        """
        return self.ijhw[2:4]

    @property
    def kl(self) -> tf.Tensor:
        r"""

        Returns
        -------

        """
        return box_tools.ijhw_to_kl(self.ijhw)

    @property
    def yx(self) -> tf.Tensor:
        r"""

        Returns
        -------

        """
        return box_tools.ijhw_to_yx(self.ijhw)

    @property
    def ijkl(self):
        r"""

        Returns
        -------

        """
        return box_tools.ijhw_to_ijkl(self.ijhw)

    @property
    def yxhw(self):
        r"""

        Returns
        -------

        """
        return box_tools.ijhw_to_yxhw(self.ijhw)

    def area(self) -> float:
        r"""

        Returns
        -------

        """
        return tf.reduce_prod(self.ijhw[..., 2:4])

    def intersection_box(self, box: 'Box', verbose=True):
        r"""

        Parameters
        ----------
        box
        verbose

        Returns
        -------

        """
        i = max(self.i, box.i)
        j = max(self.j, box.j)
        h = max(0.0, min(self.i + self.h, box.i + box.h) - i)
        w = max(0.0, min(self.j + self.w, box.j + box.w) - j)

        if (h == 0 or w == 0) and verbose:
            warnings.warn('Boxes do not intersect')

        return self.__class__(ijhw=[i, j, h, w])

    def intersection(self, box: 'Box') -> float:
        r"""

        Parameters
        ----------
        box

        Returns
        -------

        """
        return self.intersection_box(box).area()

    def union(self, box: 'Box') -> float:
        r"""

        Parameters
        ----------
        box

        Returns
        -------

        """
        return self.area() + box.area() - self.intersection(box)

    def iou(self, box: 'Box') -> float:
        r"""

        Parameters
        ----------
        box

        Returns
        -------

        """
        intersection = self.intersection(box)
        union = self.area() + box.area() - intersection
        return 0.0 if union <= 0.0 else intersection / union

    def __str__(self) -> str:
        r"""

        Returns
        -------

        """
        raise NotImplemented()

    def view(
            self,
            figure_id: Optional[int] = None,
            new_figure: Optional[bool] = False,
            **kwargs
    ) -> None:
        r"""

        Parameters
        ----------
        figure_id
        new_figure
        kwargs

        """
        BoxViewer(
            figure_id=figure_id,
            new_figure=new_figure,
            ijhw=self.ijhw,
            labels=self.labels,
            scores=self.scores
        ).render(**kwargs)


@public
class BoxCollection(Serializable):
    r"""

    Parameters
    ----------
    ijhw_tensor
    labels_list
    scores_list

    Attributes
    ----------
    ijhw_tensor
    labels_list
    scores_list
    """
    def __init__(
            self,
            ijhw_tensor: CoordsTensor,
            labels_list: Optional[List[List[str]]] = None,
            scores_list: Optional[List[List[float]]] = None,
    ) -> None:
        self.ijhw_tensor = tf.convert_to_tensor(ijhw_tensor)
        self.labels_list = labels_list
        self.scores_list = scores_list

    @classmethod
    def deserialize(cls, parsed: ParsedBoxCollection) -> 'BoxCollection':
        r"""

        Parameters
        ----------
        parsed

        Returns
        -------

        """
        return cls.from_boxes([Box.deserialize(d) for d in parsed])

    def serialize(self) -> ParsedBoxCollection:
        r"""

        Returns
        -------

        """
        return [box.serialize() for box in self.boxes()]

    @classmethod
    def from_tensor(
            cls,
            tensor: Union[tf.Tensor, List[tf.Tensor], List[List[Num]]],
            unique_labels: Optional[List[List[str]]] = None
    ) -> 'BoxCollection':
        r"""

        Parameters
        ----------
        tensor
        unique_labels

        Returns
        -------

        """
        n_boxes, n_fields = tensor.shape

        if unique_labels is not None and n_fields > 4:
            labels_list = []
            for t in tensor:
                label_list = []
                for i, j in enumerate(range(4, 4 + len(unique_labels))):
                    label = unique_labels[i][t[..., j] - 1]
                    label_list.append(label)
                labels_list.append(label_list)

            if n_fields > 4 + len(unique_labels):
                scores_list = []
                for t in tensor:
                    score_list = []
                    for j in range(4 + len(unique_labels), n_fields):
                        score = t[..., j]
                        score_list.append(score)
                    scores_list.append(score_list)
            else:
                scores_list = None
        else:
            labels_list = None

            if n_fields > 4:
                scores_list = []
                for t in tensor:
                    score_list = []
                    for j in range(4, n_fields):
                        score = t[..., j]
                        score_list.append(score)
                    scores_list.append(score_list)
            else:
                scores_list = None

        return cls(
            ijhw_tensor=tensor[..., :4],
            labels_list=labels_list,
            scores_list=scores_list,
        )

    @classmethod
    def from_yxhw_tensor(
            cls,
            yxhw_tensor: Union[tf.Tensor, List[tf.Tensor], List[List[Num]]],
            labels_list: Optional[List[List[str]]] = None,
            scores_list: Optional[List[List[float]]] = None,
    ) -> 'BoxCollection':
        r"""

        Parameters
        ----------
        yxhw_tensor
        labels_list
        scores_list

        Returns
        -------

        """
        return cls(
            ijhw_tensor=box_tools.yxhw_to_ijhw(yxhw_tensor),
            labels_list=labels_list,
            scores_list=scores_list,
        )

    @classmethod
    def from_ijkl_tensor(
            cls,
            ijkl_tensor: Union[tf.Tensor, List[tf.Tensor], List[List[Num]]],
            labels_list: Optional[List[List[str]]] = None,
            scores_list: Optional[List[List[float]]] = None,
    ) -> 'BoxCollection':
        r"""

        Parameters
        ----------
        ijkl_tensor
        labels_list
        scores_list

        Returns
        -------

        """
        return cls(
            ijhw_tensor=box_tools.ijkl_to_ijhw(ijkl_tensor),
            labels_list=labels_list,
            scores_list=scores_list,
        )

    @classmethod
    def from_boxes(cls, boxes: List[Box]) -> 'BoxCollection':
        r"""

        Parameters
        ----------
        boxes

        Returns
        -------

        """
        return cls(
            ijhw_tensor=[box.ijhw for box in boxes],
            labels_list=[box.labels for box in boxes],
            scores_list=[box.scores for box in boxes],
        )

    @property
    def unique_labels(self) -> List[List[str]]:
        r"""
        """
        unique_labels = [[] for _ in self.labels_list[0]]
        for labels in self.labels_list:
            for i, label in enumerate(labels):
                unique_labels[i].append(label)
        return [np.unique(unique).tolist() for unique in unique_labels]

    def labels_tensor(
            self,
            unique_labels: Optional[List[List[str]]] = None
    ) -> List[List[int]]:
        r"""

        Parameters
        ----------
        unique_labels

        Returns
        -------

        """
        if unique_labels is None:
            unique_labels = self.unique_labels

        return tf.convert_to_tensor([
            [u.index(label) + 1 for label, u in zip(labels, unique_labels)]
            for labels in self.labels_list
        ])

    @property
    def i_tensor(self) -> tf.Tensor:
        r"""

        Returns
        -------

        """
        return self.ijhw_tensor[..., 0]

    @property
    def j_tensor(self) -> tf.Tensor:
        r"""

        Returns
        -------

        """
        return self.ijhw_tensor[..., 1]

    @property
    def h_tensor(self) -> tf.Tensor:
        r"""

        Returns
        -------

        """
        return self.ijhw_tensor[..., 2]

    @property
    def w_tensor(self) -> tf.Tensor:
        r"""

        Returns
        -------

        """
        return self.ijhw_tensor[..., 3]

    @property
    def ij_tensor(self) -> tf.Tensor:
        r"""

        Returns
        -------

        """
        return self.ijhw_tensor[..., :2]

    @property
    def hw_tensor(self) -> tf.Tensor:
        r"""

        Returns
        -------

        """
        return self.ijhw_tensor[..., 2:4]

    @property
    def kl_tensor(self) -> tf.Tensor:
        r"""

        Returns
        -------

        """
        return box_tools.ijhw_to_kl(self.ijhw_tensor)

    @property
    def yx_tensor(self) -> tf.Tensor:
        r"""

        Returns
        -------

        """
        return box_tools.ijhw_to_yx(self.ijhw_tensor)

    @property
    def ijkl_tensor(self) -> tf.Tensor:
        r"""

        Returns
        -------

        """
        return box_tools.ijhw_to_ijkl(self.ijhw_tensor)

    @property
    def yxhw_tensor(self) -> tf.Tensor:
        r"""

        Returns
        -------

        """
        return box_tools.ijhw_to_yxhw(self.ijhw_tensor)

    # TODO: Deal with the scores
    def as_tensor(self, unique_labels: Optional[List[List[str]]] = None):
        r"""

        Parameters
        ----------
        unique_labels

        Returns
        -------

        """
        if unique_labels is None:
            unique_labels = self.unique_labels

        values = [self.ijhw_tensor]
        if self.labels_list is not None:
            labels_tensor = self.labels_tensor(unique_labels=unique_labels)
            labels_tensor = tf.cast(labels_tensor, dtype=tf.float32)
            values.append(labels_tensor)

        return tf.concat(values, axis=-1)

    def boxes(self) -> Iterable[Box]:
        r"""

        Returns
        -------

        """
        if self.labels_list is not None:
            labels_list = self.labels_list
        else:
            labels_list = [None for _ in range(len(self))]
        if self.scores_list is not None:
            scores_list = self.scores_list
        else:
            scores_list = [None for _ in range(len(self))]
        return (
            Box(ijhw=ijhw, labels=labels, scores=scores)
            for ijhw, labels, scores
            in zip(self.ijhw_tensor, labels_list, scores_list)
        )

    def area_tensor(self) -> tf.Tensor:
        r"""

        Returns
        -------

        """
        return tf.reduce_prod(
            self.ijhw_tensor[..., 2:4],
            axis=-1,
            keepdims=True
        )

    # TODO: Not sure how useful this would be...
    def intersection_box_collection(
            self,
            box_collection: 'BoxCollection',
    ) -> 'BoxCollection':
        r"""

        Parameters
        ----------
        box_collection

        Returns
        -------

        """
        raise NotImplemented()

    def _matched_up_operations(
            self,
            ijhw_tensor_other,
            operation: callable
    ) -> tf.Tensor:
        r"""

        Parameters
        ----------
        ijhw_tensor_other
        operation

        Returns
        -------

        """
        matched_up_self, matched_up_other = box_tools.match_up_tensors(
            tensor_a=self.ijhw_tensor,
            tensor_b=ijhw_tensor_other
        )
        return operation(matched_up_self, matched_up_other)

    def intersection_tensor(self, box_collection: 'BoxCollection') -> tf.Tensor:
        r"""

        Parameters
        ----------
        box_collection

        Returns
        -------

        """
        return self._matched_up_operations(
            ijhw_tensor_other=box_collection.ijhw_tensor,
            operation=box_tools.calculate_intersections
        )

    def union_tensor(self, box_collection: 'BoxCollection') -> tf.Tensor:
        r"""

        Parameters
        ----------
        box_collection

        Returns
        -------

        """
        return self._matched_up_operations(
            ijhw_tensor_other=box_collection.ijhw_tensor,
            operation=box_tools.calculate_unions
        )

    def iou_tensor(self, box_collection: 'BoxCollection') -> tf.Tensor:
        r"""

        Parameters
        ----------
        box_collection

        Returns
        -------

        """
        return self._matched_up_operations(
            ijhw_tensor_other=box_collection.ijhw_tensor,
            operation=box_tools.calculate_ious
        )

    def __str__(self) -> str:
        r"""

        Returns
        -------

        """
        raise NotImplemented()

    def __len__(self):
        return len(self.ijhw_tensor)

    def view(
            self,
            figure_id: Optional[int] = None,
            new_figure: Optional[bool] = False,
            **kwargs
    ) -> None:
        r"""

        Parameters
        ----------
        figure_id
        new_figure
        kwargs

        """
        for box in self.boxes():
            box.view(figure_id=figure_id, new_figure=new_figure, **kwargs)
