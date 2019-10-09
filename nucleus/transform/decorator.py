from copy import deepcopy
from functools import wraps
from tensorflow.python.util import tf_decorator

from nucleus.image import Image
from nucleus.box import BoxCollection
from nucleus.utils import export


@export
def image_transform(arg):
    r"""
    Decorator that makes it possible to pass a nucleus image to a function
    or class method that expects two tensors representing image pixels and
    bounding boxes.

    Notes
    -----
    This decorator is extensively used in the transform module.

    Parameters
    ----------
    arg : callable | str, optional
        If explicitly specified, `arg` is the name of the new name scope.
        If not, `arg` is the function to be decorated. In that case,
        the name of the new name scope is the decorated function's name.
    """
    @wraps(arg)
    def wrapper(*args, **kwargs):
        if isinstance(args[0], Image):
            img: Image = args[0]
            hwc, box_tensor = arg(
                img.hwc,
                img.box_collection.as_tensor() if img.box_collection else None,
                *args[1:],
                **kwargs
            )
            new_img = deepcopy(img)
            new_img.hwc = hwc
            if box_tensor is not None:
                new_img.box_collection = BoxCollection.from_tensor(
                    box_tensor, unique_labels=img.box_collection.unique_labels
                )
            return new_img
        elif isinstance(args[1], Image):
            img: Image = args[1]
            hwc, box_tensor = arg(
                args[0],
                img.hwc,
                img.box_collection.as_tensor() if img.box_collection else None,
                *args[2:],
                **kwargs
            )
            new_img = deepcopy(img)
            new_img.hwc = hwc
            if box_tensor is not None:
                new_img.box_collection = BoxCollection.from_tensor(
                    box_tensor, unique_labels=img.box_collection.unique_labels
                )
            return new_img
        else:
            return arg(*args, **kwargs)

    return tf_decorator.make_decorator(arg, wrapper)
