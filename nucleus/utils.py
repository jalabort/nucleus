import sys
import functools
import tensorflow as tf

from tqdm import tqdm, tqdm_notebook
from tensorflow.python.util import tf_decorator


__all__ = ['progress_bar', 'tf_get_shape', 'name_scope']


if 'ipykernel' in sys.modules:
    progress_bar = tqdm_notebook
else:
    progress_bar = tqdm


def tf_get_shape(tensor):
    r"""
    Get the shape of a tensor.

    Parameters
    ----------
    tensor : tf.Tensor
        Tensor.

    Returns
    -------
    shape : list[int | tf.Tensor]
        Tuple containing the dimensions of the tensor, each dimension
        can either be an `int` (if the static shape is defined) or a
        `tf.Tensor` (if the static shape is not defined).
    """
    static_shape = tensor.shape.as_list()
    dynamic_shape = tf.shape(tensor)
    shape = []
    for i, static in enumerate(static_shape):
        if static is None:
            shape.append(dynamic_shape[i])
        else:
            shape.append(static)
    return shape


def name_scope(arg):
    r"""
    Decorator that create a new tensorflow name scope.
    Parameters
    ----------
    arg : callable | str, optional
        If explicitly specified, `arg` is the name of the new name scope.
        If not, `arg` is the function to be decorated. In that case,
        the name of the new name scope is the decorated function's name.
    """
    if callable(arg):
        @functools.wraps(arg)
        def wrapper(*args, **kwargs):
            name = arg.__name__
            if name == '__call__':
                name = args[0].__class__.__name__
            else:
                name = name.title().replace('_', '')
            with tf.name_scope(name):
                return arg(*args, **kwargs)
        return tf_decorator.make_decorator(arg, wrapper)
    else:
        def decorator(func):
            @functools.wraps(func)
            def inner_wrapper(*args, **kwargs):
                with tf.name_scope(arg):
                    return func(*args, **kwargs)
            return tf_decorator.make_decorator(arg, inner_wrapper)
        return decorator

