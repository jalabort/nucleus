import sys
import functools
import stringcase
import tensorflow as tf

from tqdm import tqdm, tqdm_notebook
from tensorflow.python.util import tf_decorator


__all__ = ['progress_bar', 'export', 'name_scope', 'tf_get_shape']


if 'ipykernel' in sys.modules:
    progress_bar = tqdm_notebook
else:
    progress_bar = tqdm


def export(obj):
    r"""
    Decorator used to export function and class names from a module without
    having to manually add them to __all__.

    References
    ----------
    .. [1] http://groups.google.com/group/comp.lang.python/msg/11cbb03e09611b8a
    .. [2] http://groups.google.com/group/comp.lang.python/msg/3d400fb22d8a42e1
    """
    mod = sys.modules[obj.__module__]
    if hasattr(mod, '__all__'):
        name, all_ = obj.__name__, mod.__all__
        if name not in all_:
            all_.append(name)
    else:
        mod.__all__ = [obj.__name__]
    return obj


# TODO: Does string path work?
def name_scope(arg):
    r"""
    Decorator that creates a new tensorflow name scope.

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
            if name in ['__call__', 'call']:
                name = args[0].__class__.__name__
            else:
                name = stringcase.camelcase(name)
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


# TODO: This is deprecated and should not be used
@name_scope
@tf.function
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
