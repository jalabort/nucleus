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


# TODO: Clean up!
@export
class classproperty(property):
    """
    Similar to `property`, but allows class-level properties.  That is,
    a property whose getter is like a `classmethod`.

    The wrapped method may explicitly use the `classmethod` decorator (which
    must become before this decorator), or the `classmethod` may be omitted
    (it is implicit through use of this decorator).

    .. note::

        classproperty only works for *read-only* properties.  It does not
        currently allow writeable/deletable properties, due to subtleties of how
        Python descriptors work.  In order to implement such properties on a class
        a metaclass for that class must be implemented.

    Parameters
    ----------
    fget : callable
        The function that computes the value of this property (in particular,
        the function when this is used as a decorator) a la `property`.

    doc : str, optional
        The docstring for the property--by default inherited from the getter
        function.

    lazy : bool, optional
        If True, caches the value returned by the first call to the getter
        function, so that it is only called once (used for lazy evaluation
        of an attribute).  This is analogous to `lazyproperty`.  The ``lazy``
        argument can also be used when `classproperty` is used as a decorator
        (see the third example below).  When used in the decorator syntax this
        *must* be passed in as a keyword argument.

    Examples
    --------

    ::

        >>> class Foo:
        ...     _bar_internal = 1
        ...     @classproperty
        ...     def bar(cls):
        ...         return cls._bar_internal + 1
        ...
        >>> Foo.bar
        2
        >>> foo_instance = Foo()
        >>> foo_instance.bar
        2
        >>> foo_instance._bar_internal = 2
        >>> foo_instance.bar  # Ignores instance attributes
        2

    As previously noted, a `classproperty` is limited to implementing
    read-only attributes::

        >>> class Foo:
        ...     _bar_internal = 1
        ...     @classproperty
        ...     def bar(cls):
        ...         return cls._bar_internal
        ...     @bar.setter
        ...     def bar(cls, value):
        ...         cls._bar_internal = value
        ...
        Traceback (most recent call last):
        ...
        NotImplementedError: classproperty can only be read-only; use a
        metaclass to implement modifiable class-level properties

    When the ``lazy`` option is used, the getter is only called once::

        >>> class Foo:
        ...     @classproperty(lazy=True)
        ...     def bar(cls):
        ...         print("Performing complicated calculation")
        ...         return 1
        ...
        >>> Foo.bar
        Performing complicated calculation
        1
        >>> Foo.bar
        1

    If a subclass inherits a lazy `classproperty` the property is still
    re-evaluated for the subclass::

        >>> class FooSub(Foo):
        ...     pass
        ...
        >>> FooSub.bar
        Performing complicated calculation
        1
        >>> FooSub.bar
        1
    """

    def __new__(cls, fget=None, doc=None, lazy=False):
        if fget is None:
            # Being used as a decorator--return a wrapper that implements
            # decorator syntax
            def wrapper(func):
                return cls(func, lazy=lazy)

            return wrapper

        return super().__new__(cls)

    def __init__(self, fget, doc=None, lazy=False):
        self._lazy = lazy
        if lazy:
            self._cache = {}
        fget = self._wrap_fget(fget)

        super().__init__(fget=fget, doc=doc)

        # There is a buglet in Python where self.__doc__ doesn't
        # get set properly on instances of property subclasses if
        # the doc argument was used rather than taking the docstring
        # from fget
        # Related Python issue: https://bugs.python.org/issue24766
        if doc is not None:
            self.__doc__ = doc

    def __get__(self, obj, objtype):
        if self._lazy and objtype in self._cache:
            return self._cache[objtype]

        # The base property.__get__ will just return self here;
        # instead we pass objtype through to the original wrapped
        # function (which takes the class as its sole argument)
        val = self.fget.__wrapped__(objtype)

        if self._lazy:
            self._cache[objtype] = val

        return val

    def getter(self, fget):
        return super().getter(self._wrap_fget(fget))

    def setter(self, fset):
        raise NotImplementedError(
            "classproperty can only be read-only; use a metaclass to "
            "implement modifiable class-level properties")

    def deleter(self, fdel):
        raise NotImplementedError(
            "classproperty can only be read-only; use a metaclass to "
            "implement modifiable class-level properties")

    @staticmethod
    def _wrap_fget(orig_fget):
        if isinstance(orig_fget, classmethod):
            orig_fget = orig_fget.__func__

        # Using stock functools.wraps instead of the fancier version
        # found later in this module, which is overkill for this purpose

        @functools.wraps(orig_fget)
        def fget(obj):
            return orig_fget(obj.__class__)

        return fget


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
# TODO: Is this equivalent to ``tf.unstack(tf.shape(tensor_a))``?
@name_scope
# @tf.function
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
