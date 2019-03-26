from typing import Union, List, Iterable, Sequence

import abc
import json
import gzip
import pathlib
import warnings
import functools
import collections

from nucleus.types import Parsed


__all__ = ['Serializable', 'LazyList']


class Serializable(abc.ABC):
    r"""

    """

    @classmethod
    @abc.abstractmethod
    def deserialize(cls, parsed: Parsed) -> object:
        r"""

        Parameters
        ----------
        parsed

        Returns
        -------

        """

    @classmethod
    def load(cls, path: Union[str, pathlib.Path]) -> object:
        r"""

        Parameters
        ----------
        path

        Returns
        -------

        """
        path = pathlib.Path(path)

        if not path.exists() or path.suffix != '.json':
            tentative_path = path.parent / f'{path.stem}.json'
            if tentative_path.exists():
                path = tentative_path
            else:
                tentative_path = path.parent / f'{path.stem}.json.gz'
                if tentative_path.exists():
                    path = tentative_path

        if path.suffix == '.json':
            with open(path, 'r') as f:
                parsed = json.load(f)
        else:
            with gzip.GzipFile(path, 'r') as f:
                parsed = json.loads(f.read().decode('utf-8'))

        return cls.deserialize(parsed)

    @abc.abstractmethod
    def serialize(self, **kwargs) -> Parsed:
        r"""

        Returns
        -------

        """

    def save(
            self,
            path: Union[str, pathlib.Path],
            compress: bool = False
    ) -> None:
        r"""

        Parameters
        ----------
        path
        compress

        Returns
        -------

        """
        self._save(
            parsed=self.serialize(),
            path=path,
            compress=compress
        )

    def _save(
        self,
        parsed: Parsed,
        path: Union[str, pathlib.Path],
        compress: bool = False
    ) -> None:
        r"""

        Parameters
        ----------
        path
        compress

        Returns
        -------

        """
        path = str(self._check_path(path))

        if not compress:
            with open(path, 'w') as f:
                json.dump(parsed, f, indent=2)
        else:
            with gzip.GzipFile(path + '.gz', 'w') as f:
                f.write(json.dumps(parsed, indent=2).encode('utf-8'))

    @staticmethod
    def _check_path(path: pathlib.Path) -> pathlib.Path:
        r"""

        Parameters
        ----------
        path

        Returns
        -------

        """
        path = pathlib.Path(path)

        stem = path.stem
        suffix = path.suffix[1:]
        if suffix != 'json':
            if suffix is not None:
                warnings.warn(
                    f'Path suffix {suffix} not supported. Path suffix will be '
                    f'changed to `json`.'
                )
            suffix = 'json'
        name = '.'.join([stem, suffix])

        return path.parent / name


class LazyList(collections.Sequence):
    r"""
    An immutable sequence that provides the ability to lazily access objects.

    In truth, this sequence simply wraps a list of callables which are then
    indexed and invoked. However, if the callable represents a function that
    lazily access memory, then this list simply implements a lazy list
    paradigm.

    When slicing, another `LazyList` is returned, containing the subset
    of callables.

    Parameters
    ----------
    callables
        A list of `callable` objects that will be invoked if directly indexed.
    """

    def __init__(self, callables: List[callable]):
        self._callables = callables

    def __getitem__(self, slice_,) :
        # note that we have to check for iterable *before* __index__ as ndarray
        # has both (but we expect the iteration behavior when slicing)
        if isinstance(slice_, collections.Iterable):
            # An iterable object is passed - return a new LazyList
            return LazyList([self._callables[s] for s in slice_])
        elif isinstance(slice_, int) or hasattr(slice_, '__index__'):
            # PEP 357 and single integer index access - returns element
            # if callable(self._callables[slice_]):
            #     self._callables[slice_] = self._callables[slice_]()

            return self._callables[slice_]()
        else:
            # A slice or unknown type is passed - let List handle it
            return LazyList(self._callables[slice_])

    def __len__(self):
        return len(self._callables)

    @classmethod
    def from_iterable(
            cls,
            iterable: Iterable,
            f: callable = None
    ) -> 'LazyList':
        r"""
        Create a lazy list from an existing iterable (think Python `list`) and
        optionally a `callable` that expects a single parameter which will be
        applied to each element of the list. This allows for simply
        creating a `LazyList` from an existing list and if no `callable` is
        provided the identity function is assumed.

        Parameters
        ----------
        iterable : `collections.Iterable`
            An iterable object such as a `list`.
        f : `callable`, optional
            Callable expecting a single parameter.

        Returns
        -------
        lazy : `LazyList`
            A LazyList where each element returns each item of the provided
            iterable, optionally with `f` applied to it.
        """
        if f is None:
            # The identity function
            def f(i):
                return i
        return cls([functools.partial(f, x) for x in iterable])

    @classmethod
    def from_index_callable(cls, f: callable, n_elements: int) -> 'LazyList':
        r"""
        Create a lazy list from a `callable` that expects a single parameter,
        the index into an underlying sequence. This allows for simply
        creating a `LazyList` from a `callable` that likely wraps
        another list in a closure.

        Parameters
        ----------
        f
            Callable expecting a single integer parameter, index. This is an
            index into (presumably) an underlying sequence.
        n_elements
            The number of elements in the underlying sequence.
        Returns
        -------
        lazy
            A LazyList where each element returns the underlying indexable
            object wrapped by ``f``.
        """
        return cls([functools.partial(f, i) for i in range(n_elements)])

    def __add__(self, other: Sequence) -> 'LazyList':
        r"""
        Create a new LazyList from this list and the given list. The passed list
        items will be concatenated to the end of this list to give a new
        LazyList that contains the concatenation of the two lists.

        If a Python list is passed then the elements are wrapped in a function
        that just returns their values to maintain the callable nature of
        LazyList elements.

        Parameters
        ----------
        other
            Sequence to concatenate with this list.

        Returns
        -------
        lazy : `LazyList`
            A new LazyList formed of the concatenation of this list and
            the ``other`` list.

        Raises
        ------
        ValueError
            If other is not a LazyList or an Iterable
        """
        if isinstance(other, LazyList):
            return LazyList(self._callables + other._callables)
        elif isinstance(other, collections.Iterable):
            return self + LazyList.from_iterable(other)
        else:
            raise ValueError(
                'Can only add another LazyList or an Iterable to a LazyList '
                '- {type(other)} is neither.')

    def __str__(self):
        return 'LazyList containing {} items'.format(len(self))
