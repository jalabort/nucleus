from typing import Union

import pathlib
import stringcase
import tensorflow as tf
from public import public

from .utils import classproperty


# TODO: Think of more informative assert messages
@public
class BaseModel:
    r"""

    Parameters
    ----------
    model

    Attributes
    ----------
    model
    """
    def __init__(
            self,
            model: tf.keras.Model,
    ) -> None:
        self.model = model

    # noinspection PyMethodParameters
    @classproperty
    def name(cls) -> str:
        r"""
        """
        return stringcase.snakecase(cls.__name__)

    # noinspection PyMethodParameters
    @classproperty
    def cache(cls) -> pathlib.Path:
        r"""
        """
        cache = pathlib.Path.home() / '.hudlrd'
        cache.mkdir(parents=True, exist_ok=True)
        return cache

    # noinspection PyMethodParameters
    @classproperty
    def model_path(cls) -> pathlib.Path:
        r"""
        """
        model_path = cls.cache / stringcase.snakecase(cls.name)
        return model_path

        # TODO: Need to save the object configuration to be properly restored
    @classmethod
    def load_model(
            cls,
            path: Union[pathlib.Path, str] = None,
            save_format: str = 'hdf5',
            **kwargs
    ) -> 'BaseModel':
        r"""

        Parameters
        ----------
        path
        save_format
        kwargs

        Returns
        -------

        """
        if path is None:
            assert save_format in ['tf', 'hdf5']
            suffix = '' if save_format == 'tf' else '.h5'
            path = str(cls.model_path) + suffix

        path = pathlib.Path(path)
        suffix = path.suffix

        assert path.exists()
        assert suffix in ['', '.h5']

        if path.is_dir():
            assert (path / 'assets').exists()
            assert (path / 'variables').exists()
            assert (path / 'saved_model.pb').exists()

        model = tf.keras.models.load_model(str(path), **kwargs)
        assert model.model_name == stringcase.snakecase(cls.__name__)

        base_model: cls = BaseModel(model=model)
        base_model.__class__ = cls
        return base_model

    # TODO: What's the tf weight extension
    def save_model(
            self,
            path: Union[pathlib.Path, str] = None,
            save_format: str = 'hdf5',
            **kwargs
    ) -> None:
        r"""

        Parameters
        ----------
        path
        save_format
        kwargs

        Returns
        -------

        """
        if path is None:
            assert save_format in ['tf', 'hdf5']
            suffix = '' if save_format == 'tf' else '.h5'
            path = str(self.model_path) + suffix

        path = pathlib.Path(path)
        suffix = path.suffix

        assert suffix in ['', '.h5']

        self.model.save(str(path), **kwargs)

    @classmethod
    def load_model_arch(
            cls,
            path: Union[pathlib.Path, str] = None,
            **kwargs
    ) -> 'BaseModel':
        r"""

        Parameters
        ----------
        path
        kwargs

        Returns
        -------

        """
        if path is None:
            path = cls.model_path

        path = pathlib.Path(path)
        suffix = path.suffix

        assert path.exists()
        assert suffix in ['.yaml', '.json']

        with open(path, 'r') as f:
            encoded = f.read()

        if suffix == '.yaml':
            model = tf.keras.models.model_from_yaml(encoded, **kwargs)
        else:
            model = tf.keras.models.model_from_json(encoded, **kwargs)

        assert model.model_name == stringcase.snakecase(cls.__name__)

        return cls(model)

    def save_model_arch(
            self,
            path: Union[pathlib.Path, str] = None,
            **kwargs
    ) -> None:
        r"""

        Parameters
        ----------
        path
        kwargs

        Returns
        -------

        """
        if path is None:
            path = self.model_path

        path = pathlib.Path(path)
        suffix = path.suffix

        assert suffix in ['.yaml', '.json']

        if suffix == '.yaml':
            encoded = self.model.to_yaml(**kwargs)
        else:
            encoded = self.model.to_json(**kwargs)

        with open(path, 'w') as f:
            f.write(encoded)

    # TODO: What's the tf weight extension
    def load_model_weights(
            self,
            path: Union[pathlib.Path, str] = None
    ) -> None:
        r"""

        Parameters
        ----------
        path

        Returns
        -------

        """
        if path is None:
            path = self.model_path

        path = pathlib.Path(path)
        suffix = path.suffix

        assert suffix in ['', '.h5']

        if suffix == '.h5':
            assert path.exists()
        else:
            parent = path.parent
            name = path.name
            assert (parent / (name + '.index')).exists()
            assert (parent / (name + '.data-00000-of-00001')).exists()
            assert (parent / 'checkpoint').exists()

        self.model.load_weights(str(path))

    # TODO: What's the tf weight extension
    def save_model_weights(
            self,
            path: Union[pathlib.Path, str] = None
    ) -> None:
        r"""

        Parameters
        ----------
        path

        Returns
        -------

        """
        if path is None:
            path = self.model_path

        path = pathlib.Path(path)
        suffix = path.suffix

        assert suffix in ['', '.h5']

        self.model.save_weights(str(path))
