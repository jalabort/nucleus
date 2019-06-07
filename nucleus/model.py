from typing import Union

import pathlib
import stringcase
import tensorflow.python.keras as keras


class BaseModel:
    r"""

    Parameters
    ----------
    model

    Attributes
    ----------
    model
    """

    def __init__(self, model: keras.Model):
        self.model = model

    # TODO: What's the tf weight extension
    @classmethod
    def load_model(
            cls,
            model_path: Union[pathlib.Path, str],
            **kwargs
    ) -> 'BaseModel':
        r"""

        Parameters
        ----------
        model_path
        kwargs

        Returns
        -------

        """
        model_arch_path = pathlib.Path(model_path)
        suffix = model_arch_path.suffix

        assert model_arch_path.exists()
        assert suffix in ['.h5']

        model = keras.models.load_model(model_path, **kwargs)

        assert model.name == stringcase.snakecase(cls.__name__)

        return cls(model)

    # TODO: What's the tf weight extension
    def save_model(
            self,
            model_path: Union[pathlib.Path, str],
            **kwargs
    ) -> None:
        r"""

        Parameters
        ----------
        model_path
        kwargs

        Returns
        -------

        """
        model_arch_path = pathlib.Path(model_path)
        suffix = model_arch_path.suffix

        assert model_arch_path.exists()
        assert suffix in ['.h5']

        self.model.save(model_path, **kwargs)

    @classmethod
    def load_model_arch(
            cls,
            model_arch_path: Union[pathlib.Path, str],
            **kwargs
    ) -> 'BaseModel':
        r"""

        Parameters
        ----------
        model_arch_path
        kwargs

        Returns
        -------

        """
        model_arch_path = pathlib.Path(model_arch_path)
        suffix = model_arch_path.suffix

        assert model_arch_path.exists()
        assert suffix in ['.yaml', '.json']

        with open(model_arch_path, 'r') as f:
            encoded = f.read()

        if suffix == 'yaml':
            model = keras.models.model_from_yaml(encoded, **kwargs)
        else:
            model = keras.models.model_from_json(encoded, **kwargs)

        assert model.name == stringcase.snakecase(cls.__name__)

        return cls(model)

    def save_model_arch(
            self,
            model_arch_path: Union[pathlib.Path, str],
            **kwargs
    ) -> None:
        r"""

        Parameters
        ----------
        model_arch_path
        kwargs

        Returns
        -------

        """
        model_arch_path = pathlib.Path(model_arch_path)
        suffix = model_arch_path.suffix

        assert suffix in ['.yaml', '.json']

        if suffix == 'yaml':
            encoded = self.model.to_yaml(**kwargs)
        else:
            encoded = self.model.to_json(**kwargs)

        with open(model_arch_path, 'w') as f:
            f.write(encoded)

    # TODO: What's the tf weight extension
    def load_model_weights(
            self,
            model_weights_path: Union[pathlib.Path, str]
    ) -> None:
        r"""

        Parameters
        ----------
        model_weights_path

        Returns
        -------

        """
        model_arch_path = pathlib.Path(model_weights_path)
        suffix = model_arch_path.suffix

        assert model_arch_path.exists()
        assert suffix in ['.h5']

        self.model.load_weights(model_weights_path)

    # TODO: What's the tf weight extension
    def save_model_weights(
            self,
            model_weights_path: Union[pathlib.Path, str]
    ) -> None:
        r"""

        Parameters
        ----------
        model_weights_path

        Returns
        -------

        """
        model_arch_path = pathlib.Path(model_weights_path)
        suffix = model_arch_path.suffix

        assert suffix in ['.h5']

        self.model.save_weights(model_weights_path)
