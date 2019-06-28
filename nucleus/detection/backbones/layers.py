import tensorflow as tf

from stringcase import snakecase

from nucleus.utils import export


# TODO: All these layers could be a simple lambda layers, however we had
#  trouble saving the models if implemented like that... Possible TF2.0 bug?


# TODO: Document me!
@export
class PreprocessingLayer(tf.keras.layers.Layer):
    r"""


    Parameters
    ----------
    preprocess_fn

    **kwargs

    """
    def __init__(self, preprocess_fn: callable, **kwargs) -> None:
        name = kwargs.pop('name', False)
        if not name:
            name = snakecase(self.__class__.__name__)
        super().__init__(name=name, **kwargs)
        self.preprocess_fn = preprocess_fn

    @tf.function
    def call(self, inputs):
        r"""

        Parameters
        ----------
        inputs

        Returns
        -------

        """
        return self.preprocess_fn(inputs)


# TODO: Document me!
@export
class MobileNetPreprocessingLayer(PreprocessingLayer):
    r"""


    Parameters
    ----------
    preprocess_fn

    **kwargs

    """
    def __init__(self, **kwargs) -> None:
        super().__init__(
            preprocess_fn=tf.keras.applications.mobilenet.preprocess_input,
            **kwargs
        )


# TODO: Document me!
@export
class MobileNetV2PreprocessingLayer(PreprocessingLayer):
    r"""


    Parameters
    ----------
    preprocess_fn

    **kwargs

    """
    def __init__(self, **kwargs) -> None:
        super().__init__(
            preprocess_fn=tf.keras.applications.mobilenet_v2.preprocess_input,
            **kwargs
        )


# TODO: Document me!
@export
class NasNetMobilePreprocessingLayer(PreprocessingLayer):
    r"""


    Parameters
    ----------
    preprocess_fn

    **kwargs

    """
    def __init__(self, **kwargs) -> None:
        super().__init__(
            preprocess_fn=tf.keras.applications.nasnet.preprocess_input,
            **kwargs
        )


# TODO: Document me!
@export
class DenseNetPreprocessingLayer(PreprocessingLayer):
    r"""


    Parameters
    ----------
    preprocess_fn

    **kwargs

    """
    def __init__(self, **kwargs) -> None:
        super().__init__(
            preprocess_fn=tf.keras.applications.densenet.preprocess_input,
            **kwargs
        )


# TODO: Document me!
@export
class XceptionPreprocessingLayer(PreprocessingLayer):
    r"""


    Parameters
    ----------
    preprocess_fn

    **kwargs

    """
    def __init__(self, **kwargs) -> None:
        super().__init__(
            preprocess_fn=tf.keras.applications.xception.preprocess_input,
            **kwargs
        )


# TODO: Document me!
@export
class InceptionV3PreprocessingLayer(PreprocessingLayer):
    r"""


    Parameters
    ----------
    preprocess_fn

    **kwargs

    """
    def __init__(self, **kwargs) -> None:
        super().__init__(
            preprocess_fn=tf.keras.applications.inception_v3.preprocess_input,
            **kwargs
        )


# TODO: Document me!
@export
class ResNet50PreprocessingLayer(PreprocessingLayer):
    r"""


    Parameters
    ----------
    preprocess_fn

    **kwargs

    """
    def __init__(self, **kwargs) -> None:
        super().__init__(
            preprocess_fn=tf.keras.applications.resnet50.preprocess_input,
            **kwargs
        )
