from typing import Optional, Union, Sequence, Tuple

import tensorflow as tf
from pathlib import Path
from public import public
from abc import abstractmethod
from stringcase import pascalcase

from nucleus.detection.managers import ModelManager

from .layers import (
    MobileNetPreprocessingLayer,
    MobileNetV2PreprocessingLayer,
    NasNetMobilePreprocessingLayer,
    DenseNetPreprocessingLayer,
    XceptionPreprocessingLayer,
    ResNet50PreprocessingLayer
)


# TODO: Document me!
class BackboneManager(ModelManager):

    @property
    def custom_objects(self):
        return {
            pascalcase(self._preprocessing_layer.name):
                self._preprocessing_layer,
            'FrozenBatchNormalization':
                self._patch_keras_layers().BatchNormalization
        }

    @property
    @abstractmethod
    def _preprocessing_layer(self):
        r"""
        """

    @property
    @abstractmethod
    def _backbone_fn(self, **kwargs):
        r"""
        """

    def _patch_keras_layers(self) -> tf.keras.layers:
        r"""
        Patches the tf.keras.layers module with a modified batch normalization
        layer call method that works significantly better when using transfer
        learning; which is the intended use case for these backbone models.

        This is necessary because of the controversial way in which batch
        normalization layers are implemented in Keras. You can find a
        detailed explanation of why the patch applied by this function is a
        good idea, specially when using transfer learning, in the following
        links:

        - http://blog.datumbox.com/the-batch-normalization-layer-of-keras-is-broken/
        - https://github.com/keras-team/keras/pull/9965

        Notes
        -----
        This code was copy/pasted from:

        - https://github.com/keras-team/keras/pull/9965#issuecomment-501933060

        Returns
        -------
        The patched tk.keras.layers module.
        """
        from tensorflow.python.keras import layers

        class FrozenBatchNormalization(layers.BatchNormalization):
            def call(self, inputs, training=None):
                return super().call(inputs=inputs, training=False)

        layers.BatchNormalization = FrozenBatchNormalization

        return layers

    def create_model(
            self,
            input_shape: Tuple = (None, None, 3),
            weights: Optional[Union[str, Path]] = 'imagenet',
            trainable: Union[bool, Sequence[bool]] = True,
            **kwargs
    ) -> tf.keras.Model:
        r"""
        Creates a backbone model.

        Parameters
        ----------
        input_shape

        alpha

        weights

        trainable

        kwargs


        Returns
        -------
        The backbone model.
        """
        # Input layer
        inputs = tf.keras.Input(input_shape)

        # Define the backbone specific pre-processing layer
        x = self._preprocessing_layer(inputs)

        # Define the backbone network
        backbone: tf.keras.Model = self._backbone_fn(
            layers=self._patch_keras_layers(),
            include_top=False,
            input_shape=input_shape,
            input_tensor=x,
            weights=weights,
            **kwargs
        )

        # Determine which layers should be trainable
        n_layers = len(backbone.layers)
        if isinstance(trainable, bool):
            trainable = [trainable for _ in range(n_layers)]
        elif isinstance(trainable, Sequence):
            trainable = list(trainable)
            trainable_length = len(trainable)
            assert trainable_length <= n_layers
            if trainable_length < n_layers:
                trainable = (
                    [False for _ in range(n_layers - trainable_length)]
                    +
                    trainable
                )

        # Set the appropriate layers to trainable
        for layer, t in zip(backbone.layers, trainable):
            layer.trainable = t

        return tf.keras.Model(
            inputs=inputs,
            outputs=backbone.outputs,
            name=self.model_name
        )


# TODO: Document me!
class DenseNetManager(BackboneManager):

    @property
    def _preprocessing_layer(self):
        r"""
        """
        return DenseNetPreprocessingLayer()


# TODO: Document me!
@public
class MobileNetManager(BackboneManager):

    @property
    def _preprocessing_layer(self):
        r"""
        """
        return MobileNetPreprocessingLayer()

    @property
    def _backbone_fn(self):
        r"""
        """
        return tf.keras.applications.MobileNet

    def create_model(
            self,
            input_shape: Tuple = (None, None, 3),
            weights: Optional[Union[str, Path]] = 'imagenet',
            trainable: Union[bool, Sequence[bool]] = True,
            alpha: float = 0.75,
    ) -> tf.keras.Model:
        r"""
        Creates a MobileNet backbone model.

        Parameters
        ----------
        input_shape

        weights

        trainable

        alpha


        Returns
        -------
        The MobileNet backbone model.
        """
        return super().create_model(
            input_shape=input_shape,
            weights=weights,
            trainable=trainable,
            alpha=alpha
        )


# TODO: Document me!
@public
class MobileNetV2Manager(BackboneManager):

    @property
    def _preprocessing_layer(self):
        r"""
        """
        return MobileNetV2PreprocessingLayer()

    @property
    def _backbone_fn(self):
        r"""
        """
        return tf.keras.applications.MobileNetV2

    def create_model(
            self,
            input_shape: Tuple = (None, None, 3),
            weights: Optional[Union[str, Path]] = 'imagenet',
            trainable: Union[bool, Sequence[bool]] = True,
            alpha: float = 0.75,
    ) -> tf.keras.Model:
        r"""
        Creates a MobileNet backbone model.

        Parameters
        ----------
        input_shape

        weights

        trainable

        alpha


        Returns
        -------
        The MobileNet backbone model.
        """
        return super().create_model(
            input_shape=input_shape,
            weights=weights,
            trainable=trainable,
            alpha=alpha
        )


# TODO: Document me!
@public
class NasNetMobileManager(BackboneManager):

    @property
    def _preprocessing_layer(self):
        r"""
        """
        return NasNetMobilePreprocessingLayer()

    @property
    def _backbone_fn(self):
        r"""
        """
        return tf.keras.applications.NASNetMobile


# TODO: Document me!
@public
class DenseNet121Manager(DenseNetManager):

    @property
    def _backbone_fn(self):
        r"""
        """
        return tf.keras.applications.DenseNet121


# TODO: Document me!
@public
class DenseNet169Manager(DenseNetManager):

    @property
    def _backbone_fn(self):
        r"""
        """
        return tf.keras.applications.DenseNet169


# TODO: Document me!
@public
class DenseNet201Manager(DenseNetManager):

    @property
    def _backbone_fn(self):
        r"""
        """
        return tf.keras.applications.DenseNet201


# TODO: Document me!
@public
class XceptionManager(BackboneManager):

    @property
    def _preprocessing_layer(self):
        r"""
        """
        return XceptionPreprocessingLayer()

    @property
    def _backbone_fn(self):
        r"""
        """
        return tf.keras.applications.Xception


# TODO: Document me!
@public
class InceptionV3Manager(BackboneManager):

    @property
    def _preprocessing_layer(self):
        r"""
        """
        return XceptionPreprocessingLayer()

    @property
    def _backbone_fn(self):
        r"""
        """
        return tf.keras.applications.InceptionV3


# TODO: Document me!
@public
class ResNet50Manager(BackboneManager):

    @property
    def _preprocessing_layer(self):
        r"""
        """
        return ResNet50PreprocessingLayer()

    @property
    def _backbone_fn(self):
        r"""
        """
        return tf.keras.applications.ResNet50
