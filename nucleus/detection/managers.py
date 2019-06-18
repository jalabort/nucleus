from typing import Optional, Union, Sequence

from abc import abstractmethod
from pathlib import Path
from stringcase import snakecase
import tensorflow as tf
from public import public
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.tools.freeze_graph import freeze_graph

from .heads import create_yolo_head
from .anchors import AnchorParameters, yolo_anchor_parameters
from .matcher import Matcher, YoloMatcher
from .layers import YoloInferenceLayer


def remove_directory_contents(directory: Path) -> None:
    r"""
    Removes all the content from a directory

    Parameters
    ----------
    directory
        The directory whose content we want to remove
    """
    for element in directory.iterdir():
        if element.is_file():
            element.unlink()
        elif element.is_dir():
            remove_directory_contents(element)
            element.rmdir()


# TODO: Document me!
# TODO: Use more informative assert messages
class ModelManager:
    r"""

    Parameters
    ----------
    cache

    Attributes
    ----------
    cache
    """
    def __init__(
            self,
            cache: Union[Path, str] = Path.home() / '.hudlrd'
    ) -> None:
        self.cache = cache

    @property
    def model_name(self) -> str:
        r"""
        """
        return snakecase(self.__class__.__name__.replace('Manager', ''))

    @property
    def inference_model_name(self) -> str:
        r"""
        """
        return f'inference_{self.model_name}'

    @property
    def cache(self):
        r"""
        """
        return self._cache

    @cache.setter
    def cache(self, cache: Union[str, Path]):
        self._cache = Path(cache).absolute()
        self._model_path = self.cache / self.model_name
        self._inference_model_path = self.cache / self.inference_model_name
        self.cache.mkdir(parents=True, exist_ok=True)

    @property
    def model_path(self):
        r"""
        """
        return self._model_path

    @property
    def inference_model_path(self):
        r"""
        """
        return self._inference_model_path

    # TODO: Does not work yet!
    def export_saved_model(
            self,
            model: tf.keras.Model,
            path: Union[Path, str] = None
    ) -> None:
        r"""

        Parameters
        ----------
        model
        path

        Returns
        -------

        """
        assert (
                model.name == self.model_name
                or model.name == self.inference_model_name
        )

        if path is None:
            path = str(self.model_path) + '.pb'

        path = Path(path)
        suffix = path.suffix

        assert suffix in ['.pb']

        freeze_graph(
            input_graph='',
            input_saver='',
            input_binary=True,
            input_checkpoint='',
            output_node_names=model.output.op.name,
            restore_op_name='save/restore_all',
            filename_tensor_name='save/Const:0',
            output_graph=path,
            clear_devices=True,
            initializer_nodes='',
            variable_names_whitelist='',
            variable_names_blacklist='',
            input_meta_graph=None,
            input_saved_model_dir=str(self.model_path)
        )

    def load_model(
            self,
            inference: bool = False,
            path: Union[Path, str] = None,
            save_format: str = 'hdf5',
            **kwargs
    ) -> tf.keras.Model:
        r"""

        Parameters
        ----------
        inference
        path
        save_format
        kwargs

        Returns
        -------

        """
        if path is None:
            assert save_format in ['tf', 'hdf5']
            suffix = '' if save_format == 'tf' else '.h5'
            if not inference:
                path = str(self.model_path) + suffix
            else:
                path = str(self.inference_model_path) + suffix

        path = Path(path)
        suffix = path.suffix

        assert path.exists()
        assert suffix in ['', '.h5']

        if path.is_dir():
            assert (path / 'assets').exists()
            assert (path / 'variables').exists()
            assert (path / 'saved_model.pb').exists()

        if suffix == '.h5':
            model = tf.keras.models.load_model(str(path), **kwargs)
        else:
            model = tf.keras.experimental.load_from_saved_model(
                saved_model_path=str(path),
                **kwargs
            )

        assert (
                model.name == self.model_name
                or model.name == self.inference_model_name
        )

        return model

    def save_model(
            self,
            model: tf.keras.Model,
            path: Union[Path, str] = None,
            save_format: str = 'hdf5',
            overwrite: bool = False,
            **kwargs
    ) -> None:
        r"""

        Parameters
        ----------
        model
        path
        save_format
        overwrite
        kwargs

        Returns
        -------

        """
        assert (
                model.name == self.model_name
                or model.name == self.inference_model_name
        )

        if path is None:
            assert save_format in ['tf', 'hdf5']
            suffix = '' if save_format == 'tf' else '.h5'
            if model.name == self.model_name:
                path = str(self.model_path) + suffix
            else:
                path = str(self.inference_model_path) + suffix

        path = Path(path)
        suffix = path.suffix

        assert suffix in ['', '.h5']

        # model.save(filepath=str(path), overwrite=overwrite, **kwargs)

        if suffix == '.h5':
            model.save(filepath=str(path), overwrite=overwrite, **kwargs)
        else:
            if overwrite:
                if path.exists():
                    remove_directory_contents(directory=path)
            try:
                tf.keras.experimental.export_saved_model(
                    model=model,
                    saved_model_path=str(path),
                    **kwargs
                )
            except AssertionError:
                raise AssertionError(
                    f'Export directory: {str(path)}, already exists and is '
                    f'not empty. Please set  the parameter `overwrite=True` to '
                    f'overwrite all its content, or choose a different export '
                    f'directory.'
                )

        return path

    # TODO: There seems to be a keras bug for yaml serialization
    def load_model_arch(
            self,
            inference: bool = False,
            path: Union[Path, str] = None,
            save_format: str = 'json',
            **kwargs
    ) -> tf.keras.Model:
        r"""

        Parameters
        ----------
        inference
        path
        save_format
        kwargs

        Returns
        -------

        """
        if path is None:
            assert save_format in ['json', 'yaml']
            suffix = '.json' if save_format == 'json' else '.yaml'
            if not inference:
                path = str(self.model_path) + suffix
            else:
                path = str(self.inference_model_path) + suffix

        path = Path(path)
        suffix = path.suffix

        assert path.exists()
        assert suffix in ['.yaml', '.json']

        with open(path, 'r') as f:
            encoded = f.read()

        if suffix == '.yaml':
            model = tf.keras.models.model_from_yaml(encoded, **kwargs)
        else:
            model = tf.keras.models.model_from_json(encoded, **kwargs)

        assert (
                model.name == self.model_name
                or model.name == self.inference_model_name
        )

        return model

    # TODO: There seems to be a keras bug for yaml serialization
    def save_model_arch(
            self,
            model: tf.keras.Model,
            path: Union[Path, str] = None,
            save_format: str = 'json',
            **kwargs
    ) -> None:
        r"""

        Parameters
        ----------
        model
        path
        save_format
        kwargs

        Returns
        -------

        """
        assert (
                model.name == self.model_name
                or model.name == self.inference_model_name
        )

        if path is None:
            assert save_format in ['json', 'yaml']
            suffix = '.json' if save_format == 'json' else '.yaml'
            if model.name == self.model_name:
                path = str(self.model_path) + suffix
            else:
                path = str(self.inference_model_path) + suffix

        path = Path(path)
        suffix = path.suffix

        assert suffix in ['.yaml', '.json']

        if suffix == '.yaml':
            encoded = model.to_yaml(**kwargs)
        else:
            encoded = model.to_json(**kwargs)

        with open(path, 'w') as f:
            f.write(encoded)

        return path

    def load_model_weights(
            self,
            model: tf.keras.Model,
            path: Union[Path, str] = None,
            save_format: str = 'hdf5'
    ) -> tf.keras.Model:
        r"""

        Parameters
        ----------
        model
        path
        save_format

        Returns
        -------

        """
        assert (
                model.name == self.model_name
                or model.name == self.inference_model_name
        )

        if path is None:
            assert save_format in ['tf', 'hdf5']
            suffix = '' if save_format == 'tf' else '.h5'
            path = str(self.model_path) + suffix

        path = Path(path)
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

        model.load_weights(str(path))

        return model

    def save_model_weights(
            self,
            model: tf.keras.Model,
            path: Union[Path, str] = None,
            save_format: str = 'hdf5'
    ) -> None:
        r"""

        Parameters
        ----------
        model
        path
        save_format

        Returns
        -------

        """
        assert (
                model.name == self.model_name
                or model.name == self.inference_model_name
        )

        if path is None:
            assert save_format in ['tf', 'hdf5']
            suffix = '' if save_format == 'tf' else '.h5'
            path = str(self.model_path) + suffix

        path = Path(path)
        suffix = path.suffix

        assert suffix in ['', '.h5']

        model.save_weights(str(path))

        return path


# TODO: Document me!
class DetectorManager(ModelManager):
    r"""

    Parameters
    ----------
    anchor_parameters

    data_format

    cache


    Attributes
    ----------
    anchor_parameters

    data_format

    cache

    """
    def __init__(
            self,
            anchor_parameters: AnchorParameters,
            data_format: Optional[str] = None,
            cache: Union[Path, str] = Path.home() / '.hudlrd'
    ) -> None:
        super().__init__(cache=cache)
        self.anchor_parameters = anchor_parameters
        self.data_format = self._handle_data_format(data_format=data_format)

    @staticmethod
    def _handle_data_format(data_format):
        r"""

        Parameters
        ----------
        data_format

        Returns
        -------

        """
        return conv_utils.normalize_data_format(data_format)

    def _create_sequential_model(
            self,
            backbone: tf.keras.Model,
            head: tf.keras.Model,
    ) -> tf.keras.Model:
        r"""

        Parameters
        ----------
        backbone
        head

        Returns
        -------

        """
        return tf.keras.Sequential([backbone, head], name=self.model_name)

    def _create_functional_model(
            self,
            backbone: tf.keras.Model,
            head: tf.keras.Model,
    ) -> tf.keras.Model:
        r"""

        Parameters
        ----------
        backbone
        head

        Returns
        -------

        """
        inputs = tf.keras.Input(backbone.input_shape, name='model_input')
        x = backbone(inputs)
        outputs = head(x)

        return tf.keras.Model(
            inputs=inputs,
            outputs=outputs,
            name=self.model_name
        )

    @abstractmethod
    def create_matcher(self, **kwargs) -> Matcher:
        r"""
        Creates the unmatch_fn associated with the detector.

        Parameters
        ----------
        **kwargs
            The parameters of the unmatch_fn associated with this detector.

        Returns
        -------
        The unmatch_fn associated with this detector.
        """


@public
class YoloManager(DetectorManager):

    def __init__(
            self,
            anchor_parameters: AnchorParameters = yolo_anchor_parameters,
            data_format: Optional[str] = None,
            cache: Union[Path, str] = Path.home() / '.hudlrd'
    ) -> None:
        super().__init__(
            anchor_parameters=anchor_parameters,
            data_format=data_format,
            cache=cache
        )

    def create_model(
            self,
            backbone: tf.keras.Model,
            n_classes: int,
            n_layers: int = 3,
            n_features: Union[int, Sequence[int]] = 128,
            cls_activation: Union[str, callable] = 'softmax',
    ) -> tf.keras.Model:
        r"""
        Creates a Yolo detector model.

        Parameters
        ----------
        backbone

        n_classes

        n_layers

        n_features

        cls_activation

        Returns
        -------
        The Yolo detector model.
        """
        return self._create_sequential_model(
            backbone=backbone,
            head=create_yolo_head(
                feature_map=backbone.output,
                n_classes=n_classes,
                n_anchors=self.anchor_parameters.n_anchors,
                n_layers=n_layers,
                n_features=n_features,
                cls_activation=cls_activation
            )
        )

    def create_matcher(self, iou_threshold: float = 0.75) -> YoloMatcher:
        r"""
        Creates the unmatch_fn associated with the Yolo detector.

        Parameters
        ----------
        iou_threshold
            The iou threshold above which ground truth bounding boxes are
            associated with anchor boxes.

        Returns
        -------
        The unmatch_fn associated with the Yolo detector.
        """
        return YoloMatcher(iou_threshold=iou_threshold)

    def create_inference_model(
            self,
            model: tf.keras.Model,
            iou_threshold: float = 0.75,
            max_detections: int = 50,
            score_threshold: float = 0.5,
            nms_iou_threshold: Optional[float] = 0.75,
    ) -> tf.keras.Model:
        r"""

        Parameters
        ----------
        model
        iou_threshold
        max_detections
        score_threshold
        nms_iou_threshold

        Returns
        -------

        """
        assert model.name == self.model_name

        outputs = YoloInferenceLayer(
            iou_threshold=iou_threshold,
            scales=self.anchor_parameters.scales,
            ratios=self.anchor_parameters.ratios,
            n_anchors=self.anchor_parameters.n_anchors,
            max_detections=max_detections,
            score_threshold=score_threshold,
            nms_iou_threshold=nms_iou_threshold,
            data_format=self.data_format,
        )(model.output)

        return tf.keras.Model(
            inputs=model.inputs,
            outputs=outputs,
            name=f'inference_{self.model_name}'
        )


