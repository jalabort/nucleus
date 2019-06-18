from typing import Optional, Dict

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K

from nucleus.box import unpad_tensor


# class TensorBoardImage(tf.keras.callbacks.TensorBoard):
#     r"""
#
#     """
#     def __init__(self, ds:, create_inference_model_fn, **kwargs):
#         super().__init__(**kwargs)
#         self.ds = ds
#         self.create_inference_model_fn = create_inference_model_fn
#
#     def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
#         super().on_epoch_end(epoch=epoch, logs=logs)
#
#         for images, boxes in range(self.ds.take(1)):
#
#             inference_model = self.create_inference_model_fn(self.model)
#
#             detections = inference_model(images)
#
#             boxes_to_draw = [unpad_tensor(boxes) for boxes in detections]
#
#             drawn_images = tf.image.draw_bounding_boxes(
#                 images=images,
#                 boxes=boxes_to_draw,
#                 color=None,
#             )
#
#             summary_writer = self._get_writer(self._train_run_name)
#
#             with summary_writer.as_default():
#                 tf.summary.image('images', drawn_images)


# TODO: Should we rewrite this with pure tensorflow?
class CyclicLR(tf.keras.callbacks.Callback):
    r"""
    This callback implements a cyclical learning rate policy (CLR).

    The method cycles the learning rate between two boundaries with some
    constant frequency, as detailed in the paper "Cyclical Learning Rates for
    Training Neural Networks" by Leslie N. Smith.

    The amplitude of the cycle can be scaled on a per-iteration or
    per-cycle basis.

    This class has three built-in policies, as put forth in the paper:
    - triangular:
        A basic triangular cycle w/ no amplitude scaling.
    - triangular2:
        A basic triangular cycle that scales initial amplitude by half each
        cycle.
    - exp_range:
        A cycle that scales initial amplitude by gamma**(cycle iterations) at
        each cycle iteration.

    For more detail, please see paper.

    Notes
    -----
    This class was copy-pasted from this excellent repository
    https://github.com/bckenstler/CLR which is, unfortunately, not pip
    installable.

    References
    ----------
    ..[1] Leslie N. Smith, "Cyclical Learning Rates for Training Neural
          Networks", WACV 2017, https://arxiv.org/abs/1506.01186

    Examples
    --------
    Typical usage:
        ```python
            clr = CyclicLR(
                base_lr=0.001,
                max_lr=0.006,
                step_size=2000.,
                mode='triangular'
            )
            model.fit(X_train, Y_train, callbacks=[clr])
        ```

    The class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5 * (1 + np.sin(x * np.pi / 2.))
            clr = CyclicLR(
                base_lr=0.001,
                max_lr=0.006,
                step_size=2000.,
                scale_fn=clr_fn,
                scale_mode='cycle'
            )
            model.fit(X_train, Y_train, callbacks=[clr])
        ```

    Parameters
    ----------
    base_lr
        Initial learning rate which is the lower boundary in the cycle.
    max_lr
        Upper boundary in the cycle. Functionally, it defines the cycle
        amplitude (max_lr - base_lr). The lr at any cycle is the sum of
        base_lr and some scaling of the amplitude; therefore max_lr may not
        actually be reached depending on scaling function.
    step_size
        Number of training iterations per half cycle. Authors suggest setting
        step_size 2-8 x training iterations in epoch.
    mode
        One of {triangular, triangular2, exp_range}. Default 'triangular'.
        Values correspond to policies detailed above. If scale_fn is not
        None, this argument is ignored.
    gamma
        Constant in 'exp_range' scaling function: gamma ** cycle iterations.
    scale_fn
        Custom scaling policy defined by a single argument lambda function,
        where 0 <= scale_fn(x) <= 1 for all x >= 0. mode parameter is ignored.
    scale_mode
        {'cycle', 'iterations'}. Defines whether scale_fn is evaluated on
        cycle number or cycle iterations (training iterations since start of
        cycle). Default is 'cycle'.
    """

    def __init__(
            self,
            base_lr: float = 1e-3,
            max_lr: float = 6e-3,
            step_size: int = 2000,
            mode: str = 'triangular',
            gamma: float = 1,
            scale_fn: callable = None,
            scale_mode: str = 'cycle'
    ) -> None:
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma

        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2. ** (x - 1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** x
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

        self.clr_iterations = 0
        self.trn_iterations = 0
        self.history = {}

        self._reset()

    def _reset(
            self,
            new_base_lr: Optional[float] = None,
            new_max_lr: Optional[float] = None,
            new_step_size: Optional[float] = None
    ) -> None:
        """
        Resets cycle iterations. Optional boundary/step size adjustment.
        Parameters
        ----------
        new_base_lr
            New base learning rate.
        new_max_lr
            New maximum learning rate.
        new_step_size
            New step size.
        """
        if new_base_lr is not None:
            self.base_lr = new_base_lr
        if new_max_lr is not None:
            self.max_lr = new_max_lr
        if new_step_size is not None:
            self.step_size = new_step_size
        self.clr_iterations = 0

    @property
    def clr(self) -> float:
        r"""
        Returns the current cyclic learning rate value.
        """
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.scale_mode == 'cycle':
            return (
                self.base_lr
                + (self.max_lr - self.base_lr)
                * np.maximum(0, (1 - x))
                * self.scale_fn(cycle)
            )
        else:
            return (
                self.base_lr
                + (self.max_lr - self.base_lr)
                * np.maximum(0, (1 - x))
                * self.scale_fn(self.clr_iterations)
            )

    def on_train_begin(self, logs: Optional[Dict] = None):
        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr)

    def on_batch_end(self, epoch: int, logs: Optional[Dict] = None):
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(
            K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        K.set_value(self.model.optimizer.lr, self.clr)
