# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import abc

import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from typing import Union


class AveragedOptimizerWrapper(tf.keras.optimizers.Optimizer, metaclass=abc.ABCMeta):
    def __init__(
            self,
            optimizer: Union[tf.keras.optimizers.Optimizer, str],
            sequential_update: bool = True,
            name: str = "AverageOptimizer",
            **kwargs
    ):
        super().__init__(name, **kwargs)

        if isinstance(optimizer, str):
            optimizer = tf.keras.optimizers.get(optimizer)

        if not isinstance(optimizer, tf.keras.optimizers.Optimizer):
            raise TypeError(
                "optimizer is not an object of tf.keras.optimizers.Optimizer"
            )

        if not isinstance(sequential_update, bool):
            raise TypeError("sequential_update must be of bool type")

        self._optimizer = optimizer
        self._sequential_update = sequential_update

    def _create_slots(self, var_list):
        self._optimizer._create_slots(var_list=var_list)
        for var in var_list:
            self.add_slot(var, "average")

    def _create_hypers(self):
        self._optimizer._create_hypers()

    def _prepare(self, var_list):
        return self._optimizer._prepare(var_list=var_list)

    def apply_gradients(self, grads_and_vars, name=None):
        self._optimizer._iterations = self.iterations
        return super().apply_gradients(grads_and_vars, name)

    @abc.abstractmethod
    def average_op(self, var, average_var):
        raise NotImplementedError

    def _apply_average_op(self, train_op, var):
        average_var = self.get_slot(var, "average")
        if self._sequential_update:
            with tf.control_dependencies([train_op]):
                avg_op = self.average_op(var, average_var)
        else:
            avg_op = self.average_op(var, average_var)

        return avg_op

    def _resource_apply_dense(self, grad, var):
        train_op = self._optimizer._resource_apply_dense(grad, var)
        average_op = self._apply_average_op(train_op, var)
        return tf.group(train_op, average_op)

    def _resource_apply_sparse(self, grad, var, indices):
        train_op = self._optimizer._resource_apply_sparse(grad, var, indices)
        average_op = self._apply_average_op(train_op, var)
        return tf.group(train_op, average_op)

    def _resource_apply_sparse_duplicate_indices(self, grad, var, indices):
        train_op = self._optimizer._resource_apply_sparse_duplicate_indices(
            grad, var, indices
        )
        average_op = self._apply_average_op(train_op, var)
        return tf.group(train_op, average_op)

    def assign_average_vars(self, var_list):
        """Assign variables in var_list with their respective averages.
        Args:
            var_list: List of model variables to be assigned to their average.
        Returns:
            assign_op: The op corresponding to the assignment operation of
            variables to their average.
        Example:
        ```python
        model = tf.Sequential([...])
        opt = tfa.optimizers.SWA(
                tf.keras.optimizers.SGD(lr=2.0), 100, 10)
        model.compile(opt, ...)
        model.fit(x, y, ...)
        # Update the weights to their mean before saving
        opt.assign_average_vars(model.variables)
        model.save('model.h5')
        ```
        """
        assign_op = tf.group(
            [
                var.assign(self.get_slot(var, "average"))
                for var in var_list
                if var.trainable
            ]
        )
        return assign_op

    def get_config(self):
        config = {
            "optimizer": tf.keras.optimizers.serialize(self._optimizer),
            "sequential_update": self._sequential_update,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        optimizer = tf.keras.optimizers.deserialize(
            config.pop("optimizer"), custom_objects=custom_objects,
        )
        return cls(optimizer, **config)

    @property
    def weights(self):
        return self._weights + self._optimizer.weights

    @property
    def lr(self):
        return self._optimizer._get_hyper("learning_rate")

    @lr.setter
    def lr(self, lr):
        self._optimizer._set_hyper("learning_rate", lr)  #

    @property
    def learning_rate(self):
        return self._optimizer._get_hyper("learning_rate")

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        self._optimizer._set_hyper("learning_rate", learning_rate)


# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf
from tensorflow.python.training.moving_averages import assign_moving_average

from typing import Optional, Union


#@tf.keras.utils.register_keras_serializable(package='Addons')
class MovingAverage(AveragedOptimizerWrapper):
    """Optimizer that computes a moving average of the variables.
    Empirically it has been found that using the moving average of the trained
    parameters of a deep network is better than using its trained parameters
    directly. This optimizer allows you to compute this moving average and swap
    the variables at save time so that any code outside of the training loop
    will use by default the average values instead of the original ones.
    Example of usage:
    ```python
    opt = tf.keras.optimizers.SGD(learning_rate)
    opt = tfa.optimizers.MovingAverage(opt)
    ```
    """

    def __init__(self,
                 optimizer: Union[tf.keras.optimizers.Optimizer, str],
                 sequential_update: bool = True,
                 average_decay = 0.99,
                 num_updates: Optional[str] = None,
                 name: str = "MovingAverage",
                 **kwargs):
        r"""Construct a new MovingAverage optimizer.
        Args:
            optimizer: str or `tf.keras.optimizers.Optimizer` that will be
                used to compute and apply gradients.
            sequential_update: Bool. If False, will compute the moving average
                at the same time as the model is updated, potentially doing
                benign data races. If True, will update the moving average
                after gradient updates.
            average_decay: float. Decay to use to maintain the moving averages
                of trained variables.
            num_updates: Optional count of the number of updates applied to
                variables.
            name: Optional name for the operations created when applying
                gradients. Defaults to "MovingAverage".
            **kwargs: keyword arguments. Allowed to be {`clipnorm`,
                `clipvalue`, `lr`, `decay`}. `clipnorm` is clip gradients by
                norm; `clipvalue` is clip gradients by value, `decay` is
                included for backward compatibility to allow time inverse
                decay of learning rate. `lr` is included for backward
                compatibility, recommended to use `learning_rate` instead.
        """
        super().__init__(optimizer, sequential_update, name, **kwargs)
        self._num_updates = num_updates
        if self._num_updates is not None:
            num_updates = tf.cast(
                self._num_updates, tf.float32, name="num_updates")
            average_decay = tf.minimum(
                average_decay, (1.0 + num_updates) / (10.0 + num_updates))

        self._set_hyper("average_decay", average_decay)

    def average_op(self, var, average_var):
        decay = self._get_hyper('average_decay', tf.dtypes.float32)
        return assign_moving_average(average_var, var, decay, False)

    def get_config(self):
        config = {
            'average_decay': self._serialize_hyperparameter('average_decay'),
            'num_updates': self._num_updates,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def _create_slots(self, var_list):
        self._optimizer._create_slots(var_list=var_list)  # pylint: disable=protected-access
        for var in var_list:
            self.add_slot(var, 'average', var.read_value())

    def _create_slots_bak(self, var_list):
        self._optimizer._create_slots(var_list=var_list)  # pylint: disable=protected-access

        # for tpu can not call var.read_value in init_op
        for var in var_list:
            self.add_slot(var, 'average')
#        if not context.executing_eagerly():
#            for var in var_list:
#                self.add_slot(var, 'average', var.initial_value())
#        else:
#            for var in var_list:
#                self.add_slot(var, 'average', var.read_value())


class WarmupWrap(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, scheduler, warm_up_step, name='WarmupWrap'):
        self.scheduler = scheduler
        self.warm_up_step = warm_up_step
        self.name = name

    def get_config(self):
        return {
            "name": self.name,
            "warm_up_step": self.warm_up_step,
            'scheduler': self.scheduler,
        }

    def get_lr(self, step):
        with ops.name_scope_v2(self.name or "WarmupWrap") as name:
            if callable(self.scheduler):
                initial_learning_rate = ops.convert_to_tensor(self.scheduler.initial_learning_rate,
                                                             name="initial_learning_rate")
            else:
                initial_learning_rate = ops.convert_to_tensor(self.scheduler)
            dtype = initial_learning_rate.dtype
            decay_steps = math_ops.cast(self.warm_up_step, dtype)
            global_step_recomp = math_ops.cast(step, dtype)
            p = global_step_recomp / decay_steps
            return math_ops.multiply(
                initial_learning_rate, p, name=name)

    @tf.function
    def __call__(self, step):
        if step > self.warm_up_step:
            if callable(self.scheduler):
                return self.scheduler(step-self.warm_up_step)
            else:
                return self.scheduler
        else:
            return self.get_lr(step)
