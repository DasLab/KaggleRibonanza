import logging, os
import random
from collections import OrderedDict
import numpy as np
from tqdm import tqdm

import tensorflow as tf
from tensorflow.python.keras.callbacks import CallbackList
#import tensorflow_addons as tfa
from tensorflow_addons.optimizers import MovingAverage

from util.basic_models import CFG as BasicCFG, Model as BasicModel, LossHist
from util.tf import util as tf_util
from .optimizer import WarmupWrap
from util import util

logger = logging.getLogger(__name__)


class CFG(BasicCFG):
    def __init__(self):
        super(CFG, self).__init__()
        self.optimizer = 'adam'
        self.loss = 'mse'
        self.global_step = None
        self.gradient_clip = None
        self.dropout = None
        self.keep_checkpoint_every_n_hours = 1000000000
        self.save_keep = 1000000000
        self.save_summary_step = 1000000000
        self.use_tpu = False
        self.use_tffit = True
        self.ema = 0
        self.tpu_name = None
        self.warm_start_path = None  # used by tpu
        self.tpu_zone = None
        self.gcp_project = None
        self.num_core_per_host = 1
        self.num_hosts = 1
        self.tpu_loop_iterations = 2
        self.run_eagerly = False
        self.accumulated_batch_size = 1
        self.compile_model = False

#    def dump(self, fpath):
#        tf.io.write_file(fpath, json.dumps(self.__dict__))
#
#    def load(self, fpath):
#        cfg = json.loads(tf.io.read_file(fpath).numpy())
#        self.update(cfg)

class ModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self, filepath, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=True, mode='auto',
                save_freq=int(1e16), **kwargs):
        self._save_on_train_end = kwargs.pop('save_on_train_end', False)
        super(ModelCheckpoint, self).__init__(filepath, monitor=monitor, verbose=verbose, save_best_only=save_best_only,
                                              save_weights_only=save_weights_only, mode=mode, save_freq=int(1e32), **kwargs)

    def _save_model(self, epoch, logs):
        if isinstance(self.model.opt, MovingAverage):
            self.model.opt.assign_average_vars(self.model.variables)
        super(ModelCheckpoint, self)._save_model(epoch, logs)

    def on_train_end(self, logs=None):
        if self._save_on_train_end:
            self._save_model(epoch=self._current_epoch, logs=logs)
        super(ModelCheckpoint, self).on_train_end(logs)

def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  tf.keras.utils.set_random_seed(seed)
  tf.random.set_seed(seed)
  tf.experimental.numpy.random.seed(seed)
  # When running on the CuDNN backend, two further options must be set
  #os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
  #os.environ['TF_DETERMINISTIC_OPS'] = '1'
  os.environ["PYTHONHASHSEED"] = str(seed)

class TFModel(BasicModel):
    cfg = CFG()
    def __init__(self, name='TFModel', cfg={}):
        super(TFModel, self).__init__(name, cfg)
        self._input_features = self.get_input_features()
        self._called = False  # if run eagerly need call once to create weights
        self._restored = False
        self._accumulated_grads = None
        self._batch_id = None
        self._opt = None
        self._trainable_variables = None
        self._run_batch_graph_func = None
        self._fit_batch_graph_func = None
        self._val_batch_graph_func = None
        self._fit_func = None
        self.strategy = None
        if self.cfg.restore:
            super().restore()

        if self.cfg.restore_epoch==-99999:
            self.cfg.restore_epoch = None
    def get_input_features(self):
        """

        :return: OrderedDict with InputFeature instance. For targets and smaple_weights, must named endswith targets,
        smaple_weights
        """
        input_features = OrderedDict()
        return input_features

    def get_keras_batch_inputs_and_types(self):
        inputs, inputs_types = OrderedDict(), OrderedDict()
        for name, fea in self._input_features.items():
            inputs[fea.name] = None
            inputs_types[fea.name] = fea.dtype
        return inputs, inputs_types

    def create_keras_inputs(self, input_features):
        inputs = OrderedDict()
        for k, fea in input_features.items():
            inputs[fea.name] = tf.keras.Input(tuple(fea.shape), dtype=fea.dtype, name=fea.name)
        return inputs

    def main_nn(self, inputs):
        outputs = OrderedDict()
        outputs['dummy'] = inputs['dummy']
        return outputs

    def _create_keras_model(self, inputs, outputs):
        model = tf.keras.Model(inputs=tuple(inputs.values()), outputs=outputs)
        pred_model = model
        return model, pred_model

    def create_keras_model(self, inputs, compile=True):
        core_model = self.create_core_model()
        outputs = core_model(**inputs)
        model, pred_model = self._create_keras_model(inputs, outputs)
        if compile:
            self.compile_model(model)
            logger.info('num of parameters:%s', model.summary())
        return model, pred_model

    def compile_model(self, model):
        optimizer = self._create_optimizer(self.cfg.lr)
        loss = self.get_keras_loss()
        model.compile(optimizer=optimizer, loss=loss, run_eagerly=self.cfg.run_eagerly)

    def create_model(self, no_opt=False):
        set_seed(self.cfg.seed)
        #if self.cfg.use_fp16:
        #    tf.config.optimizer.set_jit(True)
        #    tf.config.optimizer.set_experimental_options(
        #        {"auto_mixed_precision": True})
            #policy = mixed_precision.Policy('mixed_float16')
            #mixed_precision.set_policy(policy)


        if self.cfg.use_tpu:
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=self.cfg.tpu_name, zone=self.cfg.tpu_zone,
                                                                         project=self.cfg.gcp_project)
            tf.config.experimental_connect_to_cluster(resolver)
            tf.tpu.experimental.initialize_tpu_system(resolver)
            self.strategy = tf.distribute.experimental.TPUStrategy(resolver)
            with self.strategy.scope():
                self.inputs = self.create_keras_inputs(self._input_features)
                self._model, self._pred_model = self.create_keras_model(self.inputs, compile=self.cfg.compile_model)
                self._opt = self._create_optimizer(self.cfg.opt, lr=self.cfg.lr, n_lr_decay_step=self.cfg.n_lr_decay_step,
                                                   lr_decay_rate=self.cfg.lr_decay_rate, n_lr_warmup_step=self.cfg.n_lr_warmup_step,
                                                   ema=self.cfg.ema, weight_decay=self.cfg.weight_decay
                                                   )
                if self.cfg.restore:
                    self.restore(epoch=self.cfg.restore_epoch)
        else:
            self.inputs = self.create_keras_inputs(self._input_features)
            self._model, self._pred_model = self.create_keras_model(self.inputs, compile=self.cfg.compile_model)
            if self.cfg.restore:
                self.restore(epoch=self.cfg.restore_epoch)

        if self.cfg.recompute:
            self._fit_func = tf.recompute_grad(self._model)
        else:
            self._fit_func = self._model


    def get_keras_loss(self):
        """

        :return: keras loss then compile the model
        """
        return self.cfg.loss

    def create_optimizer(self, lr=None):
        if lr is None:
            lr = self.cfg.lr
        opt = util.dynamic_import(self.cfg.opt)(learning_rate=lr, **self.cfg.opt_paras)
        return opt

    def _create_optimizer(self, lr=None):
        if lr is None:
            lr = self.cfg.lr
        if self.cfg.lr_scheduler is not None:
            lr = util.dynamic_import(self.cfg.lr_scheduler)(lr, **self.cfg.lr_scheduler_paras)
            if self.cfg.n_lr_warmup_step > 0:
                lr = WarmupWrap(lr, self.cfg.n_lr_warmup_step)

        opt = self.create_optimizer(lr)

        if self.cfg.ema > 0:
            opt = MovingAverage(opt, average_decay=self.cfg.ema)
        if self.cfg.use_fp16:
            opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)
        return opt

    def save(self, epoch=None, save_path=None):
        if save_path is None:
            save_name = 'model-{epoch:08d}.ckpt'.format(epoch=epoch) if epoch else 'model.ckpt'
            save_path = self.gen_fname(save_name)
        self.update_saved_ckpt([save_path], epoch)
        super(TFModel, self).save()
        self._model.save_weights(save_path)
        #tf.saved_model.save(self._model, save_path)
        logger.info("Model saved to file:{}".format(save_path))

    def _get_save_vars(self):
        return tf.trainable_variables()

    def _get_restore_vars(self):
        return tf.trainable_variables()

    def _get_restore_assignment_map(self, var_list, model_path):
        (assignment_map, initialized_variable_names) = tf_util.get_assignment_map_from_checkpoint(var_list, model_path)
        assignment_map = {var.op.name: var for var in var_list if var.name in initialized_variable_names}

        return assignment_map, initialized_variable_names

    def restore(self, epoch=None, var_list=None, model_dir=None, no_opt=True):
        if self._model is None:
            self.create_model()

        if not self._restored:
            if model_dir is None:
                model_dir = self.gen_fname()
            model_paths, restore_epoch = self.get_checkpoint_path(model_dir, epoch)
            model_path = model_paths[0]
            self._model.load_weights(model_path)
            self._restored = True
            logger.info('weights restored from %s', model_path)

    def get_modelcheckpoint_callback(self):
        model_path = self.gen_fname('model-{epoch:04d}.ckpt')
        cp = ModelCheckpoint(filepath=model_path, save_on_train_end=self.cfg.save, save_best_only=self.cfg.save_best)
        return cp

    def get_callbacks(self):
        callbacks = []
        if not self.cfg.no_validate:
            callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.cfg.patience,
                                                              min_delta=self.cfg.es_min_delta,
                                                              ))
        callbacks.append(self.get_modelcheckpoint_callback())
        return callbacks

    def calc_loss(self, inputs, outputs):
        """

        :param outputs:
        :param targets:
        :return:  dictionary of losses
        """
        raise {}

    def _calc_loss(self, inputs, outputs):
        losses = self.calc_loss(inputs, outputs)
        if self.strategy is not None:
            for k, v in losses.items():
                if v.get_shape().ndims==0:
                    v = tf.stack([v])
                losses[k] = tf.nn.compute_average_loss(v, global_batch_size=self.strategy.num_replicas_in_sync)
        return losses

    def fit_batch(self, batch):
        with tf.GradientTape() as tape:
            outputs = self._fit_func(batch, training=tf.constant(True))
            losses = self._calc_loss(batch, outputs)
            loss = losses['loss']
            if self.cfg.use_fp16:
                loss = self._opt.get_scaled_loss(loss)
        vars = self._get_train_vars()
        grads = self.calc_grads(tape, vars, loss)
        self.apply_grads(vars, grads)
        return grads, losses

    @tf.function
    def fit_batch_graph(self, batch):
        return self.fit_batch(batch)

    def _fit_batch_distribute(self, inputs_with_signatures):
        inputs, input_signatures = inputs_with_signatures
        if self._fit_batch_graph_func is None:
            with self.strategy.scope():
                self._fit_batch_graph_func = tf.function(self.fit_batch, input_signature=(input_signatures,))
        grads, losses = self.strategy.experimental_run_v2(self._fit_batch_graph_func, args=(inputs,))
        losses = self._agg_distribute_loss(losses)
        return grads, losses

    def _fit_batch(self, inputs):
        inputs, input_signatures = inputs
        if self.strategy is not None:
            return self._fit_batch_distribute(inputs_with_signatures)

        if self.cfg.run_eagerly:
            return self.fit_batch(inputs)
        else:
            if self._fit_batch_graph_func is None:
                self._fit_batch_graph_func = self.fit_batch_graph.get_concrete_function(input_signatures)
            return self._fit_batch_graph_func(inputs)


    def apply_grads(self, vars, grads):
        #vars = self.get_train_vars()

        if not self.cfg.accumulated_batch_size > 1:
            self._opt.apply_gradients(zip(grads, vars))

    def calc_grads(self, tape, vars, loss):
        grads = tape.gradient(loss, vars)
        if self.cfg.use_fp16:
            grads = self._opt.get_unscaled_gradients(grads)
        if self.cfg.gradient_clip is not None:
            (grads, gnorm) = tf.clip_by_global_norm(grads, clip_norm=self.cfg.gradient_clip)
        return grads

    def apply_accumulated_grads(self, vars, grads):
        if self._accumulated_grads is None:
            #self._accumulated_grads = [tf_util.flat_gradients(grad)/self.cfg.accumulated_batch_size for grad in grads]
            self._accumulated_grads = [grad/self.cfg.accumulated_batch_size for grad in grads]
        else:
            for i, grad in enumerate(grads):
                #self._accumulated_grads[i] += tf_util.flat_gradients(grad)/self.cfg.accumulated_batch_size
                self._accumulated_grads[i] += grad / self.cfg.accumulated_batch_size
        if (self._batch_id + 1) % self.cfg.accumulated_batch_size == 0:
            grads_and_vars = zip(self._accumulated_grads, vars)
            self._opt.apply_gradients(grads_and_vars)
            self._accumulated_grads = None

    def get_train_vars(self):
        return self._model.trainable_variables

    def _get_train_vars(self):
        if self._trainable_variables is None:
            self._trainable_variables = self.get_train_vars()
        return self._trainable_variables

    def outputs2preds(self, outputs):
        preds = dict()
        for output in outputs:
            for k, v in output.items():
                if k not in preds:
                    preds[k] = []
                if self.cfg.to_list:
                    preds[k].extend(list(v))
                else:
                    preds[k].append(v)
        if self.cfg.do_concat:
            for k in preds:
                preds[k] = np.concatenate(preds[k],0)
        return preds

    def predict(self, ds, data_type='val', **kwargs):
        outputs = []
        for i, batch in enumerate(ds):
            output = self.predict_batch(batch, **kwargs)
            outputs.append(output)
        preds = self.outputs2preds(outputs)
        logger.info('predict done')
        return preds

    def process_batch_output(self, output, batch):
        new_output = dict()
        for k in self.cfg.batch_keys:
            if k in output:
                new_output[k] = output[k].numpy()
            elif k in batch:
                new_output[k] = batch[k].numpy()
        return new_output

    def predict_batch(self, batch, **kwargs):
        inputs = self._get_model_inputs_tuple(batch)
        output = self._run_batch(inputs, **kwargs)
        output = self.process_batch_output(output, batch)
        return output

    def run_batch_to_init_weights(self, inputs):
        self._run_batch(inputs)
        self._called = True

    def _get_model_inputs_tuple(self, batch):
        #inputs = []
        #input_signatures = []
        inputs, input_signatures = OrderedDict(), OrderedDict()
        for name, fea in self._input_features.items():
            if name in batch:
                #inputs.append(batch[name])
                #input_signatures.append(tf.TensorSpec(shape=tuple([None] + fea.shape), dtype=fea.dtype))
                inputs[name] = batch[name]
                input_signatures[name] = tf.TensorSpec(shape=tuple([None] + fea.shape), dtype=fea.dtype)
        #return tuple(inputs), tuple(input_signatures)
        return inputs, input_signatures

#    def _get_model_inputs_tuple(self, batch):
#        inputs = []; input_signatures = []
#        for name, fea in self._input_features.items():
#            if name in batch:
#                inputs.append(batch[name])
#                input_signatures.append(tf.TensorSpec(shape=None, dtype=fea.dtype))
#        return tuple(inputs), tuple(input_signatures)

    def fit_epoch(self, ds, epoch, step, val_ds=None, save_opt=False, best_loss=None, best_epoch=None, **kwargs):
        """

        :param ds:
        :param epoch:
        :param step: global step accross epochs
        :param opt:
        :param history:
        :param barlog:
        :param kwargs:
        :return:
        """
        loss_hist = LossHist()
        should_stop = False
        desc = 'Train Epoch:{}, lr:{:.8f}, loss:{}'
        miniters, verbose, is_master = self.cfg.verbose, self.cfg.verbose, True
        if isinstance(self._opt.lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            current_lr = self._opt.lr(self._opt.iterations).numpy()
        else:
            current_lr = self._opt.lr.numpy()
        ds_itr = tqdm(enumerate(ds), total=len(ds), disable=verbose == 0, miniters=miniters, desc=desc.format(epoch, current_lr, ''))
        for i, batch in ds_itr:
            ds_itr.miniters = miniters
            input_batch = self._get_model_inputs_tuple(batch)
            self._batch_id = i
            if not self._called:
                self.run_batch_to_init_weights(input_batch)
            step += 1
            grads, losses = self._fit_batch(input_batch)
            for k, v in losses.items():
                losses[k] = v.numpy()
            loss_hist.append(losses)
            if isinstance(self._opt.lr, tf.keras.optimizers.schedules.LearningRateSchedule):
                current_lr = self._opt.lr(self._opt.iterations).numpy()
            else:
                current_lr = self._opt.lr.numpy()
            if step%verbose == 0 and is_master:
                ds_itr.set_description(desc.format(epoch, current_lr, loss_hist.avg_output()))

            if self.cfg.accumulated_batch_size>1:
                self.apply_accumulated_grads(self._get_train_vars(), grads)

            if step % self.cfg.n_save_step == 0:
                logger.info('%%%% save model to %s for step:%s', self.cfg.output_dir, step)
                self.save(epoch=step)

            if (i + 1) >= self.cfg.n_epoch_step:
                logger.info('max %s step per epoch reached', self.cfg.n_epoch_step)
                break
            if step % self.cfg.val_step == 0:
                losses = loss_hist.get_avg()
                best_loss, best_epoch, should_stop = self.validate(val_ds, step, epoch, best_loss, best_epoch, losses['loss'])
                if should_stop:
                    break
            if step % self.cfg.n_train_step == 0:
                break
        losses = loss_hist.get_avg()
        return step, losses, best_loss, should_stop

    def _on_batch_end(self, batch, i, callbacks, losses):
        if i == 0:
            callbacks.params['metrics'] = losses.keys()
            callbacks.set_params(callbacks.params)
        logs = OrderedDict()
        logs['size'] = 1
        logs.update(losses)
        callbacks.on_batch_end(i, logs=logs)

    def _create_callbacks(self, steps=None, do_validation=False):
        callback_params = {
            'batch_size': self.cfg.batch_size,
            'samples': None,
            'steps': steps,
            'verbose': self.cfg.verbose,
            'epochs': self.cfg.epochs,
            'do_validation': True,
            'metrics': None,
        }
        history = tf.keras.callbacks.History()
        barlog = tf.keras.callbacks.ProgbarLogger('steps')
        baselogger = tf.keras.callbacks.BaseLogger()
        callbacks = CallbackList([baselogger, history, barlog])
        callbacks.set_params(callback_params)
        return callbacks, history

    @tf.function
    def run_batch_graph(self, inputs):
        return self.run_batch(inputs)

    def _run_batch_distribute(self, inputs_with_signatures):
        inputs, input_signatures = inputs_with_signatures
        if self._run_batch_graph_func is None:
            with self.strategy.scope():
                self._run_batch_graph_func = tf.function(self.run_batch, input_signature=(input_signatures,))
        outputs = self.strategy.experimental_run_v2(self._run_batch_graph_func, args=(inputs,))
        outputs = self._agg_distribute_outputs(outputs)
        return outputs

    def run_batch(self, inputs):
        return self._pred_model(inputs, training=False)

    def _run_batch(self, inputs, **kwargs):
        inputs, input_signatures = inputs
        if self.strategy is not None:
            return self._run_batch_distribute(inputs)
        if self.cfg.run_eagerly:
            outputs = self.run_batch(inputs)
        else:
            if self._run_batch_graph_func is None:
                self._run_batch_graph_func = self.run_batch_graph.get_concrete_function(input_signatures)
            outputs = self._run_batch_graph_func(inputs)
        return outputs

    def val_batch(self, inputs):
        outputs = self._pred_model(inputs, training=tf.constant(False))
        losses = self._calc_loss(inputs, outputs)
        return losses, outputs

    @tf.function
    def val_batch_graph(self, inputs):
        return self.val_batch(inputs)

    @tf.function
    def _agg_distribute_loss(self, losses):
        new_losses = {}
        for k, v in losses.items():
            v = tf.stack(v.values, axis=0)
            v = tf.math.reduce_sum(v)
            new_losses[k] = v
        return new_losses

    def _agg_distribute_outputs(self, outputs):
        new_outputs = {}
        if isinstance(outputs, dict):
            for k, v in outputs.items():
                v = tf.concat(v.values, axis=0)
                new_outputs[k] = v
        elif isinstance(outputs, tuple):
            agg_outputs = []
            for output in outputs:
                agg_outputs.append(tf.concat(output.values, axis=0))
            new_outputs = tuple(agg_outputs)
        else:
            new_outputs = tf.concat(outputs.values, axis=0)
        return new_outputs

    def _val_batch_distribute(self, batch):
        inputs_with_signatures = self._get_model_inputs_tuple(batch)
        inputs, input_signatures = inputs_with_signatures
        if self._val_batch_graph_func is None:
            with self.strategy.scope():
                self._val_batch_graph_func = tf.function(self.val_batch, input_signature=(input_signatures,))
        losses, outputs = self.strategy.experimental_run_v2(self._val_batch_graph_func, args=(inputs,))
        losses = self._agg_distribute_loss(losses)
        outputs = self._agg_distribute_outputs(outputs)
        return losses, outputs

    def _val_batch(self, batch):
        if self.strategy is not None:
            return self._val_batch_distribute(batch)

        inputs, input_signatures = self._get_model_inputs_tuple(batch)
        if self.cfg.run_eagerly:
            return self.val_batch(inputs)
        else:
            if self._val_batch_graph_func is None:
                self._val_batch_graph_func = self.val_batch_graph.get_concrete_function(input_signatures)
            return self._val_batch_graph_func(*inputs)

    def val_epoch(self, ds, epoch, step=0):
        loss_hist = LossHist()
        outputs = []
        miniters, verbose, is_master = self.cfg.verbose, self.cfg.verbose, True
        desc = 'Val Epoch:{}, loss:{}'
        ds_itr = tqdm(enumerate(ds), total=len(ds), disable=verbose == 0, miniters=miniters, desc=desc.format(epoch, ''))
        for i, batch in ds_itr:
            ds_itr.miniters = miniters
            losses, output = self._val_batch(batch)
            if self.cfg.predicting or self.cfg.scoring:
                output = self.process_batch_output(output, batch)
            outputs.append(output)
            val_losses = OrderedDict([('val_' + key, v.numpy()) for key, v in losses.items()])
            loss_hist.append(val_losses)
            if (i + 1) % verbose == 0:
                ds_itr.set_description(desc.format(epoch, loss_hist.avg_output()))
            if (i + 1) >= self.cfg.n_val_epoch_step:
                self.info('max %s step per epoch reached', self.cfg.n_val_epoch_step)
                break

        if self.cfg.predicting or self.cfg.scoring:
            preds = self.outputs2preds(outputs)
        else:
            preds = None

        losses = loss_hist.get_avg()
        return i + 1, preds, losses

    def validate(self, val_ds, step, epoch, best_loss, best_epoch, loss):
        val_step, preds, val_losses = self.val_epoch(val_ds, epoch, step=step)
        val_loss = val_losses['val_loss']
        logger.info('loss for epoch %s:, train: %s, val: %s', epoch, loss, val_loss)
        if self.cfg.scoring:
            s = self.score(val_ds, preds=preds)
        if val_loss < (best_loss * (1 - self.cfg.es_min_delta)):
            best_loss = val_loss
            best_epoch = epoch
            self.cfg.best_epoch = best_epoch
            if self.cfg.save_best and (epoch + 1) > self.cfg.n_init_epoch:
                if step%self.cfg.val_step == 0:
                    save_epoch = step
                else:
                    save_epoch = epoch
                self.save(epoch=save_epoch)
        should_stop = False
        if self._should_stop(best_loss, val_loss, best_epoch, epoch):
            logger.info(
                "best_loss:%s, best_epoch:%s, current_epoch:%s, val_loss:%s, es_min_delta:%s, without improvement for %s epochs, train done",
                best_loss, best_epoch, epoch, val_loss, self.cfg.es_min_delta, self.cfg.n_es_epoch)
            save_opt = False
            should_stop = True
        elif self.cfg.only_validate:
            should_stop = True
        else:
            logger.info('best val loss:%s, best epoch:%s', best_loss, best_epoch)
        return best_loss, best_epoch, should_stop

    def tffit(self, train_ds=None, val_ds=None, opt=None, **kwargs):
        logger.info('start tffit')
        if self.strategy is not None:
            if train_ds is not None:
                train_ds = self.strategy.experimental_distribute_dataset(train_ds)
            if val_ds is not None:
                val_ds = self.strategy.experimental_distribute_dataset(val_ds)
        if self._model is None:
            self.create_model()
        if opt is not None:
            self._opt = opt
        elif self._opt is None:
            self._opt = self._create_optimizer()
        best_loss = np.inf;
        best_epoch, step = -1, 0
        for epoch in range(self.cfg.epochs):
            print(os.linesep)
            loss = None
            if train_ds is not None and not self.cfg.only_validate:
                step, losses, best_loss, should_stop = self.fit_epoch(train_ds, epoch, step, val_ds=val_ds, best_loss=best_loss, best_epoch=best_epoch, **kwargs)
                loss = losses['loss']
                if should_stop:
                    break
            preds = None
            if val_ds is not None and (epoch+1)>self.cfg.n_init_epoch and not self.cfg.no_validate:
                best_loss, best_epoch, should_stop = self.validate(val_ds, step, epoch, best_loss, best_epoch, loss)
                if should_stop:
                    break
            if ((epoch + 1) % (self.cfg.n_save_epoch) == 0) and not self.cfg.save_best and (epoch+1)>self.cfg.n_init_epoch:
                if self.cfg.swa_start_epoch > 0:
                    self.swap_swa_sgd(train_ds, self._model)
                logger.info('%%%% save model to %s for epoch:%s', self.cfg.output_dir, epoch)
                self.save(global_step=step, save_opt=save_opt, epoch=epoch)
            if not self.cfg.only_validate and step % self.cfg.n_train_step == 0:
                logger.info('total train step %s done', self.cfg.n_train_step)
                break
        if isinstance(self._opt, MovingAverage) and not self.cfg.only_validate:
            self._opt.assign_average_vars(self._get_train_vars())
            logger.info('assigned ema vars')
        if self.cfg.save or self.cfg.save_best or (self.cfg.n_save_epoch < self.cfg.N_INF) or (self.cfg.n_save_step < self.cfg.N_INF):
            if not self.cfg.only_validate and not self.cfg.save_best:
                if (not (epoch+1)%self.cfg.n_save_epoch == 0) or self.cfg.save_best:
                    self.save(epoch=epoch)
                if self.cfg.n_swa > 0:
                    self.do_swa(val_ds, epoch, best_loss)
            if preds is not None and self.cfg.save_pred:
                self.save_predict(preds, '_val')

    def fit(self, train_ds, val_ds=None, **kwargs):
        if self.cfg.use_tffit:
            return self.tffit(train_ds, val_ds, **kwargs)
        else:
            return self.kfit(train_ds, val_ds, **kwargs)

    def kfit(self, train_ds, val_ds=None, callbacks=None, **kwargs):
        if callbacks is None:
            callbacks = self.get_callbacks()
        if self._model is None:
            self.create_model()
        if self.cfg.n_epoch_step == self.cfg.N_INF:
            steps_per_epoch=None
        else:
            steps_per_epoch = self.cfg.n_epoch_step
        history = self._model.fit(train_ds, validation_data=val_ds, epochs=self.cfg.epochs, verbose=self.cfg.verbose, shuffle=False,
                                  validation_steps=self.cfg.validation_steps, validation_freq=self.cfg.validation_freq,
                                  initial_epoch=self.cfg.restore_epoch,
                                  steps_per_epoch=steps_per_epoch, callbacks=callbacks, **kwargs)
        return history

    def score(self, ds, preds=None):
        raise NotImplemented
