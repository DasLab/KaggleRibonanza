import os, sys, logging
sys.path.insert(-1, '.')
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from collections import defaultdict
from ..util.pt import PTModel
from ..util.pt import util as pt_util
from transformers import AutoTokenizer
from .. import util as ut
from . import nn
from transformers import AutoConfig
from . import util
from copy import deepcopy

logger = logging.getLogger(__name__)



class ADV(object):
    def __init__(self, cfg, model, adv_params=['weight'], adv_lr=1.0, adv_eps=0.001, adv_step=1):
        self.cfg = cfg
        self.model = model
        self.adv_params = adv_params
        self.backup = {}
        self.backup_eps = {}
        self.grad_backup = {}
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.adv_step = adv_step

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if name in self.grad_backup:
                    self.grad_backup[name] = self.grad_backup[name]+param.grad.clone()
                else:
                    self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.grad_backup:
                param.grad = self.grad_backup[name]
        self.grad_backup = {}


    def attack(self):
        for name, param in self.model.named_parameters():
            if self.should_attack(name, param):
                self.attack_param(name, param)

    def save_param(self, name, param):
        self.backup[name] = param.data.clone()

    def should_attack(self, name, param):
        #return param.requires_grad and param.grad is not None and any(adv_param in name for adv_param in self.adv_params)
        return param.requires_grad and param.grad is not None and any(re.search(adv_param, name) for adv_param in self.adv_params)

    def save(self):
        for name, param in self.model.named_parameters():
            if self.should_attack(name, param):
                self.save_param(name, param)

    def restore(self,):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}

    def accumulate_grads(self, step, opt):
        if step % self.cfg.accumulated_batch_size != 0:
            self.backup_grad()
            opt.zero_grad()
        elif self.cfg.accumulated_batch_size > 1:
            self.backup_grad()
            self.restore_grad()

    def fit_batch(self, fit_func, batch, step, phase, model, opt, lr_scheduler):
        outputs, losses = fit_func(batch, step=step, phase=phase, model=model, opt=opt, lr_scheduler=lr_scheduler, step_opt=False)
        self.save()
        self.attack()
        fit_func(batch, step=step, phase=phase, model=model, opt=opt, lr_scheduler=lr_scheduler, step_opt=False)
        self.restore()

        self.accumulate_grads(step, opt)

        return outputs, losses




class AWP(ADV):
    def should_attack(self, name, param):
        return param.requires_grad and param.grad is not None
    def attack_param(self, name, param):
        norm1 = torch.norm(param.grad)
        norm2 = torch.norm(param.data.detach())
        if norm1 != 0 and not torch.isnan(norm1):
            r_at = self.adv_lr * param.grad / (norm1 + 1e-6) * (norm2 + 1e-6)
            param.data.add_(r_at)
            param.data = torch.min(torch.max(param.data, self.backup_eps[name][0]), self.backup_eps[name][1])

    def save_param(self, name, param):
        self.backup[name] = param.data.clone()
        grad_eps = self.adv_eps * param.abs().detach()
        self.backup_eps[name] = (
            self.backup[name] - grad_eps,
            self.backup[name] + grad_eps,
        )

    def fit_batch(self, fit_func, batch, step, phase, model, opt, lr_scheduler):
        outputs, losses = fit_func(batch, step, phase, model, opt, lr_scheduler, step_opt=False)
        self.save()
        for i in range(self.adv_step):
            self.attack()
            opt.zero_grad()
            fit_func(batch, step, phase, model, opt, lr_scheduler, step_opt=False)
        self.restore()
        self.accumulate_grads(step, opt)
        return outputs, losses


class FGM(ADV):
    def attack_param(self, name, param):
        norm = torch.norm(param.grad)  # 默认为2范数
        if norm != 0:
            r_at = self.adv_lr * param.grad / norm
            param.data.add_(r_at)


class PGD():
    def __init__(self, model, alpha=0.01, emb_name=['emb']):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}
        self.alpha = alpha
        self.emb_name = emb_name

    def attack(self, epsilon=1., is_first_attack=False):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if any(emb_name in name for emb_name in self.emb_name):
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name='emb'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.grad_backup:
                param.grad = self.grad_backup[name]
        self.grad_backup = {}


class Model(PTModel):
    cfg = PTModel.cfg.copy()
    cfg.hfd = None
    cfg.ds_cls = "Dataset"
    cfg.pretrain_ds_cls = "PretrainDataset"
    cfg.val_ds_cls = "IterDataset"
    cfg.batch_keys = ['text_id', 'logits', 'label']
    cfg.backbone = 'transformer'
    cfg.verbose = 128
    cfg.to_list = True
    cfg.do_concat = False
    cfg.n_keep_ckpt = 1
    cfg.opt = 'torch.optim.AdamW'
    cfg.opt_paras = {'weight_decay':1e-4}
    cfg.epochs = 100
    cfg.lr_scheduler = 'ld'
    #cfg.lr_scheduler = "torch.optim.lr_scheduler.CosineAnnealingLR"
    cfg.lr_scheduler_paras = {}
    cfg.lr = 1e-4
    cfg.n_lr_warmup_step = 1000
    cfg.activation = 'gelu'
    cfg.initializer_range = 0.02
    cfg.max_seq_len = 4
    cfg.nan_grad = None
    cfg.adv_params = ["word_embeddings"]
    cfg.stratify = False
    cfg.use_inst = False
    cfg.enc_dim = 8
    cfg.n_layer = 2
    cfg.d_model = 8
    cfg.n_head = 2
    cfg.groupfy = False
    cfg.distill = False
    cfg.distill_model = None
    cfg.distill_pred = False
    cfg.distill_factor = 2
    cfg.use_peft = False
    cfg.lora_rank = 32
    cfg.lora_alpha = 16


    ##
    cfg.n_label = 2

    def __init__(self, name, cfg={}):
        super(Model, self).__init__(name, cfg)
        self.config = None
        self.processor = None
        if self.cfg.restore:
            if not self.cfg.use_tfm:
                self.restore_config()
            self.tokenizer = AutoTokenizer.from_pretrained(self.gen_fname("tokenizer", data_dir=self.cfg.model_dir,  name=self.cfg.restore_model), trust_remote_code=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.backbone, trust_remote_code=True)
        if self.tokenizer.sep_token_id is None:
            self.tokenizer.sep_token_id = self.tokenizer.eos_token_id
        if self.tokenizer.cls_token_id is None:
            self.tokenizer.cls_token_id = self.tokenizer.bos_token_id
        if self.tokenizer.mask_token_id is None:
            self.tokenizer.mask_token_id = self.tokenizer.unk_token_id

    def restore_config(self):
        self.config = AutoConfig.from_pretrained(self.gen_fname('config', data_dir=self.cfg.model_dir, name=self.cfg.restore_model))


    def restore(self, restore_epoch=None, epoch=None, model_dir=None, ckpt=None, no_opt=True):
        if self.cfg.use_peft:
            self._model.backbone = self._model.backbone.base_model
        super().restore(restore_epoch=restore_epoch, epoch=epoch, model_dir=model_dir, ckpt=ckpt, no_opt=no_opt)

    def save(self, global_step=None, save_path=None, epoch=None, save_opt=False, **kwargs):
        bak = self._model.backbone
        if self.cfg.use_peft:
            with ut.timer('merge lora'):
                fpath = self.gen_fname(f'peft', data_dir=self.cfg.model_dir)
                self._model.backbone.save_pretrained(fpath)
                from peft import PeftModel
                self._model.backbone = PeftModel.from_pretrained(self._model.backbone.base_model, fpath).merge_and_unload()
        super().save(global_step, save_path, epoch, save_opt, **kwargs)
        self._model.backbone = bak
        if hasattr(self._model, 'config') and self._model.config is not None:
            fpath = self.gen_fname('config', data_dir=self.cfg.model_dir)
            self._model.config.save_pretrained(fpath)
        fpath = self.gen_fname('tokenizer', data_dir=self.cfg.model_dir)
        if self.tokenizer is not None:
            os.makedirs(fpath, exist_ok=True)
            self.tokenizer.save_pretrained(fpath)


    def create_core_model(self, **kwargs):
        module = getattr(nn, self.cfg.pt_model)(self.cfg, self.config)
        logger.info('num of parameters:%s', nn.get_num_of_paras(module))
        return module

    def create_model(self, opt=None, pt_model=None, rank=None, no_opt=False, **kwargs):
        super().create_model(opt, pt_model, rank, no_opt, **kwargs)
        if self.cfg.adv_lr>0:
            self.adv = globals()[self.cfg.adv_name](self.cfg, self._model, self.cfg.adv_params, self.cfg.adv_lr/self.cfg.accumulated_batch_size, adv_eps=self.cfg.adv_eps, adv_step=self.cfg.adv_step)
            logger.info('use adv %s', self.cfg.adv_name)
        if self.cfg.compile_backbone:
            self._model = torch.compile(self._model.backbone)
        if self.cfg.distill:
            distill_model = deepcopy(self.distill_model)
            layers = self.distill_model.backbone.layers
            distill_factor = len(layers)//len(self._model.backbone.layers)
            self._model.backbone.layers = torch.nn.ModuleList([layers[i] for i in range(0, len(layers), distill_factor)])
            self._model.proj = deepcopy(distill_model.proj)
            self._model.decoder = distill_model.decoder
            nn.requires_grad(self._model.decoder, False)
            nn.requires_grad(self.distill_model, False)
            self.distill_model = self.distill_model.eval()
            if self.cfg.ema is not None and self.cfg.ema > 0:
                self._ema_model = self.create_ema(self._model, decay=self.cfg.ema)

    def _get_onnx_input(self, batch):
        inputs, input_names = [], []
        for k in self.cfg.onnx_input_names:
            if k in batch and batch[k] is not None:
                inputs.append(batch[k])
                input_names.append(k)
        return inputs, input_names

    def to_onnx(self, batchs):
        import onnx
        import onnxruntime as ort
        print(onnx.__version__)
        self.gen_use_cols()
        fpath = self.gen_fname('model.onnx')
        self._model.eval()
        rs = np.random.RandomState(9527)
        #batchs = [{'inputs': batch['x'][0]} for batch in batchs]
        #batchs = [{'inputs': torch.tensor(rs.rand(10, len(util.use_cols.flatten())).astype(np.float32))} for batch in batchs]
        inputs, input_names = self._get_onnx_input(batchs[0])
        #inputs, input_names = [batchs[0]['inputs']], ['inputs']
        dynamic_axes = {input_name: {0: 'batch_size'} for input_name in input_names}
        for k in ['inputs']:
            if k in dynamic_axes:
                dynamic_axes[k][1]='seq_len'
        dynamic_axes['outputs'] = {0: 'batch_size', 1: 'seq_len'}
        if torch.cuda.is_available():
            torch_inputs = [x.cuda() for x in inputs]
        else:
            torch_inputs = inputs
        torch.onnx.export(self._model,  # model being run
                          tuple(torch_inputs),  # model input (or a tuple for multiple inputs)
                          fpath,  # where to save the model (can be a file or file-like object)
                          export_params=True,  # store the trained parameter weights inside the model file
                          opset_version=12,  # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names=input_names,  # the model's input names
                          output_names=['outputs'],  # the model's output names
                          dynamic_axes=dynamic_axes,
                          )

        #model_onnx = onnx.load(fpath)  # load onnx model
        #onnx.checker.check_model(model_onnx)  # check onnx model
        #os.environ["OMP_NUM_THREADS"] = "1"
        opts = ort.SessionOptions()
        #opts.inter_op_num_threads = 1
        #opts.intra_op_num_threads = 1
        #opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session = ort.InferenceSession(fpath, sess_options=opts)
        for batch in batchs:
            inputs = [batch['inputs']]
            rst1 = session.run(None, dict(zip(input_names, [x.numpy() for x in inputs])))[0]
            if torch.cuda.is_available():
                pt_util.batch2device(batch, self._device)
            with torch.no_grad():
                rst2 = self._model(**batch)['logits']
            diff = np.abs(rst1-rst2.cpu().numpy())
            print('max diff to onnx:', diff.max())
            #assert diff.max()<1e-4, 'diff error'

        logger.info('onnx exported:%s', fpath)

    def onnx2tf(self, batchs):
        import tensorflow as tf
        import onnx_tf
        import onnx
        fpath = self.gen_fname('model.onnx')
        model_onnx = onnx.load(fpath)  # load onnx model
        tf_rep = onnx_tf.backend.prepare(model_onnx)
        fpath = self.gen_fname('model.tf')
        tf_rep.export_graph(fpath)
        model = tf.saved_model.load(fpath)
        for batch in batchs[:2]:
            pred1 = model(inputs=batch['inputs'])['outputs']
            if torch.cuda.is_available():
                batch['inputs'] = batch['inputs'].cuda()
            with torch.no_grad():
                pred2 = self._model(batch['inputs'])['logits'].cpu().numpy()
            diff = np.abs(pred1 - pred2)
            print('max diff load tf', diff.max())
        return model

    def pt2tf(self, input_dim, batchs):
        import nobuco
        from nobuco import ChannelOrder, ChannelOrderingStrategy
        from nobuco.layers.weight import WeightLayer
        self._model.eval()
        model = nobuco.pytorch_to_keras(
            self._model,
            args=[batchs[0]['inputs']], input_shapes={batchs[0]['inputs']: (None, None, input_dim)},
            inputs_channel_order=ChannelOrder.TENSORFLOW,
            outputs_channel_order=ChannelOrder.TENSORFLOW
        )
        return model

    def fit_adv_batch(self, batch, step, phase, model, opt, lr_scheduler):
        if opt is None:
            opt = self._opt
        if lr_scheduler is None:
            lr_scheduler = self._lr_scheduler
        outputs, losses = self.adv.fit_batch(super().fit_batch, batch, step, phase, model, opt, lr_scheduler)
        if step % self.cfg.accumulated_batch_size == 0:
            self._step_opt(opt=opt, lr_scheduler=lr_scheduler)
        return outputs, losses

    def fit_awp_batch(self, batch, step, phase, model, opt, lr_scheduler):
        if opt is None:
            opt = self._opt
        if lr_scheduler is None:
            lr_scheduler = self._lr_scheduler
        outputs, losses = super().fit_batch(batch, step, phase, model, opt, lr_scheduler, step_opt=False)
        if step % self.cfg.accumulated_batch_size == 0:
            awp = self.awp
            awp.save()
            # 对抗训练
            for t in range(self.cfg.adv_step):
                awp.attack()
                opt.zero_grad()
                super().fit_batch(batch, step, phase, model, opt, lr_scheduler, step_opt=False)
            awp.restore()

            self._step_opt(opt=opt, lr_scheduler=lr_scheduler)
        return outputs, losses

    def fit_pgd_batch(self, batch, step, phase, model, opt, lr_scheduler):
        if opt is None:
            opt = self._opt
        if lr_scheduler is None:
            lr_scheduler = self._lr_scheduler
        outputs, losses = super().fit_batch(batch, step, phase, model, opt, lr_scheduler, step_opt=False)
        if step % self.cfg.accumulated_batch_size == 0:
            K = self.cfg.pgd_k
            pgd = self.pgd
            pgd.backup_grad()  # 保存正常的grad
            # 对抗训练
            for t in range(K):
                pgd.attack(is_first_attack=(t == 0))  # 在embedding上添加对抗扰动, first attack时备份param.data
                if t != K - 1:
                    opt.zero_grad()
                else:
                    pgd.restore_grad()  # 恢复正常的grad

                super().fit_batch(batch, step, phase, model, opt, lr_scheduler, step_opt=False)
            pgd.restore()  # 恢复embedding参数

            self._step_opt(opt=opt, lr_scheduler=lr_scheduler)
        return outputs, losses


    def fit_batch(self, batch, step=None, phase='train', model=None, opt=None, lr_scheduler=None, step_opt=True, epoch=None):
        restore_epoch = self._restored_epoch if self._restored_epoch is not None else -1
        if self.cfg.distill:
            batch['distill'] = True
            outputs, _ = super().fit_batch(batch, step, 'test', self.distill_model, opt, lr_scheduler, step_opt, epoch)
            batch['teacher_atts'], batch['teacher_reps'], batch['teacher_logits'] = outputs['atts'], outputs['hhs'], outputs['logits']
        if self.cfg.adv_lr>0 and phase=='train' and (epoch-restore_epoch-1)>=self.cfg.adv_start_epoch:
            outputs, losses = self.fit_adv_batch(batch, step, phase, model, opt, lr_scheduler)
        else:
            outputs, losses = super().fit_batch(batch, step, phase, model, opt, lr_scheduler, step_opt, epoch)
        return outputs, losses


    def predict_onnx(self, ds, data_type, desc, **kwargs):
        ds = tqdm(enumerate(ds), total=len(ds), miniters=self.cfg.verbose, desc=f"{desc} for {data_type}")
        preds = defaultdict(list)
        for i, batch in ds:
            ds.miniters = self.cfg.verbose
            inputs, input_names = self._get_onnx_input(batch)
            logits = self.onnx_session.run(None, dict(zip(input_names, [x.cpu().numpy() for x in inputs])))[0]
            preds['text_id'].extend(batch['text_id'])
            preds['logits'].append(logits)

            if i==self.cfg.n_val_epoch_step:
                break
        preds['logits'] = list(np.concatenate(preds['logits'], axis=0))
        return preds

    def predict(self, ds, data_type='val', pt_model=None, phase='test', desc="predict", **kwargs):
        if self.cfg.use_onnx:
            return self.predict_onnx(ds, data_type, desc, **kwargs)
        else:
            return super().predict(ds, data_type, pt_model, phase, desc, use_multiple_gpu=self.cfg.use_multiple_gpu, **kwargs)

    def predict_rst(self, ds, data_type='val', preds=None):
        if preds is None:
            preds = self.predict(ds, data_type=data_type)
        if isinstance(preds, dict):
            preds = pd.DataFrame(preds)
        if self.cfg.use_scaler:
            preds['logits'] = list(ds.dataset.scaler.inverse_transform(np.stack(preds.logits.values, axis=0)))
        preds['text_id'] = preds.ID.map(ds.dataset.idmap)
        return preds

    def score(self, ds, data_type='val', preds=None):
        preds = self.predict_rst(ds, data_type, preds)
        s = util.score(preds)
        logger.info('score is: %s', s)
        return s


class SRF(Model):
    cfg = Model.cfg.copy()
    cfg.temp = 1
    cfg.use_mfe = False
    cfg.use_lt = False
    cfg.use_bi = False
    cfg.use_pos = False
    cfg.use_tfm = False
    cfg.no_act = False
    cfg.bpp_topk = 2
    cfg.post_norm = False
    cfg.no_sos = False
    cfg.use_rope = False
    cfg.use_ext_token = False
    cfg.gnn_layer = 2
    cfg.gnn_dim = 8
    cfg.backbone = "../data/microsoft/deberta-v3-small"
    cfg.use_bpp = False
    cfg.no_emb_bpp = False
    cfg.with_gnn_bpp = False
    cfg.use_learn = False
    cfg.pkg = 'eternafold'
    cfg.sl_window = 64
    def __init__(self, name, cfg={}):
        super().__init__(name, cfg)
        self.cfg.batch_keys = cfg.get("batch_keys", ['sequence_id', 'label', 'logits', 'label_mask', 'input_len', 'id_min', 'id_max'])
        self.cfg.pretrain_ds_cls = cfg.get("pt_ds", "PretrainDataset")
        self.cfg.ds_cls = cfg.get("ds_cls", "Dataset")
        self.cfg.val_ds_cls = cfg.get("val_ds_cls", "IterDataset")
        self.cfg.pt_model = cfg.get("pt_model", "SRF")
        self.cfg.onnx_input_names = cfg.get("onnx_input_names", ['inputs'])
        self.cfg.to_list = True
        self.cfg.do_concat = False

        if self.cfg.use_ext_token:
            self.cfg.ext_token_mapping = {
                'G' + '(': 'A',
                'G' + '.': 'B',
                'G' + ')': 'C',
                'A' + '(': 'D',
                'A' + '.': 'E',
                'A' + ')': 'F',
                'C' + '(': 'G',
                'C' + '.': 'H',
                'C' + ')': 'I',
                'U' + '(': 'J',
                'U' + '.': 'K',
                'U' + ')': 'L'
            }
            self.cfg.oids = sorted(
                [self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.mask_token_id] + self.tokenizer.convert_tokens_to_ids(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']) + [
                    self.tokenizer.sep_token_id])
        else:
            if 'roformer' in self.cfg.backbone:
                self.cfg.oids = sorted(
                    [self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.mask_token_id] + self.tokenizer.convert_tokens_to_ids(['a', 'g', 'c', 'u']) + [
                        self.tokenizer.sep_token_id])
            else:
                self.cfg.oids = sorted([self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.mask_token_id] + self.tokenizer.convert_tokens_to_ids(['A', 'G', 'C', 'U']) + [self.tokenizer.sep_token_id])

    def predict_rst(self, ds, data_type='val', preds=None):
        if preds is None:
            preds = self.predict(ds, data_type=data_type)
        logits, input_lens, labels, label_masks = preds['logits'], preds['input_len'], preds['label'], preds['label_mask']
        ids = []
        for i, input_len in enumerate(input_lens):
            if not self.cfg.no_sos:
                logits[i] = logits[i][1:input_len-1]
                label_masks[i] = label_masks[i][1:input_len-1]
                labels[i] = labels[i][1:input_len-1]
            if 'id_min' in preds:
                ids.append(np.arange(preds['id_min'][i], preds['id_max'][i]+1))
        preds['ids'] = ids

        return preds

    def val_epoch(self, ds, epoch, model=None, eval=True, step=0):
        if self.cfg.val_by_score:
            s = self.score(ds, data_type='val')
            return 0, None, {'val_loss': 1 - s}
        else:
            return super().val_epoch(ds, epoch, model, eval, step)

    def score(self, ds, data_type='val', preds=None):
        if ds is not None:
            preds = self.predict_rst(ds, data_type, preds)
        s = util.score(preds)
        return s

class Pretrain(SRF):
    cfg = SRF.cfg.copy()
    def __init__(self, name, cfg={}):
        super().__init__(name, cfg)
        self.cfg.ds_cls = cfg.get("ds_cls", "PretrainDataset")
        self.cfg.val_ds_cls = cfg.get("val_ds_cls", "PretrainDataset")
        self.cfg.pt_model = cfg.get("pt_model", "Pretrain")

class SRF1P(SRF):
    cfg = SRF.cfg.copy()
    cfg.n_label = 1
    def __init__(self, name, cfg={}):
        super().__init__(name, cfg)
        self.cfg.batch_keys = cfg.get("batch_keys", ['sequence_id', 'label', 'logits', 'label_mask', 'input_len', 'id_min', 'id_max', 'experiment_type'])
        self.cfg.pretrain_ds_cls = cfg.get("pt_ds", "PretrainDataset")
        self.cfg.ds_cls = cfg.get("ds_cls", "Dataset1P")
        self.cfg.val_ds_cls = cfg.get("val_ds_cls", "IterDataset1P")
        self.cfg.pt_model = cfg.get("pt_model", "SRF1P")
        self.cfg.onnx_input_names = cfg.get("onnx_input_names", ['inputs'])
        self.cfg.to_list = True
        self.cfg.do_concat = False

    def predict_rst(self, ds, data_type='val', preds=None):
        if preds is None:
            preds = self.predict(ds, data_type=data_type)
        sids, logits, input_lens, labels, label_masks, exps = preds['sequence_id'], preds['logits'], preds['input_len'], preds['label'], preds['label_mask'], preds['experiment_type']
        id_mins, id_maxs = preds['id_min'], preds['id_max']
        sid2inds = dict()
        for k in preds:
            preds[k] = []
        for sid, id_min, id_max, logit, input_len, label, label_mask, exp in zip(sids, id_mins, id_maxs, logits, input_lens, labels, label_masks, exps):
            if sid not in sid2inds:
                preds['logits'].append(logit)
                preds['input_len'].append(input_len)
                preds['label'].append(label)
                preds['label_mask'].append(label_mask)
                preds['id_min'].append(id_min)
                preds['id_max'].append(id_max)
                sid2inds[sid] = len(sid2inds)
            else:
                ind = sid2inds[sid]
                if exp=='2A3_MaP':
                    preds['logits'][ind] = np.concatenate([logit, preds['logits'][ind]], axis=-1)
                    preds['label'][ind] = np.concatenate([label, preds['label'][ind]], axis=-1)
                    preds['label_mask'][ind] = np.concatenate([label_mask, preds['label_mask'][ind]], axis=-1)
                else:
                    preds['logits'][ind] = np.concatenate([preds['logits'][ind], logit], axis=-1)
                    preds['label'][ind] = np.concatenate([preds['label'][ind], label], axis=-1)
                    preds['label_mask'][ind] = np.concatenate([preds['label_mask'][ind], label_mask], axis=-1)
        ids = []
        for i, input_len in enumerate(preds['input_len']):
            preds['logits'][i] = preds['logits'][i][1:input_len-1]
            preds['label_mask'][i] = preds['label_mask'][i][1:input_len-1]
            preds['label'][i] = preds['label'][i][1:input_len-1]
            if 'id_min' in preds:
                ids.append(np.arange(preds['id_min'][i], preds['id_max'][i]+1))
        preds['ids'] = ids

        return preds

class SRFRNN(SRF):
    cfg = SRF.cfg.copy()
    cfg.rnn = 'lstm'
    cfg.rnn_layer = 1
    cfg.rnn_dim = 8
    def __init__(self, name, cfg={}):
        super().__init__(name, cfg)
        self.cfg.batch_keys = cfg.get("batch_keys", ['sequence_id', 'label', 'logits', 'label_mask', 'input_len', 'id_min', 'id_max'])
        self.cfg.pretrain_ds_cls = cfg.get("pt_ds", "PretrainDataset")
        self.cfg.ds_cls = cfg.get("ds_cls", "Dataset")
        self.cfg.val_ds_cls = cfg.get("val_ds_cls", "IterDataset")
        self.cfg.pt_model = cfg.get("pt_model", "SRFRNN")
        self.cfg.onnx_input_names = cfg.get("onnx_input_names", ['inputs'])
        self.cfg.to_list = True
        self.cfg.do_concat = False


class SRFBPP(SRF):
    cfg = SRF.cfg.copy()
    def __init__(self, name, cfg={}):
        super().__init__(name, cfg)
        self.cfg.batch_keys = cfg.get("batch_keys", ['sequence_id', 'label', 'logits', 'label_mask', 'input_len', 'id_min', 'id_max'])
        self.cfg.pretrain_ds_cls = cfg.get("pt_ds", "PretrainDataset")
        self.cfg.ds_cls = cfg.get("ds_cls", "BPPDataset")
        self.cfg.val_ds_cls = cfg.get("val_ds_cls", "BPPIterDataset")
        self.cfg.pt_model = cfg.get("pt_model", "SRFBPP")
        self.cfg.onnx_input_names = cfg.get("onnx_input_names", ['inputs'])
        self.cfg.to_list = True
        self.cfg.do_concat = False

class SRFBPP1P(SRFBPP):
    cfg = SRFBPP.cfg.copy()
    def __init__(self, name, cfg={}):
        super().__init__(name, cfg)
        self.cfg.batch_keys = cfg.get("batch_keys", ['sequence_id', 'label', 'logits', 'label_mask', 'input_len', 'id_min', 'id_max'])
        self.cfg.pretrain_ds_cls = cfg.get("pt_ds", "PretrainDataset")
        self.cfg.ds_cls = cfg.get("ds_cls", "BPPDataset")
        self.cfg.val_ds_cls = cfg.get("val_ds_cls", "BPPIterDataset")
        self.cfg.pt_model = cfg.get("pt_model", "SRFBPP1P")
        self.cfg.onnx_input_names = cfg.get("onnx_input_names", ['inputs'])
        self.cfg.to_list = True
        self.cfg.do_concat = False

class SRFBPPGNN(SRFBPP):
    cfg = SRFBPP.cfg.copy()
    cfg.not_use_mfe = False

    def __init__(self, name, cfg={}):
        super().__init__(name, cfg)
        self.cfg.batch_keys = cfg.get("batch_keys", ['sequence_id', 'label', 'logits', 'label_mask', 'input_len', 'id_min', 'id_max'])
        self.cfg.pretrain_ds_cls = cfg.get("pt_ds", "PretrainDataset")
        self.cfg.ds_cls = cfg.get("ds_cls", "SRFBPPGNNDataset")
        self.cfg.val_ds_cls = cfg.get("val_ds_cls", "SRFBPPGNNIterDataset")
        self.cfg.pt_model = cfg.get("pt_model", "SRFBPPGNN")
        self.cfg.onnx_input_names = cfg.get("onnx_input_names", ['inputs'])
        self.cfg.to_list = True
        self.cfg.do_concat = False
        self.cfg.n_adj = 5
        self.cfg.n_adj += (len(self.cfg.pkg.split())-1)*2
        self.cfg.node_dim = 3
        self.cfg.node_dim += (len(self.cfg.pkg.split())-1)*3
        if self.cfg.use_pos:
            self.cfg.n_adj += 3
        if self.cfg.use_learn:
            self.cfg.n_adj += 1

class SRFGNN(SRF):
    cfg = SRF.cfg.copy()
    cfg.gnn_node_layer = 1
    cfg.gnn_dim = 8
    cfg.gnn_layer = 2
    cfg.use_lt = False

    def __init__(self, name, cfg={}):
        super().__init__(name, cfg)
        self.cfg.batch_keys = cfg.get("batch_keys", ['sequence_id', 'label', 'logits', 'label_mask', 'input_len', 'id_min', 'id_max'])
        self.cfg.pretrain_ds_cls = cfg.get("pt_ds", "PretrainDataset")
        self.cfg.ds_cls = cfg.get("ds_cls", "SRFGNNDataset")
        self.cfg.val_ds_cls = cfg.get("val_ds_cls", "SRFGNNIterDataset")
        self.cfg.pt_model = cfg.get("pt_model", "SRFGNN")
        self.cfg.onnx_input_names = cfg.get("onnx_input_names", ['inputs'])
        self.cfg.to_list = True
        self.cfg.do_concat = False
        self.cfg.n_adj = 5
        self.cfg.node_dim = 7
        if self.cfg.use_pos:
            self.cfg.n_adj += 3
        if self.cfg.use_lt:
            self.cfg.node_dim += 7
    def predict_rst(self, ds, data_type='val', preds=None):
        if preds is None:
            preds = self.predict(ds, data_type=data_type)
        logits, input_lens, labels, label_masks = preds['logits'], preds['input_len'], preds['label'], preds['label_mask']
        ids = []
        for i, input_len in enumerate(input_lens):
            if 'id_min' in preds:
                ids.append(np.arange(preds['id_min'][i], preds['id_max'][i]+1))
        preds['ids'] = ids

        return preds

    def restore_config(self):
        pass

class PretrainSRFBPPGNN(SRFBPPGNN):
    cfg = SRFBPPGNN.cfg.copy()
    def __init__(self, name, cfg={}):
        super().__init__(name, cfg)
        self.cfg.batch_keys = cfg.get("batch_keys", ['sequence_id', 'label', 'logits', 'label_mask', 'input_len', 'id_min', 'id_max'])
        self.cfg.pretrain_ds_cls = cfg.get("pt_ds", "PretrainDataset")
        self.cfg.ds_cls = cfg.get("ds_cls", "PretrainSRFBPPGNNDataset")
        self.cfg.val_ds_cls = cfg.get("val_ds_cls", "PretrainSRFBPPGNNDataset")
        self.cfg.pt_model = cfg.get("pt_model", "PretrainSRFBPPGNN")
        self.cfg.onnx_input_names = cfg.get("onnx_input_names", ['inputs'])
        self.cfg.to_list = True
        self.cfg.do_concat = False

class PretrainAESRFGNN(SRFGNN):
    cfg = SRFGNN.cfg.copy()
    def __init__(self, name, cfg={}):
        super().__init__(name, cfg)
        self.cfg.batch_keys = cfg.get("batch_keys", ['sequence_id', 'label', 'logits', 'label_mask', 'input_len', 'id_min', 'id_max'])
        self.cfg.pretrain_ds_cls = cfg.get("pt_ds", "PretrainDataset")
        self.cfg.ds_cls = cfg.get("ds_cls", "PretrainSRFGNNDataset")
        self.cfg.val_ds_cls = cfg.get("val_ds_cls", "PretrainSRFGNNDataset")
        self.cfg.pt_model = cfg.get("pt_model", "PretrainAESRFGNN")
        self.cfg.onnx_input_names = cfg.get("onnx_input_names", ['inputs'])
        self.cfg.to_list = True
        self.cfg.do_concat = False


class PretrainSRFGNN(SRFGNN):
    cfg = SRFGNN.cfg.copy()
    def __init__(self, name, cfg={}):
        super().__init__(name, cfg)
        self.cfg.batch_keys = cfg.get("batch_keys", ['sequence_id', 'label', 'logits', 'label_mask', 'input_len', 'id_min', 'id_max'])
        self.cfg.pretrain_ds_cls = cfg.get("pt_ds", "PretrainDataset")
        self.cfg.ds_cls = cfg.get("ds_cls", "PretrainSRFGNNDataset")
        self.cfg.val_ds_cls = cfg.get("val_ds_cls", "PretrainSRFGNNDataset")
        self.cfg.pt_model = cfg.get("pt_model", "PretrainSRFGNN")
        self.cfg.onnx_input_names = cfg.get("onnx_input_names", ['inputs'])
        self.cfg.to_list = True
        self.cfg.do_concat = False

class PretrainBPP(SRF):
    cfg = SRF.cfg.copy()
    def __init__(self, name, cfg={}):
        super().__init__(name, cfg)
        self.cfg.batch_keys = cfg.get("batch_keys", ['sequence_id', 'label', 'logits', 'label_mask', 'input_len', 'id_min', 'id_max'])
        self.cfg.pretrain_ds_cls = cfg.get("pt_ds", "PretrainDataset")
        self.cfg.ds_cls = cfg.get("ds_cls", "BPPDataset")
        self.cfg.val_ds_cls = cfg.get("val_ds_cls", "BPPDataset")
        self.cfg.pt_model = cfg.get("pt_model", "PretrainBPP")
        self.cfg.onnx_input_names = cfg.get("onnx_input_names", ['inputs'])
        self.cfg.to_list = True
        self.cfg.do_concat = False

class PretrainBPP1P(PretrainBPP):
    cfg = PretrainBPP.cfg.copy()
    def __init__(self, name, cfg={}):
        super().__init__(name, cfg)
        self.cfg.batch_keys = cfg.get("batch_keys", ['sequence_id', 'label', 'logits', 'label_mask', 'input_len', 'id_min', 'id_max'])
        self.cfg.pretrain_ds_cls = cfg.get("pt_ds", "PretrainBPPDataset")
        self.cfg.ds_cls = cfg.get("ds_cls", "PretrainBPPDataset")
        self.cfg.val_ds_cls = cfg.get("val_ds_cls", "PretrainBPPDataset")
        self.cfg.pt_model = cfg.get("pt_model", "PretrainBPP1P")
        self.cfg.onnx_input_names = cfg.get("onnx_input_names", ['inputs'])
        self.cfg.to_list = True
        self.cfg.do_concat = False


class PretrainMFE(PretrainBPP):
    cfg = PretrainBPP.cfg.copy()
    def __init__(self, name, cfg={}):
        super().__init__(name, cfg)
        self.cfg.batch_keys = cfg.get("batch_keys", ['sequence_id', 'label', 'logits', 'label_mask', 'input_len', 'id_min', 'id_max'])
        self.cfg.pretrain_ds_cls = cfg.get("pt_ds", "PretrainMFEDataset")
        self.cfg.ds_cls = cfg.get("ds_cls", "PretrainMFEDataset")
        self.cfg.val_ds_cls = cfg.get("val_ds_cls", "PretrainMFEDataset")
        self.cfg.pt_model = cfg.get("pt_model", "PretrainMFE")
        self.cfg.onnx_input_names = cfg.get("onnx_input_names", ['inputs'])
        self.cfg.to_list = True
        self.cfg.do_concat = False
