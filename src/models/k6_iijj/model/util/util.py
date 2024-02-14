import os, sys, logging, json, time, re
import hashlib
import pickle
import numpy as np
import psutil
import math
import inspect
import argparse
from collections import OrderedDict
from contextlib import contextmanager
from collections import abc
import subprocess


logger = logging.getLogger(__name__)


def wait_gpu(seconds=300):
    with timer(f'wait gpu(availabe for at least {seconds} seconds)'):
        cmd = ''' nvidia-smi --query-compute-apps=name --format=csv '''
        num, available = 10, False
        while True:
            for i in range(num):
                output = subprocess.check_output(cmd, shell=True)
                if len(output.split())>1:
                    available = False
                    break
                else:
                    available = True
                    time.sleep(seconds/num)
            if available:
                break
            else:
                time.sleep(5)

def sleep(n, f=10):
    start = time.time()
    i = 0
    while i<n:
        left = n-i
        print("\r", "senconds left {}".format(left), end="", flush=True)
        time.sleep(min(left, f))
        i = int(time.time()-start)


class NestedObject(argparse.Namespace):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
               setattr(self, a, [NestedObject(x) if isinstance(x, dict) else x for x in b])
            else:
               setattr(self, a, NestedObject(b) if isinstance(b, dict) else b)


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
       logger.info('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def update_dict(d, u):
    for k, v in u.items():
        if isinstance(v, abc.Mapping):
            d[k] = update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def get_default_args(func):
    signature = inspect.signature(func)
    return { k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty}


def fill_kwargs(func, args, kwargs):
    default_args = get_default_args(func)
    default_args.update(kwargs)
    arg_names = [k for k, v in inspect.signature(func).parameters.items() if k!='self']
    for k, v in zip(arg_names, args):
        default_args[k] = v
    return default_args


def dynamic_import(kls):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m


def run_cmd(cmd, echo=True, asserting=True):
    if echo:
        logger.info('cmd is %s', cmd)
    rst = os.system(cmd)
    if asserting:
        assert rst == 0

def df2dict(df):
    data = {}
    for k, v in df.to_dict('list').items():
        data[k] = np.array(v)
    return data


def onehot(labels, dim):
    """
    :param labels: one dimension array of int
    :param dim: one hot dimension, i.e. total num of class
    :return:
    """
    num = len(labels)
    onehot = np.zeros((num, dim), np.int32)
    onehot[np.arange(num), labels] = 1
    return onehot


def get_default_num_job():
    from multiprocessing import cpu_count
    return max(1, cpu_count()-1)


def get_hash(fpath, hasher=None):
    if hasher is None:
        hasher = hashlib.md5()
    with open(fpath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_dump(fpath):
    with open(fpath, 'rb') as f:
        data = pickle.load(f)
    return data


def dump(data, fpath, protocol=2):
    fdir = os.path.dirname(fpath)
    if not os.path.exists(fdir):
        os.makedirs(fdir)
    with open(fpath, 'wb') as f:
        pickle.dump(data, f)


def load_json(fpath):
    with open(fpath) as f:
        dictionary = json.load(f)
    return dictionary


def load_json_lines(fpath, num=1e16, exclude_keys=None, post_process=None):
    if exclude_keys is None:
        exclude_keys = []
    data = []
    with open(fpath) as f:
        for l in f:
            dic = json.loads(l)
            for k in exclude_keys:
                if k in dic:
                    _ = dic.pop(k)
            if post_process is not None:
                dic = post_process(dic)
            data.append(dic)
            if len(data)>=num:
                break
    return data


def dump_json(dictionary, fpath, ensure_ascii=False):
    with open(fpath, 'w') as f:
        json.dump(dictionary, f, ensure_ascii=ensure_ascii)


def dump_json_lines(dicts, fpath, ensure_ascii=False):
    with open(fpath, 'w', encoding='utf8') as f:
        for d in dicts:
            json.dump(d, f, ensure_ascii=ensure_ascii)
            f.write(os.linesep)


def pad_sequence(seqs, pad=0, seqs_len=None, dtype=np.int32, max_seq_len=None):
    seqs = np.array(seqs)
    shape = seqs.shape
    if seqs.dtype == object:
        flatten_seqs = seqs.flatten()
        seqs_len = np.reshape([len(x) for x in flatten_seqs], shape)
        seqs = np.concatenate(flatten_seqs)
        if max_seq_len is None:
            max_seq_len = np.max(seqs_len)
        mask = np.arange(max_seq_len) < np.expand_dims(seqs_len, -1)
        padded_seqs = np.ones(mask.shape, dtype) * pad
        padded_seqs[mask] = seqs
    else:
        padded_seqs = seqs
        shape = padded_seqs.shape
        pads = []
        if max_seq_len is not None:
            assert shape[-1]<=max_seq_len
            if shape[-1] < max_seq_len:
                for dim in shape[:-1]:
                    pads.append((0, 0))
                pads.append((0, max_seq_len-shape[-1]))
                padded_seqs = np.pad(padded_seqs, tuple(pads), 'constant', constant_values=pad)
        seqs_len = np.ones(shape[:-1], dtype)*shape[-1]
    return padded_seqs, seqs_len


def timestamp():
    return time.strftime('%Y%m%d%H%M%S')


@contextmanager
def timer(name):
    t0 = time.time()
    #print('{} start'.format(name))
    logger.info('%s start', name)
    yield
    #print('{} done in {} seconds'.format(name, time.time() - t0))
    logger.info('%s done in %s seconds', name, time.time()-t0)


@contextmanager
def trace(name, sleep=1):
    t0 = time.time()
    p = psutil.Process(os.getpid())
    m0 = p.memory_info()[0] / 2. ** 20
    #print('{} start'.format(name))
    logger.info('%s start', name)
    yield
    time.sleep(sleep)
    m1 = p.memory_info()[0] / 2. ** 20

    #print('{} done in {} seconds'.format(name, time.time() - t0))
    logger.info('%s done in %s seconds, memory percent:%s, delta:%sM', name, time.time()-t0, psutil.virtual_memory().percent, m1-m0)


def merge_args(args, cfg={}):
    for k, v in args.__dict__.items():
        if v is not None:
            cfg[k] = v
    return cfg


def parse_model_name(model_names):
    names2cls = OrderedDict()
    for name in re.split('[, ]+', model_names):
        names2cls[name] = name.split('_')[0]
    return names2cls


def create_model(name, args, cls, suffix='', cfg={}, **kwargs):
    name = name + suffix
    cfg = cfg.copy()
    cfg = merge_args(args, cfg)
    model = cls(name=name, cfg=cfg, **kwargs)
    return model


def set_logger(level=logging.INFO):
    logger = logging.getLogger()
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s -   %(message)s')
    handler.setFormatter(formatter)
    logger.handlers = [handler]
    logger.setLevel(level)

def unpack_args(func):
    from functools import wraps
    @wraps(func)
    def wrapper(args):
        if isinstance(args, dict):
            return func(**args)
        else:
            return func(*args)
    return wrapper


def update_deepspeed_cfg(cfg):
    if not cfg.use_deepspeed:
        del cfg.deepspeed
        return
    config = cfg.deepspeed
    if not cfg.use_fp16:
        logger.warning('automatically enable fp16 for deepspeed')
    cfg.use_fp16 = False
    config['fp16']['enabled'] = True

    config['train_micro_batch_size_per_gpu'] = cfg.batch_size
    config['gradient_accumulation_steps'] = cfg.accumulated_batch_size
    if cfg.gradient_clip is not None:
        config['gradient_clipping'] = cfg.gradient_clip
    #config['optimizer']['type'] = cfg.opt
    config['optimizer']['params'] = cfg.opt_paras
    config['optimizer']['params']['lr'] = cfg.lr

    scheduler = "WarmupLR"
    params = {
        "warmup_min_lr": 0,
        "warmup_max_lr": cfg.lr,
        "warmup_num_steps": cfg.n_lr_warmup_step,
    }
    config["scheduler"] = {
        "type": scheduler,
        "params": params,
    }


def init_deepspeed(config, model):
    class Object(object):
        pass
    args = Object()
    args.local_rank = -1
    import deepspeed
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        args=args,  # expects an obj
        model=model,
        model_parameters=model_parameters,
        config_params=config,
    )
    print(111, optimizer.optimizer.state_dict().keys())
    logger.info('deepspeed model initialized')

    return model, optimizer, lr_scheduler


## arg parser
class ArgParser(argparse.ArgumentParser):
    def __init__(self, **kwargs):
        super(ArgParser, self).__init__(prog=os.path.basename(sys.argv[0]),
                                         formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__, **kwargs)

        self.add_argument("-m", "--method_name", help="run method name")
        self.add_argument("-num", type=int, default=int(1e16), help="num of examples")
        self.add_argument("-val_num", type=int, default=int(1e16), help="num of examples")
        self.add_argument("-test_num", type=int, default=int(1e16), help="num of examples")
        self.add_argument("-d", "--debug", action="store_true", help="debug")
        self.add_argument("-data_dir", default='../data', help="data dir")
        self.add_argument("-output_dir", default='../data', help="output dir")
        self.add_argument("-model_dir", default="../data", help="model dir")
        self.add_argument("-save", action="store_true", help="save")


class TrainArgParser(ArgParser):
    def __init__(self, **kwargs):
        super(TrainArgParser, self).__init__(**kwargs)

        self.add_argument("-seed", type=int , default=9527)
        self.add_argument("-show_cfg", action="store_true", help="show model cfg")
        self.add_argument("-ms", "--model_names", help="model names")
        self.add_argument("-scoring", "--scoring", action="store_true", help="run score after train")
        self.add_argument("-ov", "--only_validate", action="store_true", help="only do validation")
        self.add_argument("-nt", "--no_train", action="store_true", help="do not do train")
        self.add_argument("-np", "--no_predicting", action="store_true", help="do not do predicting")
        self.add_argument("-ntest", "--no_test", action="store_true", help="do not pred test")
        self.add_argument("-predicting", "--predicting", action="store_true", help="run predict after train")
        self.add_argument("-brcfg", "--batch_reader_cfg", type=json.loads, help="data reader cfg")
        self.add_argument("-tf_loglevel", "--tf_loglevel", type=int, default=1, help="tf log level.0:DEBUG, 1:INFO, 2:WARN, 3: ERROR")
        self.add_argument("-use_tf_estimator", "--use_tf_estimator", action="store_true", help="use tensorflow estimator api")
        self.add_argument("-regen_tfrecord", "--regen_tfrecord", action="store_true", help="regeneration of tf records")
        self.add_argument("-fake_colab", "--fake_colab", action="store_true", help="used for debug at local")
        self.add_argument("-warm_start_path", "--warm_start_path", help="tpu warm start")
        self.add_argument("-tfrecords_repeat", type=int, help="when generate tf records, repeat the data")
        self.add_argument("-num_core_per_host", type=int,  help="num_core_per_host")
        self.add_argument("-save_opt", action="store_true", help="save optimizer")
        self.add_argument("-save_half", action="store_true", help="save half precision(fp16)")
        self.add_argument("-save_best", action="store_true", help="save best epoch")
        self.add_argument("-save_log", action="store_true", help="save log to file")
        self.add_argument("-save_processed", "--save_processed", action="store_true", help="save processed data")
        self.add_argument("-save_epoch", "--save_epoch", type=int, help="save per epochs")
        self.add_argument("-use_gc", action="store_true", help="use gradient checkpointing")

        # dataset
        self.add_argument("-shuffle_buffer_size", type=int, default=128, help="dataset shuffle buffer size")
        self.add_argument("-n_dl_worker", type=int, help="data loader works")

        # distributed
        self.add_argument("-device", nargs='+', help="specify gpus")
        self.add_argument("-world_size", type=int, default=0)
        self.add_argument("-use_mg", "--use_multiple_gpu", action="store_true", help="enable multiple gpu if available")
        self.add_argument("-find_unused_parameters", action="store_true")

        # tpu
        self.add_argument("-use_tpu", "--use_tpu", action="store_true", help="use tpu")
        self.add_argument("-tpu_name", "--tpu_name",  help="tpu name")
        self.add_argument("-gsurl", "--gsurl", help="google cloud storage url for save tfrecords and model, used for tpu")
        self.add_argument("-tpu_zone", "--tpu_zone",  help="tpu zone")
        self.add_argument("-tpu_loop_iterations", type=int,  help="tpu loop iterations")
        self.add_argument("-gcp_project", "--gcp_project",  help="gcp project")

        # xla
        self.add_argument("-xla_procs", type=int,  help="num of process for xla_multiprocessing")
        self.add_argument("-xla_spawn_method", help="spawn method for xla_multiprocessing")
        self.add_argument("-xla_share_model", action="store_true", help="share model for multiprocessing to save memory")

        # quantize
        self.add_argument("-use_qat", action="store_true", help="use quantize aware training")


        # training
        self.add_argument("-batch_keys", nargs='+', help="collect data from batch output as predict output, if key not in output batch, will get from the input batch")
        self.add_argument("-dp", "--dropout", type=float, default=0.0)
        self.add_argument("-save_pred", action="store_true", help="save pred")
        self.add_argument("-compile", action="store_true", help="comiple pytorch")
        self.add_argument("-compile_dynamic", action="store_true", help="comiple pytorch")
        self.add_argument("-compile_mode", default="default")
        self.add_argument("-use_fp16", action="store_true", help="mixed precision fp16 training")
        self.add_argument("-use_deepspeed", action="store_true")
        self.add_argument("-recompute", action="store_true", help="recompute grads")
        self.add_argument("-run_eagerly", action="store_true", help="tenesorflow eager mode", default=None)
        self.add_argument("-bs", "--batch_size", type=int, default=2, help="batch size")
        self.add_argument("-vbs", "--val_batch_size", type=int, default=2, help="validate batch size")
        self.add_argument("-abs", "--accumulated_batch_size", type=int, help="accumulated batch training")
        self.add_argument("-n_save_step", type=int, help="save per steps")
        self.add_argument("-n_save_epoch", type=int, help="save per epochs")
        self.add_argument("-n_init_epoch", type=int, help="ignore such as validation stuff for first n epochs")
        self.add_argument("-n_es_epoch", type=int, help="early stop after n epochs without improve")
        self.add_argument("-restore_model", default="")
        self.add_argument("-restore_epoch", type=int, default=-99999, help="restore epoch training from")
        self.add_argument("-map_gpu", action="store_true", help="load torch checkping directly to gpu")
        self.add_argument("-n_epoch_step", type=int, help="num of steps per train epoch")
        self.add_argument("-n_val_epoch_step", type=int, help="num of steps per val epoch")
        self.add_argument("-nv", "--no_validate", action="store_true", help="do not do validation")
        self.add_argument("-validation_steps", type=int)
        self.add_argument("-validation_frea", type=int)
        self.add_argument("-initial_epoch", type=int, default=0)
        self.add_argument("-patience", type=int, help='Number of epochs with no improvement after which training will be stopped')
        self.add_argument("-es", "--es_min_delta", type=float, help="Minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute change of less than min_delta, will count as no improvement.")
        self.add_argument("-ema", type=float, help="if >0 use exponential moving average of the weights")
        self.add_argument("-copy_ema", action="store_true")
        self.add_argument("-use_ema", action="store_true")
        self.add_argument("-save_ema", action="store_true", help="save the ema model seperately, usefully for resume training")
        self.add_argument("-use_lml", action="store_true", help="low memory loading weight")
        self.add_argument("-n_swa", type=int, default=0, help="swa last n ckpt")
        self.add_argument("-swa_bs", "--swa_batch_size", type=int, help="swa batch size for bn update. pt only")
        # cai
        self.add_argument("-use_cai", action="store_true")
        ## deepspeed
        self.add_argument("-local_rank", "--local_rank", type=int, default=-1)

        # save and  restore
        self.add_argument("-n_keep_ckpt", type=int, help="num of saved ckpts to keep")
        self.add_argument("-restore", action="store_true", help="restore")
        self.add_argument("-restore_opt", action="store_true", help="restore opt")
        self.add_argument("-restore_rng", action="store_true", help="restore torch random state")
        self.add_argument("-restore_strict", action="store_true", help="stric to exactly match state dict")
        self.add_argument("-restore_ignore_keys")
        self.add_argument("-not_restore_cfg", action="store_true", default=False, help="not restore cfg")

        # self distill
        self.add_argument("-sd_weight", type=float, help="self distill loss weight")
        self.add_argument("-sd_warmup_step", type=int, help="self distill warmup step")

        # torchscript
        self.add_argument("-torchscript", action="store_true")
        ## lr
        self.add_argument("-lr", "--lr", type=float, help="learning rate")
        self.add_argument("-n_lr_warmup_step", type=int, help="lr warmup steps")
        self.add_argument("-n_lr_decay_step", type=int, help="lr decay steps")
        self.add_argument("-lr_decay_rate", type=float, help="lr decay ratio")
        self.add_argument("-lr_scheduler", help='lr scheduler')
        self.add_argument("-lr_scheduler_paras", type=json.loads, help='lr scheduler parameters')
        self.add_argument("-lr_scheduler_ld_paras", type=json.loads, help='lr scheduler parameters for ld')


        self.add_argument("-save_summary_step", "--save_summary_step", type=int, help="save per steps")
        self.add_argument("-save_keep", "--save_keep", type=int, help="keep number of saved copy")
        #self.add_argument("-weight_decay", type=float, help="weight decay")
        self.add_argument("-opt", help="optimizer:adam, radam, adamW, sgd")
        self.add_argument("-opt_paras", type=json.loads, help='parameters for optimizer')
        self.add_argument("-kernel_initializer", help="kernel initializer")
        self.add_argument("-gradient_clip", type=float, help="global norm gradient clip")
        self.add_argument("-nan_grad_to_zero", action="store_true")
        self.add_argument("-warmup_steps", type=int, help="lr warm up steps")
        self.add_argument("-val_step", "--val_step", type=int, help="do validate per steps", default=int(1e15))
        self.add_argument("-ckpt_hours", "--keep_checkpoint_every_n_hours", type=int, help="")
        self.add_argument("-val_test", "--val_test", action="store_true", help="run validation on test split")
        self.add_argument("-fsuffix", default='', help="prefix name")
        self.add_argument("-fprefix", default='', help="prefix of generated name")
        self.add_argument("-l2", "--l2", type=float, help="l2 regularize")
        self.add_argument("-kfid", "--kfid", help="k fold ID")
        self.add_argument("-kn", "--kn", type=int, help="k fold num")
        self.add_argument("-epochs", "--epochs", type=int, help="training epochs")
        self.add_argument("-tp", "--train_pct", type=float, help="data percent used for train")
        self.add_argument("-verbose", "--verbose", type=int, help="verbose per num of batch")

