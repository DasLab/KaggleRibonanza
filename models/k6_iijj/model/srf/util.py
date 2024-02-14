import os, re, logging
import numpy as np
from glob import glob
import pandas as pd
from collections import defaultdict
import json
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
from ast import literal_eval
from copy import deepcopy
from sklearn.metrics import mean_absolute_error
try:
    import networkx as nx
except:
    pass

from .. import util as ut

logger = logging.getLogger(__name__)

parser = ut.TrainArgParser(conflict_handler='resolve')
parser.add_argument("-ds", "--dataset", default='srf')
parser.add_argument("-test_ds", default='srf')
parser.add_argument("-stack_ds")
parser.add_argument("-ds_cls", help="dataset class")
parser.add_argument("-pretrain_ds_cls", help="dataset class")
parser.add_argument("-val_ds_cls", help="dataset class")
parser.add_argument("-ems", "--eval_model_names")
parser.add_argument("-sms", "--stack_model_names")
parser.add_argument("-prefix", default="")
parser.add_argument("-activation")
parser.add_argument("-backbone")
parser.add_argument("-n_label", type=int)
parser.add_argument("-n_label", type=int)
parser.add_argument("-enc_dim", type=int)
parser.add_argument("-n_layer", type=int)
parser.add_argument("-n_head", type=int)
parser.add_argument("-d_model", type=int)
parser.add_argument("-n_repeat", type=int, default=1)
parser.add_argument("-pretrain", action="store_true")
parser.add_argument("-use_inst", action="store_true")
parser.add_argument("-use_cache", action="store_true")
parser.add_argument("-use_semi", action="store_true")
parser.add_argument("-use_semi2", action="store_true")
parser.add_argument("-semi_ratio", type=float, default=1.0)
parser.add_argument("-semi_weight", type=float, default=1.0)
parser.add_argument("-semi2_ratio", type=float, default=1.0)
parser.add_argument("-semi2_weight", type=float, default=1.0)
parser.add_argument("-val_by_score", action="store_true")
parser.add_argument("-val_all", action="store_true")
parser.add_argument("-use_wwm", action="store_true")
parser.add_argument("-use_peft", action="store_true")
parser.add_argument("-use_onnx", action="store_true")
parser.add_argument("-lora_rank", type=int)
parser.add_argument("-lora_alpha", type=int)
parser.add_argument("-restore_distill", action="store_true")
parser.add_argument("-distill_model_dir")
parser.add_argument("-stratify", action="store_true", default=None)
parser.add_argument("-stratify_col", default='split')
parser.add_argument("-groupfy", action="store_true", default=None)
parser.add_argument("-groupfy_col", default='sequence_id')
parser.add_argument("-bs", "--batch_size", type=int, default=64, help="batch size")
parser.add_argument("-vbs", "--val_batch_size", type=int, default=64, help="validate batch size")
parser.add_argument("-pt_model", help="pytorch module to train")
parser.add_argument("-task_cnt", type=int, default=4)
parser.add_argument("-n_dl_worker", type=int, default=4)
parser.add_argument("-data_type", default='train')
parser.add_argument("-max_seq_len", type=int)
parser.add_argument("-min_cnt", type=int)
parser.add_argument("-data_seed", type=int, default=9527)
parser.add_argument("-lr_x", type=int, default=1)
parser.add_argument("-lr_decay", type=float, default=0)
parser.add_argument("-initializer_range", type=float)
parser.add_argument("-predict_val", action="store_true")
parser.add_argument("-predict_train", action="store_true")
parser.add_argument("-predict_raw", action="store_true")
parser.add_argument("-predict_test", action="store_true")
parser.add_argument("-backbone_dp", default=-1, type=float)
parser.add_argument("-output_dp", default=0, type=float)
parser.add_argument("-mlm_node", default=0.3, type=float)
parser.add_argument("-mlm_prob", default=0.15, type=float)
parser.add_argument("-dp_start", default=0, type=int)
parser.add_argument("-dp_end", default=1000000000000, type=int)
parser.add_argument("-use_pretrain", action="store_true")
parser.add_argument("-adv_lr", type=float, default=0)
parser.add_argument("-adv_name")
parser.add_argument("-adv_params", nargs='+')
parser.add_argument("-adv_eps", type=float, default=0.1)
parser.add_argument("-adv_step", type=float, default=1)
parser.add_argument("-adv_start_epoch", type=float, default=0)
parser.add_argument("-label_smooth", type=float, default=0)
parser.add_argument("-mixup", type=float, default=0)
parser.add_argument("-sd_start_step", type=int, default=0)
parser.add_argument("-sd_weight", type=float, default=0.0)
parser.add_argument("-avg_pool", action="store_true", default=None)
parser.add_argument("-rdrop", type=float, default=0.0)
# distill
parser.add_argument("-distill", action="store_true", default=None)
parser.add_argument("-distill_factor", type=int)
parser.add_argument("-distill_pred", action="store_true", default=None)
#
parser.add_argument("-temp", type=float, default=None)
parser.add_argument("-bpp_emb_rand", type=float, default=0.1)
parser.add_argument("-disable_val_kf", action="store_true")

##
parser.add_argument("-use_snf0", action="store_true")
parser.add_argument("-use_ext", action="store_true")
parser.add_argument("-use_gen", action="store_true")
parser.add_argument("-use_rmdb", action="store_true")
parser.add_argument("-use_mfe", action="store_true")
parser.add_argument("-not_use_mfe", action="store_true", default=None)
parser.add_argument("-use_bpp", action="store_true", default=None)
parser.add_argument("-no_emb_bpp", action="store_true", default=None)
parser.add_argument("-with_gnn_bpp", action="store_true", default=None)
parser.add_argument("-use_lt", action="store_true", default=None)
parser.add_argument("-use_bi", action="store_true", default=None)
parser.add_argument("-use_pos", action="store_true", default=None)
parser.add_argument("-use_learn", action="store_true", default=None)
parser.add_argument("-use_tfm", action="store_true", default=None)
parser.add_argument("-post_norm", action="store_true", default=None)
parser.add_argument("-use_sn_weight", action="store_true")
parser.add_argument("-no_act", action="store_true", default=None)
parser.add_argument("-no_sos", action="store_true", default=None)
parser.add_argument("-use_rope", action="store_true", default=None)
parser.add_argument("-with_test", action="store_true")
parser.add_argument("-compile_backbone", action="store_true")
parser.add_argument("-pkg")
parser.add_argument("-use_ext_token", action="store_true", default=None)
parser.add_argument("-use_label_error", action="store_true")
parser.add_argument("-aug_reverse", type=float, default=0)
parser.add_argument("-aug_cat", type=float, default=0)
parser.add_argument("-aug_mask", type=float, default=0)
parser.add_argument("-aug_cut", type=float, default=0)
parser.add_argument("-shuffle_label", type=float, default=0)
parser.add_argument("-bpp_thr", type=float, default=0.5)
parser.add_argument("-bpp_topk", type=int)
parser.add_argument("-max_error", type=float, default=10000000)
parser.add_argument("-min_sn2", type=float, default=-1000)
parser.add_argument("-n_char_max", type=int, default=10000000)
parser.add_argument("-n_char_min", type=int, default=0)
parser.add_argument("-min_sn", type=float, default=1)
parser.add_argument("-filter_sn", type=float, default=-1)
parser.add_argument("-min_reads", type=int, default=100)
parser.add_argument("-sl_window", type=int)
parser.add_argument("-rnn_dim", type=int)
parser.add_argument("-gnn_dim", type=int)
parser.add_argument("-gnn_node_layer", type=int)
parser.add_argument("-gnn_layer", type=int)


def load_preds(model_names, data_type='val', model_dir='../data', is_raw=True, prefix='pred'):
    preds = defaultdict(list)
    for model_name in model_names.split(' '):
        if is_raw:
            pred = ut.load_dump(os.path.join(model_dir, model_name, '_'.join([prefix, 'raw', data_type]) + '.dump'))
        else:
            pred = ut.load_dump(os.path.join(model_dir, model_name, '_'.join([prefix, data_type]) + '.dump'))
        model_name = '_'.join(model_name.split('_')[:-1])
        preds[model_name].append(pred)
    if data_type=='val':
        for k, v in preds.items():
            preds[k] = pd.concat(v)
    else:
        pass
    return preds


def load_bpp(inputs):
    fpath, n = inputs
    bpp = np.zeros([n, n], dtype=np.float16)
    with open(fpath) as f:
        text = f.read()
        for l in text.split('\n'):
            l = l.split()
            if len(l) > 0:
                s, e, p = l
                s, e, p = int(s), int(e), float(p)
                bpp[s, e] = p
                bpp[e, s] = p
    return bpp


def load_bpp_pos(inputs, thr=0.5):
    fpath, n = inputs
    g = nx.Graph()
    with open(fpath) as f:
        text = f.read()
        for l in text.split('\n'):
            l = l.split()
            if len(l) > 0:
                s, e, p = l
                s, e, p = int(s), int(e), float(p)
                if p>thr:
                    g.add_edge(s, e)
    for i in range(n-1):
        g.add_edge(i, i+1)
    poses = np.zeros([n, n], dtype=np.int16)
    for i in range(n):
        for j in range(i+1, n):
            d = nx.shortest_path_length(g, i, j)
            poses[i, j] = d
            poses[j, i] = d
    return poses

def load_semi_data(args, fpath):
    df = pd.read_csv(fpath, dtype={"reactivity_2A3_MaP":np.float16, "reactivity_DMS_MaP":np.float16})
    logits = np.stack([df["reactivity_2A3_MaP"].values, df["reactivity_DMS_MaP"].values], axis=1).astype(np.float16)
    ids = df.id.values
    inds = np.argsort(ids)
    logits = np.ascontiguousarray(logits[inds])
    logger.info('logits dtype %s, %s', logits.dtype, logits.shape)
    test = pd.read_csv('../data/srf/test_sequences.csv', nrows=args.num)
    test['ID'] = np.arange(len(test))
    test['SN_filter'] = 1
    test['src'] = "srf"
#    max_len = test.sequence.apply(len).max()
#    labels = np.zeros([len(test), max_len, 2], dtype=np.float16)
#    for ID, id_min, id_max in zip(test.ID, test.id_min, test.id_max):
#        labels[ID, :id_max-id_min+1] = logits[id_min:id_max+1]

    return test, logits

def load_data(args):
    if args.dataset=='srf':
        if args.data_type=='train':
            if args.debug:
                df = pd.read_csv(f"{args.data_dir}/{args.dataset}/train_data.csv", nrows=args.num)
            else:
                df = pd.read_csv(f"{args.data_dir}/{args.dataset}/train_data.csv")
                sids = sorted(df.sequence_id.unique())
                rs = np.random.RandomState(9527)
                rs.shuffle(sids)
                sids = sids[:args.num]
                df = df[df.sequence_id.isin(sids)]
        elif args.data_type=='test':
            df = pd.read_csv(f"{args.data_dir}/{args.dataset}/test_sequences.csv", nrows=args.test_num)
        df['src'] = 'srf'
    elif args.dataset=='snf0':
        args = deepcopy(args)
        args.dataset='srf'
        args.data_type = 'train'
        df = load_data(args)
        df = df.groupby('sequence_id').head(1)
        df = df[df.signal_to_noise<=1]
        df = df[["sequence_id", "sequence", "src"]]
        df['SN_filter'] = 1
        df = df.sort_values('sequence_id').reset_index(drop=True)
        df['ID'] = np.arange(len(df))
    elif args.dataset == 'srf200':
        df = pd.read_csv(f"{args.data_dir}/srf/train_data.csv")
        df = df[df.sequence.apply(len)>200]
        df['src'] = 'srf'
    elif args.dataset == 'rmdb':
        df = pd.read_csv('../data/rmdb/rmdb_data.v1.3.0.csv', nrows=args.num)
        df = df[df.experiment_type.isin(['1M7', 'NMIA', 'BzCN', 'DMS'])].reset_index(drop=True)
        df['src'] = 'rmdb'

    elif args.dataset == 'srftxt':
        use_cols = ['sequence_id', 'sequence', 'SN_filter']
        if args.debug:
            df = pd.read_csv(f"{args.data_dir}/srf/train_data.csv", nrows=args.num, usecols=use_cols)
        else:
            df = pd.read_csv(f"{args.data_dir}/srf/train_data.csv", usecols=use_cols)
        df = df.groupby('sequence_id').head(1)
        df['SN_filter'] = 1
        df = df.sample(frac=1, random_state=9527)
        df = df[:args.num]
        if args.with_test:
            df2 = pd.read_csv(f"{args.data_dir}/srf/test_sequences.csv", nrows=args.num)
            df2['SN_filter'] = 1
            df = pd.concat([df, df2])
        df['src'] = 'srf'
        if args.use_rmdb:
            df2 = pd.read_csv(f"{args.data_dir}/rmdb/rmdb_data.v1.3.0.csv", usecols=['sequence_id', 'sequence'], nrows=args.num)
            df2 = df2[~df2.sequence_id.isin(df.sequence_id)]
            df2['SN_filter'] = 1
            df2['src'] = 'rmdb'
            df = pd.concat([df, df2])

    df = reduce_mem_usage(df)
    df['n_char'] = df.sequence.apply(len)
    df['ID'] = np.arange(len(df))
    df = df[(df.n_char>=args.n_char_min) & (df.n_char<=args.n_char_max)].reset_index(drop=True)
    cids = ut.load_dump(f'{args.data_dir}/cluster.dump')
    df['cid'] = df.sequence_id.map(cids)
    return df


def score(preds):
    losses = []
    for logits, label, label_mask in zip(preds['logits'], preds['label'], preds['label_mask']):
        mask = label_mask>0
        losses.append(np.abs(logits[mask]-label[mask]))
    loss = np.mean(np.concatenate(losses))
    return 1-loss

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
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df



def get_base_pair(structure, bi=False):
    cue, pairs = [], []
    for i in range(len(structure)):
        if structure[i] == "(":
            cue.append(i)
        elif structure[i] == ")":
            s = cue.pop()
            pairs.append([s, i])
            if bi:
                pairs.append([i, s])
    return pairs


def get_dist_matrix(l):
    d = np.arange(l)
    d = np.abs(d[:, None] - d[None, :]) + 1
    d = 1 / d
    ds = np.stack([d**i for i in [1, 2, 4]], axis=2).astype(np.float32)
    ds = ds/ds.sum(axis=1, keepdims=True)
    return ds

def get_edge(structure):
    cue = []
    edges = []
    for i in range(len(structure)):
        if structure[i] == "(":
            cue.append(i)
        elif structure[i] == ")":
            s = cue.pop()
            edges.append([s, i])
    return edges

def get_pos(structure, plus=1):
    cue = []
    pos = np.arange(len(structure))
    pos = pos[:, None] - pos[None, :]
    sign = np.sign(pos)
    pos = np.abs(pos)
    for i in range(len(structure)):
        if structure[i] == "(":
            cue.append(i)
        elif structure[i] == ")":
            s = cue.pop()
            pos[s] = np.minimum(pos[s], pos[i] + plus)
            pos[i] = np.minimum(pos[i], pos[s] + plus)

            pos = np.minimum(pos[:, s:s + 1] + pos[s:s + 1, :], pos)
            pos = np.minimum(pos[:, i:i + 1] + pos[i:i + 1, :], pos)
    return pos * sign

if __name__ == "__main__":
    args = parser.parse_args([])
    args.task_cnt = 4
    args.num = 100
    args.data_type = 'train'
    ut.set_logger(logging.INFO)

    args.num = 100
    args.dataset = 'srf'
    df = load_data(args)
