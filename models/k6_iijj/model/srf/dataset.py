import os, sys, logging
#os.environ['TOKENIZERS_PARALLELISM'] = 'False'
import numpy as np
import pandas as pd
from collections import defaultdict
from glob import glob
import math
from ..util.pt.util import sequence_mask
from . import util
from .. import util as ut
from copy import deepcopy
import torch
import torch.distributed as dist
from torch.utils.data.dataloader import default_collate
from .....util.feature_gen import get_feature

logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

def worker_init_fn(worker_id):
    np.random.seed(np.random.randint(1000000000) + worker_id)

class Sampler(torch.utils.data.Sampler):
    def __init__(self, cfg, data_type, ds):
        self.cfg = cfg
        self.data_type = data_type
        self.ds = ds
        self.inds = np.arange(len(ds))
        cid2cnt = {cid:len(self.ds.cid2data[cid]) for cid in self.ds.cid2data}
        self.weight = np.array([1/cid2cnt[cid] for cid in self.ds.cluster_id])
        self.weight = self.weight/np.sum(self.weight)
    def __len__(self):
        return len(self.ds)*self.cfg.n_repeat

    def __iter__(self):
        for ind in self.gen_inds():
            yield ind

    def gen_inds(self):
        if self.data_type=='train':
            inds = np.random.choice(self.inds, self.__len__(), p=self.weight)
        else:
            raise NotImplementedError(self.data_type)
        return inds

class DatasetMix():
    def __init__(self, cfg, data_type, data, tokenizer=None):
        self.cfg = cfg
        self.data_type = data_type
        self.tokenizer=tokenizer
        self.oid2new = dict(zip(self.cfg.oids, np.arange(len(self.cfg.oids))))
        if 'roformer' in self.cfg.backbone:
            vocab = ["a", "c", "g", "u"]
        else:
            vocab = ["A", "C", "G", "U"]
        #self.c2id = dict(zip(vocab, self.tokenizer.convert_tokens_to_ids(vocab)))
        special_token_ids = [self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.mask_token_id]
        self.rand_token_ids = [id for id in self.cfg.oids if id not in special_token_ids]

        with ut.timer('preprocess'):
            self.data = self.preprocess_data(data)
        self.base_type_map = {"(":0, ".":1, ")":2}
        self.token_type_map = {"A":0, "C":1, "G":2, "U":3}
        if self.cfg.use_lt:
            lt = ['B', 'E', 'H', 'I', 'M', 'S', 'X']
            self.lt_map = dict(zip(lt, np.arange(len(lt))))
            sid2lt = ut.load_dump(f'{self.cfg.data_dir}/sid2lt_eternafold.dump')
            sids = set(data.sequence_id.values)
            self.sid2lt = {k: v for k, v in sid2lt.items() if k in sids}
        if self.cfg.use_mfe or self.cfg.use_pos or 'MFE' in self.cfg.pt_model or 'GNN' in self.cfg.pt_model or self.cfg.use_ext_token:
            if 'GNN' in self.cfg.pt_model:
                sid2mfe = []
                for pkg in self.cfg.pkg.split():
                    if pkg == 'vienna_2':
                        sid2mfe.append({
                            row['sequence_id']: get_feature('mfe_vienna2', row['sequence']) for _, row in data.iterrows()
                        })
                    elif pkg == 'eternafold':
                        sid2mfe.append({
                            row['sequence_id']: get_feature('mfe_eternafold', row['sequence']) for _, row in data.iterrows()
                        })
                    else:
                        raise NotImplementedError(pkg)
            else:
                if pkg == 'vienna_2':
                    sid2mfe = {
                        row['sequence_id']: get_feature('mfe_vienna2', row['sequence']) for _, row in data.iterrows()
                    }
                elif pkg == 'eternafold':
                    sid2mfe = {
                        row['sequence_id']: get_feature('mfe_eternafold', row['sequence']) for _, row in data.iterrows()
                    }
                else:
                    raise NotImplementedError(pkg)
            sids = set(data.sequence_id.values)
            self.sid2mfe = sid2mfe
            if self.cfg.aug_reverse>0:
                sid2mfe_reverse = ut.load_dump(f'{self.cfg.data_dir}/sid2mfe_{self.cfg.pkg}_reverse.dump')
                self.sid2mfe_reverse = {k: v for k, v in sid2mfe_reverse.items() if k in sids}
            if self.cfg.use_gen:
                gen_sid2mfe = ut.load_dump(f'{self.cfg.data_dir}/sid2mfe_gen_{self.cfg.pkg}.dump')
                self.sid2mfe.update(gen_sid2mfe)
            if self.cfg.use_rmdb:
                rmdb_sid2mfe = ut.load_dump(f'{self.cfg.data_dir}/sid2mfe_rmdb_{self.cfg.pkg}.dump')
                self.sid2mfe.update(rmdb_sid2mfe)

    def __len__(self):
        num = len(self.data)
        if self.data_type=='train' and self.cfg.use_semi:
            num = int(num*(1+self.cfg.semi_ratio))
        if self.data_type=='train' and self.cfg.use_semi2:
            num = num +int(len(self.data)*self.cfg.semi2_ratio)
        return num

    def __bakgetitem__(self, index):
        raise NotImplementedError()
        #item = self.getitem(index)
        #if item is None:
        #    return None
        #if self.data_type=='train' and self.cfg.aug_reverse>0 and np.random.rand()<self.cfg.aug_reverse:
        #    item = self.aug_reverse(item)
        #    if self.cfg.use_bpp:
        #        sid = item['sequence_id']
        #        bpp_fpath = f"{self.cfg.data_dir}/preprocessed/bpps/{self.cfg.pkg}/{sid}_reverse.txt"
        #        if not os.path.exists(bpp_fpath):
        #            item['bpp'] = np.zeros([item['input_len'], item['input_len']], dtype=np.float32)
        #        else:
        #            item['bpp'] = util.load_bpp([bpp_fpath, item['input_len']]).astype(np.float32)
        #else:
        #    sid = item['sequence_id']
        #    if self.cfg.use_bpp and 'bpp' not in item:
        #        if item['src'] in ['gen', 'rmdb']:
        #            bpp_fpath = f"{self.cfg.data_dir}/preprocessed/bpps/{self.cfg.pkg}/{sid}.txt"
        #            if os.path.exists(bpp_fpath):
        #                bpp = util.load_bpp([bpp_fpath, item['input_len']]).astype(np.float32)
        #            else:
        #                bpp = np.zeros([item['input_len'], item['input_len']], dtype=np.float32)
        #        else:
        #            if sid in self.sid2bpp_fpath and self.cfg.pkg=='eternafold' and item['src'] in ['srf']:
        #                bpp_fpath = self.sid2bpp_fpath[sid]
        #                bpp = util.load_bpp([bpp_fpath, item['input_len']]).astype(np.float32)
        #            else:
        #                bpp_fpath = f"{self.cfg.data_dir}/preprocessed/bpps/{self.cfg.pkg}/{sid}.txt"
        #                if os.path.exists(bpp_fpath):
        #                    bpp = util.load_bpp([bpp_fpath, item['input_len']]).astype(np.float32)
        #                else:
        #                    bpp = np.zeros([item['input_len'], item['input_len']], dtype=np.float32)
        #        item['bpp'] = bpp
        #if self.data_type=='train' and self.cfg.aug_cat>0 and np.random.rand()<self.cfg.aug_cat:
        #    item = self.aug_cat(item)
        #if self.data_type=='train' and self.cfg.aug_cut>0 and np.random.rand()<self.cfg.aug_cut:
        #    item = self.aug_cut(item)
        #if self.data_type=='train' and self.cfg.aug_mask>0:
        #    item = self.aug_mask(item)
        #return item

    def __getitem__(self, index):
        item = self.getitem(index)
        if item is None:
            return None
        if self.data_type=='train' and self.cfg.aug_reverse>0 and np.random.rand()<self.cfg.aug_reverse:
            item = self.aug_reverse(item)
            if self.cfg.use_bpp:
                sid = item['sequence_id']
                bpp_fpath = f"{self.cfg.data_dir}/preprocessed/bpps/{self.cfg.pkg}/{sid}_reverse.txt"
                if not os.path.exists(bpp_fpath):
                    item['bpp'] = np.zeros([item['input_len'], item['input_len']], dtype=np.float32)
                else:
                    item['bpp'] = util.load_bpp([bpp_fpath, item['input_len']]).astype(np.float32)
        else:
            sid = item['sequence_id']
            assert 'bpp' not in item
            if "GNN" in self.cfg.pt_model:
                bpps = []
                if item['src'] in ['gen', 'rmdb']:
                    raise NotImplementedError(item['src'])
                    bpp_fpath = f"{self.cfg.data_dir}/preprocessed/bpps/{self.cfg.pkg}/{sid}.txt"
                    if os.path.exists(bpp_fpath):
                        bpp = util.load_bpp([bpp_fpath, item['input_len']]).astype(np.float32)
                    else:
                        bpp = np.zeros([item['input_len'], item['input_len']], dtype=np.float32)
                else:
                    for pkg in sorted(self.cfg.pkg.split()):
                        if pkg == 'vienna_2':
                            bpps.append(
                                np.pad(
                                    get_feature('bpps_vienna2', self.sid2rec[self.data[index]]['sequence']).astype(np.float16).astype(np.float32),
                                    ((0, item['input_len'] - len(item['sequence'])))
                                )
                            )
                        elif pkg == 'eternafold':
                            bpps.append(
                                np.pad(
                                    get_feature('bpps_eternafold', self.sid2rec[self.data[index]]['sequence']).astype(np.float16).astype(np.float32),
                                    ((0, item['input_len'] - len(item['sequence'])))
                                )
                            )
                        else:
                            raise NotImplementedError(pkg)
                        # if sid in self.sid2bpp_fpath and pkg=='eternafold' and item['src'] in ['srf', 'semi']:
                        #     bpp_fpath = self.sid2bpp_fpath[sid]
                        #     bpp = util.load_bpp([bpp_fpath, item['input_len']]).astype(np.float32)
                        #     bpps.append(bpp)
                        # else:
                        #     bpp_fpath = f"{self.cfg.data_dir}/preprocessed/bpps/{pkg}/{sid}.txt"
                        #     if os.path.exists(bpp_fpath):
                        #         bpp = util.load_bpp([bpp_fpath, item['input_len']]).astype(np.float32)
                        #     else:
                        #         bpp = np.zeros([item['input_len'], item['input_len']], dtype=np.float32)
                        #         logging.info('zeros bpp %s, %s', pkg, sid)
                        #     bpps.append(bpp)
                item['bpp'] = bpps
            elif self.cfg.use_bpp and 'bpp' not in item:
                if item['src'] in ['gen', 'rmdb']:
                    raise NotImplementedError()
                    # bpp_fpath = f"{self.cfg.data_dir}/preprocessed/bpps/{self.cfg.pkg}/{sid}.txt"
                    # if os.path.exists(bpp_fpath):
                    #     bpp = util.load_bpp([bpp_fpath, item['input_len']]).astype(np.float32)
                    # else:
                    #     bpp = np.zeros([item['input_len'], item['input_len']], dtype=np.float32)
                else:
                    n_pkg = 0
                    for pkg in self.cfg.pkg.split():
                        if pkg == 'vienna_2':
                            pkg_bpp = np.pad(
                                get_feature('bpps_vienna2', self.sid2rec[self.data[index]]['sequence']).astype(np.float16).astype(np.float32),
                                ((0, item['input_len'] - len(item['sequence'])))
                            )
                        elif pkg == 'eternafold':
                            pkg_bpp = np.pad(
                                get_feature('bpps_eternafold', self.sid2rec[self.data[index]]['sequence']).astype(np.float16).astype(np.float32),
                                ((0, item['input_len'] - len(item['sequence'])))
                            )
                        else:
                            raise NotImplementedError(pkg)
                        if n_pkg == 0:
                            bpp = pkg_bpp
                        else:
                            bpp += pkg_bpp
                        n_pkg += 1
                        #if sid in self.sid2bpp_fpath and pkg=='eternafold' and item['src'] in ['srf', 'semi']:
                        #    bpp_fpath = self.sid2bpp_fpath[sid]
                        #    if n_pkg==0:
                        #        bpp = util.load_bpp([bpp_fpath, item['input_len']]).astype(np.float32)
                        #    else:
                        #        bpp += util.load_bpp([bpp_fpath, item['input_len']]).astype(np.float32)
                        #    n_pkg += 1
                        #else:
                        #    bpp_fpath = f"{self.cfg.data_dir}/preprocessed/bpps/{pkg}/{sid}.txt"
                        #    if os.path.exists(bpp_fpath):
                        #        if n_pkg==0:
                        #            bpp = util.load_bpp([bpp_fpath, item['input_len']]).astype(np.float32)
                        #        else:
                        #            bpp += util.load_bpp([bpp_fpath, item['input_len']]).astype(np.float32)
                        #        n_pkg += 1
                        #    else:
                        #        if n_pkg==0:
                        #            bpp = np.zeros([item['input_len'], item['input_len']], dtype=np.float32)
                        #            logging.info('zeros bpp %s, %s', pkg, sid)
                if n_pkg>0:
                    item['bpp'] = bpp/n_pkg
                else:
                    item['bpp'] = bpp

        if self.data_type=='train' and self.cfg.aug_cat>0 and np.random.rand()<self.cfg.aug_cat:
            item = self.aug_cat(item)
        if self.data_type=='train' and self.cfg.aug_cut>0 and np.random.rand()<self.cfg.aug_cut:
            item = self.aug_cut(item)
        if self.data_type=='train' and self.cfg.aug_mask>0:
            item = self.aug_mask(item)
        return item

    def set_rank(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size

    def get_distribute_data(self, data, world_size=None, rank=None):
        rank = dist.get_rank() if rank is None else rank
        world_size = dist.get_world_size() if world_size is None else world_size
        per_rank = int(math.ceil(len(data) / world_size))
        return data[rank * per_rank:(rank + 1) * per_rank]

    def load_bpp_fpath(self, sids):
        bpp_fpaths = [fpath for fpath in glob(f"../data/srf/Ribonanza_bpp_files/extra_data/*/*/*/*.txt")]
        fpaths, nums, bpp_sids = [], [], []
        for fpath in bpp_fpaths:
            sid = os.path.basename(fpath).split('.txt')[0]
            #if sid in sids:
            if 1==1:
                fpaths.append(fpath)
                bpp_sids.append(sid)
        logger.info("sids:%s", bpp_sids[:5])
        self.sid2bpp_fpath = dict(zip(bpp_sids, fpaths))

    def preprocess_rmdb_data(self, data):
        if self.data_type=='train' and not self.cfg.use_sn_weight:
            num = len(data)
            #data = data[data.sequence_id.isin(data[(data.reads > self.cfg.min_reads) & (data.signal_to_noise > self.cfg.min_sn)].sequence_id)]
            data = data[data.signal_to_noise > self.cfg.min_sn]
            logger.info('rmdb num filtered for min_sn and min_reads:%s', num-len(data))
        data['ID'] = np.arange(len(data))
        if 1==1:
            label_cols = [c for c in data.columns if 'reactivity_0' in c]
            label_errors = [c for c in data.columns if 'reactivity_error_0' in c]
            self.rmdb_labels = data[label_cols].values
            self.rmdb_label_errors = data[label_errors].values
            data = data[['ID', 'sequence_id', 'sequence', 'signal_to_noise', 'reads','SN_filter', 'experiment_type', 'src']]

            sid2rec = defaultdict(dict)
            sid2len = defaultdict(int)
            for sid , gp in data.groupby('sequence_id'):
                for rec in gp.to_records(index=False):
                    if rec.experiment_type not in sid2rec[sid]:
                        sid2rec[sid][rec.experiment_type] = []
                    sid2rec[sid][rec.experiment_type].append(rec)
                    sid2len[sid] = len(rec.sequence)
            self.rmdb_sid2rec = sid2rec
        sids = sorted(self.rmdb_sid2rec.keys())
        return sids

    def preprocess_semi2_data(self, data):
        data = data[['ID', 'sequence_id', 'sequence', 'SN_filter', 'src']]

        self.semi2_sid2rec = dict(zip(data.sequence_id, data.to_records(index=False)))
        sids = sorted(self.semi2_sid2rec.keys())
        return sids

    def preprocess_semi_data(self, data):
        data = data[['ID', 'sequence_id', 'sequence', 'SN_filter', 'id_min', 'id_max', 'src']]

        self.semi_sid2rec = dict(zip(data.sequence_id, data.to_records(index=False)))
        sids = sorted(self.semi_sid2rec.keys())
#        with Pool(self.cfg.task_cnt) as p:
#            self.semi_input_ids = []
#            for input_ids in tqdm(p.imap(self.get_input_ids, data.sequence, chunksize=32), total=len(data), desc=f"{self.cfg.task_cnt} get semi input ids"):
#                input_ids = [self.oid2new[id] for id in input_ids]
#                self.semi_input_ids.append(input_ids)
        return sids

    def preprocess_data(self, data):
        if self.data_type=='train':
            data = data[data.signal_to_noise>self.cfg.filter_sn]
        if self.data_type=='val':
            data = data[data.SN_filter==1]
            num = len(data)
            data = data.groupby(['sequence_id', 'SN_filter']).filter(lambda x: x.experiment_type.nunique()==2)
            logger.info('filter both SN_filter==1, %s, %s', num, len(data))
        elif self.data_type=='train' and not self.cfg.use_sn_weight:
            num = len(data)
            #data = data[data.sequence_id.isin(data[(data.reads > self.cfg.min_reads) & (data.signal_to_noise > self.cfg.min_sn)].sequence_id)]
            data = data[(data.reads > self.cfg.min_reads) & (data.signal_to_noise > self.cfg.min_sn)]
            logger.info('num filtered for min_sn and min_reads:%s', num-len(data))
        data['ID'] = np.arange(len(data))
        if self.data_type=='train' and self.cfg.use_semi:
            with ut.timer('load_semi_data'):
                semi, self.semi_labels = util.load_semi_data(self.cfg, f"{self.cfg.data_dir}/semi.zip")
            self.semi_data = self.preprocess_semi_data(semi)
        if self.data_type=='train' and self.cfg.use_semi2:
            with ut.timer('load_semi2_data'):
                cfg = self.cfg.copy()
                cfg.dataset = 'snf0'
                semi2 = util.load_data(cfg)
                self.semi2_labels = ut.load_dump(f"{self.cfg.data_dir}/semi_snf0.dump")
            self.semi2_data = self.preprocess_semi2_data(semi2)
        if self.data_type!='test':
            label_cols = [c for c in data.columns if 'reactivity_0' in c]
            label_errors = [c for c in data.columns if 'reactivity_error_0' in c]
            self.labels = data[label_cols].values
            self.label_errors = data[label_errors].values
            data = data[['ID', 'sequence_id', 'sequence', 'signal_to_noise', 'reads','SN_filter', 'experiment_type', 'src']]

            sid2rec = defaultdict(dict)
            sid2len = defaultdict(int)
            for sid , gp in data.groupby('sequence_id'):
                for rec in gp.to_records(index=False):
                    if rec.experiment_type not in sid2rec[sid]:
                        sid2rec[sid][rec.experiment_type] = []
                    sid2rec[sid][rec.experiment_type].append(rec)
                    sid2len[sid] = len(rec.sequence)
            self.sid2rec = sid2rec
        else:
            self.sid2rec = dict(zip(data.sequence_id, data.to_records(index=False)))
        if self.data_type=='test':
            sids = sorted(self.sid2rec.keys(), key=lambda x: len(self.sid2rec[x].sequence), reverse=True)
        elif self.data_type=='val':
            sids = sorted(self.sid2rec.keys(), key=lambda x: sid2len[x], reverse=True)
        else:
            sids = sorted(self.sid2rec.keys())
        return sids

    def get_input_ids(self, text, sid=None, use_reverse=False):
        if 'roformer' in self.cfg.backbone:
            text = text.lower()
        if self.cfg.use_ext_token:
            if use_reverse:
                mfe = self.sid2mfe_reverse[sid]
                text = ''.join([self.cfg.ext_token_mapping[c + s] for c, s in zip(text[::-1], mfe)])
            else:
                mfe = self.sid2mfe[sid]
                text = ''.join([self.cfg.ext_token_mapping[(c + s)] for c, s in zip(text, mfe)])
        input_ids = self.tokenizer.convert_tokens_to_ids(list(text))
        input_ids = [self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.sep_token_id]
        return input_ids

    def get_ext_feature(self, item):
        sid = item['sequence_id']
        if self.cfg.use_mfe:
            mfe = '.' + self.sid2mfe[sid] + '.'
            item['base_type'] = np.array([self.base_type_map[c] for c in mfe])
        if self.cfg.use_lt:
            lt = 'E' + self.sid2lt[sid] + 'E'
            item['lt'] = np.array([self.lt_map[c] for c in lt])
        if self.cfg.use_pos:
            mfe = '.' + self.sid2mfe[sid] + '.'
            pos = util.get_pos(mfe)
            item['pos'] = pos
        if 'GNN' in self.cfg.pt_model:
            edges = []
            for sid2mfe in self.sid2mfe:
                mfe = '.' + sid2mfe[sid] + '.'
                edge = util.get_edge(mfe)
                edges.append(edge)
            item['edge'] = edges

    def getitem_rmdb(self, index):
        sid = self.rmdb_data[index]
        recs = self.rmdb_sid2rec[sid]
        if 1==1:
            for k in recs:
                break
            rec = recs[k]
            if self.data_type=='train':
                rec = np.random.choice(rec)
            else:
                rec = rec[0]
        else:
            rec = recs

        sequence_id, sequence, src = rec.sequence_id, rec.sequence, rec.src
        label = np.zeros([len(sequence)+2, 2], dtype=np.float32)
        label_error = np.zeros([len(sequence)+2, 2], dtype=np.float32)
        label_mask = np.zeros([len(sequence)+2, 2], dtype=np.int64)
        weight = np.ones([2], dtype=np.float32)
        if self.data_type!='test':
            for i in range(2):
                if i==0:
                    if '1M7' in recs or 'NMIA' in recs or 'BzCN' in recs:
                        k = np.random.choice(sorted(recs.keys()))
                    else:
                        continue
                else:
                    k = 'DMS'
                if k in recs:
                    if self.data_type=='train':
                        _rec = np.random.choice(recs[k])
                    else:
                        _rec = recs[k][0]
                    if self.cfg.use_sn_weight and self.data_type=='train':
                        weight[i] = np.clip(_rec.signal_to_noise, 0, self.cfg.min_sn)/self.cfg.min_sn
                    _label = self.rmdb_labels[_rec.ID][:len(sequence)]
                    _label_error = self.rmdb_label_errors[_rec.ID][:len(sequence)]
                    label[1:-1, i] = _label
                    label_error[1:-1, i] = _label_error
                    label_mask[1:-1, i] = ~np.isnan(_label)

                    if self.data_type=='train':
                        label_mask[1:-1, i] = np.logical_and(label_mask[1:-1, i], _label_error<self.cfg.max_error)
                        label_mask[1:-1, i] = np.logical_and(label_mask[1:-1, i], (_label/(_label_error+1e-4) > self.cfg.min_sn2))
            if self.data_type=='train' and self.cfg.shuffle_label>0 and np.random.rand()<self.cfg.shuffle_label:
                bak_label = deepcopy(label)
                bak_label_mask = deepcopy(label_mask)
                rd = np.random.rand(label.shape[0])
                _shuffle_mask = rd<np.clip(label_error[:, 0], 0, 3)/3
                label[:, 0][_shuffle_mask] = bak_label[:, 1][_shuffle_mask]
                label_mask[:, 0][_shuffle_mask] = bak_label_mask[:, 1][_shuffle_mask]
                rd = np.random.rand(label.shape[0])
                _shuffle_mask = rd<np.clip(label_error[:, 1], 0, 3)/3
                label[:, 1][_shuffle_mask] = bak_label[:, 0][_shuffle_mask]
                label_mask[:, 1][_shuffle_mask] = bak_label_mask[:, 0][_shuffle_mask]

        input_ids = self.get_input_ids(sequence, sequence_id)
        input_ids = [self.oid2new[id] for id in input_ids]
        input_len = len(input_ids)
        item = dict(index=index, src=src, sequence_id=rec.sequence_id,  sequence=sequence, weight=weight, input_ids=np.array(input_ids), input_len=input_len,
                    label=np.clip(label, 0, 1), label_mask=label_mask, label_error=np.clip(label_error, 0, 1))
        if self.data_type=='test':
            item['id_min'] = rec.id_min
            item['id_max'] = rec.id_max
            item['reads'] = np.array([0, 1]).astype(np.float32)
            item['SN_filter'] = np.array([0, 1]).astype(np.float32)
        else:
            item['reads'] = np.array([0, 1]).astype(np.float32)
            if rec.SN_filter>1:
                item['SN_filter'] = np.array([0, 1]).astype(np.float32)
            else:
                item['SN_filter'] = np.array([1, 0]).astype(np.float32)
        return item

    def getitem_semi2(self, index):
        sid = self.semi2_data[np.random.randint(len(self.semi2_data))]
        rec = self.semi2_sid2rec[sid]

        sequence_id, sequence, src = rec.sequence_id, rec.sequence, rec.src
        label = np.zeros([len(sequence)+2, 2], dtype=np.float32)
        label_error = np.zeros([len(sequence)+2, 2], dtype=np.float32)
        label_mask = np.ones([len(sequence)+2, 2], dtype=np.int64)
        label_mask[:27, :] = 0
        label_mask[-22:,:] = 0
        weight = np.ones([2], dtype=np.float32)*self.cfg.semi2_weight
        if 1==1:
            _rec = rec
            _label = self.semi2_labels[_rec.ID][:len(sequence)]
            label[1:-1] = _label

        input_ids = self.get_input_ids(sequence, sequence_id)
        input_ids = [self.oid2new[id] for id in input_ids]
        input_len = len(input_ids)

        item = dict(index=index, src=src, sequence_id=rec.sequence_id,  sequence=sequence, weight=weight, input_ids=np.array(input_ids), input_len=input_len,
                    label=np.clip(label, 0, 1), label_mask=label_mask, label_error=np.clip(label_error, 0, 1))
        if self.data_type=='test':
            item['id_min'] = rec.id_min
            item['id_max'] = rec.id_max
            item['reads'] = np.array([0, 1]).astype(np.float32)
            item['SN_filter'] = np.array([0, 1]).astype(np.float32)
        else:
            item['reads'] = np.array([0, 1]).astype(np.float32)
            if rec.SN_filter>1:
                item['SN_filter'] = np.array([0, 1]).astype(np.float32)
            else:
                item['SN_filter'] = np.array([1, 0]).astype(np.float32)
        return item
    def getitem_semi(self, index):
        sid = self.semi_data[np.random.randint(len(self.semi_data))]
        rec = self.semi_sid2rec[sid]

        sequence_id, sequence, src = rec.sequence_id, rec.sequence, rec.src
        label = np.zeros([len(sequence)+2, 2], dtype=np.float32)
        label_error = np.zeros([len(sequence)+2, 2], dtype=np.float32)
        label_mask = np.ones([len(sequence)+2, 2], dtype=np.int64)
        label_mask[:27, :] = 0
        label_mask[-22:,:] = 0
        weight = np.ones([2], dtype=np.float32)*self.cfg.semi_weight
        if self.data_type!='test':
            _rec = rec
            #_label = self.semi_labels[_rec.ID][:len(sequence)]
            _label = self.semi_labels[_rec.id_min:_rec.id_max+1]
            label[1:-1] = _label

        input_ids = self.get_input_ids(sequence, sequence_id)
        input_ids = [self.oid2new[id] for id in input_ids]
        input_len = len(input_ids)

        item = dict(index=index, src=src, sequence_id=rec.sequence_id,  sequence=sequence, weight=weight, input_ids=np.array(input_ids), input_len=input_len,
                    label=np.clip(label, 0, 1), label_mask=label_mask, label_error=np.clip(label_error, 0, 1))
        if self.data_type=='test':
            item['id_min'] = rec.id_min
            item['id_max'] = rec.id_max
            item['reads'] = np.array([0, 1]).astype(np.float32)
            item['SN_filter'] = np.array([0, 1]).astype(np.float32)
        else:
            item['reads'] = np.array([0, 1]).astype(np.float32)
            if rec.SN_filter>1:
                item['SN_filter'] = np.array([0, 1]).astype(np.float32)
            else:
                item['SN_filter'] = np.array([1, 0]).astype(np.float32)
        return item
    def getitem(self, index):
        if self.data_type=='train' and index>=len(self.data):
            if self.cfg.use_semi and index < int(len(self.data)*(1+self.cfg.semi_ratio)):
                return self.getitem_semi(index - len(self.data))
            else:
                return self.getitem_semi2(index - int(len(self.data)*(1+self.cfg.semi_ratio)))
        sid = self.data[index]
        recs = self.sid2rec[sid]
        if self.data_type!='test':
            rec = recs['2A3_MaP'] if '2A3_MaP' in recs else recs['DMS_MaP']
            if self.data_type=='train':
                rec = np.random.choice(rec)
            else:
                rec = rec[0]
        else:
            rec = recs

        sequence_id, sequence, src = rec.sequence_id, rec.sequence, rec.src
        label = np.zeros([len(sequence)+2, 2], dtype=np.float32)
        label_error = np.zeros([len(sequence)+2, 2], dtype=np.float32)
        label_mask = np.zeros([len(sequence)+2, 2], dtype=np.int64)
        weight = np.ones([2], dtype=np.float32)
        if self.data_type!='test':
            for i, k in enumerate(['2A3_MaP', 'DMS_MaP']):
                if k in recs:
                    if self.data_type=='train':
                        _rec = np.random.choice(recs[k])
                    else:
                        _rec = recs[k][0]
                    if self.cfg.use_sn_weight and self.data_type=='train':
                        weight[i] = np.clip(_rec.signal_to_noise, 0, self.cfg.min_sn)/self.cfg.min_sn
                    _label = self.labels[_rec.ID][:len(sequence)]
                    _label_error = self.label_errors[_rec.ID][:len(sequence)]
                    label[1:-1, i] = _label
                    label_error[1:-1, i] = _label_error
                    label_mask[1:-1, i] = ~np.isnan(_label)

                    if self.data_type=='train':
                        label_mask[1:-1, i] = np.logical_and(label_mask[1:-1, i], _label_error<self.cfg.max_error)
                        label_mask[1:-1, i] = np.logical_and(label_mask[1:-1, i], (_label/(_label_error+1e-4) > self.cfg.min_sn2))
            if self.data_type=='train' and self.cfg.shuffle_label>0 and np.random.rand()<self.cfg.shuffle_label:
                bak_label = deepcopy(label)
                bak_label_mask = deepcopy(label_mask)
                rd = np.random.rand(label.shape[0])
                _shuffle_mask = rd<np.clip(label_error[:, 0], 0, 3)/3
                label[:, 0][_shuffle_mask] = bak_label[:, 1][_shuffle_mask]
                label_mask[:, 0][_shuffle_mask] = bak_label_mask[:, 1][_shuffle_mask]
                rd = np.random.rand(label.shape[0])
                _shuffle_mask = rd<np.clip(label_error[:, 1], 0, 3)/3
                label[:, 1][_shuffle_mask] = bak_label[:, 0][_shuffle_mask]
                label_mask[:, 1][_shuffle_mask] = bak_label_mask[:, 0][_shuffle_mask]

        input_ids = self.get_input_ids(sequence, sequence_id)
        input_ids = [self.oid2new[id] for id in input_ids]
        input_len = len(input_ids)
        item = dict(index=index, src=src, sequence_id=rec.sequence_id,  sequence=sequence, weight=weight, input_ids=np.array(input_ids), input_len=input_len,
                    label=np.clip(label, 0, 1), label_mask=label_mask, label_error=np.clip(label_error, 0, 1))
        if self.data_type=='test':
            if self.cfg.test_ds!='snf0':
                item['id_min'] = rec.id_min
                item['id_max'] = rec.id_max
            item['reads'] = np.array([0, 1]).astype(np.float32)
            item['SN_filter'] = np.array([0, 1]).astype(np.float32)
        else:
            if rec.reads>100:
                item['reads'] = np.array([0, 1]).astype(np.float32)
            else:
                item['reads'] = np.array([1, 0]).astype(np.float32)
            if rec.SN_filter>1:
                item['SN_filter'] = np.array([0, 1]).astype(np.float32)
            else:
                item['SN_filter'] = np.array([1, 0]).astype(np.float32)
        return item


    def get_iter_items(self, index):
        yield self.__getitem__(index)

    def sample_index(self, index):
        num = self.__len__()
        idxes = np.random.choice(num, 10)
        for idx in idxes:
            if idx != index:
                return idx

    def sample_rec(self, index, max_audio_len):
        num = self.__len__()
        idxes = np.random.choice(num, 20)
        for idx in idxes:
            if idx!=index:
                rec = self.data[idx]
                if rec.audio_len<max_audio_len:
                    return rec
        return None

    def sample_item(self, index):
        for i in range(10):
            index = self.sample_index(index)
            item = self.getitem(index)
            if item is not None:
                break
        return item

    def aug_cut(self, item):
        item2 = self.sample_item(item['index'])
        len1, len2 = item['input_len'], item2['input_len']
        cut1 = np.random.randint(50, len1-50)
        left = max(0, len2-(len1-cut1))
        cut2 = np.random.randint(0, left+1)
        for k in ['input_ids', 'label', 'label_mask']:
            if k in item:
                item[k] = np.concatenate([item[k][:cut1], item2[k][cut2:cut2+len1-cut1]], axis=0)
        item['input_len'] = len(item['input_ids'])
        return item

    def aug_mask(self, item):
        l = item['input_len']
        mask = np.random.rand(l)<self.cfg.aug_mask
        ids = np.random.choice(self.rand_token_ids, mask.sum())
        ids = [self.oid2new[id] for id in ids]
        item['input_ids'][mask] = ids
        return item

    def aug_cat(self, item):
        item2 = self.sample_item(item['index'])
        for k in ['input_ids', 'label', 'label_mask']:
            if k in item:
                item[k] = np.concatenate([item[k][0:1], item[k][1:-1], item2[k][1:-1], item[k][-1:]], axis=0)
        item['input_len'] = len(item['input_ids'])
        return item

    def aug_reverse(self, item):
        if self.cfg.use_ext_token:
            input_ids = np.array(self.get_input_ids(item['sequence'], item['sequence_id'], use_reverse=True))
            input_ids = [self.oid2new[id] for id in input_ids]
            item['input_ids'] = np.array(input_ids)
        else:
            item['input_ids'] = np.concatenate([item['input_ids'][0:1], item['input_ids'][1:-1][::-1], item['input_ids'][-1:]], axis=0)
        if 'label' in item:
            item['label'] = item['label'][::-1]
        if 'label_mask' in item:
            item['label_mask'] = item['label_mask'][::-1]
        if 'pos' in item:
            sid = item['sequence_id']
            mfe = '.' + self.sid2mfe_reverse[sid] + '.'
            item['pos'] = util.get_pos(mfe)
        if 'base_pair' in item:
            sid = item['sequence_id']
            mfe = '.' + self.sid2mfe_reverse[sid] + '.'
            item['base_pair'] = util.get_base_pair(mfe, bi=self.cfg.use_bi)
        if 'base_type' in item:
            sid = item['sequence_id']
            mfe = '.' + self.sid2mfe_reverse[sid] + '.'
            item['base_type'] = np.array([self.base_type_map[c] for c in mfe])



        return item

    def collate(self, batch):
        batch = [item for item in batch if item is not None]
        if self.cfg.no_sos:
            for item in batch:
                for k in ['input_ids', 'label', 'label_mask', 'bpp', 'base_type', 'label_error', 'pos']:
                    if k in item:
                        if k=='bpp' or k=='pos':
                            item[k] = item[k][1:-1, 1:-1]
                        else:
                            item[k] = item[k][1:-1]
                item['input_len'] = item['input_len'] - 2
        max_input_len = max([item['input_len'] for item in batch])
        bp_inds, bp_labels, edges = [], [], []
        for i, item in enumerate(batch):
            item['input_ids'] = np.pad(item['input_ids'], ((0, max_input_len-item['input_len'])), "constant")
            if 'base_type' in item:
                item['base_type'] = np.pad(item['base_type'], ((0, max_input_len-item['input_len'])), "constant")
            if 'lt' in item:
                item['lt'] = np.pad(item['lt'], ((0, max_input_len-item['input_len'])), "constant")
            if 'node_feature' in item:
                item['node_feature'] = np.pad(item['node_feature'], ((0, max_input_len-item['input_len']), (0, 0)), "constant")
            if 'label' in item:
                item['label'] = np.pad(item['label'], ((0, max_input_len-item['input_len']), (0, 0)), "constant")
            if 'label_mask' in item:
                if len(item['label_mask'].shape)==1:
                    item['label_mask'] = np.pad(item['label_mask'], ((0, max_input_len-item['input_len'])), "constant")
                else:
                    item['label_mask'] = np.pad(item['label_mask'], ((0, max_input_len-item['input_len']), (0, 0)), "constant")
            if 'label_error' in item:
                item['label_error'] = np.pad(item['label_error'], ((0, max_input_len-item['input_len']), (0, 0)), "constant")
            if 'bpp' in item:
                item['bpp'] = np.pad(item['bpp'], ((0, max_input_len - item['input_len']), (0, max_input_len-item['input_len'])), "constant", constant_values=-1000)
            if 'adj' in item:
                item['adj'] = np.pad(item['adj'], ((0, max_input_len - item['input_len']), (0, max_input_len-item['input_len']), (0, 0)), "constant", constant_values=0)
            if 'pos' in item:
                item['pos'] = np.pad(item['pos'], ((0, max_input_len - item['input_len']), (0, max_input_len - item['input_len'])), "constant", constant_values=0)
            if 'base_pair' in item:
                s = i*max_input_len
                bp_ind, bp_label = zip(*item.pop('base_pair'))
                bp_inds.append(np.array(bp_ind)+s)
                bp_labels.append(np.array(bp_label))
            if 'edge' in item:
                s = i * max_input_len
                edges.append(np.array(item.pop('edge'))+s)

        batch = default_collate(batch)
        batch['mask'] = sequence_mask(batch['input_len'], max_input_len, dtype=torch.int64)
        if len(bp_inds)>0:
            batch['bp_ind'] = torch.from_numpy(np.concatenate(bp_inds))
            batch['bp_label'] = torch.from_numpy(np.concatenate(bp_labels))
        if len(edges)>0:
            batch['edge'] = torch.from_numpy(np.concatenate(edges, axis=0))
        return batch

    def get_token_mask_inds(self, word_offsets, char2token, indices_masked):
        #word_offsets = word_offsets[indices_masked]
        word_offsets = [word_offsets[ind] for ind in indices_masked]
        token_inds = []
        for (start, end) in word_offsets:
            inds = np.arange(start, end)
            token_inds.append(char2token[inds])
        token_inds = np.concatenate(token_inds)
        token_inds = np.unique(token_inds)
        token_inds = token_inds[token_inds>=0]
        return token_inds

    def get_masked_indices(self, num, shift=0, exclude_indices=None):
        n_mask = max(1, int(self.cfg.mlm_prob*num))
        masked_indices = np.random.choice(num, n_mask, replace=False) + shift
        if exclude_indices is not None:
            mask = np.full(num+shift+1, False)
            mask[masked_indices] = True
            mask[exclude_indices] = False
            masked_indices = np.where(mask)[0]
        n = len(masked_indices)
        n1 = max(int(n*0.8), 1)
        n2 = max(int((n-n1)*0.5), 1) if n>n1 else 0
        indices_masked = masked_indices[:n1]
        indices_random = masked_indices[n1:n1+n2]
        indices_replaced = masked_indices[n1+n2:]
        return masked_indices, indices_masked, indices_random, indices_replaced

    def get_wwm_indicies(self, word_offsets, char2token, has_special=True, exclude_indices=None):
        _, indices_masked, indices_random, indices_replaced = self.get_masked_indices(len(word_offsets))

        if len(indices_masked)>0:
            indices_masked = self.get_token_mask_inds(word_offsets, char2token, indices_masked)
        if len(indices_random)>0:
            indices_random = self.get_token_mask_inds(word_offsets, char2token, indices_random)
        if len(indices_replaced)>0:
            indices_replaced = self.get_token_mask_inds(word_offsets, char2token, indices_replaced)

        mask = np.full(char2token.max()+1, False)
        if exclude_indices is not None:
            mask[exclude_indices] = True
            indices_masked = indices_masked[~mask[indices_masked]]
        mask[indices_masked] = True
        indices_random = indices_random[~mask[indices_random]]
        mask[indices_random] = True
        indices_replaced = indices_replaced[~mask[indices_replaced]]
        mask[indices_replaced] = True

        masked_indices = np.where(mask)[0]
        return masked_indices, indices_masked, indices_replaced, indices_random

    def wwm_seq(self, seq, word_offsets, char2token, has_special=True, exclude_indices=None):
        masked_indices, indices_masked, indices_random, indices_replaced = self.get_wwm_indicies(word_offsets, char2token, has_special=has_special, exclude_indices=exclude_indices)

        seq[indices_masked] = self.tokenizer.mask_token_id
        random_words = np.random.randint( low=0, high=len(self.tokenizer), size=len(indices_random), dtype=np.int64 )
        seq[indices_random] = random_words
        return seq, masked_indices

    def mlm_seq(self, seq, has_special=True, exclude_indices=None):
        num, shift = (len(seq) - 2, 1) if has_special else (len(seq), 0)
        masked_indices, indices_masked, indices_random, indices_replaced = self.get_masked_indices(num, shift=shift, exclude_indices=exclude_indices)
        seq[indices_masked] = self.tokenizer.mask_token_id
        seq[indices_random] = np.random.randint(low=0, high=len(self.tokenizer), size=len(indices_random), dtype=np.int64)
        return seq, masked_indices

class Mix1P():
    def getitem(self, index):
        rec = self.data[index]
        sequence_id, sequence = rec.sequence_id, rec.sequence
        label = np.zeros([len(sequence)+2, 1], dtype=np.float32)
        label_mask = np.zeros([len(sequence)+2, 1], dtype=np.int64)
        weight = np.ones([1], dtype=np.float32)
        if self.data_type!='test':
            _label = self.labels[rec.ID][:len(sequence)]
            label[1:-1, 0] = _label
            label_mask[1:-1, 0] = ~np.isnan(_label)
        input_ids = self.get_input_ids(sequence, sequence_id)
        input_ids = [self.oid2new[id] for id in input_ids]
        input_len = len(input_ids)
        item = dict(index=index, sequence_id=rec.sequence_id, input_ids=input_ids, input_len=input_len,
                    label=label, label_mask=label_mask, experiment_type=rec.experiment_type, weight=weight)
        if rec.experiment_type=='2A3_MaP':
            item['exp_id'] = 0
        else:
            item['exp_id'] = 1
        if self.data_type=='test':
            item['id_min'] = rec.id_min
            item['id_max'] = rec.id_max
        else:
            item['id_min'] = 0
            item['id_max'] = 0
        return item

    def preprocess_data(self, data):
        if self.data_type=='val':
            num = len(data)
            data = data.groupby(['sequence_id', 'SN_filter']).filter(lambda x: len(x)==2)
            logger.info('filter both SN_filter==1, %s, %s', num, len(data))
        data['ID'] = np.arange(len(data))
        if self.data_type!='test':
            label_cols = [c for c in data.columns if 'reactivity_0' in c]
            label_errors = [c for c in data.columns if 'reactivity_error_0' in c]
            self.labels = np.clip(data[label_cols].values, 0, 1)
            self.label_errors = data[label_cols].values
            data = data[['ID', 'sequence_id', 'sequence', 'experiment_type']]

        else:
            data2 = deepcopy(data)
            data2['experiment_type'] = '2A3_MaP'
            data = deepcopy(data)
            data['experiment_type'] = 'DMS_MaP'
            data = pd.concat([data, data2])
        recs = data.to_records(index=False)
        if self.data_type!='train':
            recs = sorted(recs, key=lambda x: len(x.sequence), reverse=True)
        else:
            recs = sorted(recs, key=lambda x: x.sequence_id)
        return recs

class BPPMix():
    def __init__(self, cfg, data_type, data, tokenizer=None):
        super().__init__(cfg, data_type, data, tokenizer=tokenizer)

    def preprocess_data(self, data):
        data = super().preprocess_data(data)
        sids = set(data)
        #self.load_bpp_fpath(sids)
        return data


    def getitem(self, index):
        item = super().getitem(index)
        sid = item['sequence_id']
        n = item['input_len']
        self.get_ext_feature(item)

        return item

class SRFBPPGNNMix():
    def get_adj(self, item):
        adjs = []
        input_len = item['input_len']
        for bpp in item.get('bpp'):
            adjs.append(bpp[:, :, None])
        if self.cfg.with_gnn_bpp:
            item['bpp'] = item['bpp'][0]
        else:
            _ = item.pop('bpp')
        adjs.append(util.get_dist_matrix(input_len))
        for edge in item.pop('edge'):
            _edge = np.zeros([input_len, input_len], dtype=np.float32)
            for e in edge:
                s, e = e
                _edge[s, e] = 1
                _edge[e, s] = 1
            adjs.append(_edge[:, :, None])
        if self.cfg.use_pos:
            pos = item.pop('pos')
            pos = np.abs(pos) + 1
            pos = 1 / pos
            pos = np.stack([pos ** i for i in [1, 2, 4]], axis=2).astype(np.float32)
            pos = pos / pos.sum(axis=1, keepdims=True)
            adjs.append(pos)
        adj = np.concatenate(adjs, axis=-1)
        item['adj'] = adj

    def get_node_feature(self, item):
        sid = item['sequence_id']
        node_feature = []
        for sid2mfe in self.sid2mfe:
            mfe = '.' + sid2mfe[sid] + '.'
            base_type = np.zeros([len(mfe), 3], dtype=np.float32)
            for i, c in enumerate(mfe):
                base_type[i, self.base_type_map[c]] = 1
            node_feature.append(base_type)

        item['node_feature'] = np.concatenate(node_feature, axis=-1)


    def __getitem__(self, index):
        item = super().__getitem__(index)
        self.get_adj(item)
        self.get_node_feature(item)
        return item

class SRFGNNMix(SRFBPPGNNMix):
    def __getitem__(self, index):
        item = super().__getitem__(index)
        if 'adj' not in item:
            self.get_adj(item)
            self.get_node_feature(item)
        item['adj'] = item['adj'][1:-1, 1:-1]
        if 'label_mask' in item:
            item['label_mask'] = item['label_mask'][1:-1]
            item['label'] = item['label'][1:-1]
        if 'bpp' in item:
            item['bpp'] = item['bpp'][1:-1, 1:-1]
        if 'base_type' in item:
            item['base_type'] = item['base_type'][1:-1]
        return item

    def get_node_feature(self, item):
        sid = item['sequence_id']
        mfe = self.sid2mfe[sid]
        node_features = []
        base_type = np.zeros([len(mfe), 3], dtype=np.float32)
        for i, c in enumerate(mfe):
            base_type[i, self.base_type_map[c]] = 1
        node_features.append(base_type)
        sequence = item['sequence']
        token_type = np.zeros([len(sequence), 4], dtype=np.float32)
        for i, c in enumerate(sequence):
            token_type[i, self.token_type_map[c]] = 1
        node_features.append(token_type)
        if self.cfg.use_lt:
            for i, j in enumerate(item.pop('lt')):
                token_type[i, j] = 1
            lt_type = np.zeros([len(mfe), 7], dtype=np.float32)
            node_features.append(lt_type)
        item['node_feature'] = np.concatenate(node_features, axis=-1)
        item['input_len'] = len(sequence)
        _ = item.pop('input_ids', None)

    def collate(self, batch):
        batch = [item for item in batch if item is not None]
        max_input_len = max([item['input_len'] for item in batch])
        bp_inds, bp_labels, edges = [], [], []
        for i, item in enumerate(batch):
            if 'input_ids' in item:
                item['input_ids'] = np.pad(item['input_ids'], ((0, max_input_len-item['input_len'])), "constant")
            if 'base_type' in item:
                item['base_type'] = np.pad(item['base_type'], ((0, max_input_len-item['input_len'])), "constant")
            if 'lt' in item:
                item['lt'] = np.pad(item['lt'], ((0, max_input_len-item['input_len'])), "constant")
            if 'node_feature' in item:
                item['node_feature'] = np.pad(item['node_feature'], ((0, max_input_len-item['input_len']), (0, 0)), "constant")
            if 'label' in item:
                item['label'] = np.pad(item['label'], ((0, max_input_len-item['input_len']), (0, 0)), "constant")
                item['label_mask'] = np.pad(item['label_mask'], ((0, max_input_len-item['input_len']), (0, 0)), "constant")
            if 'label_error' in item:
                item['label_error'] = np.pad(item['label_error'], ((0, max_input_len-item['input_len']), (0, 0)), "constant")
            if 'bpp' in item:
                item['bpp'] = np.pad(item['bpp'], ((0, max_input_len - item['input_len']), (0, max_input_len-item['input_len'])), "constant", constant_values=-1000)
            if 'adj' in item:
                item['adj'] = np.pad(item['adj'], ((0, max_input_len - item['input_len']), (0, max_input_len-item['input_len']), (0, 0)), "constant", constant_values=0)
            if 'pos' in item:
                item['pos'] = np.pad(item['pos'], ((0, max_input_len - item['input_len']), (0, max_input_len - item['input_len'])), "constant", constant_values=0)
            if 'base_pair' in item:
                s = i*max_input_len
                bp_ind, bp_label = zip(*item.pop('base_pair'))
                bp_inds.append(np.array(bp_ind)+s)
                bp_labels.append(np.array(bp_label))
            if 'edge' in item:
                s = i * max_input_len
                edges.append(np.array(item.pop('edge'))+s)

        batch = default_collate(batch)
        batch['mask'] = sequence_mask(batch['input_len'], max_input_len, dtype=torch.int64)
        if len(bp_inds)>0:
            batch['bp_ind'] = torch.from_numpy(np.concatenate(bp_inds))
            batch['bp_label'] = torch.from_numpy(np.concatenate(bp_labels))
        if len(edges)>0:
            batch['edge'] = torch.from_numpy(np.concatenate(edges, axis=0))
        return batch


class PretrainBPPMix():
    def getitem(self, index):
        rec = self.data[index]
        item = dict(index=index, src=rec.src, sequence_id=rec.sequence_id, sequence=rec.sequence)
        seq = self.get_input_ids(rec.sequence, rec.sequence_id)
        seq = [self.oid2new[id] for id in seq]
        item['input_ids'] = seq
        item['input_len'] = len(seq)
        #bpp = util.load_bpp([self.sid2bpp_fpath[rec.sequence_id], len(seq)]).astype(np.float32)
        #item['bpp'] = bpp
        label_mask = np.ones([len(seq)], dtype=np.int64)
        label_mask[:26] = 0
        label_mask[-21:] = 0
        item['label_mask'] = label_mask
        self.get_ext_feature(item)
        if self.cfg.use_mfe and self.sid2mfe[rec.sequence_id]== '':
            return None
        return item

    def preprocess_data(self, data):
        if self.data_type=='train' and self.cfg.use_gen:
            df = pd.read_csv(f'{self.cfg.data_dir}/gen.csv', nrows=self.cfg.num)
            df['SN_filter'] = 1
            df['src'] = 'semi'
            data = pd.concat([df, data])

        sids = set(data.sequence_id.values)
        data = data.to_records(index=False)
        self.sid2rec = {rec.sequence_id: rec for rec in data}
        self.load_bpp_fpath(sids)
        return data

class PretrainMFEMix():
    def getitem(self, index):
        rec = self.data[index]
        item = dict(index=index, sequence_id=rec.sequence_id)
        seq = self.get_input_ids(rec.sequence, rec.sequence_id)
        seq = [self.oid2new[id] for id in seq]
        item['input_ids'] = seq
        item['input_len'] = len(seq)
        self.get_ext_feature(item)
        sid = item['sequence_id']
        mfe = '.' + self.sid2mfe[sid] + '.'
        item['base_pair'] = util.get_base_pair(mfe, bi=self.cfg.use_bi)
        return item

    def preprocess_data(self, data):
        data = data.to_records(index=False)
        self.sid2rec = {rec.sequence_id: rec for rec in data}
        return data


class PretrainMix():
    def __init__(self, cfg, data_type, data, tokenizer=None):
        super().__init__(cfg, data_type, data, tokenizer=tokenizer)

    def preprocess_data(self, data):
        return data.to_records(index=False)

    def convert_seg_ids(self, seg_inds, indices):
        indices2 = []
        for ind in indices:
            inds = np.arange(seg_inds[ind-1]+1, seg_inds[ind]+1)
            indices2.append(inds)
        return np.concatenate(indices2)

    def mlm_seq(self, seq, has_special=True, exclude_indices=None):
        seg_inds = np.where(seq[:-1]!=seq[1:])[0]
        num, shift = len(seg_inds) - 1, 1
        masked_indices, indices_masked, indices_random, indices_replaced = self.get_masked_indices(num, shift=shift, exclude_indices=None)
        masked_indices = self.convert_seg_ids(seg_inds, masked_indices)
        indices_masked = self.convert_seg_ids(seg_inds, indices_masked)
        indices_random = self.convert_seg_ids(seg_inds, indices_random)
        seq[indices_masked] = self.tokenizer.mask_token_id
        seq[indices_random] = np.random.choice(self.rand_token_ids, len(indices_random))

        return seq, masked_indices

    def getitem(self, index):
        rec = self.data[index]
        seq = np.array(self.get_input_ids(rec.sequence, rec.sequence_id))

        label = deepcopy(seq)
        if self.cfg.use_wwm:
            char2token = np.full(len(text), -100, dtype=np.int32)
            last_end = offsets[0][0]
            max_offset = -1
            for i, offset in enumerate(offsets):
                char2token[last_end:offset[1]] = i
                last_end = offset[1]
                if last_end > max_offset:
                    max_offset = last_end
            word_offsets = [x[1] for x in self.word_tokenize(text[:max_offset + 1])]
            if len(word_offsets) > 1:
                seq, masked_indices = self.wwm_seq(deepcopy(seq), word_offsets, char2token)
            else:
                seq, masked_indices = deepcopy(seq), np.empty(0, np.int64)
        else:
            exclude_indices = np.where(np.logical_or(seq == self.tokenizer.cls_token_id, seq == self.tokenizer.sep_token_id))[0]
            seq, masked_indices = self.mlm_seq(seq, exclude_indices=exclude_indices)

        seq = [self.oid2new[id] for id in seq]
        label = [self.oid2new[id] for id in label]
        item = dict()
        item['input_ids'] = seq
        item['input_len'] = len(seq)
        item['mlm_label'] = label
        item['masked_indices'] = masked_indices
        return item

    def collate(self, batch):
        seq_lens = [item['input_len'] for item in batch]
        max_seq_len = max(seq_lens)

        masked_indices = []
        for i, item in enumerate(batch):
            item['input_ids'] = np.pad(item['input_ids'], (0, max_seq_len - item['input_len']), 'constant', constant_values=0)
            item['mlm_label'] = np.pad(item['mlm_label'], (0, max_seq_len - len(item['mlm_label'])), 'constant', constant_values=0)
            masked_indices.append(item.pop('masked_indices') + max_seq_len * i)

        batch = default_collate(batch)
        batch['mask'] = sequence_mask(batch['input_len'], max_seq_len, torch.int32)
        batch['masked_indices'] = torch.tensor(np.concatenate(masked_indices, axis=0)).to(torch.int64)
        return batch




class Dataset(DatasetMix, torch.utils.data.Dataset):
    pass

class Dataset1P(Mix1P, Dataset):
    pass

class BPPDataset(BPPMix, Dataset):
    pass

class SRFBPPGNNDataset(SRFBPPGNNMix, BPPDataset):
    pass

class SRFGNNDataset(SRFGNNMix, SRFBPPGNNDataset):
    pass

class PretrainDataset(PretrainMix, Dataset):
    pass

class PretrainBPPDataset(PretrainBPPMix, Dataset):
    pass

class PretrainSRFBPPGNNDataset(SRFBPPGNNMix, PretrainBPPDataset):
    pass

class PretrainSRFGNNDataset(SRFGNNMix, PretrainBPPDataset):
    pass

class PretrainMFEDataset(PretrainMFEMix, Dataset):
    pass

class TestDataset(Dataset):
    pass


class IterMix(DatasetMix):
    pass
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        data = self.data
        if self.cfg.use_multiple_gpu and hasattr(self, 'world_size'):
            data = self.get_distribute_data(data, self.world_size, self.rank)
        if worker_info is not None:
            self.data = self.get_distribute_data(data, worker_info.num_workers, worker_info.id)
        for i in range(self.cfg.n_repeat):
            for index in range(len(self.data)):
                items = self.get_iter_items(index)
                for item in items:
                    if item is not None:
                        yield item

class IterDataset(IterMix, torch.utils.data.IterableDataset):
    pass

class IterDataset1P(Mix1P, IterDataset):
    pass

class BPPIterDataset(BPPMix, IterDataset):
    pass

class SRFBPPGNNIterDataset(SRFBPPGNNMix, BPPIterDataset):
    pass

class SRFGNNIterDataset(SRFGNNMix, SRFBPPGNNIterDataset):
    pass


def gen_ds(args, data_type, data, **kwargs):
    drop_last, shuffle, num_workers, sampler, batch_size, collate_func = False, False, args.n_dl_worker, None, args.batch_size, None
    if data_type=='train':
        ds_cls = globals()[args.ds_cls]
        drop_last, shuffle = True, True
    else:
        ds_cls = globals()[args.val_ds_cls]
        batch_size = args.val_batch_size
    if args.pretrain:
        ds_cls = globals()[args.pretrain_ds_cls]
    ds = ds_cls(args, data_type, data, **kwargs)
    collate_func = ds.collate
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers, shuffle=shuffle, drop_last=drop_last, collate_fn=collate_func, worker_init_fn=worker_init_fn, sampler=sampler)
    return dl


if __name__ == '__main__':
    from util import *
    from tokenizer import get_tokenizer
    args = parser.parse_args([])
    args.batch_size=2
    args.n_dl_worker = 2
    args.ds_cls = 'Dataset'
    args.val_ds_cls = 'IterDataset'
    args.pretrain_ds_cls = 'PretrainDataset'
    args.pretrain = False
    args.num = 100
    args.max_seq_len = 8
