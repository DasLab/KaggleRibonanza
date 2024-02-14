import itertools
import numpy as np
import torch
from torch.utils.data import Dataset
from ....util.feature_gen import get_feature

class DatasetType1(Dataset):
    def __init__(self, df, mask_only=False, **kwargs):
        self.seq_map = {"A": 0, "C": 1, "G": 2, "U": 3}
        self.structure_map = {".": 1, "(": 2, ")": 3}  # Add
        df["L"] = df.sequence.apply(len)
        self.Lmax = df["L"].max()
        self.df = df
        self.mask_only = mask_only

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        id_min, id_max, seq = self.df.loc[
            idx, ["id_min", "id_max", "sequence"]
        ]
        mask = torch.zeros(self.Lmax, dtype=torch.bool)
        L = len(seq)
        mask[:L] = True
        if self.mask_only:
            return {"mask": mask}, {}
        ids = np.arange(id_min, id_max + 1)

        bp_matrix = np.float32(get_feature('bpps_eternafold', seq))
        bp_matrix = np.pad(
            bp_matrix,
            ((0, self.Lmax - len(bp_matrix)), (0, self.Lmax - len(bp_matrix))),
        )  # Add
        bp_matrix_contrafold = np.float32(get_feature('bpps_contrafold', seq))
        bp_matrix_contrafold = np.pad(
            bp_matrix_contrafold,
            ((0, self.Lmax - len(bp_matrix_contrafold)), (0, self.Lmax - len(bp_matrix_contrafold))),
        )  # Add
        structure = get_feature('mea_eternafold_bpps', seq)
        structure = [self.structure_map[s] for s in structure]  # Add
        structure = np.array(structure)  # Add
        structure = np.pad(structure, (0, self.Lmax - len(structure)))  # Add
        ids = np.pad(ids, (0, self.Lmax - L), constant_values=-1)

        seq = [self.seq_map[s] for s in seq]
        seq = np.array(seq)
        seq = np.pad(seq, (0, self.Lmax - L))

        return {
            "seq": torch.from_numpy(seq),
            "mask": mask,
            "bp_matrix": torch.from_numpy(bp_matrix),  # Add
            "bp_matrix_contrafold": torch.from_numpy(bp_matrix_contrafold),
            "structure": torch.from_numpy(structure),  # Add
        }, {"ids": ids}

class DatasetType2(Dataset):
    def __init__(self, cfg, df):
        df["L"] = df.sequence.apply(len)
        self.Lmax = df["L"].max()
        self.df = df

        '''
        df = pd.read_parquet(parquet_path)
        self.cfg = cfg
        self.mode = mode
        fold_id = cfg.data.fold_id
        fold_num = cfg.data.fold_num
        '''
        self.cfg = cfg

        if cfg.task.ngram == 1:
            self.seq_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
        else:
            self.seq_map = {k: i for i, k in enumerate(itertools.product("ACGUX", repeat=cfg.task.ngram))}

        '''
        if mode != "test":
            df_2A3 = df.loc[df.experiment_type == "2A3_MaP"]
            df_DMS = df.loc[df.experiment_type == "DMS_MaP"]

            split = list(KFold(n_splits=fold_num, random_state=42,
                            shuffle=True).split(df_2A3))[fold_id][0 if mode == 'train' else 1]
            df_2A3 = df_2A3.iloc[split].reset_index(drop=True)
            df_DMS = df_DMS.iloc[split].reset_index(drop=True)

            if not cfg.task.oof:
                if mode == "train":
                    sn_th = cfg.task.sn_th
                    m = (df_2A3["signal_to_noise"].values > sn_th) & (df_DMS["signal_to_noise"].values > sn_th)
                else:
                    m = (df_2A3['SN_filter'].values > 0) & (df_DMS['SN_filter'].values > 0)

                df_2A3 = df_2A3.loc[m].reset_index(drop=True)
                df_DMS = df_DMS.loc[m].reset_index(drop=True)

            self.seq = df_2A3['sequence'].values
            self.seq_id = df_2A3['sequence_id'].values
            self.bpp_dir = data_root.joinpath("Ribonanza_bpp_files", "extra_data")
            bpp_df = pd.read_csv(data_root.joinpath("train_seq_id_to_bpp_path.csv"))
            merged_df = pd.merge(df_2A3, bpp_df, on="sequence_id", how="left")
            self.bpp_paths = merged_df["bpp_path"].values

            self.react_2A3 = df_2A3[[c for c in df_2A3.columns if "reactivity_0" in c]].values
            self.react_DMS = df_DMS[[c for c in df_DMS.columns if "reactivity_0" in c]].values
            return
        '''

        self.seq = df["sequence"].values
        self.seq_id = df["sequence_id"].values

    def __len__(self):
        return len(self.seq)

    def encode_seq(self, seq):
        seq_len = len(seq)
        n = self.cfg.task.ngram
        seq = "X" * (n // 2) + seq + "X" * (n // 2)
        return [self.seq_map[tuple(seq[i:i + n])] for i in range(seq_len)]

    def __getitem__(self, idx):
        seq = self.seq[idx]

        if self.cfg.task.ngram == 1:
            seq = [self.seq_map[s] for s in seq]
        else:
            seq = self.encode_seq(seq)

        seq = np.array(seq)
        mask = torch.zeros(self.Lmax, dtype=torch.bool)
        mask[:len(seq)] = True
        seq = np.pad(seq, (0, self.Lmax - len(seq)), constant_values=5 ** self.cfg.task.ngram - 1)

        '''
        if self.mode != 'test':
            react = torch.from_numpy(np.stack([self.react_2A3[idx],
                                            self.react_DMS[idx]], -1))

            if self.cfg.task.pseudo:
                react = np.pad(react, ((0, self.Lmax - len(react)), (0, 0)), constant_values=np.nan)

            bpp_path = self.bpp_dir.joinpath(self.bpp_paths[idx])
            df = pd.read_csv(bpp_path, sep=" ", header=None)
            bpp_matrix = np.eye(self.Lmax, dtype=np.float32)

            for i, j, v in df.values:
                bpp_matrix[int(i) - 1, int(j) - 1] = v
                bpp_matrix[int(j) - 1, int(i) - 1] = v

            x = {'seq': torch.from_numpy(seq), 'mask': mask, "bpp": bpp_matrix}
            y = {'react': react, 'mask': mask}

            if self.cfg.task.oof:
                x["seq_id"] = self.seq_id[idx]

            return x, y
        '''

        base_matrix = np.half(get_feature('bpps_eternafold', self.seq[idx]))
        base_matrix = np.pad(
            base_matrix,
            ((0, self.Lmax - len(base_matrix)), (0, self.Lmax - len(base_matrix))),
        ) 

        bpp_matrix = np.eye(self.Lmax, dtype=np.float32)
        np.copyto(bpp_matrix, base_matrix, where=base_matrix != 0)

        id_min, id_max, L = self.df.loc[
            idx, ["id_min", "id_max", "L"]
        ]
        ids = np.arange(id_min, id_max + 1)
        ids = np.pad(ids, (0, self.Lmax - L), constant_values=-1)

        return {'seq': torch.from_numpy(seq), 'mask': mask, "bpp": bpp_matrix}, {"ids": ids}

def dict_to(x, device="cuda"):
    return {k: x[k].to(device) for k in x}


def to_device(x, device="cuda"):
    return tuple(dict_to(e, device) for e in x)


class DeviceDataLoader:
    def __init__(self, dataloader, device="cuda"):
        self.dataloader = dataloader
        self.device = device

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        for batch in self.dataloader:
            yield tuple(dict_to(x, self.device) for x in batch)
