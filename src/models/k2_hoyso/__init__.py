from typing import Union
import warnings
import gc
import glob
from os import path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from .model.model import RNARegModel
from .model.dataset import RNADataset, collate_fn
from ...util.progress import get_progress_manager
from ...util.feature_gen import FeatureName
from ...util.data_format import format_input, format_output
from ...util.torch import DeviceDataLoader

REQUIRED_FEATURES: list[FeatureName] = ['bpps_linearfolde', 'sstype_capr', 'mfe_eternafold', 'sstype_bprna_eternafold']

def infer_model(path: str, input_df: pd.DataFrame, batch_size: int):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = RNARegModel(192, 12, 4)
    model.load_state_dict(torch.load(path)['model'])
    model = model.to(device)
    model.eval()

    ds = RNADataset(input_df, mode='test', SN_filter=False)
    dl = DeviceDataLoader(
        DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            sampler=None,
            num_workers=0,
            drop_last=False,
            pin_memory=True,
            collate_fn=collate_fn
        ),
        device
    )

    ids, preds = [], []

    with torch.no_grad(), torch.cuda.amp.autocast():
        for x, y in get_progress_manager().iterator(dl, desc='batch'):
            p = model(x)

            for idx, mask, pi in zip(y['ids'].cpu(), x['mask'].cpu(), p['react_pred'].cpu()):
                ids.append(idx[mask])
                preds.append(pi[mask[:pi.shape[0]]])
    
    ids = torch.concat(ids)
    preds = torch.concat(preds)

    return pd.DataFrame({
        'id': ids.numpy(),
        'reactivity_DMS_MaP': preds[:,1].numpy(), 
        'reactivity_2A3_MaP': preds[:,0].numpy()
    })

def infer(sequences: Union[str, list[str], pd.DataFrame], batch_size=128):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', ".*'has_cuda' is deprecated.*")
        warnings.filterwarnings('ignore', ".*'has_cudnn' is deprecated.*")
        warnings.filterwarnings('ignore', ".*'has_mps' is deprecated.*")
        warnings.filterwarnings('ignore', ".*'has_mkldnn' is deprecated.*")

        input_df = format_input(sequences)
        subs = []

        for p in get_progress_manager().iterator(glob.glob(f'{path.dirname(__file__)}/model_weights/*.pth'), desc='k2_hoyso submodels'):
            sub = infer_model(p, input_df.copy(deep=False), batch_size)
            subs.append(sub)
            gc.collect()
            torch.cuda.empty_cache()
        
        N = len(subs)
        weights = [1./N for _ in subs]

        init = False
        for df, w in zip(subs, weights):
            if not init:
                df_len = len(df)
                p_DMS = np.zeros((df_len,), dtype=np.float32)
                p_2A3 = np.zeros((df_len,), dtype=np.float32)
                init = True
            
            p_DMS += df['reactivity_DMS_MaP'].values * w
            p_2A3 += df['reactivity_2A3_MaP'].values * w
            del df
        
        p_DMS = np.clip(p_DMS, 0, 1)
        p_2A3 = np.clip(p_2A3, 0, 1)
        
        ensemble_pred = pd.DataFrame({
            'id': np.arange(0, df_len, 1), 
            'reactivity_DMS_MaP': p_DMS, 
            'reactivity_2A3_MaP': p_2A3
        })

        format_output(input_df, ensemble_pred)

        return ensemble_pred
