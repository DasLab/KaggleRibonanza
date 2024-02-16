from contextlib import nullcontext
from os import path
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from .model.model import RNAdjNetBrk, RNA_Dataset_Test, RNAdjNetBrkReactive, RNA_Dataset_Test_Reactive
from ...util.progress import get_progress_manager
from ...util.feature_gen import FeatureName
from ...util.data_format import format_input, format_output
from ...util.torch import DeviceDataLoader

REQUIRED_FEATURES: list[FeatureName] = ['mfe_eternafold', 'mea_ipknot', 'bpps_eternafold']

def infer_standard(
    input_df: pd.DataFrame,
    model_path: str,
    batch_size: int,
    use_ss: bool = False,
    use_se: bool = False,
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = RNAdjNetBrk(
        positional_embedding='dyn',
        brk_names=['eterna', 'ipknot'] if use_ss else [],
        depth=12,
        num_convs=12,
        adj_ks=3,
        not_slice=False,
        use_se=use_se
    )
    model.load_state_dict(torch.load(model_path, map_location='cpu')['model'])
    model = model.to(device)
    model = model.eval()

    ds = RNA_Dataset_Test(input_df)
    dl = DeviceDataLoader(
        DataLoader(
            ds, 
            batch_size=batch_size, 
            shuffle=False,
            drop_last=False, 
            num_workers=0
        ),
        device
    )

    ids, preds = [],[]

    with torch.no_grad(), torch.cuda.amp.autocast() if torch.cuda.is_available() else nullcontext():
        for x, y in get_progress_manager().iterator(dl, desc='batch'):
            p = model(x).clip(0,1)

            for idx, mask, pi in zip(y['ids'].cpu(), x['mask'].cpu(), p.cpu()):
                ids.append(idx[mask])
                preds.append(pi[mask[:pi.shape[0]]])

    ids = torch.concat(ids)
    preds = torch.concat(preds)

    return pd.DataFrame({
        'id': ids.numpy(),
        'reactivity_DMS_MaP': preds[:,1].numpy(), 
        'reactivity_2A3_MaP': preds[:,0].numpy()
    })

def infer_reactive(
    input_df: pd.DataFrame,
    model_path: str,
    reactive_df: pd.DataFrame,
    batch_size: int,
):
    seqid_map = {}
    for i in input_df.itertuples():
        for idx in range(i.id_min, i.id_max + 1):
            seqid_map[idx] = i.sequence_id

    reactive_df['seq_id'] = reactive_df.id.apply(lambda x: seqid_map[x])
    Lmax = input_df.sequence.apply(len).max()

    pred_data = {i: np.full((2,Lmax), fill_value=np.nan) for i in input_df.sequence_id}
    coords_data = {i.sequence_id: (i.id_min, i.id_max) for i in input_df.itertuples()}

    for data in reactive_df.itertuples():
        seq_id = data.seq_id
        coords = coords_data[seq_id]
        pos_idx = data.id - coords[0]
    
        pred_data[seq_id][0][pos_idx] = data.reactivity_DMS_MaP
        pred_data[seq_id][1][pos_idx] = data.reactivity_2A3_MaP

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = RNAdjNetBrkReactive(
        positional_embedding='dyn',
        depth=12, 
        num_convs=12,
        adj_ks=3,
        not_slice=False,
    )
    model.load_state_dict(torch.load(model_path, map_location='cpu')['model'])
    model = model.to(device)
    model.eval()

    ds = RNA_Dataset_Test_Reactive(
        input_df,
        pred_data,
        pred_mode='dms_2a3'
    )
    dl = DeviceDataLoader(
        DataLoader(
            ds, 
            batch_size=batch_size, 
            shuffle=False,
            drop_last=False, 
            num_workers=0
        ),
        device
    )

    ids, preds = [], []

    with torch.no_grad(), torch.cuda.amp.autocast() if torch.cuda.is_available() else nullcontext():
        for x, y in get_progress_manager().iterator(dl, desc='batch'):
            p = model(x).clip(0,1)

            for idx, mask, pi in zip(y['ids'].cpu(), x['mask'].cpu(), p.cpu()):
                ids.append(idx[mask])
                preds.append(pi[mask[:pi.shape[0]]])

    ids = torch.concat(ids)
    preds = torch.concat(preds)

    return pd.DataFrame({
        'id':ids.numpy(),
        'reactivity_2A3_MaP':preds[:].numpy()
    })

def infer(sequences: str | list[str] | pd.DataFrame, batch_size=128):
    input_df = format_input(sequences)

    model_dir = path.join(path.dirname(__file__), 'model_weights')

    preds: list[pd.DataFrame] = []

    with get_progress_manager().updater(total=28, desc='k1_vigg submodels') as pbar:
        # models trained on kfold 1000 split without se-blocks and brackets
        for i in range(0, 10):
            preds.append(infer_standard(input_df.copy(deep=False), path.join(model_dir, 'weights_thousands', f'model_{i}.pth'), batch_size))
            pbar.update(1)

            gc.collect()
            torch.cuda.empty_cache()

        # models trained on kfold 1000 split with se-blocks and without brackets
        for i in range(0, 15):
            preds.append(infer_standard(input_df.copy(deep=False), path.join(model_dir, 'se_thousands', f'model_{i}.pth'), batch_size, use_se=True))
            pbar.update(1)

            gc.collect()
            torch.cuda.empty_cache()

        # model trained on split by length without se-blocks and without brackets
        preds.append(infer_standard(input_df.copy(deep=False), path.join(model_dir, 'lengths', 'model_0.pth'), batch_size))
        pbar.update(1)

        gc.collect()
        torch.cuda.empty_cache()

        # model trained on split by length without se-blocks and with brackets
        preds.append(infer_standard(input_df.copy(deep=False), path.join(model_dir, 'brks_lengths', 'model_0.pth'), batch_size, use_ss=True))
        pbar.update(1)

        gc.collect()
        torch.cuda.empty_cache()

        # calculates main prediction by averaging individual predictions made by each model
        ensemble_pred = preds[0]
        for pred in preds[1:]:
            ensemble_pred['reactivity_DMS_MaP'] = ensemble_pred['reactivity_DMS_MaP'] + pred['reactivity_DMS_MaP']
            ensemble_pred['reactivity_2A3_MaP'] = ensemble_pred['reactivity_2A3_MaP'] + pred['reactivity_2A3_MaP']
        ensemble_pred['reactivity_DMS_MaP'] = ensemble_pred['reactivity_DMS_MaP']/len(preds)
        ensemble_pred['reactivity_2A3_MaP'] = ensemble_pred['reactivity_2A3_MaP']/len(preds)

        # predicts and adds correction predicted by dms-to-2a3 model
        reacive_pred = infer_reactive(input_df.copy(deep=False), path.join(model_dir, 'dms2a3', 'model_0.pth'), ensemble_pred.copy(deep=False), batch_size)
        pbar.update(1)

        a = len(preds)/(len(preds) + 1)
        b = 1/(len(preds) + 1)

        ensemble_pred['reactivity_2A3_MaP'] = a * ensemble_pred['reactivity_2A3_MaP'] + b * reacive_pred['reactivity_2A3_MaP']

        format_output(input_df, ensemble_pred)

        return ensemble_pred
