from os import path
import gc
import math
import warnings
import logging
from argparse import Namespace
import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ProgressBar
from .models.aayan.data import RNA_DM as RNA_DM_Aayan
from .models.aayan.bottle import RNA_Lightning as RNA_Lightning_Aayan
from .models.aayan.utils import collate_preds as collate_preds_aayan
from .models.junseong.data import RNA_DM as RNA_DM_Junseong
from .models.junseong.bottle import RNA_Lightning as RNA_Lightning_Junseong
from .models.junseong.utils import collate_preds as collate_preds_jungseong
from .models.roger.data import RNA_DM as RNA_DM_Roger
from .models.roger.bottle import RNA_Lightning as RNA_Lightning_Roger
from .models.roger.utils import collate_preds as collate_preds_roger
from ...util.progress import get_progress_manager
from ...util.feature_gen import FeatureName
from ...util.data_format import format_input, format_output

SOL_DIR = path.join(path.dirname(__file__), 'model_weights')
NUM_WORKERS = 0

REQUIRED_FEATURES: list[FeatureName] = ['bpps_linearfolde']

logging.getLogger('lightning.pytorch').setLevel(logging.ERROR)

class CustomProgressBar(ProgressBar):
    def __init__(self):
        super().__init__()
        self.enable = True

    def disable(self):
        self.enable = False

    def enable(self):
        self.enable = True

    def on_predict_start(self, trainer, pl_module) -> None:
        self.progress_updater = get_progress_manager().updater(None, 'batch')
        self.progress_updater.__enter__()

    def on_predict_end(self, trainer, pl_module) -> None:
        self.progress_updater.__exit__(None, None, None)

    def on_predict_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        if not self.has_dataloader_changed(dataloader_idx):
            return
        
        x = self.total_predict_batches_current_dataloader
        self.progress_updater.reset(
            None if x is None or math.isinf(x) or math.isnan(x) else x
        )
    
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if self.enable:
            self.progress_updater.update(1)

def infer_aayan(input_df: pd.DataFrame, batch_size: int):
    torch.set_float32_matmul_precision("medium")
    
    ckpt_paths  = [
        f'{SOL_DIR}/ayaan/version_26/epoch=1-step=1034.ckpt',
        f'{SOL_DIR}/ayaan/version_27/epoch=3-step=2072.ckpt',
        f'{SOL_DIR}/ayaan/version_28/epoch=4-step=2585.ckpt',
        f'{SOL_DIR}/ayaan/version_29/epoch=5-step=3102.ckpt',
        f'{SOL_DIR}/ayaan/version_30/epoch=6-step=3626.ckpt',
    ]

    fold_results = []

    def run_fold(ckpt_path, input_df, batch_size):
        ckpt = torch.load(ckpt_path)
        hp = Namespace(**ckpt['hyper_parameters'])
        hp.batch_size = batch_size

        dm = RNA_DM_Aayan(hp, NUM_WORKERS, input_df)
        model = RNA_Lightning_Aayan.load_from_checkpoint(ckpt_path, hp=hp, strict=False)
        model.eval()
        res = pl.Trainer(
            precision='16-mixed',
            accelerator='gpu',
            benchmark=True,
            enable_model_summary=False,
            logger=False,
            callbacks=[CustomProgressBar()]
        ).predict(model, datamodule=dm)

        ids, preds = [], []

        for p, y in res:
            for idx, mask, pi in zip(y['ids'], y['mask'], p):
                ids.append(idx[mask])
                preds.append(pi.clip(0,1)[mask])

        ids = torch.concat(ids)
        preds = torch.concat(preds)

        return pd.DataFrame({
            'id': ids.numpy(),
            'reactivity_DMS_MaP': preds[:,0].numpy(), 
            'reactivity_2A3_MaP': preds[:,1].numpy()
        })

    for ckpt_path in get_progress_manager().iterator(ckpt_paths, desc='fold'):
        fold_results.append(run_fold(ckpt_path, input_df, batch_size))
        gc.collect()
        torch.cuda.empty_cache()

    # Reset to default
    torch.set_float32_matmul_precision("highest")

    ensemble_pred = fold_results[0]
    for pred in fold_results[1:]:
        ensemble_pred['reactivity_DMS_MaP'] = ensemble_pred['reactivity_DMS_MaP'] + pred['reactivity_DMS_MaP']
        ensemble_pred['reactivity_2A3_MaP'] = ensemble_pred['reactivity_2A3_MaP'] + pred['reactivity_2A3_MaP']
    ensemble_pred['reactivity_DMS_MaP'] = ensemble_pred['reactivity_DMS_MaP']/len(fold_results)
    ensemble_pred['reactivity_2A3_MaP'] = ensemble_pred['reactivity_2A3_MaP']/len(fold_results)

    return ensemble_pred

def infer_junseong(input_df: pd.DataFrame, batch_size: int):
    ckpt_paths  = [
        f'{SOL_DIR}/junseong/version_00/epoch=106-step=55212.ckpt',
        f'{SOL_DIR}/junseong/version_01/epoch=184-step=95460.ckpt',
        f'{SOL_DIR}/junseong/version_02/epoch=152-step=78948.ckpt',
        f'{SOL_DIR}/junseong/version_03/epoch=124-step=64625.ckpt',
        f'{SOL_DIR}/junseong/version_04/epoch=78-step=40922.ckpt',
    ]

    fold_results = []

    def run_fold(ckpt_path, input_df, batch_size):
        ckpt = torch.load(ckpt_path)
        hp = Namespace(**ckpt['hyper_parameters'])
        hp.batch_size = batch_size

        dm = RNA_DM_Junseong(hp, NUM_WORKERS, input_df)
        model = RNA_Lightning_Junseong.load_from_checkpoint(ckpt_path, hp=hp, strict=False)
        model.eval()
        res = pl.Trainer(
            precision='16-mixed',
            accelerator="gpu",
            benchmark=True,
            enable_model_summary=False,
            logger=False,
            callbacks=[CustomProgressBar()]
        ).predict(model, datamodule=dm)

        ids, preds = [], None

        if isinstance(res[0], tuple): res = [res]
        for sub in res:
            v = torch.concat([p.clip(0,1)[y['mask']] for p, y in sub])
            preds = v if preds is None else preds + v
        preds = preds / len(res)

        for _, y in res[0]:
            for idx, mask in zip(y['ids'], y['mask']):
                ids.append(idx[mask])

        ids = torch.concat(ids)

        return pd.DataFrame({
            'id': ids.numpy(),
            'reactivity_DMS_MaP': preds[:,0].numpy(), 
            'reactivity_2A3_MaP': preds[:,1].numpy()
        })

    for ckpt_path in get_progress_manager().iterator(ckpt_paths, desc='fold'):
        fold_results.append(run_fold(ckpt_path, input_df, batch_size))
        gc.collect()
        torch.cuda.empty_cache()

    ensemble_pred = fold_results[0]
    for pred in fold_results[1:]:
        ensemble_pred['reactivity_DMS_MaP'] = ensemble_pred['reactivity_DMS_MaP'] + pred['reactivity_DMS_MaP']
        ensemble_pred['reactivity_2A3_MaP'] = ensemble_pred['reactivity_2A3_MaP'] + pred['reactivity_2A3_MaP']
    ensemble_pred['reactivity_DMS_MaP'] = ensemble_pred['reactivity_DMS_MaP']/len(fold_results)
    ensemble_pred['reactivity_2A3_MaP'] = ensemble_pred['reactivity_2A3_MaP']/len(fold_results)

    return ensemble_pred

def infer_roger(input_df: pd.DataFrame, ckpt_paths: list[str], batch_size: int):
    fold_results = []

    def run_fold(ckpt_path, input_df, batch_size):
        ckpt = torch.load(ckpt_path)
        hp = Namespace(**ckpt['hyper_parameters'])
        hp.batch_size = batch_size

        torch._dynamo.reset()
        dm = RNA_DM_Roger(hp, NUM_WORKERS, input_df)
        model = RNA_Lightning_Roger.load_from_checkpoint(ckpt_path, hp=hp, strict=False)
        model.eval()
        res = pl.Trainer(
            precision='16-mixed',
            accelerator='gpu',
            benchmark=True,
            max_epochs=hp.n_epochs,
            gradient_clip_val=hp.grad_clip,
            num_sanity_val_steps=0,
            enable_model_summary=False,
            reload_dataloaders_every_n_epochs=1,
            callbacks=[CustomProgressBar()],
            logger=False,
        ).predict(model, datamodule=dm)

        ids, preds = [], None

        # One of the versions of this model has multiple predictions that need to be averaged
        if isinstance(res[0], tuple): res = [res]
        for sub in res:
            v = torch.concat([p.clip(0,1)[y['mask']] for p, y in sub])
            preds = v if preds is None else preds + v
        preds = preds / len(res)

        for _, y in res[0]:
            for idx, mask in zip(y['ids'], y['mask']):
                ids.append(idx[mask])

        ids = torch.concat(ids)

        return pd.DataFrame({
            'id': ids.numpy(),
            'reactivity_DMS_MaP': preds[:,0].numpy(), 
            'reactivity_2A3_MaP': preds[:,1].numpy()
        })

    for ckpt_path in get_progress_manager().iterator(ckpt_paths, desc='fold'):
        fold_results.append(run_fold(ckpt_path, input_df, batch_size))
        gc.collect()
        torch.cuda.empty_cache()

    ensemble_pred = fold_results[0]
    for pred in fold_results[1:]:
        ensemble_pred['reactivity_DMS_MaP'] = ensemble_pred['reactivity_DMS_MaP'] + pred['reactivity_DMS_MaP']
        ensemble_pred['reactivity_2A3_MaP'] = ensemble_pred['reactivity_2A3_MaP'] + pred['reactivity_2A3_MaP']
    ensemble_pred['reactivity_DMS_MaP'] = ensemble_pred['reactivity_DMS_MaP']/len(fold_results)
    ensemble_pred['reactivity_2A3_MaP'] = ensemble_pred['reactivity_2A3_MaP']/len(fold_results)

    return ensemble_pred

def infer(sequences: str | list[str] | pd.DataFrame, batch_size=128):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', '.*does not have many workers.*')
        warnings.filterwarnings('ignore', '.*dropout option adds dropout after all but last recurrent layer.*')
        warnings.filterwarnings('ignore', ".*'has_cuda' is deprecated.*")
        warnings.filterwarnings('ignore', ".*'has_cudnn' is deprecated.*")
        warnings.filterwarnings('ignore', ".*'has_mps' is deprecated.*")
        warnings.filterwarnings('ignore', ".*'has_mkldnn' is deprecated.*")

        input_df = format_input(sequences)

        preds = []
        with get_progress_manager().updater(total=3, desc='k5_rlj submodels') as pbar:
            preds.append(infer_aayan(input_df.copy(deep=False), batch_size))
            pbar.update(1)

            # The original submission described two models from junseong, but it provided results
            # closer to the original submission to omit the junseong part instead of including
            # a model using the provided re-trained weights
            #preds.append(infer_junseong(input_df.copy(deep=False), batch_size))
            #pbar.update(1)

            roger_ckpt_a = [
                f'{SOL_DIR}/roger/fold_0_epoch18-step9500.ckpt',
                f'{SOL_DIR}/roger/fold_1_epoch14-step7500.ckpt',
                f'{SOL_DIR}/roger/fold_2_epoch26-step13500.ckpt',
                f'{SOL_DIR}/roger/fold_3_epoch14-step7500.ckpt',
                f'{SOL_DIR}/roger/fold_4_epoch19-step10000.ckpt',
            ]

            preds.append(infer_roger(input_df.copy(deep=False), roger_ckpt_a, batch_size))
            pbar.update(1)

            roger_ckpt_b = [
                f'{SOL_DIR}/roger/e31v05.ckpt',
                f'{SOL_DIR}/roger/e31v06.ckpt',
                f'{SOL_DIR}/roger/e31v07.ckpt',
                f'{SOL_DIR}/roger/e31v08.ckpt',
                f'{SOL_DIR}/roger/e31v09.ckpt',
            ]

            preds.append(infer_roger(input_df.copy(deep=False), roger_ckpt_b, batch_size))
            pbar.update(1)

        ensemble_pred = preds[0]
        for pred in preds[1:]:
            ensemble_pred['reactivity_DMS_MaP'] = ensemble_pred['reactivity_DMS_MaP'] + pred['reactivity_DMS_MaP']
            ensemble_pred['reactivity_2A3_MaP'] = ensemble_pred['reactivity_2A3_MaP'] + pred['reactivity_2A3_MaP']
        ensemble_pred['reactivity_DMS_MaP'] = ensemble_pred['reactivity_DMS_MaP']/len(preds)
        ensemble_pred['reactivity_2A3_MaP'] = ensemble_pred['reactivity_2A3_MaP']/len(preds)

        format_output(input_df, ensemble_pred)

        return ensemble_pred
