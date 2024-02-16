import warnings
from os import path
import gc
from glob import glob
from types import SimpleNamespace
from contextlib import nullcontext
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from .models.dataset import DatasetType1, DatasetType2, DeviceDataLoader
from .models.monnu.exp112 import RibonanzaLightningModel as RLModel112
from .models.monnu.exp300 import RibonanzaLightningModel as RLModel300
from .models.monnu.exp302 import RibonanzaLightningModel as RLModel302
from .models.monnu.exp312 import RibonanzaLightningModel as RLModel312
from .models.monnu.exp317 import RibonanzaLightningModel as RLModel317
from .models.tattaka.exp064 import RibonanzaLightningModel as RLModel064
from .models.tattaka.exp070 import RibonanzaLightningModel as RLModel070
from .models.tattaka.exp071 import RibonanzaLightningModel as RLModel071
from .models.tattaka.exp072 import RibonanzaLightningModel as RLModel072
from .models.yu4u.model import RNAModel as Type2Model
from ...util.progress import get_progress_manager
from ...util.feature_gen import FeatureName
from ...util.data_format import format_input, format_output

REQUIRED_FEATURES: list[FeatureName] = ['bpps_eternafold', 'bpps_contrafold', 'mea_eternafold_bpps']

def infer_type1(input_df: pd.DataFrame, Model: pl.LightningModule, checkpoint_dir: str, batch_size: int):
    fold_results = []

    def run_fold(ckpt_path, input_df, batch_size):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        model = Model.load_from_checkpoint(ckpt_path).model_ema.module
        model = model.to(device)
        model = model.eval()

        ds = DatasetType1(input_df)
        dl = DeviceDataLoader(
            DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=0,
            ),
            device,
        )

        ids, preds = [], []
        with torch.no_grad(), torch.cuda.amp.autocast():
            for x, y in get_progress_manager().iterator(dl, desc='batch'):
                p = torch.nan_to_num(model(x)).clip(0, 1)
                for idx, mask, pi in zip(y['ids'].cpu(), x['mask'].cpu(), p.cpu()):
                    ids.append(idx[mask])
                    preds.append(pi[mask[: pi.shape[0]]])

        ids = torch.concat(ids)
        preds = torch.concat(preds)

        return pd.DataFrame({
            'id': ids.numpy(), 
            'reactivity_DMS_MaP': preds[:, 1].numpy(),
            'reactivity_2A3_MaP': preds[:, 0].numpy()
        })

    ckpt_paths = glob(f'{checkpoint_dir}/fold*/**/*.ckpt', recursive=True)
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

def infer_type2(input_df: pd.DataFrame, checkpoint_dir: str, batch_size: int):
    fold_results = []

    def run_fold(ckpt_path, input_df, batch_size):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        checkpoint = torch.load(str(ckpt_path), map_location="cpu")
        state_dict = {k[6:]: v for k, v in checkpoint["state_dict"].items() if k.startswith("model.")}

        cfg = SimpleNamespace(**{})
        cfg.model = SimpleNamespace(**{})
        cfg.model.arch = 'transformer'
        cfg.model.num_layers = 12
        cfg.model.hidden_dim = 192
        cfg.model.nhead = 6
        cfg.model.drop_path_rate = 0.2
        cfg.model.norm_first = True
        cfg.model.ema = False
        cfg.model.swa = False
        cfg.model.freeze_backbone = False
        cfg.model.freeze_end_epoch = 16
        cfg.task = SimpleNamespace(**{})
        cfg.task.mode = 'train'
        cfg.task.pseudo = False
        cfg.task.oof = False
        cfg.task.sn_th = 1.0
        cfg.task.ngram = 1

        model = Type2Model(cfg)
        model = model.to(device)
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        ds = DatasetType2(cfg, input_df)
        dl = DeviceDataLoader(
            DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=0,
                collate_fn=None,
            ),
            device,
        )

        ids, preds = [], []

        with torch.no_grad(), torch.cuda.amp.autocast(dtype=(torch.bfloat16)) if 'fold0' in ckpt_path else nullcontext():
            for x, y in get_progress_manager().iterator(dl, desc='batch'):
                p = model(x).float().clip(0, 1)

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

    ckpt_paths = glob(f'{checkpoint_dir}/**/*.ckpt', recursive=True)
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
        warnings.filterwarnings('ignore', '.*Support for mismatched key_padding_mask and attn_mask is deprecated.*')

        input_df = format_input(sequences)

        model_dir = path.join(path.dirname(__file__), 'model_weights')

        preds = []

        type1_args = [
            (RLModel112, path.join(model_dir, 'monnu/exp112_finetune/conv_gnn_l12')),
            (RLModel300, path.join(model_dir, 'monnu/exp300_finetune/conv_gnn_l12')),
            (RLModel302, path.join(model_dir, 'monnu/exp302_finetune/conv_gnn_l12')),
            (RLModel312, path.join(model_dir, 'monnu/exp312_finetune/conv_gnn_l12')),
            (RLModel317, path.join(model_dir, 'monnu/exp317_finetune/conv_gnn_l12')),
            (RLModel064, path.join(model_dir, 'tattaka/exp064_finetune/biased_conv_transformer_192_12_32_7_ALiBi_swiGLU_ep300_lr1e_3_bs256')),
            (RLModel070, path.join(model_dir, 'tattaka/exp070_finetune/biased_conv_transformer_192_12_32_7_RMSNorm_ALiBi_swiGLU_ep300_lr1e_3_bs256')),
            # exp070_tiny
            (RLModel070, path.join(model_dir, 'tattaka/exp070_finetune/biased_conv_transformer_128_8_16_7_RMSNorm_ALiBi_swiGLU_ep300_lr2e_3_bs256')),
            (RLModel071, path.join(model_dir, 'tattaka/exp071_finetune/biased_conv_transformer_192_12_24_7_RMSNorm_ALiBi_GeGLU_ep300_lr1e_3_bs256')),
            (RLModel072, path.join(model_dir, 'tattaka/exp072_finetune/biased_conv_transformer_192_12_24_7_RMSNorm_ALiBi_GeGLU_ep300_lr1e_3_bs256')),
        ]
        type2_checkpoint_dirs = [
            path.join(model_dir, 'yu4u/rna_models2')
            # yu4u_yu4upl2's weights for both DMS and 2A3 were 0, so no need to evaluate
            # path.join(WEIGHTS_DIR, 'yu4u/rna_models_pl'),
        ]

        rdms_weights = [
            0.0399,
            0.0272,
            0.0619,
            0.0303,
            0.0362,
            0.1422,
            0.1806,
            0.0800,
            0.1205,
            0.1976,
            0.0849
        ]
        r2a3_weights = [
            0.0481,
            0.0247,
            0.0992,
            0.0444,
            0.0406,
            0.1147,
            0.1979,
            0.0000,
            0.0512,
            0.2316,
            0.1488
        ]

        with get_progress_manager().updater(total=11, desc='k4_gorna submodels') as pbar:
            for args in type1_args:
                preds.append(infer_type1(input_df.copy(deep=False), *args, batch_size))
                pbar.update(1)

            for cpdir in type2_checkpoint_dirs:
                preds.append(infer_type2(input_df.copy(deep=False), cpdir, batch_size))
                pbar.update(1)

        ensemble_pred = preds[0].loc[:, ['id']]
        ensemble_pred['reactivity_DMS_MaP'] = sum([ pred['reactivity_DMS_MaP'] * weight for pred, weight in zip(preds, rdms_weights)]) / sum(rdms_weights)
        ensemble_pred['reactivity_2A3_MaP'] = sum([ pred['reactivity_2A3_MaP'] * weight for pred, weight in zip(preds, r2a3_weights)]) / sum(r2a3_weights)
        
        format_output(input_df, ensemble_pred)

        return ensemble_pred
