import warnings
import gc
from os import path
from collections import OrderedDict
from types import SimpleNamespace
import pandas as pd
import torch
from torch.utils.data import DataLoader
from .models.squeezeformer import CustomDataset as SequeezeformerDataset, Net as SqueezeformerNet
from .models.twintower import TestDataset as TwintowerDataset, Net as TwintowerNet, collate_fn as twintower_collate_fn
from ...util.progress import get_progress_manager
from ...util.feature_gen import FeatureName
from ...util.format_input import format_input
from ...util.torch import DeviceDataLoader

REQUIRED_FEATURES: list[FeatureName] = ['bpps_linearfolde']

def infer_sequeezeformer(input_df: pd.DataFrame, batch_size: int):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    weights_path = path.join(path.dirname(__file__), 'model_weights/cfg_1/squeezeformer_pretrained.pt')
    model_state_dict = torch.load(weights_path, map_location=torch.device('cpu'))['model']

    # Remove '_orig_mod.' prefix from state_dict keys
    new_state_dict = OrderedDict()
    for key, value in model_state_dict.items():
        new_key = key.replace('_orig_mod.', '')
        new_state_dict[new_key] = value

    cfg = SimpleNamespace(**{})
    cfg.d_model = 192
    cfg.encoder_config = SimpleNamespace(**{})
    cfg.encoder_config.input_dim=192
    cfg.encoder_config.encoder_dim=192
    cfg.encoder_config.num_layers=14
    cfg.encoder_config.num_attention_heads= 6
    cfg.encoder_config.feed_forward_expansion_factor=4
    cfg.encoder_config.conv_expansion_factor= 2
    cfg.encoder_config.input_dropout_p= 0.1
    cfg.encoder_config.feed_forward_dropout_p= 0.1
    cfg.encoder_config.attention_dropout_p= 0.1
    cfg.encoder_config.conv_dropout_p= 0.1
    cfg.encoder_config.conv_kernel_size= 51

    model = SqueezeformerNet(cfg)
    model = model.to(device)
    model.load_state_dict(new_state_dict)
    model.eval()

    dataset = SequeezeformerDataset(input_df)
    dl = DeviceDataLoader(
        DataLoader(
            dataset,
            shuffle=False,
            drop_last=False,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=False,
        ),
        device
    )

    ids, preds = [],[]

    with torch.no_grad(), torch.cuda.amp.autocast():
        for x, y in get_progress_manager().iterator(dl, desc='batch'):
            p = model(x).clip(0,1)
                
            for idx, mask, pi in zip(y['ids'].cpu(), x['input_mask'].cpu(), p.cpu()):
                ids.append(idx[mask])
                preds.append(pi[mask[:pi.shape[0]]])

    ids = torch.concat(ids)
    preds = torch.concat(preds)

    return pd.DataFrame({
        'id': ids.numpy(),
        'reactivity_DMS_MaP': preds[:,1].numpy(),
        'reactivity_2A3_MaP': preds[:,0].numpy()
    })

def infer_twintower(input_df: pd.DataFrame, batch_size: int):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    weights_path = path.join(path.dirname(__file__), 'model_weights/cfg_2/twintower_pretrained.pt')
    model_state_dict = torch.load(weights_path, map_location='cpu')['model']

    # Remove '_orig_mod.' prefix from state_dict keys
    new_state_dict = OrderedDict()
    for key, value in model_state_dict.items():
        new_key = key.replace('_orig_mod.', '')
        new_state_dict[new_key] = value

    cfg = SimpleNamespace(**{})
    cfg.model = 'mdl_2_twintower'
    cfg.msa_depth = 64
    cfg.d_model = 128
    cfg.ce_ignore_index = -100
    cfg.padding_index = 0
    cfg.vocab_size = 5
    cfg.msa_embedder = SimpleNamespace(**{})
    cfg.msa_embedder.c_m = 128
    cfg.msa_embedder.c_z = 64
    cfg.msa_embedder.rna_fm = True
    cfg.chemformer_stack = SimpleNamespace(**{})
    cfg.chemformer_stack.blocks_per_ckpt =  1,
    cfg.chemformer_stack.c_m = 128
    cfg.chemformer_stack.c_z = 64
    cfg.chemformer_stack.c_hidden_msa_att = 32
    cfg.chemformer_stack.c_hidden_opm = 32
    cfg.chemformer_stack.c_hidden_mul = 64
    cfg.chemformer_stack.c_hidden_pair_att = 32
    cfg.chemformer_stack.c_s = 384
    cfg.chemformer_stack.no_heads_msa = 8
    cfg.chemformer_stack.no_heads_pair = 8
    cfg.chemformer_stack.no_blocks = 8
    cfg.chemformer_stack.transition_n = 4
    cfg.bpp_head = SimpleNamespace(**{})
    cfg.bpp_head.c_in = 64
    cfg.ss_head = SimpleNamespace(**{})
    cfg.ss_head.c_in = 384
    cfg.ss_head.c_hidden = 384
    cfg.plddt_head = SimpleNamespace(**{})
    cfg.plddt_head.c_in = 384
    cfg.plddt_head.no_bins = 50

    model = TwintowerNet(cfg)
    model = model.to(device)
    model.load_state_dict(new_state_dict)
    model.eval()

    dataset = TwintowerDataset(input_df)
    dl = DeviceDataLoader(
        DataLoader(
            dataset,
            shuffle=False,
            drop_last=False,
            batch_size=batch_size,
            collate_fn=twintower_collate_fn,
            num_workers=0,
            pin_memory=False,
        ),
        device
    )

    ids, preds = [], []

    with torch.no_grad(), torch.cuda.amp.autocast():
        for x, y in get_progress_manager().iterator(dl, desc='batch'):
            p = model(x).clip(0,1)
                
            for idx, mask, pi in zip(y['ids'].cpu(), x['mask_tokens'].bool().cpu(), p.cpu()):
                ids.append(idx[mask])
                preds.append(pi[mask[:pi.shape[0]]])

    ids = torch.concat(ids)
    preds = torch.concat(preds)

    df = pd.DataFrame({
        'id':ids.numpy(),
        'reactivity_DMS_MaP':preds[:,1].numpy(), 
        'reactivity_2A3_MaP':preds[:,0].numpy()
    })

    return df

def infer(sequences: str | list[str] | pd.DataFrame, batch_size_squeezeformer=128, batch_size_twintower=8):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', ".*'has_cuda' is deprecated.*")
        warnings.filterwarnings('ignore', ".*'has_cudnn' is deprecated.*")
        warnings.filterwarnings('ignore', ".*'has_mps' is deprecated.*")
        warnings.filterwarnings('ignore', ".*'has_mkldnn' is deprecated.*")

        input_df = format_input(sequences)

        with get_progress_manager().updater(total=2, desc='k3_aek submodels') as pbar:
            squeezeformer_pred = infer_sequeezeformer(input_df.copy(deep=False), batch_size_squeezeformer)
            pbar.update(1)
            gc.collect()
            torch.cuda.empty_cache()

            twintower_pred = infer_twintower(input_df.copy(deep=False), batch_size_twintower)
            pbar.update(1)
            gc.collect()
            torch.cuda.empty_cache()

        ensemble_pred = squeezeformer_pred.loc[:, ['id']]
        ensemble_pred['reactivity_DMS_MaP'] = (squeezeformer_pred['reactivity_DMS_MaP'] + twintower_pred['reactivity_DMS_MaP']) / 2
        ensemble_pred['reactivity_2A3_MaP'] = (squeezeformer_pred['reactivity_2A3_MaP'] + twintower_pred['reactivity_2A3_MaP']) / 2

        return ensemble_pred
