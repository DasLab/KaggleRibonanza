from os import path
import gc
import numpy as np
import pandas as pd
from copy import deepcopy
import torch
from .model.util import parse_model_name, create_model
from .model.srf.util import parser
from .model.srf.dataset import gen_ds
from .model.srf import models
from ...util.progress import get_progress_manager
from ...util.feature_gen import FeatureName
from ...util.data_format import format_input, format_output

REQUIRED_FEATURES: list[FeatureName] = ['mfe_vienna2', 'mfe_eternafold', 'bpps_vienna2', 'bpps_eternafold']
NUM_WORKERS = 0

def infer(sequences: str | list[str] | pd.DataFrame, batch_size=64):
    input_df = format_input(sequences)

    MODEL_NAMES = "SRFBPP_semi2debxsd01_KF0 SRFBPPGNN_semi2debxsd02_KF0 SRFBPPGNN_semi2debxsd01_KF0 SRFBPPGNN_semidebxsd04_KF0 SRFBPPGNN_semidebxsd03_KF0 SRFBPP_semidebxsd01_KF0 SRFBPP_semidebxsd04_KF0 SRFBPPGNN_semidebxsd02_KF0 SRFBPPGNN_semidebxsd01_KF0 SRFBPP_semidebxsd03_KF0"
    args = parser.parse_args([
        '-use_fp16',
        '-nt',
        '-restore',
        '-predict_test',
        '-n_init_epoch', '0',
        '-ds', 'srf',
        '-n_dl_worker', str(NUM_WORKERS),
        '-task_cnt', '8',
        '-epochs', '1',
        '-verbose', '640',
        '-bs', str(batch_size),
        '-vbs', str(batch_size),
        '-model_dir', path.join(path.dirname(__file__), 'model_weights'),
    ])
    test_args = deepcopy(args)
    test_args.data_type = 'test'
    test_args.dataset = test_args.test_ds

    processed_df = input_df.copy(deep=False)
    processed_df['src'] = 'srf'

    test_preds = []

    def run_model(name, args, cls, processed_df):
        model = create_model(name, args, getattr(models, cls))
        test_ds = gen_ds(model.cfg, 'test', processed_df, tokenizer=model.tokenizer)
        test_pred = model.predict_rst(test_ds, data_type='test')
        return test_pred

    for name, cls in get_progress_manager().iterator(parse_model_name(MODEL_NAMES).items(), desc='k6_iijj submodels'):
        test_preds.append(run_model(name, args, cls, processed_df))
        gc.collect()
        torch.cuda.empty_cache()

    for j, preds in enumerate(test_preds):
        sids = np.array(preds['sequence_id'])
        inds = np.argsort(sids)
        ids = [preds['ids'][ind] for ind in inds]
        logits = [preds['logits'][ind] for ind in inds]
        ids, logits = np.concatenate(ids), np.concatenate(logits)

        if j==0:
            ensemble_logits = logits
        else:
            ensemble_logits += logits
    ensemble_logits = ensemble_logits / len(test_preds)

    ensemble_pred = pd.DataFrame({
        'id': ids,
        'reactivity_DMS_MaP': ensemble_logits[:, 1],
        'reactivity_2A3_MaP': ensemble_logits[:, 0],
    })

    format_output(input_df, ensemble_pred)

    return ensemble_pred
