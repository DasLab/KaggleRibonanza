from typing import Literal, Iterable
from .progress import get_progress_manager

MODEL_NAMES = Literal['k1_vigg', 'k2_hoyso', 'k3_aek', 'k4_gorna', 'k5_rlj', 'k6_iijj']

def load_models(model_names: Iterable[MODEL_NAMES], display_progress=True):
    # We do this instead of just importing all of them up front since transative imports can add nontrivial
    # additional latency (eg, it takes a couple extra seconds to import models which import pytorch lightning)
    models = {}
    iter = get_progress_manager().iterator(model_names, 'load model') if display_progress else model_names
    for name in iter:
        if name == 'k1_vigg':
            from ..models import k1_vigg
            models['k1_vigg'] = k1_vigg
        if name == 'k2_hoyso':
            from ..models import k2_hoyso
            models['k2_hoyso'] = k2_hoyso
        if name == 'k3_aek':
            from ..models import k3_aek
            models['k3_aek'] = k3_aek
        if name == 'k4_gorna':
            from ..models import k4_gorna
            models['k4_gorna'] = k4_gorna
        if name == 'k5_rlj':
            from ..models import k5_rlj
            models['k5_rlj'] = k5_rlj
        if name == 'k6_iijj':
            from ..models import k6_iijj
            models['k6_iijj'] = k6_iijj

    return models
