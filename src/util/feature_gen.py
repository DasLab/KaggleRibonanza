from typing import Literal
from collections.abc import Iterable
import os
from os import path
import subprocess
from joblib import Parallel, delayed

# Needs to be done before importing arnie
ROOT = path.abspath(path.join(path.dirname(__file__), '../../'))
os.environ['ARNIEFILE'] = path.join(ROOT, 'external/arniefile.txt')
os.environ['PERL5LIB'] = path.join(ROOT, 'external/perl/lib/perl5')

from arnie.bpps import bpps
from arnie.pk_predictors import pk_predict
from arnie.mfe import mfe
from arnie.mea.mea import MEA
from arnie.utils import filename
from . import feature_cache
from . import progress

def capr(seq):
    capr_executable = path.join(ROOT, 'external/CapR/CapR')

    fasta_loc = f'{filename()}.fasta'
    outfile_loc = f'{filename()}.txt'

    with open(fasta_loc, 'w') as file:
        file.write(f'>test\n{seq}')
    
    subprocess.run(
        [capr_executable, fasta_loc, outfile_loc, '512'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=path.dirname(fasta_loc)
    )

    with open(outfile_loc) as file:
        data = file.read()
    
    return data

def bprna(sequence, structure):
    tmpname = filename()
    dbn_loc = f'{tmpname}.dbn'
    st_loc = f'{tmpname}.st'

    with open(dbn_loc, 'w') as file:
        file.write(sequence + '\n')
        file.write(structure + '\n')
        
    subprocess.run(
        ['perl', path.join(ROOT, "external/bpRNA/bpRNA.pl"), dbn_loc],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=path.dirname(dbn_loc)
    )

    with open(st_loc) as file:
        result = [l.strip('\n') for l in file]

    return result[5]

FeatureName = Literal['bpps_contrafold', 'bpps_eternafold', 'bpps_linearfolde', 'bpps_vienna2', 'mea_eternafold_bpps', 'mfe_eternafold', 'mfe_vienna2', 'mfe_ipknot', 'sstype_capr', 'sstype_bprna_eternafold']

def compute_feature(feat: FeatureName, sequence: str):
    if feat == 'bpps_contrafold':
        return bpps(sequence, package='contrafold')
    elif feat == 'bpps_eternafold':
        return bpps(sequence, package='eternafold')
    elif feat == 'bpps_linearfolde':
        return bpps(sequence, package='eternafold', linear=True)
    elif feat == 'bpps_vienna2':
        return bpps(sequence, package='vienna_2')
    elif feat == 'mea_eternafold_bpps':
        return MEA(get_feature('bpps_eternafold', sequence)).structure
    elif feat == 'mfe_eternafold':
        return mfe(sequence, package='eternafold')
    elif feat == 'mfe_vienna2':
        try:
            return mfe(sequence, package='vienna_2')
        except FileNotFoundError:
            # There is currently a race condition in arnie where if mfe is called multile times,
            # multiple processes may attempt to remove a file vienna creates in the cwd called rna.ps.
            # If we run into that, just retry
            return compute_feature(feat, sequence)
    elif feat == 'mea_ipknot':
        return pk_predict(sequence, 'ipknot', refinement=1, cpu=1)
    elif feat == 'sstype_capr':
        return capr(sequence)
    elif feat == 'sstype_bprna_eternafold':
        return bprna(sequence, get_feature('mfe_eternafold', sequence))
    else:
        raise ValueError(f'Invalid feature name {feat}')

def get_feature(feat: FeatureName, sequence: str):
    cached = feature_cache.cache.get(feat, sequence)
    if cached is not None: return cached

    res = compute_feature(feat, sequence)

    feature_cache.cache.set(feat, sequence, res)

    return res

def precompute(feat: FeatureName, sequences: Iterable[str], n_jobs: int):
    if isinstance(feature_cache.cache, feature_cache.NullFeatureCache):
        return
    
    # Note all interactions with the feature cache have to happen outside the parallel call,
    # as when the parallelized function is run, it will be in a separate process without
    # access to the cache
    to_compute = [
        seq for seq in progress.progress_manager.iterator(sequences, f'Find uncached for feature: {feat}')
        if not feature_cache.cache.exists(feat, seq)
    ]

    if len(to_compute) == 0: return

    for (res, seq) in progress.progress_manager.iterator(
        Parallel(return_as="generator", n_jobs=n_jobs)(
            delayed(lambda seq: (compute_feature(feat, seq), seq))(seq) for seq in to_compute
        ),
        total=len(to_compute),
        desc=f'Precompute feature: {feat}'
    ):
        feature_cache.cache.set(feat, seq, res)

def precompute_multi(feats: Iterable[FeatureName], sequences: Iterable[str], n_jobs: int):
    for feat in progress.progress_manager.iterator(feats, desc='features'):
        precompute(feat, sequences, n_jobs)
