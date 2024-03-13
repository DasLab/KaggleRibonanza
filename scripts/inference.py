#!/usr/bin/env python3
import os, sys
import io
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from itertools import product
import pandas as pd
from matplotlib import pyplot as plt
from src.util.progress import ConsoleProgress, NullProgress
from src.util.feature_cache import NullFeatureCache, MemoryFeatureCache, FSFeatureCache
from src.util.feature_gen import precompute_features
from src.util.load_models import load_models
from src.util.data_format import format_input, read_fasta
from src.util.plot import plot_reactivity

def run(args):
    progress = ConsoleProgress() if not args.no_progress else NullProgress()
    if args.cache_type == 'none':
        cache = NullFeatureCache()
    elif args.cache_type == 'memory':
        cache = MemoryFeatureCache()
    else:
        cache = FSFeatureCache(args.cache_location, args.cache_compression_level)

    # When provided an input file, prefer inferring file content from the extension.
    # Otherwise, read a line and infer based on the contents
    sequences = []
    if args.sequence:
        sequences = [args.sequence]
    elif args.infile and args.infile.endswith('.parquet'):
        sequences = pd.read_parquet(args.infile)
    elif args.infile and  args.infile.endswith(('.fasta', '.fas', '.fa', '.fna', '.ffn', '.frn')):
        sequences = read_fasta(args.infile)
    elif args.infile and  args.infile.endswith('.csv'):
        sequences = pd.read_csv(args.infile)
    elif args.infile and  args.infile.endswith('.tsv'):
        sequences = pd.read_csv(args.infile, sep='\t')
    else:
        if args.infile:
            infile = args.infile
            with open(args.infile, 'f') as f:
                firstline = f.readline().strip()
        else:
            infile = io.StringIO(sys.stdin.read())
            firstline = infile.readline().strip()
            infile.seek(0)

        if firstline == 'sequence':
            sequences = pd.read_csv(infile)
        elif firstline.startswith(('>', ';', '#')):
            sequences = read_fasta(infile)
        elif ',' in firstline:
            sequences = pd.read_csv(infile)
        elif '\t' in firstline:
            sequences = pd.read_csv(infile, sep='\t')
        else:
            sequences = infile.read().splitlines()
    
    # Consistently give us a dataframe with sequence IDs, etc which we'll need later
    sequences = format_input(sequences)

    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)

    with progress, cache:
        models = load_models(sorted(set(args.include_models) - set(args.exclude_models)))
        if len(models) == 0: return
        
        if args.cache_type != 'none':
            precompute_features(models.keys(), sequences['sequence'], args.parallel_feature_jobs)
    
        # Run inference for each enabled model
        inferences = {
            model_name: model.infer(
                sequences,
                args.batch_size_small_model,
                *(
                    (args.batch_size_twintower,) if model_name == 'k3_aek' else ()
                )
            ) for (model_name, model) in models.items()
        }

        # Output raw data
        if not args.no_outfile:
            # Depending on our settings, split up the data into multiple files/sections, then save/show it
            for (split_sequence, split_model, split_experiment) in product(
                sequences['sequence_id'] if args.outfile_split_sequences else (None,),
                models.keys() if args.outfile_split_models else (None,),
                ('dms', '2a3') if args.outfile_split_experiments else (None,)
            ):
                output_name = '_'.join([part for part in [split_sequence, split_model, split_experiment, 'inference'] if part is not None])

                df = pd.DataFrame()

                # If we have more than one sequence per file, include the sequence ID and sequence-local index.
                # In all cases, include the global index (scoped to the content of the file)
                first_inference = next(iter(inferences.values()))
                if split_sequence is None and len(sequences) > 1:
                    df['id'] = first_inference['id']
                    df['sequence_id'] = first_inference['sequence_id']
                    df['index_in_sequence'] = first_inference['index_in_sequence']
                elif split_sequence is None:
                    df['id'] = first_inference['id']
                else:
                    df['id'] = first_inference[first_inference['sequence_id'] == split_sequence]['index_in_sequence']

                # If we're not splitting on model/experiment/sequence, add all of them to the dataframe.
                # Otherwise, just add the subset for the file we're currently working on
                for model in (split_model,) if split_model is not None else models.keys():
                    for exp in (split_experiment,) if split_experiment is not None else ('dms', '2a3'):
                        colname_parts = []
                        if split_model is None:
                            colname_parts.append(model)
                        if split_experiment is None:
                            colname_parts.append(exp)
                        colname_parts.append('reactivity')
                        colname = '_'.join(colname_parts)

                        if split_sequence is None:
                            df[colname] = inferences[model][f'reactivity_{exp.upper()}_MaP']
                        else:
                            df[colname] = inferences[model][inferences[model]['sequence_id'] == split_sequence][f'reactivity_{exp.upper()}_MaP']

                if not args.outdir and (split_sequence is not None or split_model is not None or split_sequence is not None):
                    print(f'**** {output_name} ****')

                out = os.path.join(args.outdir, f'{output_name}.{args.outfile_format}') if args.outdir is not None else sys.stdout

                if args.outfile_format == 'parquet':
                    df.to_parquet(out)
                elif args.outfile_format == 'csv':
                    df.to_csv(out, index=False)
                elif args.outfile_format == 'tsv':
                    df.to_csv(out, sep='\t', index=False)

        # Plot
        if args.plot:
            # Depending on our settings, generate multiple plots, then save/show them
            for (split_sequence, split_model, split_experiment) in product(
                sequences['sequence_id'] if args.plot_split_sequences else (None,),
                models.keys() if args.plot_split_models else (None,),
                ('dms', '2a3') if args.plot_split_experiments else (None,)
            ):
                output_name = '_'.join([part for part in [split_sequence, split_model, split_experiment, 'inference'] if part is not None])

                # If we're not splitting on model/experiment/sequence, include all of them in the plot.
                # Otherwise, just include the subset for the file we're currently working on
                infs = inferences if split_model is None else {split_model: inferences[split_model]}
                infs = {
                    name: inf if split_sequence is None else inf[inf['sequence_id'] == split_sequence] for name, inf in infs.items()
                }
                fig = plot_reactivity(infs, split_experiment)

                if args.outdir is None:
                    fig.canvas.manager.set_window_title(output_name)
                    fig.show()
                else:
                    fig.savefig(os.path.join(args.outdir, f'{output_name}.png'))
                    plt.close(fig)
        
            if args.outdir is None:
                plt.show(block=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    input_group = parser.add_argument_group('Input').add_mutually_exclusive_group()
    input_group.add_argument(
        '--infile',
        help='''
        Where to read list of sequences to run. Input may be newline-separated list of sequences, FASTA, or TSV/CSV/Parquet
        (with column headers for 'sequence' and optionally 'sequence_id', 'id_min', and 'id_max'). Default: read from stdin
        (if sequence parameter is not specified)
        '''
    )
    input_group.add_argument('sequence', nargs='?', help='Single sequence to run inference for')

    output_group = parser.add_argument_group('Output')
    output_group.add_argument(
        '--outdir',
        help='Where to write the inference results, plots, etc. Default: stdout/interactive'
    )
    output_group.add_argument(
        '--outfile-format',
        choices=['parquet', 'tsv', 'csv'],
        default='tsv'
    )
    output_group.add_argument(
        '--outfile-split-sequences',
        action='store_true',
        help='Split output into separate files, per sequence'
    )
    output_group.add_argument(
        '--outfile-split-models',
        action='store_true',
        help='Split output into separate files, per model'
    )
    output_group.add_argument(
        '--outfile-split-experiments',
        action='store_true',
        help='Split output into separate files, per experiment type (DMS/2A3)'
    )
    output_group.add_argument(
        '--no-outfile',
        action='store_true',
        help='Skip generating output file with raw data (only useful with --plot)'
    )
    output_group.add_argument(
        '--plot',
        action='store_true',
        help='Generate heatmap images'
    )
    output_group.add_argument(
        '--plot-split-sequences',
        action='store_true',
        help='Split output into separate files, per sequence'
    )
    output_group.add_argument(
        '--plot-split-models',
        action='store_true',
        help='Split output into separate files, per model'
    )
    output_group.add_argument(
        '--plot-split-experiments',
        action='store_true',
        help='Split output into separate files, per experiment type (DMS/2A3)'
    )
    output_group.add_argument(
        '--no-progress',
        action='store_true',
        help='Do not display tqdm progress bars'
    )

    model_group = parser.add_argument_group(
        'Model Selection',
        '''
            By default, all models will be included. Manually setting included models will override
            this behavior, and excluded models will override the resulting included models. Multiple
            models may be specified either by passing these arguments multiple times or by using a comma
            separated list Valid models include: 'k1_vigg', 'k2_hoyso', 'k3_aek', 'k4_gorna', 'k5_rlj',
            and 'k6_iijj' (you can also specify them like 'k1' or 'vigg')
        '''
    )

    model_choices = ['k1_vigg', 'k2_hoyso', 'k3_aek', 'k4_gorna', 'k5_rlj', 'k6_iijj']
    class ModelSelector(argparse.Action):
        first_call = True

        def __call__(self, parser, namespace, values, option_string):
            candidates = values.split(',')
            models = []
            for candidate in candidates:
                model_name = next(
                    (
                        name for name in model_choices
                        if candidate == name or candidate in name.split('_')
                    ),
                    None
                )
                if model_name is None:
                    raise argparse.ArgumentError(self, f'Invalid choice: {candidate} (valid chocies: {".".join(model_choices)})')
                
                models.append(model_name)
            

            if self.first_call:
                setattr(namespace, self.dest, models)
                self.first_call = False
            else:
                prev = getattr(namespace, self.dest, [])
                prev.extend(models)
    
    model_group.add_argument(
        '--include-models',
        action=ModelSelector,
        metavar='MODEL[,MODEL]',
        default=['k1_vigg', 'k2_hoyso', 'k3_aek', 'k4_gorna', 'k5_rlj', 'k6_iijj'],
        help='Models to include inference for. Default: all models'
    )
    model_group.add_argument(
        '--exclude-models',
        action=ModelSelector,
        metavar='MODEL[,MODEL]',
        default=[],
        help='Included models to omit running inference for (default: none)'
    )
    model_group.add_argument(
        '--batch-size-small-model',
        default=128,
        type=int,
        help='Number of sequences to run in a single pytorch batch for most models (default: 128)'
    )
    model_group.add_argument(
        '--batch-size-twintower',
        default=8,
        type=int,
        help='Number of sequences to run in a single pytorch batch for k3_aek\'s "twintower" submodel, which requires substantialy more VRAM (default: 8)'
    )

    cache_group = parser.add_argument_group('Feature Caching')
    cache_group.add_argument(
        '--cache-type',
        choices=['none', 'memory', 'fs'],
        default='memory',
        help='''
            When computing input features (eg MFE and BPPs) needed by multiple (sub)models,
            how these features should be cached. If 'none', features will always be recomputed.
            If 'memory', features will be cached in memory. If 'fs', features will be cached in
            a local file (customizable with additional parameters). If 'memory' or 'fs', features
            will be pre-computed (in parallel, if possible). Default: memory
        ''',
    )
    cache_group.add_argument(
        '--parallel-feature-jobs',
        default=os.cpu_count(),
        type=int,
        help='If using memory or fs feature caching, number of parallel jobs to use when precomputing features. Default: cpu count'
    )
    cache_group.add_argument(
        '--cache-location',
        default='./memo',
        help='If using fs feature caching, directory where the cache file is to be placed. Default: ./memo'
    )
    cache_group.add_argument(
        '--cache-compression-level',
        default=4,
        type=int,
        help='If using fs feature caching, gzip compression level to use (where compression is possible) Default: 4'
    )

    run(parser.parse_args())
