from math import sin
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, get_cmap

def plot_reactivity(inferences: dict[str, pd.DataFrame], only_exp: str = None) -> plt.Figure:
    norm = Normalize(0, 1)
    cmap = get_cmap('gist_heat_r')
    cmap.set_bad(color='grey')

    first_inference = next(iter(inferences.values()))
    max_seqlen = first_inference['index_in_sequence'].max()
    total_seqs = first_inference['sequence_id'].nunique()

    with plt.ioff():
        fig, axs = plt.subplots(
            len(inferences),
            2 if only_exp is None else 1,
            squeeze=False,
            layout="constrained",
            figsize=(
                # Width scales linearly with max sequence length up to 16in
                min(max_seqlen / (2 if only_exp is None else 4), 16),
                # Height scales directly with #sequences*#models, clamped
                max(
                    min(
                        (
                            # Allocate space for labels on each chart
                            len(inferences) * 0.55
                            # Instead of scaling the height linearly up to the max height,
                            # we want to make the row size shrink a bit as we add more rows,
                            # so that we can keep our figure size shorter for
                            # relatively-low numbers of sequences
                            + total_seqs * len(inferences) / (3 + min(1, total_seqs / 60) * 21)
                        ),
                        32
                    ),
                    0.5 + 5.5 * len(inferences) / 6
                )
            )
        )
        
        for model_idx, (model_name, inference) in enumerate(inferences.items()):
            exp_idx = 0

            if not only_exp or only_exp == 'dms':
                axs[model_idx, exp_idx].imshow(
                    inference.pivot(index='sequence_id', columns='index_in_sequence', values='reactivity_DMS_MaP'),
                    norm=norm, cmap=cmap, aspect='auto'
                )
                axs[model_idx, exp_idx].locator_params(steps=[1, 5, 10], integer=True)
                axs[model_idx, exp_idx].set_yticks([])
                axs[model_idx, exp_idx].set_title(f'{model_name}_dms')

                exp_idx += 1

            if not only_exp or only_exp == '2a3':
                axs[model_idx, exp_idx].imshow(
                    inference.pivot(index='sequence_id', columns='index_in_sequence', values='reactivity_2A3_MaP'),
                    norm=norm, cmap=cmap, aspect='auto'
                )
                axs[model_idx, exp_idx].locator_params(steps=[1, 5, 10], integer=True)
                axs[model_idx, exp_idx].set_yticks([])
                axs[model_idx, exp_idx].set_title(f'{model_name}_2a3')
        
        fig.colorbar(
            ScalarMappable(norm=norm, cmap=cmap),
            orientation='horizontal',
            label='Predicted Reactivity',
            ax=axs,
            shrink=0.6,
            pad=0.01
        )

    return fig
