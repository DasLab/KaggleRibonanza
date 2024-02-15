from math import sin
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, get_cmap

def plot_reactivity(sequences: str | list[str], inferences: dict[str, pd.DataFrame]):
    sequences = [sequences] if isinstance(sequences, str) else sequences
    norm = Normalize(0, 1)
    cmap = get_cmap('gist_heat_r')
    cmap.set_bad(color='grey')

    max_seqlen = max([len(seq) for seq in sequences])

    with plt.ioff():
        fig, axs = plt.subplots(
            len(inferences),
            2,
            squeeze=False,
            layout="constrained",
            figsize=(
                # Width scales linearly with max sequence length up to 16in
                min(max_seqlen / 2, 16),
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
                            + len(sequences) * len(inferences) / (3 + min(1, len(sequences) / 60) * 21)
                        ),
                        32
                    ),
                    0.5 + 5.5 * len(inferences) / 6
                )
            )
        )
        
        for model_idx, (model_name, inference) in enumerate(inferences.items()):
            rdms = []
            reactive_idx = 0
            for seq in sequences:
                rdms.append(
                    inference['reactivity_DMS_MaP'][reactive_idx:reactive_idx + len(seq)].reset_index(drop=True).reindex(range(max_seqlen))
                )
                reactive_idx += len(seq)

            axs[model_idx, 0].imshow(rdms, norm=norm, cmap=cmap, aspect='auto')
            axs[model_idx, 0].locator_params(steps=[1, 5, 10], integer=True)
            axs[model_idx, 0].set_yticks([])
            axs[model_idx, 0].set_title(f'{model_name}_dms')

            r2a3 = []
            reactive_idx = 0
            for seq in sequences:
                r2a3.append(
                    inference['reactivity_DMS_MaP'][reactive_idx:reactive_idx + len(seq)].reset_index(drop=True).reindex(range(max_seqlen))
                )
                reactive_idx += len(seq)

            axs[model_idx, 1].imshow(r2a3, norm=norm, cmap=cmap, aspect='auto')
            axs[model_idx, 1].locator_params(steps=[1, 5, 10], integer=True)
            axs[model_idx, 1].set_yticks([])
            axs[model_idx, 1].set_title(f'{model_name}_2a3')
        
        fig.colorbar(
            ScalarMappable(norm=norm, cmap=cmap),
            orientation='horizontal',
            label='Predicted Reactivity',
            ax=axs,
            shrink=0.6,
            pad=0.01
        )

    return fig
