from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_tiled_heatmap(tensor, layer_sequence: List[int], head_sequence: List[int], fixed_scale: bool = True):
    tensor = tensor[layer_sequence, :][
        :, head_sequence, :, :
    ]  # Slice the tensor according to the provided sequences and sequence_count
    num_layers = len(layer_sequence)
    num_heads = len(head_sequence)

    x_size = num_heads * 2
    y_size = num_layers * 2
    fig, axes = plt.subplots(num_layers, num_heads, figsize=(x_size, y_size), squeeze=False)
    for i in range(num_layers):
        for j in range(num_heads):
            if fixed_scale:
                im = axes[i, j].imshow(
                    tensor[i, j].detach().numpy(), cmap="viridis", aspect="equal", vmin=0, vmax=1
                )
            else:
                im = axes[i, j].imshow(
                    tensor[i, j].detach().numpy(), cmap="viridis", aspect="equal"
                )
            axes[i, j].axis("off")

            # Enumerate the axes
            if i == 0:
                axes[i, j].set_title(f"Head {head_sequence[j] + 1}", fontsize=10, y=1.05)

    # Calculate the row label offset based on the number of columns
    offset = 0.02 + (12 - num_heads) * 0.0015
    for i, ax_row in enumerate(axes):
        row_label = f"{layer_sequence[i]+1}"
        row_pos = ax_row[num_heads - 1].get_position()
        fig.text(row_pos.x1 + offset, (row_pos.y1 + row_pos.y0) / 2, row_label, va="center")
    
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    return fig


def plot_single_heatmap(
    tensor,
    layer: int,
    head: int,
    tokens: list[str],
    fixed_scale : bool = True
):
    single_heatmap = tensor[layer, head, :, :].detach().numpy()

    fig, ax = plt.subplots(figsize=(10, 10))
    if fixed_scale:
        heatmap = ax.imshow(single_heatmap, cmap="viridis", aspect="equal", vmin=0, vmax=1)
    else:
        heatmap = ax.imshow(single_heatmap, cmap="viridis", aspect="equal")

    # Function to adjust font size based on the number of labels
    def get_font_size(labels):
        if len(labels) <= 60:
            return 8
        else:
            return 8 * (60 / len(labels))

    # Adjust font size
    font_size = get_font_size(tokens)

    # Set the x and y axis ticks
    ax.xaxis.set_major_locator(FixedLocator(np.arange(0, len(tokens))))
    ax.yaxis.set_major_locator(FixedLocator(np.arange(0, len(tokens))))

    # Set tick labels as sequence values
    ax.set_xticklabels(tokens, fontsize=font_size, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(tokens, fontsize=font_size)

    # Set the axis labels
    ax.set_xlabel("Sequence tokens")
    ax.set_ylabel("Sequence tokens")

    # Create custom colorbar axes with the desired dimensions
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    # Add a colorbar to show the scale
    cbar = fig.colorbar(heatmap, cax=cax)
    cbar.ax.set_ylabel("Attention Weight", rotation=-90, va="bottom")

    # Set the title of the plot
    ax.set_title(f"Layer {layer + 1} - Head {head + 1}")

    return fig
