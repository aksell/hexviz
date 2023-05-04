from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_tiled_heatmap(tensor, layer_sequence: List[int], head_sequence: List[int]):
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
            axes[i, j].imshow(tensor[i, j].detach().numpy(), cmap="viridis", aspect="equal")
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
    slice_start: int,
    slice_end: int,
    max_labels: int = 40,
):
    single_heatmap = tensor[layer, head, :, :].detach().numpy()

    fig, ax = plt.subplots(figsize=(10, 10))
    heatmap = ax.imshow(single_heatmap, cmap="viridis", aspect="equal")

    # Set the x and y axis major ticks and labels
    ax.xaxis.set_major_locator(
        MaxNLocator(integer=True, steps=[1, 2, 5], prune="both", nbins=max_labels)
    )
    ax.yaxis.set_major_locator(
        MaxNLocator(integer=True, steps=[1, 2, 5], prune="both", nbins=max_labels)
    )

    tick_indices_x = np.clip((ax.get_xticks()).astype(int), 0, slice_end - slice_start)
    tick_indices_y = np.clip((ax.get_yticks()).astype(int), 0, slice_end - slice_start)
    ax.set_xticklabels(
        np.arange(slice_start, slice_end + 1)[tick_indices_x], fontsize=8
    )
    ax.set_yticklabels(
        np.arange(slice_start, slice_end + 1)[tick_indices_y], fontsize=8
    )

    # Set the x and y axis minor ticks
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(1))

    # Set the axis labels
    ax.set_xlabel("Residue Number")
    ax.set_ylabel("Residue Number")

    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Create custom colorbar axes with the desired dimensions
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    # Add a colorbar to show the scale
    cbar = fig.colorbar(heatmap, cax=cax)
    cbar.ax.set_ylabel("Attention Weight", rotation=-90, va="bottom")

    # Set the title of the plot
    ax.set_title(f"Layer {layer + 1} - Head {head + 1}")

    return fig
