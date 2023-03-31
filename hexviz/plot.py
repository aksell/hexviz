from typing import List

import matplotlib.pyplot as plt


def plot_tiled_heatmap(tensor, layer_sequence: List[int], head_sequence: List[int]):
    tensor = tensor[layer_sequence, :][:, head_sequence, :, :]  # Slice the tensor according to the provided sequences and sequence_count
    num_layers = len(layer_sequence)
    num_heads = len(head_sequence)
    fig, axes = plt.subplots(num_layers, num_heads, figsize=(12, 12))
    for i in range(num_layers):
        for j in range(num_heads):
            axes[i, j].imshow(tensor[i, j].detach().numpy(), cmap='viridis', aspect='auto')
            axes[i, j].axis('off')

            # Enumerate the axes
            if i == 0:
                axes[i, j].set_title(f'Head {head_sequence[j] + 1}', fontsize=10, y=1.05)

    # Add layer labels on the right Y-axis
    for i in range(num_layers):
        fig.text(0.98, (num_layers - i - 1) / num_layers + 0.025, f'Layer {layer_sequence[i]+1}', fontsize=10, rotation=0, ha='right', va='center')

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    return fig