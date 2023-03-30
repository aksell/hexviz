import matplotlib.pyplot as plt


def plot_tiled_heatmap(tensor, layer_count=12, head_count=12):
    tensor = tensor[:layer_count:2, :head_count:2, :, :]  # Slice the tensor according to the provided arguments
    num_layers = layer_count // 2
    num_heads = head_count // 2
    fig, axes = plt.subplots(num_layers, num_heads, figsize=(12, 12))
    for i in range(num_layers):
        for j in range(num_heads):
            axes[i, j].imshow(tensor[i, j].detach().numpy(), cmap='viridis', aspect='auto')
            axes[i, j].axis('off')

            # Enumerate the axes
            if i == 0:
                axes[i, j].set_title(f'Head {j * 2 + 1}', fontsize=10, y=1.05)

    # Add layer labels on the right Y-axis
    for i in range(num_layers):
        fig.text(0.98, (num_layers - i - 1) / num_layers + 0.025, f'Layer {i * 2 + 1}', fontsize=10, rotation=0, ha='right', va='center')

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    return fig
