import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_latent_pca(latent_pca, y_data, annotations, pca, first_index=0, second_index=1):
    fig, ax = plt.subplots(1, 2, figsize=(16, 3))

    # Generate a colormap that has as many colors as you have unique labels
    unique_labels = torch.unique(y_data.cpu())
    n_unique_labels = len(unique_labels)
    cmap = ListedColormap(plt.colormaps.get('tab10').colors[:n_unique_labels])

    # Map each label to a color
    label_to_color = {label.item(): cmap(i) for i, label in enumerate(unique_labels)}

    # Color each point in the scatter plot according to its label
    colors = [label_to_color[label.item()] for label in y_data.cpu()]

    # Plot the scatter plot
    ax[0].scatter(latent_pca[:, first_index], latent_pca[:, second_index], c=colors)
    ax[0].set_xlabel(f"Principal Component {first_index}")
    ax[0].set_ylabel(f"Principal Component {second_index}")
    ax[0].set_title("Latent Space after PCA")

    # Create legend
    legend_labels = [list(annotations.keys())[value] for value in unique_labels]
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=label_to_color[value.item()], markersize=10) for label, value in zip(legend_labels, unique_labels)]
    ax[0].legend(handles=legend_elements)

    # plot the explained variance ratio
    ax[1].plot(pca.explained_variance_ratio_)
    ax[1].set_xlabel("Principal Component")
    ax[1].set_ylabel("Explained Variance Ratio")
    ax[1].set_title("Explained Variance Ratio of Principal Components")

    plt.show()