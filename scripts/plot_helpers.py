import matplotlib.pyplot as plt
import numpy as np
import cmasher as cmr
import seaborn as sns


def get_colors(num_colors, cmap="cmr.pride", cmap_range=(0.15, 0.85)):
    return cmr.take_cmap_colors(cmap, num_colors, cmap_range=cmap_range, return_fmt='hex')


def plot_stabilization_across_resolutions(data_resolutions,
                                          title,
                                          y_label,
                                          precision=0.1,
                                          stability_steps=10,
                                          marker_size=5):
    """
    General function to plot data across resolutions with stabilization checks.

    Parameters:
        data_resolutions (dict): Data values for each resolution.
        title (str): Title for the plot.
        y_label (str): Label for the y-axis.
        precision (float): Precision band for stabilization.
        stability_steps (int): Steps within the band for stabilization.
    """
    resolutions = sorted(data_resolutions.keys())
    num_bins = len(data_resolutions[resolutions[0]])

    colors = get_colors(num_bins)
    fig, axes = plt.subplots(num_bins, 1, figsize=(8, 2. * num_bins), sharex=True)
    fig.suptitle(title, fontsize=16)

    for i in range(num_bins):
        bin_values = [data_resolutions[res][i] for res in resolutions]
        avg_value = np.mean(bin_values)
        margin = avg_value * (precision / 100)
        upper_band, lower_band = avg_value + margin, avg_value - margin

        # Plot data with stabilization check
        axes[i].plot(resolutions, bin_values, '-o', markersize=marker_size, color=colors[i], label=f"Bin {i + 1}")
        axes[i].fill_between(resolutions, lower_band, upper_band, color='lightgray', alpha=0.3)

        stable_count = 0
        for res, value in zip(resolutions, bin_values):
            if lower_band <= value <= upper_band:
                stable_count += 1
                if stable_count >= stability_steps:
                    axes[i].axvline(res, color='red', linestyle='--')
                    axes[i].text(res, avg_value, f'Stable at {res}', color='red',
                                 va='top', ha='right', fontsize=10, rotation=90)
                    break
            else:
                stable_count = 0

        axes[i].set_ylabel(f"{y_label} {i + 1}")
        axes[i].legend()

    axes[-1].set_xlabel("Resolution")
    plt.tight_layout()
    plt.show()


def plot_stabilization_heatmap(data_by_zmax_and_resolution,
                               title,
                               y_label,
                               precision=0.1,
                               stability_steps=10,
                               annotate_max=False):
    """
    General function to create a heatmap of stabilization resolution across `zmax` values for each bin.

    Parameters:
        data_by_zmax_and_resolution (dict): Nested dictionary with data values for each `zmax` and resolution.
        title (str): Title for the heatmap.
        y_label (str): Label for the y-axis.
        precision (float): Precision band for stabilization.
        stability_steps (int): Steps within the band for stabilization.
        annotate_max (bool): Whether to annotate only the maximum stabilization resolution.
    """
    zmax_values = sorted(data_by_zmax_and_resolution.keys())
    sample_zmax = zmax_values[0]
    resolution_values = sorted(data_by_zmax_and_resolution[sample_zmax].keys())
    num_bins = len(data_by_zmax_and_resolution[sample_zmax][resolution_values[0]])

    # Initialize heatmap data
    heatmap_data = np.full((num_bins, len(zmax_values)), np.nan)

    for zmax_idx, zmax in enumerate(zmax_values):
        for bin_idx in range(num_bins):
            bin_values = [data_by_zmax_and_resolution[zmax][res][bin_idx] for res in resolution_values]
            avg_value = np.mean(bin_values)
            margin = avg_value * (precision / 100)
            upper_band, lower_band = avg_value + margin, avg_value - margin

            stable_count = 0
            for res_idx, value in enumerate(bin_values):
                if lower_band <= value <= upper_band:
                    stable_count += 1
                    if stable_count >= stability_steps:
                        heatmap_data[bin_idx, zmax_idx] = resolution_values[res_idx]
                        break
                else:
                    stable_count = 0

    # Plot heatmap
    cmap = cmr.get_sub_cmap('cmr.pride', 0.15, 0.85)
    plt.figure(figsize=(10, num_bins * 0.5))
    ax = sns.heatmap(heatmap_data, annot=not annotate_max, fmt=".0f", cmap=cmap,
                     cbar_kws={'label': 'Stabilization Resolution'},
                     xticklabels=np.round(zmax_values, 2), yticklabels=[f"Bin {i + 1}" for i in range(num_bins)])

    # Annotate max values if requested
    if annotate_max:
        for zmax_idx in range(len(zmax_values)):
            max_row_idx = np.nanargmax(heatmap_data[:, zmax_idx])
            max_value = heatmap_data[max_row_idx, zmax_idx]
            ax.text(zmax_idx + 0.5, max_row_idx + 0.5, f"{int(max_value)}",
                    ha='center', va='center', color='white', fontsize=10)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("$z_\\mathrm{max}$", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    plt.tight_layout()
    plt.show()
