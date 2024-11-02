import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import cmasher as cmr


def get_colors(data, cmap="cmr.pride", cmap_range=(0.15, 0.85)):
    num = len(data)
    colors = cmr.take_cmap_colors(cmap, num, cmap_range=cmap_range, return_fmt='hex')

    return colors

def plot_resolution_sweep_subplots(data_resolutions, labels, y_label, title, forecast_year,
                                   precision=None, stability_steps=10, marker_size=5, fig_format=".pdf", extra_info=""):
    """
    General function to plot data across resolutions in separate subplots with an optional precision band.

    Parameters:
        data_resolutions (dict): Nested dictionary containing data for each resolution.
        labels (list of str): Labels for each subplot (e.g., bin names or kernel names).
        y_label (str): Label for the y-axis.
        title (str): Title for the figure.
        forecast_year (str): Forecast year for labeling.
        precision (float): Precision for the averaged band (optional).
        stability_steps (int): Number of consecutive steps within the band for stabilization.
        marker_size (int): Size of markers.
        fig_format (str): File format for saving the figure.
        extra_info (str): Extra information to add to the figure name.
    """
    resolutions = sorted(data_resolutions.keys())
    num_plots = len(labels)

    # Colors for each subplot
    colors = cmr.take_cmap_colors("cmr.pride", num_plots, cmap_range=(0.15, 0.85), return_fmt='hex')

    fig, axes = plt.subplots(num_plots, 1, figsize=(8, 2. * num_plots), sharex=True)
    fig.suptitle(f"{title} - Forecast Year {forecast_year}", fontsize=16)

    # Ensure axes is iterable if only one subplot
    if num_plots == 1:
        axes = [axes]

    for i, label in enumerate(labels):
        data_values = [data_resolutions[res][i] for res in resolutions]
        axes[i].plot(resolutions, data_values, '-o', markersize=marker_size, color=colors[i], label=label)

        avg_value = np.mean(data_values)
        if precision:
            margin = avg_value * (precision / 100)
            upper_band = avg_value + margin
            lower_band = avg_value - margin
            axes[i].fill_between(resolutions, lower_band, upper_band, color='lightgray', alpha=0.3)

        stable_count = 0
        for res, value in zip(resolutions, data_values):
            if lower_band <= value <= upper_band:
                stable_count += 1
                if stable_count >= stability_steps:
                    axes[i].axvline(res, color='red', linestyle='--')
                    axes[i].text(res, avg_value, f'{res}', color='red', va='top', ha='right', fontsize=10, rotation=90)
                    break
            else:
                stable_count = 0

        axes[i].set_ylabel(f"{y_label} {label}")
        axes[i].legend()

    axes[-1].set_xlabel("Redshift Resolution")
    plt.tight_layout()

    extra = f"_{extra_info}" if extra_info else ""
    fig_name = f"{title.replace(' ', '_').lower()}_{forecast_year}{extra}{fig_format}"
    plt.savefig(f"plots_output/{fig_name}")
    plt.show()


def plot_stabilization_heatmap(heatmap_data, x_labels, y_labels, title, forecast_year, annotate_max=False, fig_format=".pdf"):
    """
    General function to create a heatmap showing stabilization resolution for different variables.

    Parameters:
        heatmap_data (np.ndarray): 2D array of stabilization resolutions.
        x_labels (list): Labels for the x-axis (e.g., zmax values).
        y_labels (list): Labels for the y-axis (e.g., bin or kernel indices).
        title (str): Title for the heatmap.
        forecast_year (str): Forecast year for labeling.
        annotate_max (bool): Whether to annotate only maximum values.
        fig_format (str): File format for saving the figure.
    """
    cmap = cmr.get_sub_cmap('cmr.pride', 0.15, 0.85)
    plt.figure(figsize=(10, len(y_labels) * 0.5))
    ax = sns.heatmap(heatmap_data, annot=not annotate_max, fmt=".0f", cmap=cmap,
                     cbar_kws={'label': 'Stabilization Resolution'},
                     xticklabels=x_labels, yticklabels=y_labels)

    if annotate_max:
        for x_idx in range(len(x_labels)):
            max_row_idx = np.nanargmax(heatmap_data[:, x_idx])
            max_value = heatmap_data[max_row_idx, x_idx]
            ax.text(x_idx + 0.5, max_row_idx + 0.5, f"{int(max_value)}",
                    ha='center', va='center', color='white', fontsize=10)

    ax.set_title(f"{title} - Forecast Year {forecast_year}", fontsize=14)
    ax.set_xlabel("zmax", fontsize=12)
    ax.set_ylabel("Index", fontsize=12)

    fig_name = f"{title.replace(' ', '_').lower()}_{forecast_year}{fig_format}"
    plt.savefig(f"plots_output/{fig_name}")
    plt.show()
