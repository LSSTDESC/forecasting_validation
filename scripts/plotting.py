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
    resolutions = sorted(data_resolutions.keys())
    num_plots = len(labels)
    colors = get_colors(range(num_plots))

    fig, axes = plt.subplots(num_plots, 1, figsize=(8, 2. * num_plots), sharex=True)
    fig.suptitle(f"{title} - Forecast Year {forecast_year}", fontsize=16)

    if num_plots == 1:
        axes = [axes]

    for i, label in enumerate(labels):
        data_values = [data_resolutions[res][i] for res in resolutions]
        axes[i].plot(resolutions, data_values, '-o', markersize=marker_size, color=colors[i])

        avg_value = np.mean(data_values)
        if precision:
            margin = avg_value * (precision / 100)
            upper_band = avg_value + margin
            lower_band = avg_value - margin
            axes[i].fill_between(resolutions,
                                 lower_band,
                                 upper_band,
                                 color='lightgray',
                                 alpha=0.3,
                                 label=f"±{precision}% margin")
        axes[i].legend(frameon=False, loc='upper right', fontsize=11)

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

    axes[-1].set_xlabel("Redshift Resolution")
    plt.tight_layout()

    extra = f"_{extra_info}" if extra_info else ""
    fig_name = f"{title.replace(' ', '_').lower()}_{forecast_year}{extra}{fig_format}"
    plt.savefig(f"plots_output/{fig_name}")
    plt.show()


def plot_stabilization_heatmap(heatmap_data, x_labels, y_labels, title, forecast_year, annotate_max=False,
                               fig_format=".pdf"):
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
#########################################

def plot_tomo_peaks_zres_sweep(bin_centers_resolutions, zmax, forecast_year, bin_type, precision=None,
                               marker_size=5, stability_steps=10, extra_info=""):
    # Assuming the bin keys are integers, use them directly
    bin_keys = list(bin_centers_resolutions[next(iter(bin_centers_resolutions))][f"{bin_type}_bin_centers"].keys())
    labels = [f"Bin {k + 1}" for k in range(len(bin_keys))]  # Keep labels as formatted strings

    # Use numeric keys to retrieve data directly in data_resolutions
    plot_resolution_sweep_subplots(
        data_resolutions={
            res: [bin_centers_resolutions[res][f"{bin_type}_bin_centers"][k] for k in bin_keys]
            for res in bin_centers_resolutions
        },
        labels=labels,
        y_label=f"{bin_type.capitalize()}",
        title=f"{bin_type.capitalize()} Bin Centers Across Redshift Resolutions",
        forecast_year=forecast_year,
        precision=precision,
        stability_steps=stability_steps,
        marker_size=marker_size,
        extra_info=f"{extra_info}_zmax{zmax}"
    )


def plot_tomo_peaks_zres_and_zmax_sweep(bin_centers_by_zmax, forecast_year, bin_type, precision=0.1, stability_steps=10,
                                        annotate_max=False):
    zmax_values = sorted(bin_centers_by_zmax.keys())
    sample_zmax = zmax_values[0]
    resolution_values = sorted(bin_centers_by_zmax[sample_zmax].keys())

    # Retrieve actual bin keys directly from the data structure
    bin_keys = list(bin_centers_by_zmax[sample_zmax][resolution_values[0]][f"{bin_type}_bin_centers"].keys())
    labels = [f"Bin {k + 1}" for k in range(len(bin_keys))]  # Only used for y-axis labels

    heatmap_data = np.full((len(bin_keys), len(zmax_values)), np.nan)

    for zmax_idx, zmax in enumerate(zmax_values):
        for bin_idx, bin_key in enumerate(bin_keys):  # Access each bin by its actual key
            # Extract values using bin_key directly (no formatted labels)
            bin_center_values = [bin_centers_by_zmax[zmax][res][f"{bin_type}_bin_centers"][bin_key] for res in resolution_values]

            avg_value = np.mean(bin_center_values)
            margin = avg_value * (precision / 100)
            upper_band = avg_value + margin
            lower_band = avg_value - margin

            stable_count = 0
            for res_idx, value in enumerate(bin_center_values):
                if lower_band <= value <= upper_band:
                    stable_count += 1
                    if stable_count >= stability_steps:
                        heatmap_data[bin_idx, zmax_idx] = resolution_values[res_idx]
                        break
                else:
                    stable_count = 0

    plot_stabilization_heatmap(heatmap_data, x_labels=np.round(zmax_values, 2), y_labels=labels,
                               title=f"{bin_type.capitalize()} Bin Stabilization Resolution",
                               forecast_year=forecast_year, annotate_max=annotate_max)


def plot_gbias_values_zres_sweep(galaxy_bias_resolutions, forecast_year, precision=0.1, stability_steps=10,
                                 marker_size=5):
    labels = [f"Bin {i + 1}" for i in range(len(galaxy_bias_resolutions[next(iter(galaxy_bias_resolutions))]))]
    plot_resolution_sweep_subplots(
        data_resolutions={res: galaxy_bias_resolutions[res] for res in galaxy_bias_resolutions},
        labels=labels,
        y_label="Galaxy Bias",
        title="Galaxy Bias Across Redshift Resolutions",
        forecast_year=forecast_year,
        precision=precision,
        stability_steps=stability_steps,
        marker_size=marker_size
    )


def plot_kernel_peaks_zres_sweep(peaks_by_resolution, forecast_year, kernel_type="wl", precision=0.1,
                                 stability_steps=10, marker_size=5, extra_info=""):
    """
    Plot kernel peaks across different redshift resolutions.

    Parameters:
        peaks_by_resolution (dict): Nested dictionary with resolutions as keys. Format:
                                    {resolution: {"wl": [(z_peak, value_peak), ...], "nc": [(z_peak, value_peak), ...]}}
        forecast_year (str): Forecasting year for plot title.
        kernel_type (str): Which kernel to plot: "wl" for weak lensing, "nc" for number counts.
        precision (float): Precision band percentage for stabilization check (default 0.1 %).
        stability_steps (int): Steps required within precision band for stabilization.
        marker_size (int): Size of the markers for the peaks.
        extra_info (str): Additional information to add to the plot title.
    """
    resolutions = sorted(peaks_by_resolution.keys())

    # Determine number of kernels for the given kernel type
    num_kernels = len(peaks_by_resolution[resolutions[0]][kernel_type])
    labels = [f"Kernel {i + 1}" for i in range(num_kernels)]

    # Extract data for each kernel type
    data_resolutions = {
        res: [peaks_by_resolution[res][kernel_type][i][0] for i in range(num_kernels)]
        for res in resolutions
    }

    plot_resolution_sweep_subplots(
        data_resolutions=data_resolutions,
        labels=labels,
        y_label="",
        title=f"{kernel_type.upper()} Kernel Peaks Across Redshift Resolutions",
        forecast_year=forecast_year,
        precision=precision,
        stability_steps=stability_steps,
        marker_size=marker_size,
        extra_info=extra_info
    )


def plot_kernel_peaks_zres_zmax_sweep(kernel_peaks_by_zmax_and_resolution, forecast_year, kernel_type, precision=0.1,
                                      stability_steps=10, annotate_max=False):
    zmax_values = sorted(kernel_peaks_by_zmax_and_resolution.keys())
    sample_zmax = zmax_values[0]
    resolution_values = sorted(kernel_peaks_by_zmax_and_resolution[sample_zmax].keys())
    num_kernels = len(kernel_peaks_by_zmax_and_resolution[sample_zmax][resolution_values[0]][kernel_type])
    heatmap_data = np.full((num_kernels, len(zmax_values)), np.nan)

    for zmax_idx, zmax in enumerate(zmax_values):
        for kernel_idx in range(num_kernels):
            kernel_peak_values = [kernel_peaks_by_zmax_and_resolution[zmax][res][kernel_type][kernel_idx][0] for res in
                                  resolution_values]
            avg_value = np.mean(kernel_peak_values)
            margin = avg_value * (precision / 100)
            upper_band = avg_value + margin
            lower_band = avg_value - margin

            stable_count = 0
            for res_idx, value in enumerate(kernel_peak_values):
                if lower_band <= value <= upper_band:
                    stable_count += 1
                    if stable_count >= stability_steps:
                        heatmap_data[kernel_idx, zmax_idx] = resolution_values[res_idx]
                        break
                else:
                    stable_count = 0

    plot_stabilization_heatmap(heatmap_data, x_labels=np.round(zmax_values, 2),
                               y_labels=[f"Kernel {i + 1}" for i in range(num_kernels)],
                               title=f"{kernel_type.upper()} Kernel Stabilization Resolution",
                               forecast_year=forecast_year, annotate_max=annotate_max)


def plot_gbias_value_zres_zmax_sweep(galaxy_bias_by_zmax_and_resolution, forecast_year, precision=0.1,
                                     stability_steps=10, annotate_max=False):
    zmax_values = sorted(galaxy_bias_by_zmax_and_resolution.keys())
    sample_zmax = zmax_values[0]
    resolution_values = sorted(galaxy_bias_by_zmax_and_resolution[sample_zmax].keys())
    num_bins = len(galaxy_bias_by_zmax_and_resolution[sample_zmax][resolution_values[0]])
    heatmap_data = np.full((num_bins, len(zmax_values)), np.nan)

    for zmax_idx, zmax in enumerate(zmax_values):
        for bin_idx in range(num_bins):
            bin_bias_values = [galaxy_bias_by_zmax_and_resolution[zmax][res][bin_idx] for res in resolution_values]
            avg_value = np.mean(bin_bias_values)
            margin = avg_value * (precision / 100)
            upper_band = avg_value + margin
            lower_band = avg_value - margin

            stable_count = 0
            for res_idx, value in enumerate(bin_bias_values):
                if lower_band <= value <= upper_band:
                    stable_count += 1
                    if stable_count >= stability_steps:
                        heatmap_data[bin_idx, zmax_idx] = resolution_values[res_idx]
                        break
                else:
                    stable_count = 0

    plot_stabilization_heatmap(heatmap_data, x_labels=np.round(zmax_values, 2),
                               y_labels=[f"Bin {i + 1}" for i in range(num_bins)],
                               title="Galaxy Bias Stabilization Resolution", forecast_year=forecast_year,
                               annotate_max=annotate_max)
