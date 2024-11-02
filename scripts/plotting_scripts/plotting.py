import numpy as np
import scripts.plotting_scripts.plot_helpers as ph
import matplotlib.pyplot as plt


def plot_tomo_peaks_zres_sweep(bin_centers_resolutions,
                               zmax,
                               forecast_year,
                               bin_type,
                               precision=None,
                               marker_size=5,
                               stability_steps=10,
                               extra_info=""):
    bin_keys = list(bin_centers_resolutions[next(iter(bin_centers_resolutions))][f"{bin_type}_bin_centers"].keys())
    labels = [f"{k + 1}" for k in range(len(bin_keys))]
    data_resolutions = {res: [bin_centers_resolutions[res][f"{bin_type}_bin_centers"][k] for k in bin_keys] for res in
                        bin_centers_resolutions}

    ph.plot_resolution_sweep_subplots(data_resolutions,
                                      labels,
                                      f"bin",
                                      f"{bin_type.capitalize()} bin centers across redshift resolutions",
                                      forecast_year,
                                      precision,
                                      stability_steps,
                                      marker_size,
                                      extra_info=f"{extra_info}_zmax{zmax}")


def plot_tomo_peaks_zres_and_zmax_sweep(bin_centers_by_zmax,
                                        forecast_year,
                                        bin_type,
                                        precision=0.1,
                                        stability_steps=10,
                                        annotate_max=False):
    zmax_values = sorted(bin_centers_by_zmax.keys())
    sample_zmax = zmax_values[0]
    resolution_values = sorted(bin_centers_by_zmax[sample_zmax].keys())
    bin_keys = list(bin_centers_by_zmax[sample_zmax][resolution_values[0]][f"{bin_type}_bin_centers"].keys())
    labels = [f"{k + 1}" for k in range(len(bin_keys))]

    heatmap_data = np.full((len(bin_keys), len(zmax_values)), np.nan)

    for zmax_idx, zmax in enumerate(zmax_values):
        for bin_idx, bin_key in enumerate(bin_keys):
            bin_center_values = [bin_centers_by_zmax[zmax][res][f"{bin_type}_bin_centers"][bin_key] for res in
                                 resolution_values]
            avg_value = np.mean(bin_center_values)
            margin = avg_value * (precision / 100)
            upper_band, lower_band = avg_value + margin, avg_value - margin

            stable_count = 0
            for res_idx, value in enumerate(bin_center_values):
                if lower_band <= value <= upper_band:
                    stable_count += 1
                    if stable_count >= stability_steps:
                        heatmap_data[bin_idx, zmax_idx] = resolution_values[res_idx]
                        break
                else:
                    stable_count = 0

    ph.plot_stabilization_heatmap(heatmap_data,
                                  x_labels=np.round(zmax_values, 2),
                                  y_labels=labels,
                                  title=f"{bin_type.capitalize()} bin stabilization resolution",
                                  forecast_year=forecast_year,
                                  annotate_max=annotate_max)


def plot_kernel_peaks_zres_sweep(peaks_by_resolution,
                                 forecast_year,
                                 kernel_type,
                                 precision=0.1,
                                 stability_steps=10,
                                 marker_size=5,
                                 extra_info=""):
    resolutions = sorted(peaks_by_resolution.keys())
    num_kernels = len(peaks_by_resolution[resolutions[0]][kernel_type])
    labels = [f"{i + 1}" for i in range(num_kernels)]

    data_resolutions = {
        res: [peaks_by_resolution[res][kernel_type][i][0] for i in range(num_kernels)]
        for res in resolutions
    }

    ph.plot_resolution_sweep_subplots(
        data_resolutions=data_resolutions,
        labels=labels,
        y_label="kernel",
        title=f"{kernel_type.upper()} kernel peaks across redshift resolutions",
        forecast_year=forecast_year,
        precision=precision,
        stability_steps=stability_steps,
        marker_size=marker_size,
        extra_info=extra_info
    )


def plot_kernel_peaks_zres_zmax_sweep(kernel_peaks_zres_zmax_sweep,
                                      forecast_year,
                                      kernel_type,
                                      precision=0.1,
                                      stability_steps=10,
                                      annotate_max=False):
    zmax_values = sorted(kernel_peaks_zres_zmax_sweep.keys())
    sample_zmax = zmax_values[0]
    resolution_values = sorted(kernel_peaks_zres_zmax_sweep[sample_zmax].keys())
    num_kernels = len(kernel_peaks_zres_zmax_sweep[sample_zmax][resolution_values[0]][kernel_type])
    heatmap_data = np.full((num_kernels, len(zmax_values)), np.nan)

    for zmax_idx, zmax in enumerate(zmax_values):
        for kernel_idx in range(num_kernels):
            kernel_peak_values = [kernel_peaks_zres_zmax_sweep[zmax][res][kernel_type][kernel_idx][0] for res in
                                  resolution_values]
            avg_value = np.mean(kernel_peak_values)
            margin = avg_value * (precision / 100)
            upper_band, lower_band = avg_value + margin, avg_value - margin

            stable_count = 0
            for res_idx, value in enumerate(kernel_peak_values):
                if lower_band <= value <= upper_band:
                    stable_count += 1
                    if stable_count >= stability_steps:
                        heatmap_data[kernel_idx, zmax_idx] = resolution_values[res_idx]
                        break
                else:
                    stable_count = 0

    ph.plot_stabilization_heatmap(heatmap_data,
                                  x_labels=np.round(zmax_values, 2),
                                  y_labels=[f"{i + 1}" for i in range(num_kernels)],
                                  title=f"{kernel_type.upper()} kernel stabilization resolution",
                                  forecast_year=forecast_year,
                                  annotate_max=annotate_max)


def plot_gbias_values_zres_sweep(galaxy_bias_resolutions, forecast_year, precision=0.1, stability_steps=10,
                                 marker_size=5):
    labels = [f"{i + 1}" for i in range(len(galaxy_bias_resolutions[next(iter(galaxy_bias_resolutions))]))]
    ph.plot_resolution_sweep_subplots(
        data_resolutions={res: galaxy_bias_resolutions[res] for res in galaxy_bias_resolutions},
        labels=labels,
        y_label="bias at bin",
        title="Galaxy bias across redshift resolutions",
        forecast_year=forecast_year,
        precision=precision,
        stability_steps=stability_steps,
        marker_size=marker_size
    )


def plot_gbias_value_zres_zmax_sweep(gbias_value_zres_zmax_sweep,
                                     forecast_year,
                                     precision=0.1,
                                     stability_steps=10,
                                     annotate_max=False):
    zmax_values = sorted(gbias_value_zres_zmax_sweep.keys())
    sample_zmax = zmax_values[0]
    resolution_values = sorted(gbias_value_zres_zmax_sweep[sample_zmax].keys())
    num_bins = len(gbias_value_zres_zmax_sweep[sample_zmax][resolution_values[0]])
    heatmap_data = np.full((num_bins, len(zmax_values)), np.nan)

    for zmax_idx, zmax in enumerate(zmax_values):
        for bin_idx in range(num_bins):
            bin_bias_values = [gbias_value_zres_zmax_sweep[zmax][res][bin_idx] for res in resolution_values]
            avg_value = np.mean(bin_bias_values)
            margin = avg_value * (precision / 100)
            upper_band, lower_band = avg_value + margin, avg_value - margin

            stable_count = 0
            for res_idx, value in enumerate(bin_bias_values):
                if lower_band <= value <= upper_band:
                    stable_count += 1
                    if stable_count >= stability_steps:
                        heatmap_data[bin_idx, zmax_idx] = resolution_values[res_idx]
                        break
                else:
                    stable_count = 0

    ph.plot_stabilization_heatmap(heatmap_data,
                                  x_labels=np.round(zmax_values, 2),
                                  y_labels=[f"{i + 1}" for i in range(num_bins)],
                                  title="Galaxy bias stabilization resolution",
                                  forecast_year=forecast_year,
                                  annotate_max=annotate_max)


def compare_two_data_vector_sets_absolute(data_vector_1,
                                          data_vector_2,
                                          ells_1,
                                          ells_2,
                                          cmap_1="cmr.pride",
                                          cmap_2="viridis", ):
    data_len = len(data_vector_1.shape[1])
    colors_1 = ph.get_colors(data_len, cmap=cmap_1, cmap_range=(0.15, 0.85))
    colors_2 = ph.get_colors(data_len, cmap=cmap_2, cmap_range=(0.15, 0.85))

    for i in range(data_vector_1.shape[1]):  # Loop over 15 cls
        plt.loglog(ells_1, data_vector_1[:, i], c=colors_1[i])
        plt.loglog(ells_2, data_vector_2[:, i], c=colors_2[i], ls=":", lw=3)

    plt.xlabel("Multipole scale $\\ell$", fontsize=18)
    plt.ylabel("Angular power spectrum $C_{\\ell}$", fontsize=18)
    # plt.legend()
    plt.show()



def compare_two_data_vector_sets_relative(data_vector_1,
                                          data_vector_2,
                                          ells,
                                          cmap_1="cmr.pride",
                                          label_1="X",
                                          label_2="Y"):
    num_colors = data_vector_1.shape[1]  # Number of colors needed for each `cls`
    colors = ph.get_colors(num_colors, cmap=cmap_1, cmap_range=(0.15, 0.85))

    for i in range(num_colors):
        plt.plot(ells, data_vector_1[:, i] / data_vector_2[:, i] - 1, c=colors[i])
        plt.axhline(0, c="gray")

    plt.plot([], [], c="white", label=f"X: {label_1} \n Y: {label_2}")

    plt.xlabel("Multipole Scale $\\ell$", fontsize=18)
    plt.ylabel("$C_{\\ell}^X / C_{\\ell}^Y - 1 $", fontsize=18)
    plt.legend(fontsize=18, frameon=False)
    plt.show()

def plot_stabilization_vs_precision(bin_centers_resolutions,
                                     bin_type,
                                     precisions=(1, 5, 10, 15),
                                     stability_steps=10):
    """
    Plot the resolution at which bin centers stabilize across different precision bands.

    Parameters:
        bin_centers_resolutions (dict): Nested dictionary containing bin centers for each redshift resolution.
        bin_type (str): Either "source" or "lens" to specify which bin centers to plot.
        precisions (tuple): A sequence of precision values to loop over (e.g., (1, 5, 10, 15)).
        stability_steps (int): Number of consecutive steps within the band required to consider it stable (default: 5).
    """
    # Extract resolutions and sort them
    resolutions = sorted(map(int, bin_centers_resolutions.keys()))

    # Get the number of bins from the first resolution's bin centers
    bin_keys = list(bin_centers_resolutions[(resolutions[0])][f"{bin_type}_bin_centers"].keys())
    colors = ph.get_colors(len(bin_keys))

    # Prepare a figure
    fig, ax = plt.subplots(figsize=(8, 3))

    # Loop over each bin and precision
    for bin_key in bin_keys:
        stabilization_points = []

        for precision in precisions:
            bin_center_values = [
                bin_centers_resolutions[res][f"{bin_type}_bin_centers"][bin_key] for res in resolutions
            ]
            avg_bin_center = np.mean(bin_center_values)
            margin = avg_bin_center * (precision / 100)
            upper_band = avg_bin_center + margin
            lower_band = avg_bin_center - margin

            # Find the first resolution with stable values within the band
            stable_count = 0
            stabilization_resolution = None
            for res, value in zip(resolutions, bin_center_values):
                if lower_band <= value <= upper_band:
                    stable_count += 1
                else:
                    stable_count = 0  # Reset count if the value goes outside the band

                if stable_count >= stability_steps:
                    stabilization_resolution = res
                    break

            stabilization_points.append(stabilization_resolution)

        # Plot stabilization resolution as a function of precision
        ax.plot(precisions, stabilization_points, marker='o', label=f"{bin_key + 1}", c=colors[bin_key])

    # Add labels and legend
    ax.set_xlabel("precision band", fontsize=14)
    ax.set_ylabel("stabilization resolution", fontsize=14)
    ax.legend(title="tomographic bin", fontsize=10)

    plt.tight_layout()
    plt.show()