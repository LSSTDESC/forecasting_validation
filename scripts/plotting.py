import matplotlib.pyplot as plt
import numpy as np
import cmasher as cmr
import seaborn as sns


def get_colors(data, cmap="cmr.pride", cmap_range=(0.15, 0.85)):
    num = len(data)
    colors = cmr.take_cmap_colors(cmap, num, cmap_range=cmap_range, return_fmt='hex')

    return colors


def compare_two_data_vector_sets_absolute(data_vector_1,
                                          data_vector_2,
                                          ells_1,
                                          ells_2,
                                          cmap_1="cmr.pride",
                                          cmap_2="viridis", ):
    colors_1 = get_colors(data_vector_1, cmap=cmap_1, cmap_range=(0.15, 0.85))
    colors_2 = get_colors(data_vector_2, cmap=cmap_2, cmap_range=(0.15, 0.85))

    for i in range(data_vector_1.shape[1]):  # Loop over 15 cls
        plt.loglog(ells_1, data_vector_1[:, i], c=colors_1[i])
        plt.loglog(ells_2, data_vector_2[:, i], c=colors_2[i], ls=":", lw=3)

    plt.xlabel("Multipole Scale $\\ell$", fontsize=18)
    plt.ylabel("Angular Power Spectrum $C_{\\ell}$", fontsize=18)
    # plt.legend()
    plt.show()


def compare_two_data_vector_sets_relative(data_vector_1,
                                          data_vector_2,
                                          ells,
                                          cmap_1="cmr.pride",
                                          label_1="X",
                                          label_2="Y"):
    num_colors = data_vector_1.shape[1]  # Number of colors needed for each `cls`
    colors = get_colors(data_vector_1, cmap=cmap_1, cmap_range=(0.15, 0.85))

    for i in range(num_colors):
        plt.plot(ells, data_vector_1[:, i] / data_vector_2[:, i] - 1, c=colors[i])
        plt.axhline(0, c="gray")

    plt.plot([], [], c="white", label=f"X: {label_1} \n Y: {label_2}")

    plt.xlabel("Multipole Scale $\\ell$", fontsize=18)
    plt.ylabel("$C_{\\ell}^X / C_{\\ell}^Y - 1 $", fontsize=18)
    plt.legend(fontsize=18, frameon=False)
    plt.show()


def plot_comoving_distance_comparison(abs_diff, rel_diff, redshift_fine):
    """
    Plots absolute and relative differences in comoving distances.

    Args:
        abs_diff (np.ndarray): Absolute differences on the fine grid.
        rel_diff (np.ndarray): Relative differences on the fine grid.
        redshift_fine (np.ndarray): The fine redshift grid for plotting.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)

    colors = cmr.take_cmap_colors("cmr.pride", 5, cmap_range=(0.15, 0.85), return_fmt='hex')

    # Plot absolute difference
    ax1.plot(redshift_fine, abs_diff, color=colors[1], label='Absolute Difference')
    ax1.set_ylabel('Absolute Difference [Mpc]')
    ax1.legend()

    # Plot relative difference
    ax2.plot(redshift_fine, rel_diff, color=colors[3], label='Relative Difference')
    ax2.set_ylabel('Relative Difference')
    ax2.set_xlabel('Redshift')
    ax2.legend()

    plt.tight_layout()
    plt.show()


def plot_bin_centers_subplots(bin_centers_resolutions,
                              zmax,
                              forecast_year,
                              bin_type="source",
                              precision=None,
                              marker_size=5,
                              stability_steps=10,
                              include_error_bars=False,
                              fig_format=".pdf",
                              title_pad=0.95):
    """
    Plot bin centers across different redshift resolutions in separate subplots with an optional averaged precision band.
    Adds a vertical line at the resolution where the bin center stabilizes within the desired band for `stability_steps` consecutive steps.

    Parameters:
        zmax (float): Maximum redshift value for the bin centers.
        forecast_year (str): Forecast year for the bin centers.
        bin_centers_resolutions (dict): Nested dictionary containing bin centers for each redshift resolution.
        bin_type (str): Either "source" or "lens" to specify which bin centers to plot.
        precision (float): precision for the averaged band (default: None).
        marker_size (int): Size of the markers for the bin centers (default: 5).
        stability_steps (int): Number of consecutive steps within the band required for stabilization (default: 10).
        include_error_bars (bool): Whether to include error bars in the plot (default: False).
        fig_format (str): File format for saving the figure (default: ".pdf").
        title_pad (float): Padding for the title above the subplots (default: 1.02).
    """
    # Extract resolutions and sort them
    resolutions = sorted(map(int, bin_centers_resolutions.keys()))

    # Get the number of bins from the first resolution's bin centers
    bin_keys = list(bin_centers_resolutions[resolutions[0]][f"{bin_type}_bin_centers"].keys())
    num_bins = len(bin_keys)
    colors = get_colors(bin_keys)

    # Set up subplots
    fig, axes = plt.subplots(num_bins, 1, figsize=(8, 2. * num_bins), sharex=True)
    # Add a title above all subplots with padding
    fig.suptitle(f"{bin_type.capitalize()} Bin Centers Across Redshift Resolutions",
                 fontsize=16,
                 y=title_pad)

    # If there's only one bin, ensure axes is actually iterable
    if num_bins == 1:
        axes = [axes]

    # Plot each bin center in a separate subplot with optional averaged band
    for i, bin_key in enumerate(bin_keys):
        # Access `bin_centers_resolutions` with `res` as an integer key
        bin_center_values = [
            bin_centers_resolutions[res][f"{bin_type}_bin_centers"][bin_key] for res in resolutions
        ]
        bin_center_values = np.array(bin_center_values)
        std_dev = np.std(bin_center_values)  # Calculate standard deviation for error bars

        # Plot with error bars if requested
        if include_error_bars:
            axes[i].errorbar(resolutions, bin_center_values, yerr=std_dev, fmt='o', markersize=marker_size,
                         color=colors[i], label=f"Bin {bin_key + 1}")

        axes[i].plot(resolutions, bin_center_values, marker='o', markersize=marker_size, c=colors[i])
        axes[i].set_ylabel(f"bin {bin_key + 1} centre")
        axes[i].locator_params(axis='x', nbins=20)

        # include ticks everywhere top bottom left right and inwards
        axes[i].tick_params(axis='both', which='both', direction='in', top=True, right=True)

        #axes[i].legend(fontsize=18)

        # If precision is provided, calculate and plot the averaged band
        if precision:
            avg_bin_center = np.mean(bin_center_values)
            margin = avg_bin_center * (precision / 100)
            upper_band = avg_bin_center + margin
            lower_band = avg_bin_center - margin
            axes[i].fill_between(resolutions, lower_band, upper_band, color='gray', alpha=0.2,
                                 label=f"±{precision}% Band (around average)")



            # Stabilization check: Find the first resolution with stable values within the band
            stable_count = 0
            for res, value in zip(resolutions, bin_center_values):
                if lower_band <= value <= upper_band:
                    stable_count += 1
                else:
                    stable_count = 0  # Reset count if the value goes outside the band

                # If we have enough consecutive stable points, mark the stabilization point
                if stable_count >= stability_steps:
                    axes[i].axvline(x=res, color='red', linestyle='--', label=f'Stable at {res}')
                    axes[i].text(res, avg_bin_center, f'{res}', color='red', va='top', ha='right',
                                 fontsize=10, rotation=90)
                    break

    # Set x-axis label on the last subplot
    axes[-1].set_xlabel("redshift resolution", fontsize=18)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig_name = f"{bin_type}_bin_centers_sweep_zmax{zmax}_y{forecast_year}{fig_format}"
    plt.savefig(f"plots_output/{fig_name}")


def plot_stabilization_vs_precision(bin_centers_resolutions,
                                     bin_type="source",
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
    colors = get_colors(bin_keys)

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


def plot_tomobin_stabilization_resolution_heatmap(bin_centers_by_zmax,
                                                  forecast_year="1",
                                                  bin_type="source",
                                                  precision=0.1,
                                                  stability_steps=10,
                                                  annotate_max=False,
                                                  fig_format=".pdf"):
    """
    Create a seaborn heatmap of the redshift resolution where stabilization occurs for each bin and `zmax` value,
    with an option to annotate only the maximum stabilization resolution for each `zmax` column.

    Parameters:
        bin_centers_by_zmax (dict): Nested dictionary containing bin centers for each `zmax` and resolution.
        bin_type (str): Either "source" or "lens" to specify which bin centers to plot.
        precision (float): Desired precision (%) for stabilization band (default 0.1 %).
        stability_steps (int): Number of consecutive steps within the band required for stabilization (default 10).
        forecast_year (str): The forecast year (default "1").
        fig_format (str): File format for saving the figure (default ".pdf").
        annotate_max (bool): Whether to annotate only the maximum stabilization resolution in each column (default: False).
    """
    # Extract sorted `zmax` values and resolutions
    zmax_values = sorted(bin_centers_by_zmax.keys())
    sample_zmax = zmax_values[0]
    resolution_values = sorted(bin_centers_by_zmax[sample_zmax].keys())

    # Extract bin keys
    bin_keys = list(bin_centers_by_zmax[sample_zmax][resolution_values[0]][f"{bin_type}_bin_centers"].keys())

    # Initialize a matrix for stabilization resolution heatmap data
    heatmap_data = np.full((len(bin_keys), len(zmax_values)), np.nan)

    # Find the stabilization resolution for each `zmax` and bin
    for zmax_idx, zmax in enumerate(zmax_values):
        for bin_idx, bin_key in enumerate(bin_keys):
            # Retrieve bin center values across resolutions for current `zmax` and bin
            bin_center_values = [
                bin_centers_by_zmax[zmax][res][f"{bin_type}_bin_centers"][bin_key] for res in resolution_values
            ]

            # Calculate stabilization based on precision band
            avg_bin_center = np.mean(bin_center_values)
            margin = avg_bin_center * (precision / 100)
            upper_band = avg_bin_center + margin
            lower_band = avg_bin_center - margin

            # Identify first stabilization resolution
            stable_count = 0
            for res_idx, value in enumerate(bin_center_values):
                if lower_band <= value <= upper_band:
                    stable_count += 1
                    if stable_count >= stability_steps:
                        heatmap_data[bin_idx, zmax_idx] = resolution_values[res_idx]
                        break
                else:
                    stable_count = 0  # Reset if outside band

    # Create the heatmap plot
    cmap = cmr.get_sub_cmap('cmr.pride', 0.15, 0.85)
    plt.figure(figsize=(10, len(bin_keys) * 0.5))
    ax = sns.heatmap(heatmap_data, annot=False if annotate_max else True, fmt=".0f", cmap=cmap,
                     cbar_kws={'label': 'Stabilization Resolution'},
                     xticklabels=np.round(zmax_values, 2), yticklabels=[f"{i + 1}" for i in bin_keys])

    # If annotating only max values, find max value in each column and annotate it
    if annotate_max:
        for zmax_idx in range(len(zmax_values)):
            max_row_idx = np.nanargmax(heatmap_data[:, zmax_idx])  # Find the row with the max value in this column
            max_value = heatmap_data[max_row_idx, zmax_idx]
            ax.text(zmax_idx + 0.5, max_row_idx + 0.5, f"{int(max_value)}",
                    ha='center', va='center', color='white', fontsize=10)

    # Add title and labels
    ax.set_title(f"Stabilization $z_\mathrm{{res}}$ across $z_\mathrm{{max}}$ values LSST Y{forecast_year}",
                 fontsize=14)
    ax.set_xlabel("$z_\mathrm{{max}}$", fontsize=12)
    ax.set_ylabel("tomo bin", fontsize=12)

    # Save the figure
    fig_name = f"{bin_type}_stabilization_resolution_heatmap_zmax_sweep_y{forecast_year}{fig_format}"
    plt.savefig(f"plots_output/{fig_name}")
    plt.show()


def plot_kernel_peaks_z_resolution(peaks_by_resolution,
                                   forecast_year,
                                   kernel_type="wl",
                                   precision=0.1,
                                   stability_steps=10,
                                   marker_size=3,
                                   fig_format=".pdf",
                                   title_pad=0.95):
    """
    Plot kernel peaks across different redshift resolutions.

    Parameters:
        peaks_by_resolution (dict): Output from kernel_peaks_z_resolution_sweep, a nested dictionary with
                resolutions as keys. Format: {resolution: {"wl": [(z_peak, value_peak), ...], "nc": [(z_peak, value_peak), ...]}}
        forecast_year (str): Forecasting year for the plot title.
        kernel_type (str): Which kernel to plot: "wl" for weak lensing, "nc" for number counts.
        precision (float): Desired precision (%) for stabilization band (default 0.1 %).
        stability_steps (int): Number of consecutive steps within the band required for stabilization (default 10).
        marker_size (int): Size of the markers for the peaks (default: 5).
        fig_format (str): File format for saving the figure (default: ".pdf").
        title_pad (float): Padding for the title above the subplots (default: 0.95).
    """
    # Extract resolutions and sort them for consistent plotting
    resolutions = sorted(peaks_by_resolution.keys())

    # Determine number of subplots based on kernel type
    num_kernels = len(peaks_by_resolution[resolutions[0]][kernel_type]) if kernel_type in ["wl", "nc"] else max(
        len(peaks_by_resolution[resolutions[0]]["wl"]), len(peaks_by_resolution[resolutions[0]]["nc"]))

    # Generate distinct colors for each subplot/kernel
    colors = get_colors(range(num_kernels))
    fig, axes = plt.subplots(num_kernels, 1, figsize=(8, 2. * num_kernels), sharex=True)
    prefix = f"LSST Y{forecast_year} WL" if kernel_type == "wl" else f"LSST Y{forecast_year} NC"
    fig.suptitle(f"{prefix} Kernel Peaks Across Redshift Resolutions", fontsize=16, y=title_pad)

    # Make sure axes is always iterable
    if num_kernels == 1:
        axes = [axes]

    # Plot each kernel peak in a separate subplot
    for i in range(num_kernels):
        z_peaks = []
        for res in resolutions:
            if kernel_type == "wl":
                wl_peaks = peaks_by_resolution[res]["wl"]
                if i < len(wl_peaks):  # Check if kernel i exists
                    z_peak, _ = wl_peaks[i]
                    z_peaks.append(z_peak)
            elif kernel_type == "nc":
                nc_peaks = peaks_by_resolution[res]["nc"]
                if i < len(nc_peaks):  # Check if kernel i exists
                    z_peak, _ = nc_peaks[i]
                    z_peaks.append(z_peak)

        # Plot the line connecting all z_peaks for the current kernel
        axes[i].plot(resolutions, z_peaks, '-o', markersize=marker_size, color=colors[i])

        # Labeling each subplot
        axes[i].set_ylabel(rf"$W^{i + 1} \: z_\mathrm{{peak}}$")
        if i == num_kernels - 1:
            axes[i].set_xlabel("Redshift Resolution", fontsize=18)

        # include ticks everywhere top bottom left right and inwards
        axes[i].tick_params(axis='both', which='both', direction='in', top=True, right=True)
        # Set more ticks and finer detail on axes
        axes[i].locator_params(axis='x', nbins=20)  # Adjusts the number of x-axis ticks

        # Calculate average peak and precision band
        if precision:
            z_peaks = [peaks_by_resolution[res][kernel_type][i][0] for res in resolutions if
                       i < len(peaks_by_resolution[res][kernel_type])]
            avg_peak = np.mean(z_peaks)
            margin = avg_peak * (precision / 100)
            upper_band = avg_peak + margin
            lower_band = avg_peak - margin
            axes[i].fill_between(resolutions, lower_band, upper_band, color='gray', alpha=0.2,
                                 label=f"±{precision}% Band (around average)")

        # Optional stabilization check and line
        stable_count = 0
        for res, z_peak in zip(resolutions, z_peaks):
            if lower_band <= z_peak <= upper_band:
                stable_count += 1
            else:
                stable_count = 0
            if stable_count >= stability_steps:
                axes[i].axvline(x=res, color='red', linestyle='--', label=f'Stable at {res}')
                axes[i].text(res, avg_peak, f'{res}', color='red', va='top', ha='right', fontsize=10, rotation=90)
                break

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fname_prefix = f"wl_y{forecast_year}_" if kernel_type == "wl" else f"nc_y{forecast_year}_"
    fig_name = f"plots_output/{fname_prefix}kernel_peaks_resolution_sweep{fig_format}"
    plt.savefig(fig_name)
    plt.show()


def plot_kernel_stabilization_resolution_heatmap(kernel_peaks_by_zmax_and_resolution,
                                                 forecast_year,
                                                 kernel_type="wl",
                                                 precision=0.1,
                                                 stability_steps=10,
                                                 annotate_max=False,
                                                 fig_format=".pdf"):
    """
    Create a heatmap of the redshift resolution where kernel stabilization occurs for each kernel and `zmax` value.

    Parameters:
        kernel_peaks_by_zmax_and_resolution (dict): Nested dictionary containing kernel peaks for each `zmax` and resolution.
        kernel_type (str): Either "wl" for weak lensing or "nc" for number counts kernels.
        precision (float): Desired precision (%) for stabilization band (default 0.1 %).
        stability_steps (int): Number of consecutive steps within the band required for stabilization (default 10).
        annotate_max (bool): Whether to annotate only the maximum stabilization resolution in each column (default: False).
        forecast_year (str): The forecast year (default "1").
        fig_format (str): File format for saving the figure (default ".pdf").
    """
    # Extract sorted `zmax` values and resolutions
    zmax_values = sorted(kernel_peaks_by_zmax_and_resolution.keys())
    sample_zmax = zmax_values[0]
    resolution_values = sorted(kernel_peaks_by_zmax_and_resolution[sample_zmax].keys())

    # Get the number of kernels for the specified kernel type
    num_kernels = len(kernel_peaks_by_zmax_and_resolution[sample_zmax][resolution_values[0]][kernel_type])

    # Initialize a matrix for stabilization resolution heatmap data
    heatmap_data = np.full((num_kernels, len(zmax_values)), np.nan)

    # Find the stabilization resolution for each `zmax` and kernel
    for zmax_idx, zmax in enumerate(zmax_values):
        for kernel_idx in range(num_kernels):
            # Retrieve kernel peak values across resolutions for current `zmax` and kernel
            kernel_peak_values = [
                kernel_peaks_by_zmax_and_resolution[zmax][res][kernel_type][kernel_idx][0]
                for res in resolution_values
                if kernel_idx < len(kernel_peaks_by_zmax_and_resolution[zmax][res][kernel_type])
            ]

            # Calculate stabilization based on precision band
            avg_peak_value = np.mean(kernel_peak_values)
            margin = avg_peak_value * (precision / 100)
            upper_band = avg_peak_value + margin
            lower_band = avg_peak_value - margin

            # Identify first stabilization resolution
            stable_count = 0
            for res_idx, value in enumerate(kernel_peak_values):
                if lower_band <= value <= upper_band:
                    stable_count += 1
                    if stable_count >= stability_steps:
                        heatmap_data[kernel_idx, zmax_idx] = resolution_values[res_idx]
                        break
                else:
                    stable_count = 0  # Reset if outside band

    # Create the heatmap plot
    cmap = cmr.get_sub_cmap('cmr.pride', 0.15, 0.85)
    plt.figure(figsize=(10, num_kernels * 0.5))
    ax = sns.heatmap(heatmap_data, annot=not annotate_max, fmt=".0f", cmap=cmap,
                     cbar_kws={'label': 'Stabilization Resolution'},
                     xticklabels=np.round(zmax_values, 2), yticklabels=[f"{i + 1}" for i in range(num_kernels)])

    # If annotating only max values, find max value in each column and annotate it
    if annotate_max:
        for zmax_idx in range(len(zmax_values)):
            max_row_idx = np.nanargmax(heatmap_data[:, zmax_idx])  # Find the row with the max value in this column
            max_value = heatmap_data[max_row_idx, zmax_idx]
            ax.text(zmax_idx + 0.5, max_row_idx + 0.5, f"{int(max_value)}",
                    ha='center', va='center', color='white', fontsize=10)

    # Add title and labels
    kernel_label = "Weak Lensing" if kernel_type == "wl" else "Number Counts"
    ax.set_title(f"{kernel_label} kernel stabilization across $z_\\mathrm{{max}}$ LSST Y{forecast_year}",
                 fontsize=14)
    ax.set_xlabel("$z_\\mathrm{max}$", fontsize=12)
    ax.set_ylabel("kernel index", fontsize=12)

    # Save the figure
    fig_name = f"{kernel_type}_kernel_stabilization_resolution_heatmap_zmax_sweep_y{forecast_year}{fig_format}"
    plt.savefig(f"plots_output/{fig_name}")
    plt.show()


def plot_galaxy_bias_resolutions(galaxy_bias_resolutions,
                                 forecast_year,
                                 precision=0.1,
                                 stability_steps=10,
                                 marker_size=5):
    """
    Plot galaxy bias across different redshift resolutions.

    Parameters:
        galaxy_bias_resolutions (dict): Galaxy bias values across resolutions.
        forecast_year (str): Year of forecast for plot labeling.
        precision (float): Precision band percentage for stabilization check.
        stability_steps (int): Steps required within precision band for stabilization.
    """
    resolutions = sorted(galaxy_bias_resolutions.keys())
    num_bins = len(galaxy_bias_resolutions[resolutions[0]])

    colors = cmr.take_cmap_colors("cmr.pride", num_bins, cmap_range=(0.15, 0.85), return_fmt='hex')
    fig, axes = plt.subplots(num_bins, 1, figsize=(8, 2. * num_bins), sharex=True)
    fig.suptitle(f"Galaxy Bias Across Redshift Resolutions - Forecast Year {forecast_year}", fontsize=16)

    for i in range(num_bins):
        bin_bias_values = [galaxy_bias_resolutions[res][i] for res in resolutions]
        axes[i].plot(resolutions, bin_bias_values, '-o', markersize=marker_size, color=colors[i], label=f"Bin {i + 1}")

        # Precision band calculation and plotting
        avg_bias = np.mean(bin_bias_values)
        margin = avg_bias * (precision / 100)
        upper_band, lower_band = avg_bias + margin, avg_bias - margin
        axes[i].fill_between(resolutions, lower_band, upper_band, color='lightgray', alpha=0.3)

        # Stabilization check
        stable_count = 0
        for res, bias in zip(resolutions, bin_bias_values):
            if lower_band <= bias <= upper_band:
                stable_count += 1
                if stable_count >= stability_steps:
                    axes[i].axvline(res, color='red', linestyle='--')
                    # Add text label on the stabilization line
                    axes[i].text(res, avg_bias, f'{res}', color='red',
                                 va='top', ha='right', fontsize=10, rotation=90)
                    break
            else:
                stable_count = 0

        axes[i].set_ylabel(f"Bin {i + 1} Bias")
        axes[i].legend()

    axes[-1].set_xlabel("Redshift Resolution")
    plt.tight_layout()
    plt.show()


def plot_galaxy_bias_stabilization_heatmap(galaxy_bias_by_zmax_and_resolution,
                                           forecast_year,
                                           precision=0.1,
                                           stability_steps=10,
                                           annotate_max=False,
                                           fig_format=".pdf"):
    """
    Create a heatmap showing the redshift resolution where galaxy bias stabilization occurs for each bin and `zmax`.

    Parameters:
        galaxy_bias_by_zmax_and_resolution (dict): Nested dictionary containing galaxy bias for each `zmax` and resolution.
        precision (float): Desired precision (%) for stabilization band.
        stability_steps (int): Number of consecutive steps within the band required for stabilization.
        annotate_max (bool): Whether to annotate only the maximum stabilization resolution in each column.
        forecast_year (str): The forecast year (default "1").
        fig_format (str): File format for saving the figure.
    """
    # Extract sorted `zmax` values and resolutions
    zmax_values = sorted(galaxy_bias_by_zmax_and_resolution.keys())
    sample_zmax = zmax_values[0]
    resolution_values = sorted(galaxy_bias_by_zmax_and_resolution[sample_zmax].keys())

    # Get the number of bins
    num_bins = len(galaxy_bias_by_zmax_and_resolution[sample_zmax][resolution_values[0]])

    # Initialize matrix for heatmap data
    heatmap_data = np.full((num_bins, len(zmax_values)), np.nan)

    # Calculate stabilization resolution for each `zmax` and bin
    for zmax_idx, zmax in enumerate(zmax_values):
        for bin_idx in range(num_bins):
            # Collect galaxy bias values across resolutions for the current `zmax` and bin
            bin_bias_values = [galaxy_bias_by_zmax_and_resolution[zmax][res][bin_idx] for res in resolution_values]
            avg_bias = np.mean(bin_bias_values)
            margin = avg_bias * (precision / 100)
            upper_band, lower_band = avg_bias + margin, avg_bias - margin

            # Identify first stabilization resolution
            stable_count = 0
            for res_idx, value in enumerate(bin_bias_values):
                if lower_band <= value <= upper_band:
                    stable_count += 1
                    if stable_count >= stability_steps:
                        heatmap_data[bin_idx, zmax_idx] = resolution_values[res_idx]
                        break
                else:
                    stable_count = 0  # Reset if outside band

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

    # Add title and labels
    ax.set_title(f"Galaxy Bias Stabilization across $z_\\mathrm{{max}}$ - LSST Y{forecast_year}", fontsize=14)
    ax.set_xlabel("$z_\\mathrm{max}$", fontsize=12)
    ax.set_ylabel("Bin Index", fontsize=12)

    # Save the figure
    fig_name = f"galaxy_bias_stabilization_heatmap_zmax_sweep_y{forecast_year}{fig_format}"
    plt.savefig(f"plots_output/{fig_name}")
    plt.show()
