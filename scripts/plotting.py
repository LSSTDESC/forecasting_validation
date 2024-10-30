import matplotlib.pyplot as plt
import numpy as np
import cmasher as cmr


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
                                          label_2="Y"
                                          ):
    colors = get_colors(data_vector_1, cmap=cmap_1, cmap_range=(0.15, 0.85))

    for i in range(data_vector_1.shape[1]):  # Loop over 15 cls
        plt.plot(ells, data_vector_1[:, i] / data_vector_2[:, i] - 1, c=colors[3])
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
                              percentage=None,
                              marker_size=5,
                              stability_steps=10,
                              fig_format=".pdf"):
    """
    Plot bin centers across different redshift resolutions in separate subplots with an optional averaged percentage band.
    Adds a vertical line at the resolution where the bin center stabilizes within the desired band for `stability_steps` consecutive steps.

    Parameters:
        bin_centers_resolutions (dict): Nested dictionary containing bin centers for each redshift resolution.
        bin_type (str): Either "source" or "lens" to specify which bin centers to plot.
        percentage (float, optional): Percentage for the band around the average bin center. If None, no band is plotted.
        marker_size (int): Size of the markers in the plot (default: 6).
        stability_steps (int): Number of consecutive steps within the band required to consider it stable (default: 5).
    """
    # Extract resolutions and sort them
    resolutions = sorted(map(int, bin_centers_resolutions.keys()))

    # Get the number of bins from the first resolution's bin centers
    bin_keys = list(bin_centers_resolutions[resolutions[0]][f"{bin_type}_bin_centers"].keys())
    num_bins = len(bin_keys)
    colors = get_colors(bin_keys)

    # Set up subplots
    fig, axes = plt.subplots(num_bins, 1, figsize=(8, 2. * num_bins), sharex=True)
    fig.suptitle(f"{bin_type.capitalize()} Bin Centers Across Redshift Resolutions", fontsize=16)

    # If there's only one bin, ensure axes is actually iterable
    if num_bins == 1:
        axes = [axes]

    # Plot each bin center in a separate subplot with optional averaged band
    for i, bin_key in enumerate(bin_keys):
        # Access `bin_centers_resolutions` with `res` as an integer key
        bin_center_values = [
            bin_centers_resolutions[res][f"{bin_type}_bin_centers"][bin_key] for res in resolutions
        ]
        axes[i].plot(resolutions, bin_center_values, marker='o', markersize=marker_size, c=colors[i])
        axes[i].set_ylabel(f"bin {bin_key + 1} centre")

        #axes[i].legend(fontsize=18)

        # If percentage is provided, calculate and plot the averaged band
        if percentage:
            avg_bin_center = np.mean(bin_center_values)
            margin = avg_bin_center * (percentage / 100)
            upper_band = avg_bin_center + margin
            lower_band = avg_bin_center - margin
            axes[i].fill_between(resolutions, lower_band, upper_band, color='gray', alpha=0.2,
                                 label=f"±{percentage}% Band (around average)")

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


def plot_stabilization_vs_percentage(bin_centers_resolutions,
                                     bin_type="source",
                                     percentages=(1, 5, 10, 15),
                                     stability_steps=10):
    """
    Plot the resolution at which bin centers stabilize across different percentage bands.

    Parameters:
        bin_centers_resolutions (dict): Nested dictionary containing bin centers for each redshift resolution.
        bin_type (str): Either "source" or "lens" to specify which bin centers to plot.
        percentages (tuple): A sequence of percentage values to loop over (e.g., (1, 5, 10, 15)).
        stability_steps (int): Number of consecutive steps within the band required to consider it stable (default: 5).
    """
    # Extract resolutions and sort them
    resolutions = sorted(map(int, bin_centers_resolutions.keys()))

    # Get the number of bins from the first resolution's bin centers
    bin_keys = list(bin_centers_resolutions[(resolutions[0])][f"{bin_type}_bin_centers"].keys())
    colors = get_colors(bin_keys)

    # Prepare a figure
    fig, ax = plt.subplots(figsize=(8, 3))

    # Loop over each bin and percentage
    for bin_key in bin_keys:
        stabilization_points = []

        for percentage in percentages:
            bin_center_values = [
                bin_centers_resolutions[(res)][f"{bin_type}_bin_centers"][bin_key] for res in resolutions
            ]
            avg_bin_center = np.mean(bin_center_values)
            margin = avg_bin_center * (percentage / 100)
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

        # Plot stabilization resolution as a function of percentage
        ax.plot(percentages, stabilization_points, marker='o', label=f"Bin {bin_key + 1}", c=colors[bin_key])

    # Add labels and legend
    ax.set_xlabel("Percentage Band", fontsize=14)
    ax.set_ylabel("Stabilization Resolution", fontsize=14)
    ax.set_title(f"Stabilization Resolution vs. Percentage Band for {bin_type.capitalize()} Bins", fontsize=16)
    ax.legend(title="Bin", fontsize=10)

    plt.tight_layout()
    plt.show()

