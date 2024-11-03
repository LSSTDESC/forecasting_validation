import numpy as np
from scripts.tomographic_binning import TomographicBinning
from scripts.presets import Presets
from scipy.integrate import cumulative_trapezoid

from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz


class NZMetrics:
    def __init__(self, presets):
        if not isinstance(presets, Presets):
            raise TypeError(f"Expected a Presets object, but received {type(presets).__name__}.")
        # Set attributes from presets
        self.cosmology = presets.cosmology
        self.ells = presets.ells
        self.forecast_year = presets.forecast_year
        self.redshift_max = presets.redshift_max
        self.redshift_resolution = presets.redshift_resolution
        self.redshift_range = presets.redshift_range
        self.perform_binning = presets.perform_binning
        self.robust_binning = presets.robust_binning
        self.ell_min = presets.ell_min
        self.ell_max = presets.ell_max
        self.ell_num = presets.ell_num
        self.save_data = presets.save_data

        self.source_bins = TomographicBinning(presets).source_bins()
        self.lens_bins = TomographicBinning(presets).lens_bins()

    def tomo_peaks_zres_sweep(self,
                              res_start=300,
                              res_end=10000,
                              step=50,
                              decimal_places=4):
        """
        Compare tomographic bin centers across a range of redshift resolutions.

        Parameters:
            res_start (int): Starting resolution for redshift range.
            res_end (int): Ending resolution for redshift range.
            step (int): Step increment.
            decimal_places (int): Number of decimal places to round the bin centers.
                                    Note that decreasing the number of decimal places will matter as if there are
                                    too few, the rounding will mask the fact there are differences in the bin centers.

        Returns:
            dict: Nested dictionary with bin centers for each redshift resolution.
        """
        bin_centers_resolutions = {}

        for resolution in range(res_start, res_end + 1, step):
            # Update Presets with the current resolution
            presets = Presets(
                cosmology=self.cosmology,
                redshift_max=self.redshift_max,
                redshift_resolution=resolution,  # Update resolution here
                ell_min=self.ell_min,
                ell_max=self.ell_max,
                ell_num=self.ell_num,
                forecast_year=self.forecast_year,
                perform_binning=self.perform_binning
            )

            # Reinitialize TomographicBinning with updated presets
            tomographic_binning = TomographicBinning(presets)

            # Calculate bin centers for source and lens bins
            source_bin_centers = tomographic_binning.source_bin_centers(decimal_places=decimal_places)
            lens_bin_centers = tomographic_binning.lens_bin_centers(decimal_places=decimal_places)

            # Store results in the dictionary, using the resolution as the key
            bin_centers_resolutions[resolution] = {
                "source_bin_centers": source_bin_centers,
                "lens_bin_centers": lens_bin_centers
            }

        # Save the full dictionary to a file
        extra_info = f"zmax{self.redshift_max}"
        self.save_data("tomo_peaks_zres_sweep",
                       bin_centers_resolutions,
                       "bin_centers",
                       extra_info=extra_info,
                       include_ccl_version=True)

        return bin_centers_resolutions

    def tomo_peaks_zres_and_zmax_sweep(self,
                                       zmax_start=3.0,
                                       zmax_end=4.0,
                                       zmax_step=0.1,
                                       res_start=300,
                                       res_end=10000,
                                       res_step=50,
                                       decimal_places=4):
        """
        Compare tomographic bin centers across a range of redshift resolutions and zmax values.

        Parameters:
            zmax_start (float): Starting value for zmax.
            zmax_end (float): Ending value for zmax.
            zmax_step (float): Step increment for zmax.
            res_start (int): Starting resolution for redshift range.
            res_end (int): Ending resolution for redshift range.
            res_step (int): Step increment for resolution.
            decimal_places (int): Number of decimal places to round the bin centers.

        Returns:
            dict: Nested dictionary with bin centers for each zmax and resolution.
        """
        bin_centers_by_zmax = {}

        for zmax in np.arange(zmax_start, zmax_end + zmax_step, zmax_step):
            # Round zmax to 1 decimal place for dictionary key
            # otherwise, floating point errors may cause issues
            # when calling the dictionary keys
            zmax_key = round(zmax, 1)
            bin_centers_by_zmax[zmax_key] = {}

            for resolution in range(res_start, res_end + 1, res_step):
                # Update Presets with the current resolution and zmax
                presets = Presets(
                    cosmology=self.cosmology,
                    redshift_max=zmax,
                    redshift_resolution=resolution,
                    ell_min=self.ell_min,
                    ell_max=self.ell_max,
                    ell_num=self.ell_num,
                    forecast_year=self.forecast_year,
                    perform_binning=self.perform_binning,
                    robust_binning=self.robust_binning
                )

                # Reinitialize TomographicBinning with updated presets
                tomographic_binning = TomographicBinning(presets)

                # Calculate bin centers for source and lens bins
                source_bin_centers = tomographic_binning.source_bin_centers(decimal_places=decimal_places)
                lens_bin_centers = tomographic_binning.lens_bin_centers(decimal_places=decimal_places)

                # Store results in the dictionary, organized by zmax and resolution
                bin_centers_by_zmax[zmax_key][resolution] = {
                    "source_bin_centers": source_bin_centers,
                    "lens_bin_centers": lens_bin_centers
                }

        # Save the full nested dictionary to a file
        extra_info = f"zmax_range_{zmax_start}_to_{zmax_end}"
        self.save_data("tomo_peaks_zres_and_zmax_sweep",
                       bin_centers_by_zmax,
                       "bin_centers",
                       extra_info=extra_info,
                       include_ccl_version=True)

        return bin_centers_by_zmax

    def _generate_draws(self, bin_dict, num_draws, seed):
        """
        Generate random draws based on a dictionary of bins.

        Parameters:
            bin_dict (dict): A dictionary where each key is a bin index and each value is the PDF for that bin.
            num_draws (int): The number of random draws to generate for each bin.
            seed (int): Seed for random number generator.

        Returns:
            dict: A dictionary where each bin index maps to an array of random draws.
        """
        np.random.seed(seed)  # Set a fixed random seed for reproducibility

        random_draws = {}

        # Use numpy vectorization to generate CDFs and perform interpolation for all bins
        for bin_key, pdf in bin_dict.items():
            # Compute the CDF using cumulative trapezoidal integration
            cdf = cumulative_trapezoid(pdf, self.redshift_range, initial=0)
            cdf /= cdf[-1]  # Normalize the CDF

            # Generate random draws using numpy.interp
            random_values = np.random.rand(num_draws)  # Uniform random values between 0 and 1
            random_draws[bin_key] = np.interp(random_values, cdf, self.redshift_range)

        return random_draws

    def generate_tomo_draws(self, bin_type, num_draws=10**3, seed=42, num_points=200):
        """
            Generate and analyze random tomographic redshift draws for a specified bin type (either source or lens bins).

            Parameters:
                bin_type (str): Type of bin to generate draws for. Options are "source" or "lens".
                num_draws (int): The number of random draws to generate for each bin. Default is 1000.
                seed (int): Seed for random number generator for reproducibility. Default is 42.
                num_points (int): Number of different bin sizes to analyze within each histogram. Default is 200.

            Returns:
                np.ndarray: A 3D array of shape (num_bins, num_points, 2) containing peak redshift information for each
                            tomographic bin and binning configuration.

            Output Array Structure:
                The output array, `max_values`, has dimensions `(num_bins, num_points, 2)`, where:
                - `num_bins` is the number of tomographic bins in `bin_type`.
                - `num_points` represents the number of different binning configurations (histogram bin sizes) analyzed.
                - Each entry `max_values[bin_index][idx]` contains two values:
                    - `num_bins_in_hist` (int): Number of bins in the histogram for this configuration, given by
                      `50 * (np.arange(num_points) + 1)`.
                    - `peak_bin_center` (float): Center of the bin with the highest density (peak) in the histogram for
                      this configuration, representing the most probable redshift in that binning scheme.

            Description:
                1. For each tomographic bin (source or lens), this method generates random redshift draws using
                   inverse transform sampling based on the bin's PDF.
                2. Histograms are created for each bin using different bin sizes (defined by `50 * (np.arange(num_points) + 1)`).
                3. For each histogram, the bin center with the highest density is recorded to capture the peak redshift.
                4. Results are saved to a file with metadata specifying the number of draws, seed, and number of points.

            Example:
                max_values = generate_tomo_draws("source", num_draws=1000, seed=42, num_points=200)
            """
        bin_type_dict = {
            "source": self.source_bins,
            "lens": self.lens_bins
        }

        bins = bin_type_dict[bin_type]
        num_bins = len(bins.keys())  # Determine the number of bins dynamically

        max_values = np.empty((num_bins, num_points, 2), float)
        random_draws = self._generate_draws(bins, num_draws, seed)

        # Define bin sizes once, reducing repeated computation in the inner loop
        bin_sizes = 50 * (np.arange(num_points) + 1)

        # Predefine bin centers array outside loop to avoid redundant calculations
        for bin_index, (bin_key, draws) in enumerate(random_draws.items()):
            # Precompute histograms and centers in a vectorized way if possible
            for idx, num_bins_in_hist in enumerate(bin_sizes):
                # Generate histograms only if the bin size is new; avoid re-computation
                hist, bin_edges = np.histogram(draws, bins=num_bins_in_hist, density=True)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                max_idx = np.argmax(hist)
                max_values[bin_index][idx] = [num_bins_in_hist, bin_centers[max_idx]]

        # Save the data with efficient handling
        self.save_data(f"{bin_type}_tomo_bin_draws",
                       max_values,
                       "bin_centers",
                       extra_info=f"numdraws{num_draws:.0e}_seed{seed}_numpts{num_points}",
                       include_ccl_version=True)

        return max_values
