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

    def load_and_bootstrap_peak_estimates(self,
                                          bin_type,
                                          forecast_year,
                                          num_bootstrap=100,
                                          seed=42,
                                          num_points=200):
        """
        Load precomputed tomographic bin draws for a given forecast year and perform bootstrapping
        to estimate confidence intervals on peak redshifts for different histogram configurations.

        Parameters:
            bin_type (str): Type of bin to generate draws for ("source" or "lens").
            forecast_year (int): Forecast year to load precomputed data (e.g., 1, 5, 10).
            num_bootstrap (int): Number of bootstrap samples to generate.
            seed (int): Seed for reproducibility.
            num_points (int): Number of different bin sizes to analyze.

        Returns:
            dict: Dictionary of peak redshift distributions for each histogram bin size.
        """
        np.random.seed(seed)

        # Load precomputed data based on forecast year and bin type
        file_path = f"data_input/{bin_type}_tomo_bin_draws_forecast_year_{forecast_year}.npz"
        with np.load(file_path, allow_pickle=True) as data:
            max_values = data['bin_centers']  # Assuming "bin_centers" holds the 3D array max_values

        # Initialize dictionary to store bootstrapped peak estimates for each histogram bin size
        bin_sizes = 50 * (np.arange(num_points) + 1)
        bootstrap_peaks = {bin_size: [] for bin_size in bin_sizes}

        # Loop over each bin in the max_values array
        for bin_index in range(max_values.shape[0]):
            # Extract original draws for each bin
            draws = max_values[bin_index, :, 1]  # Extract peak bin centers from max_values (2nd column)

            for bin_size in bin_sizes:
                peaks = []
                for _ in range(num_bootstrap):
                    # Resample with replacement from precomputed draws
                    bootstrap_sample = np.random.choice(draws, size=len(draws), replace=True)

                    # Generate histogram and find peak
                    hist, bin_edges = np.histogram(bootstrap_sample, bins=bin_size, density=True)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    peak_bin_center = bin_centers[np.argmax(hist)]
                    peaks.append(peak_bin_center)

                # Store all bootstrap peaks for the current bin size
                bootstrap_peaks[bin_size].append(np.array(peaks))

        return bootstrap_peaks
 
    def load_and_bootstrap_resolution_intervals(self,
                                                forecast_year,
                                                num_bootstrap=100,
                                                seed=42,
                                                decimal_places=4):
        """
        Load precomputed bin centers for different resolutions for a given forecast year and perform bootstrapping
        to estimate confidence intervals on these centers.

        Parameters:
            forecast_year (int): Forecast year to load precomputed data. Accepts only 1 and 10 for now.
            num_bootstrap (int): Number of bootstrap samples for each resolution.
            seed (int): Seed for reproducibility.
            decimal_places (int): Number of decimal places for rounding.

        Returns:
            dict: Nested dictionary with mean and confidence intervals for each resolution.
        """

        np.random.seed(seed)

        # Load precomputed bin centers across resolutions based on forecast year
        file_path = f"data_input/forecast_year_{forecast_year}_bin_centers_resolutions.npz"
        with np.load(file_path, allow_pickle=True) as data:
            bin_centers_resolutions = data['bin_centers_resolutions'].item()

        res_intervals = {}

        for resolution, centers in bin_centers_resolutions.items():
            source_bin_centers = np.round(centers["source_bin_centers"], decimals=decimal_places)
            lens_bin_centers = np.round(centers["lens_bin_centers"], decimals=decimal_places)

            # Perform bootstrapping on source and lens bin centers
            source_bootstrap_means = []
            lens_bootstrap_means = []

            for _ in range(num_bootstrap):
                source_resample = np.random.choice(source_bin_centers, size=len(source_bin_centers), replace=True)
                lens_resample = np.random.choice(lens_bin_centers, size=len(lens_bin_centers), replace=True)
                source_bootstrap_means.append(np.mean(source_resample))
                lens_bootstrap_means.append(np.mean(lens_resample))

            # Calculate mean and confidence intervals for this resolution
            source_mean = np.mean(source_bootstrap_means)
            source_conf_interval = np.percentile(source_bootstrap_means, [2.5, 97.5])
            lens_mean = np.mean(lens_bootstrap_means)
            lens_conf_interval = np.percentile(lens_bootstrap_means, [2.5, 97.5])

            # Store results for current resolution
            res_intervals[resolution] = {
                "source_bin_center_mean": source_mean,
                "source_conf_interval": source_conf_interval,
                "lens_bin_center_mean": lens_mean,
                "lens_conf_interval": lens_conf_interval
            }

        return res_intervals
