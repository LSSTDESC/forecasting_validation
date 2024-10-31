import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.special import erf
from scripts.srd_redshift_distributions import SRDRedshiftDistributions
from scripts.presets import Presets


# noinspection PyMethodMayBeStatic,DuplicatedCode
class TomographicBinning:
    """
    Performs the slicing of the input redshift distribution into tomographic bins.
    The binning algorithm follows the LSST DESC prescription. For more details, see
    the LSST DESC Science Requirements Document (SRD) Appendix D (link to paper:
    https://arxiv.org/abs/1809.01669).
    The methods allow for slicing of the initial redshift distribution into a source or
    lens galaxy sample for the appropriate LSST DESC forecast year (year 1 or year 10).

    ...
    Attributes
    ----------
    redshift_range: array
        An interval of redshifts for which
        the redshift distribution is generated
        redshift_range
    forecast_year: str
        year that corresponds to the SRD forecast. Accepted values
        are "1" and "10"
    """

    def __init__(self, presets: Presets):
        if not isinstance(presets, Presets):
            raise TypeError(f"Expected a Presets object, but received {type(presets).__name__}.")
        self.forecast_year = presets.forecast_year
        self.redshift_range = presets.redshift_range

        self.lens_params = presets.lens_parameters
        self.source_params = presets.source_parameters

        self.save_data = presets.save_data
        self.perfom_binning = presets.perform_binning

        # Load the redshift distributions with high precision
        self.nz = SRDRedshiftDistributions(presets)
        self.source_nz = np.array(self.nz.source_sample(save_file=False), dtype=np.float64)
        self.lens_nz = np.array(self.nz.lens_sample(save_file=False), dtype=np.float64)

        self.save_data = presets.save_data

    def true_redshift_distribution(self, upper_edge, lower_edge, variance, bias, redshift_distribution):
        """A function that returns the true redshift distribution of a galaxy sample.
         The true distribution of galaxies is defined as a convolution of an overall galaxy redshift distribution and
         a probability distribution p(z_{ph}|z)  at a given z (z_{ph} is a photometric distribution at a given z).
         Overall galaxy redshift distribution is a Smail type distribution (n(z) = (z/z_0)^alpha exp[-(z/z_0)^beta]).
         The true distribution defined here is following Ma, Hu & Huterer 2018
          (see https://arxiv.org/abs/astro-ph/0506614 eq. 6).

           Arguments:
               upper_edge (float): upper edge of the redshift bin
               lower_edge (float): lower edge of the redshift bin
               variance (float): variance of the photometric distribution
               bias (float): bias of the photometric distribution
               redshift_distribution (array): overall galaxy redshift distribution
            Returns:
                true_redshift_distribution (array): true redshift distribution of a galaxy sample"""
        # Calculate the scatter
        scatter = variance * (1 + self.redshift_range)
        # Calculate the upper and lower limits of the integral
        lower_limit = (upper_edge - self.redshift_range + bias) / np.sqrt(2) / scatter
        upper_limit = (lower_edge - self.redshift_range + bias) / np.sqrt(2) / scatter

        # Calculate the true redshift distribution
        true_redshift_distribution = 0.5 * np.array(redshift_distribution) * (erf(upper_limit) - erf(lower_limit))

        return true_redshift_distribution

    def compute_equal_number_bounds(self, redshift_range, redshift_distribution, n_bins):
        """
        Determines the redshift values that divide the distribution into bins
        with an equal number of galaxies.

        Arguments:
            redshift_range (array): an array of redshift values
            redshift_distribution (array): the corresponding redshift distribution defined over redshift_range
            n_bins (int): the number of tomographic bins

        Returns:
            An array of redshift values that are the boundaries of the bins.
        """

        # Calculate the cumulative distribution
        cumulative_distribution = cumulative_trapezoid(redshift_distribution, redshift_range, initial=0)
        total_galaxies = cumulative_distribution[-1]

        # Find the bin edges
        bin_edges = []
        for i in range(1, n_bins):
            fraction = i / n_bins * total_galaxies
            # Find the redshift value where the cumulative distribution crosses this fraction
            bin_edge = np.interp(fraction, cumulative_distribution, redshift_range)
            bin_edges.append(bin_edge)

        return [redshift_range[0]] + bin_edges + [redshift_range[-1]]

    def source_bins(self, normalized=True, perform_binning=True, save_file=True, tolerance=0.01):
        """
        Split the initial redshift distribution of source galaxies into tomographic bins or return the original distribution.

        For LSST DESC, sources are split into 5 tomographic bins for both year 1 and year 10 forecasts,
        with equal numbers of galaxies in each bin.

        Parameters:
            normalized (bool): Normalize each bin's redshift distribution independently (default True).
            perform_binning (bool): Perform binning (default True). If False, return the original
                                    distribution in dictionary format.
            save_file (bool): Option to save the output file (default True).
            ttolerance (float): Acceptable deviation for galaxy fraction between bins as a percentage
                                (e.g., 1 for 1% tolerance).

        Returns:
            dict: Source galaxy sample, binned or original distribution.

        Raises:
            ValueError: If the number of galaxies in each bin deviates beyond the specified tolerance.
        """
        if not perform_binning:
            # Return the original distribution in a dictionary format without binning
            return {0: self.source_nz}

        # Perform binning
        bins = self.compute_equal_number_bounds(self.redshift_range,
                                                self.source_nz,
                                                self.source_params["n_tomo_bins"])

        # Get the bias and variance values for each bin
        source_z_bias_list = np.repeat(self.source_params["z_bias"], self.source_params["n_tomo_bins"])
        source_z_variance_list = np.repeat(self.source_params["sigma_z"], self.source_params["n_tomo_bins"])

        # Create a dictionary of the redshift distributions for each bin
        source_redshift_distribution_dict = {}
        for index, (x1, x2) in enumerate(zip(bins[:-1], bins[1:])):
            z_bias = source_z_bias_list[index]
            z_variance = source_z_variance_list[index]
            dist = self.true_redshift_distribution(x1, x2, z_variance, z_bias, self.source_nz)

            # Normalize each bin individually if normalisation is True
            if normalized:
                norm_factor = np.trapz(dist, x=self.redshift_range)
                dist /= norm_factor  # Normalize the distribution

            source_redshift_distribution_dict[index] = dist

        # After constructing bins, check if the galaxy counts are approximately equal
        fractions = self.get_galaxy_fraction_in_bin(source_redshift_distribution_dict)
        mean_fraction = np.mean(fractions)

        # Convert tolerance from percentage to fraction
        tolerance_fraction = tolerance / 100

        # Verify each bin's fraction is within tolerance
        if not np.all(np.abs(fractions - mean_fraction) <= mean_fraction * tolerance_fraction):
            raise ValueError(
                "Galaxy counts across source bins are not approximately equal within the specified tolerance. "
                f"Bin fractions: {fractions}, Mean fraction: {mean_fraction}"
            )

        cumulative_count = np.sum(fractions) * np.trapz(self.source_nz, self.redshift_range)
        expected_total_count = np.trapz(self.source_nz, self.redshift_range)
        if not np.isclose(cumulative_count, expected_total_count, atol=1e-3):
            raise ValueError("Cumulative galaxy count across bins does not match the expected total galaxy count.")

        # Create a combined dictionary with the redshift range and bins
        combined_data = {'redshift_range': self.redshift_range, 'bins': source_redshift_distribution_dict}

        # Save the data if required
        if save_file:
            self.save_data("source_bins",
                           combined_data,
                           dir="redshift_distributions",
                           include_ccl_version=False)

        return source_redshift_distribution_dict


    def lens_bins(self, normalized=True, perform_binning=True, save_file=True):
        """
        Split the initial redshift distribution of lens galaxies (lenses) into tomographic bins or return the original distribution.

        For LSST DESC, lenses are split into 5 or 10 tomographic bins depending on the forecast year,
        with variance 0.03 and z_bias of zero.

        Parameters:
            normalized (bool): Normalize each bin's redshift distribution independently (default True).
            perform_binning (bool): Perform binning (default True). If False, return the original distribution in dictionary format.
            save_file (bool): Option to save the output file (default True).

        Returns:
            dict: Lens galaxy sample, binned or original distribution.
        """

        if not perform_binning:
            # Return the original distribution in a dictionary format without binning
            return {0: self.lens_nz}

        # Define the bin edges
        bins = np.arange(self.lens_params["bin_start"],
                         self.lens_params["bin_stop"] + self.lens_params["bin_spacing"],
                         self.lens_params["bin_spacing"])

        # Get the bias and variance values for each bin
        lens_z_bias_list = np.repeat(self.lens_params["z_bias"], self.lens_params["n_tomo_bins"])
        lens_z_variance_list = np.repeat(self.lens_params["sigma_z"], self.lens_params["n_tomo_bins"])

        # Create a dictionary of the redshift distributions for each bin
        lens_redshift_distribution_dict = {}
        for index, (x1, x2) in enumerate(zip(bins[:-1], bins[1:])):
            z_bias = lens_z_bias_list[index]
            z_variance = lens_z_variance_list[index]
            dist = self.true_redshift_distribution(x1, x2, z_variance, z_bias, self.lens_nz)

            # Normalize each bin individually if normalisation is True
            if normalized:
                norm_factor = np.trapz(dist, x=self.redshift_range)
                dist /= norm_factor  # Normalize the distribution

            lens_redshift_distribution_dict[index] = dist

        # Combine the data
        combined_data = {'redshift_range': self.redshift_range, 'bins': lens_redshift_distribution_dict}

        # Save the distributions to a file if specified
        if save_file:
            self.save_data("lens_bins",
                           combined_data,
                           dir="redshift_distributions",
                           include_ccl_version=False)

        return lens_redshift_distribution_dict

    def get_bin_centers(self, bins_dict, decimal_places=2):
        """
        Calculate and round the bin centers for all bins in a dictionary based on the maximum of each bin's distribution.

        Parameters:
            bins_dict (dict): A dictionary where each key represents a bin and each value is the redshift distribution for that bin.
            decimal_places (int): Number of decimal places to round the bin centers.

        Returns:
            dict: A dictionary with bin keys and their corresponding rounded bin centers.
        """
        bin_centers = {}
        for bin_key, bin_distribution in bins_dict.items():
            max_index = np.argmax(bin_distribution)
            bin_centers[bin_key] = round(self.redshift_range[max_index], decimal_places)
        return bin_centers

    def lens_bin_centers(self, decimal_places=2):
        """
        Calculate and round the bin centers for the lens bins.

        Parameters:
            decimal_places (int): Number of decimal places to round the bin centers.

        Returns:
            dict: A dictionary with lens bin keys and their corresponding rounded bin centers.
        """
        lens_bins = self.lens_bins(save_file=False)
        return self.get_bin_centers(lens_bins, decimal_places)

    def source_bin_centers(self, decimal_places=2):
        """
        Calculate and round the bin centers for the source bins.

        Parameters:
            decimal_places (int): Number of decimal places to round the bin centers.

        Returns:
            dict: A dictionary with source bin keys and their corresponding rounded bin centers.
        """
        source_bins = self.source_bins(save_file=False)
        return self.get_bin_centers(source_bins, decimal_places)

    def get_galaxy_fraction_in_bin(self, bin_distribution):
        """
        Calculate the fraction of the total galaxies within each bin's redshift range.

        Parameters:
            bin_distribution (np.ndarray or dict): The redshift distribution for each bin.
                If a dict, the function will compute for each bin.

        Returns:
            fraction_in_bin (np.ndarray): Array of fractions of galaxies in each bin.
        """

        def compute_fraction(distribution, total_distribution):
            # Integrate the bin distribution over the redshift range
            bin_integral = np.trapz(distribution, self.redshift_range)

            # Calculate the fraction for this bin relative to the total distribution
            bin_fraction = bin_integral / np.trapz(total_distribution, self.redshift_range)
            return bin_fraction

        # Infer the number of bins based on the input bin_distribution
        if isinstance(bin_distribution, dict):
            bin_distributions = np.array(list(bin_distribution.values()))  # Convert dict to array of bins
        else:
            bin_distributions = bin_distribution  # If already an array, use directly

        # Calculate the total distribution by summing over all inferred bins
        total_distribution = np.sum(bin_distributions, axis=0)

        # Compute the fraction for each bin
        fractions = [compute_fraction(distr, total_distribution) for distr in bin_distributions]

        return np.array(fractions)

    def fraction_of_lens_galaxies_in_bin(self):
        """
        Calculate the fraction of lens galaxies in each lens bin.

        Returns:
            np.ndarray: Array of fractions of lens galaxies in each lens bin.
        """
        lens_bins = self.lens_bins(save_file=False)
        return self.get_galaxy_fraction_in_bin(lens_bins)

    def fraction_of_source_galaxies_in_bin(self):
        """
        Calculate the fraction of source galaxies in each source bin.

        Returns:
            np.ndarray: Array of fractions of source galaxies in each source bin.
        """
        source_bins = self.source_bins(save_file=False)
        return self.get_galaxy_fraction_in_bin(source_bins)

    def calculate_average_galaxies_per_bin(self, binned_distribution, number_density):
        """
        Calculate the average number of galaxies for each tomo bin using the binned distribution.

        Parameters:
            binned_distribution (dict or np.ndarray): Redshift distribution for each bin.
                If a dict, the function will compute for each bin.
            number_density (float): The number density of galaxies per arcminute squared.

        Returns:
            avg_galaxies_per_bin (np.ndarray): Array of average number of galaxies for each bin.
        """

        def compute_nz_avg(distribution):
            # Integrate the distribution over the redshift range
            total_integral = np.trapz(distribution, self.redshift_range)

            # Calculate fraction and total number of galaxies in the bin
            fraction_in_bin = total_integral
            fraction_in_bin /= np.trapz(np.concatenate(list(binned_distribution.values())), self.redshift_range)
            total_number_of_galaxies = total_integral * number_density

            # Calculate average number of galaxies in this bin
            return fraction_in_bin * total_number_of_galaxies

        # If binned_distribution is a dictionary (multiple bins), calculate for each bin
        if isinstance(binned_distribution, dict):
            avg_galaxies_per_bin = [compute_nz_avg(distr) for distr in binned_distribution.values()]
            return np.array(avg_galaxies_per_bin)

        # If a single bin is passed (as an array), compute directly
        return np.array([compute_nz_avg(binned_distribution)])

    def number_of_galaxies_in_lens_bins(self):
        """
        Calculate the average number of galaxies in each lens bin.

        Returns:
            np.ndarray: Array of average number of galaxies in each lens bin.
        """
        lens_bins = self.lens_bins(save_file=False)
        return self.calculate_average_galaxies_per_bin(lens_bins, self.lens_params["number_density"])

    def number_of_galaxies_in_source_bins(self):
        """
        Calculate the average number of galaxies in each source bin.

        Returns:
            np.ndarray: Array of average number of galaxies in each source bin.
        """
        source_bins = self.source_bins(save_file=False)
        return self.calculate_average_galaxies_per_bin(source_bins, self.source_params["number_density"])

    def get_redshift_for_nz(self, nz_value, distribution_type):
        """
        Get the redshift corresponding to a specified n(z) value for either lens or source galaxies.

        Parameters:
            nz_value (float): The target n(z) value.
            distribution_type (str): Type of distribution, either "source" or "lens".

        Returns:
            float: The redshift value at which n(z) matches nz_value, or None if not found.
        """


        # Select the appropriate redshift distribution based on the distribution type
        if distribution_type == "sources":
            nz_distribution = self.source_nz
        elif distribution_type == "lenses":
            nz_distribution = self.lens_nz
        else:
            raise ValueError("Invalid distribution type. Use 'source' or 'lens'.")

        # Find the indices where n(z) matches nz_value
        indices = np.where(nz_distribution == nz_value)[0]

        if len(indices) == 0:
            print("Specified n(z) value not found in the distribution.")
            return None

        # Return the corresponding redshift value
        return self.redshift_range[indices[0]]
