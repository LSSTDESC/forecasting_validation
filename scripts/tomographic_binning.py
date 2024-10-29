#  Niko Sarcevic
#  nikolina.sarcevic@gmail.com
#  github.com/nikosarcevic
#  ----------

import os
import numpy as np
import pandas
from scipy.integrate import simpson, cumulative_trapezoid
from scipy.special import erf
import yaml
from scripts.srd_redshift_distributions import SRDRedshiftDistributions


class TomographicBinning:

    def __init__(self,
                 redshift_range,
                 forecast_year="1",
                 perform_binning=True):
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

        supported_forecast_years = {"1", "10"}
        if forecast_year in supported_forecast_years:
            self.forecast_year = forecast_year
        else:
            raise ValueError(f"forecast_year must be one of {supported_forecast_years}.")

        self.redshift_range = np.array(redshift_range, dtype=np.float64)
        self.forecast_year = forecast_year

        current_dir = os.path.dirname(__file__)
        yaml_path = os.path.join(current_dir, "lsst_desc_parameters.yaml")

        # Load the YAML file
        with open(yaml_path, "r") as f:
            self.lsst_desc_parameters = yaml.load(f, Loader=yaml.FullLoader)

        self.lens_params = self.lsst_desc_parameters["lens_sample"][self.forecast_year]
        self.source_params = self.lsst_desc_parameters["source_sample"][self.forecast_year]

        # Load the redshift distributions with high precision
        self.nz = SRDRedshiftDistributions(self.redshift_range, forecast_year)
        self.source_nz = np.array(self.nz.source_sample(), dtype=np.float64)
        self.lens_nz = np.array(self.nz.lens_sample(), dtype=np.float64)

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

    def source_bins(self, normalised=True, perform_binning=True, save_file=True, file_format='npy'):
        """Split the initial redshift distribution of source galaxies into tomographic bins or return the original distribution.

        For LSST DESC, sources are split into 5 tomographic bins for both year 1 and year 10 forecasts,
        with equal numbers of galaxies in each bin.

        Parameters:
            normalised (bool): Normalize each bin's redshift distribution independently (default True).
            perform_binning (bool): Perform binning (default True). If False, return the original distribution in dictionary format.
            save_file (bool): Option to save the output file (default True).
            file_format (str): Format of the output file, 'csv' or 'npy' (default 'npy').

        Returns:
            dict: Source galaxy sample, binned or original distribution.
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
            if normalised:
                norm_factor = simpson(dist, self.redshift_range)
                dist /= norm_factor  # Normalize the distribution

            source_redshift_distribution_dict[index] = dist

        # Create a combined dictionary with the redshift range and bins
        combined_data = {'redshift_range': self.redshift_range, 'bins': source_redshift_distribution_dict}

        # Save the data if required
        if save_file:
            self.save_to_file(combined_data, "source", file_format)

        return source_redshift_distribution_dict

    def lens_bins(self, normalised=True, perform_binning=True, save_file=True, file_format='npy'):
        """
        Split the initial redshift distribution of lens galaxies (lenses) into tomographic bins or return the original distribution.

        For LSST DESC, lenses are split into 5 or 10 tomographic bins depending on the forecast year,
        with variance 0.03 and z_bias of zero.

        Parameters:
            normalised (bool): Normalize each bin's redshift distribution independently (default True).
            perform_binning (bool): Perform binning (default True). If False, return the original distribution in dictionary format.
            save_file (bool): Option to save the output file (default True).
            file_format (str): Format of the output file, 'csv' or 'npy' (default 'npy').

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
            if normalised:
                norm_factor = simpson(dist, self.redshift_range)
                dist /= norm_factor  # Normalize the distribution

            lens_redshift_distribution_dict[index] = dist

        # Combine the data
        combined_data = {'redshift_range': self.redshift_range, 'bins': lens_redshift_distribution_dict}

        # Save the distributions to a file if specified
        if save_file:
            self.save_to_file(combined_data, "lens", file_format)

        return lens_redshift_distribution_dict

    def get_bin_centers(self, decimal_places=2, save_file=True):
        """Method to calculate the bin centers for the source and lens galaxy samples.
        The bin centers are calculated as the redshift value where
        the redshift distribution is maximised.
        The bin centers are rounded to the specified number of decimal places.

        Arguments:
            decimal_places (int): number of decimal places to round the bin centers to (defaults to 2)
            save_file (bool): option to save the output as a .npy file (defaults to True)
        Returns: a nested dictionary of bin centers for source and lens galaxy samples
         for year 1 and year 10 forecast (keys are the forecast years).
            """
        bin_centers = {"sources": [], "lenses": []}

        # Calculate bin centers for sources
        source_bins = self.source_bins(normalised=True, save_file=False)
        for index in range(self.source_params["n_tomo_bins"]):
            bin_center = self.find_bin_center(source_bins[index], self.redshift_range, decimal_places)
            bin_centers["sources"].append(bin_center)

        # Calculate bin centers for lenses
        lens_bins = self.lens_bins(normalised=True, save_file=False)
        for index in range(self.lens_params["n_tomo_bins"]):
            bin_center = self.find_bin_center(lens_bins[index], self.redshift_range, decimal_places)
            bin_centers["lenses"].append(bin_center)

        if save_file:
            # Save to .npy file if save_file is True
            np.save(f'data_output/srd_bin_centers_y_{self.forecast_year}.npy', bin_centers)

        return bin_centers

    def find_bin_center(self, bin_distribution, redshift_range, decimal_places=2):
        """Helper method to calculate and round the bin center."""
        max_index = np.argmax(bin_distribution)
        return round(redshift_range[max_index], decimal_places)

    def save_to_file(self, data, name, file_format="npy"):

        if file_format == "npy":
            np.save(f"data_output/srd_{name}_bins_year_{self.forecast_year}.npy", data)
        elif file_format == "csv":
            dndz_df = pandas.DataFrame(data)
            dndz_df.to_csv(f"data_output/srd_{name}_bins_year_{self.forecast_year}.csv", index=False)