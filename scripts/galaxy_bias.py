import numpy as np
import pyccl as ccl
from .srd_redshift_distributions import SRDRedshiftDistributions
from .tomographic_binning import TomographicBinning
from .presets import Presets
import itertools


class GalaxyBias:

    def __init__(self, presets: Presets):
        if not isinstance(presets, Presets):
            raise TypeError(f"Expected a Presets object, but received {type(presets).__name__}.")
        # Set the cosmology to the user-defined value or a default if not provided
        self.cosmology = presets.cosmology
        self.redshift_range = presets.redshift_range
        self.perform_binning = presets.perform_binning
        self.forecast_year = presets.forecast_year
        self.redshift_max = presets.redshift_max
        self.redshift_resolution = presets.redshift_resolution
        self.lens_params = presets.lens_parameters

        self.save_data = presets.save_data


        self.lens_bins = TomographicBinning(presets).lens_bins(save_file=False)
        self.lens_bin_centers = TomographicBinning(presets).lens_bin_centers()

    def get_galaxy_bias(self):
        """
        Calculates the galaxy bias dictionary based on lens parameters.

        Returns:
            dict: A dictionary mapping bin indices to galaxy bias values.
        """

        # Extracting lens parameters
        n_tomo_bins = self.lens_params["n_tomo_bins"]

        galaxy_bias_values = self.linear_galaxy_bias()

        # Using bin indices for dict keys
        bin_indices = list(range(n_tomo_bins))

        # Creating a dictionary mapping bin indices to galaxy bias values
        # Formatting the values to 6 decimal places
        galaxy_bias_dict = {bin_idx: round(gbias, 6) for bin_idx, gbias in
                            zip(bin_indices, galaxy_bias_values[:n_tomo_bins])}

        return galaxy_bias_dict

    def averaged_galaxy_bias(self, redshift_distributions, num_points=100):
        """
        Calculates the galaxy bias averaged over each bin, weighted by the provided redshift distribution.

        Args:
            redshift_distributions (dict): A dictionary of redshift distributions for each bin.
                                           Each entry is an array representing the normalized n(z) values
                                           within the bin, corresponding to sampled redshifts.
            num_points (int): Number of redshift points to sample within each bin for averaging.

        Returns:
            dict: A dictionary mapping bin indices to averaged galaxy bias values.
        """
        b_prefactor_dict = {
            "1": 1.05,
            "10": 0.95
        }
        b_prefactor = b_prefactor_dict[self.forecast_year]

        # Initialize the dictionary to store averaged biases for each bin
        averaged_gbias_dict = {}

        for bin_idx, z_center in self.lens_bin_centers.items():
            # Define the bin edges from lens_bins for the current bin
            z_min, z_max = self.lens_bins[bin_idx]

            # Sample `num_points` redshifts within the bin
            z_samples = np.linspace(z_min, z_max, num_points)
            scale_factors = 1 / (1 + z_samples)

            # Compute the galaxy bias at each sampled redshift
            growth_factors = ccl.growth_factor(self.cosmology, scale_factors)
            galaxy_bias_samples = b_prefactor / growth_factors

            # Retrieve the pre-computed redshift distribution for this bin
            redshift_distribution = redshift_distributions[bin_idx]

            # Ensure the distribution length matches the sampling points
            if len(redshift_distribution) != num_points:
                raise ValueError(
                    f"Redshift distribution for bin {bin_idx} has length {len(redshift_distribution)}, "
                    f"expected {num_points}.")

            # Weight the galaxy bias by the redshift distribution and integrate over the bin
            weighted_bias = galaxy_bias_samples * redshift_distribution
            averaged_bias = np.trapz(weighted_bias, z_samples) / np.trapz(redshift_distribution, z_samples)

            # Store the averaged galaxy bias for this bin
            averaged_gbias_dict[bin_idx] = round(averaged_bias, 6)

        return averaged_gbias_dict

    def linear_galaxy_bias(self):
        b_prefactor_dict = {
            "1": 1.05,
            "10": 0.95
        }

        b_prefactor = b_prefactor_dict[self.forecast_year]

        # Convert the lens_bin_centers dictionary values to an array of redshift values
        z_array = np.array(list(self.lens_bin_centers.values()))

        scale_factor = 1 / (1 + z_array)
        # Calculate the growth factor for each z
        growth_factor = ccl.growth_factor(self.cosmology, scale_factor)

        # Calculate the galaxy bias
        gbias = b_prefactor / growth_factor

        # Formatting the galaxy bias values to 6 decimal places
        gbias = np.round(gbias, 6)

        return gbias
