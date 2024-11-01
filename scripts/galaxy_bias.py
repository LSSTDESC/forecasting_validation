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
