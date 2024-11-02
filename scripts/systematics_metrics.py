import numpy as np
from scripts.tomographic_binning import TomographicBinning
from scripts.presets import Presets
from scripts.galaxy_bias import GalaxyBias  # Assuming GalaxyBias class is in galaxy_bias.py

class SystematicsMetrics:
    def __init__(self, presets):
        if not isinstance(presets, Presets):
            raise TypeError(f"Expected a Presets object, but received {type(presets).__name__}.")
        # Set attributes from presets
        self.cosmology = presets.cosmology
        self.forecast_year = presets.forecast_year
        self.redshift_max = presets.redshift_max
        self.redshift_resolution = presets.redshift_resolution
        self.lens_params = presets.lens_parameters
        self.save_data = presets.save_data

    def galaxy_bias_zres_sweep(self, res_start=300, res_end=10000, step=50):
        """
        Compare galaxy bias values across a range of redshift resolutions.

        Parameters:
            res_start (int): Starting resolution for redshift range.
            res_end (int): Ending resolution for redshift range.
            step (int): Step increment.

        Returns:
            dict: Dictionary with galaxy bias for each redshift resolution.
        """
        galaxy_bias_resolutions = {}

        for resolution in range(res_start, res_end + 1, step):
            # Update Presets with the current resolution
            presets = Presets(
                cosmology=self.cosmology,
                redshift_max=self.redshift_max,
                redshift_resolution=resolution,
                forecast_year=self.forecast_year,
            )

            # Calculate galaxy bias with the updated presets
            galaxy_bias = GalaxyBias(presets).get_galaxy_bias()
            galaxy_bias_resolutions[resolution] = galaxy_bias

        # Save the galaxy bias data
        extra_info = f"zmax{self.redshift_max}"
        self.save_data("galaxy_bias_zres_sweep",
                       galaxy_bias_resolutions,
                       "galaxy_bias",
                       extra_info=extra_info,
                       include_ccl_version=True)

        return galaxy_bias_resolutions

    def galaxy_bias_zres_and_zmax_sweep(self,
                                        zmax_start=3.0,
                                        zmax_end=4.0,
                                        zmax_step=0.1,
                                        res_start=300,
                                        res_end=10000,
                                        res_step=50):
        """
        Compare galaxy bias values across a range of redshift resolutions and zmax values.

        Parameters:
            zmax_start (float): Starting value for zmax.
            zmax_end (float): Ending value for zmax.
            zmax_step (float): Step increment for zmax.
            res_start (int): Starting resolution for redshift range.
            res_end (int): Ending resolution for resolution range.
            res_step (int): Step increment for resolution.

        Returns:
            dict: Nested dictionary with galaxy bias for each zmax and resolution.
        """
        galaxy_bias_by_zmax = {}

        for zmax in np.arange(zmax_start, zmax_end + zmax_step, zmax_step):
            galaxy_bias_by_zmax[zmax] = {}

            for resolution in range(res_start, res_end + 1, res_step):
                # Update Presets with the current resolution and zmax
                presets = Presets(
                    cosmology=self.cosmology,
                    redshift_max=zmax,
                    redshift_resolution=resolution,
                    forecast_year=self.forecast_year,
                )

                # Calculate galaxy bias with the updated presets
                galaxy_bias = GalaxyBias(presets).get_galaxy_bias()
                galaxy_bias_by_zmax[zmax][resolution] = galaxy_bias

        # Save the galaxy bias data
        extra_info = f"zmax_range_{zmax_start}_to_{zmax_end}"
        self.save_data("galaxy_bias_zres_zmax_sweep",
                       galaxy_bias_by_zmax,
                       "galaxy_bias",
                       extra_info=extra_info,
                       include_ccl_version=True)

        return galaxy_bias_by_zmax
