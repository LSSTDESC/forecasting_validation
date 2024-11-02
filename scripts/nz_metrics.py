import numpy as np
from scripts.tomographic_binning import TomographicBinning
from scripts.presets import Presets


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
        self.perform_binning = presets.perform_binning
        self.ell_min = presets.ell_min
        self.ell_max = presets.ell_max
        self.ell_num = presets.ell_num
        self.save_data = presets.save_data

    def compare_bin_centers_over_zresolutions(self,
                                              res_start=300,
                                              res_end=10000,
                                              step=50,
                                              decimal_places=4,
                                              robust=False):
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
            source_bin_centers = tomographic_binning.source_bin_centers(decimal_places=decimal_places,
                                                                        robust=robust)
            lens_bin_centers = tomographic_binning.lens_bin_centers(decimal_places=decimal_places,
                                                                    robust=robust)

            # Store results in the dictionary, using the resolution as the key
            bin_centers_resolutions[resolution] = {
                "source_bin_centers": source_bin_centers,
                "lens_bin_centers": lens_bin_centers
            }

        # Save the full dictionary to a file
        robust_info = "_robust" if robust else ""
        extra_info = f"zmax{self.redshift_max}{robust_info}"
        self.save_data("bin_centers_resolutions",
                       bin_centers_resolutions,
                       "bin_centers",
                       extra_info=extra_info,
                       include_ccl_version=True)

        return bin_centers_resolutions

    def compare_bin_centers_over_zresolutions_and_zmax(self,
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
            bin_centers_by_zmax[zmax] = {}

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
                    perform_binning=self.perform_binning
                )

                # Reinitialize TomographicBinning with updated presets
                tomographic_binning = TomographicBinning(presets)

                # Calculate bin centers for source and lens bins
                source_bin_centers = tomographic_binning.source_bin_centers(decimal_places=decimal_places)
                lens_bin_centers = tomographic_binning.lens_bin_centers(decimal_places=decimal_places)

                # Store results in the dictionary, organized by zmax and resolution
                bin_centers_by_zmax[zmax][resolution] = {
                    "source_bin_centers": source_bin_centers,
                    "lens_bin_centers": lens_bin_centers
                }

        # Save the full nested dictionary to a file
        extra_info = f"zmax_range_{zmax_start}_to_{zmax_end}"
        self.save_data("bin_centers_resolutions_by_zmax",
                       bin_centers_by_zmax,
                       "bin_centers",
                       extra_info=extra_info,
                       include_ccl_version=True)

        return bin_centers_by_zmax

