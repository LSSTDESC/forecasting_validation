import numpy as np
from scripts.data_vectors import DataVectors
from scripts.presets import Presets


class KernelMetrics:

    def __init__(self, presets):
        if not isinstance(presets, Presets):
            raise TypeError(f"Expected a Presets object, but received {type(presets).__name__}.")
        # Initialize presets and data vectors with initial redshift range from presets
        self.presets = presets
        self.dv = DataVectors(presets)  # Store an instance of DataVectors for access to its methods
        self.redshift_range = self.dv.redshift_range
        self.save_data = presets.save_data

    def get_kernel_peaks(self, kernel_array, redshift_range):
        """
        Calculate the peak redshifts and values for each kernel in the given kernel array.
        """
        peaks = []
        for kernel in kernel_array:
            # Flatten the kernel in case it has extra dimensions
            kernel = np.ravel(kernel)

            # Check if kernel length matches redshift_range
            if len(kernel) != len(redshift_range):
                raise ValueError(
                    f"Kernel length {len(kernel)} does not match redshift_range length {len(redshift_range)}.")

            # Find peak index and values
            peak_index = np.argmax(kernel)
            peak_redshift = redshift_range[peak_index]
            peak_value = kernel[peak_index]
            peaks.append((peak_redshift, peak_value))

        return peaks

    def kernel_peaks_zresolution_sweep(self, z_resolutions=None, include_ia=True, include_gbias=True):
        """
        Perform a parametric sweep of redshift resolutions, calculating kernel peaks for each resolution
        for both weak lensing (WL) and number counts (NC) kernels.

        Parameters:
            z_resolutions (list, optional): List of redshift resolutions to iterate over. If None, uses default range.
            include_ia (bool): Include intrinsic alignment for WL kernels.
            include_gbias (bool): Include galaxy bias for NC kernels.

        Returns:
            dict: A dictionary with resolutions as keys, each containing sub-dictionaries with WL and NC kernel peaks.
                  Format: {resolution: {"wl": [(z_peak, value_peak), ...], "nc": [(z_peak, value_peak), ...]}}
        """
        # Use default range if z_resolutions is not provided
        if z_resolutions is None:
            z_resolutions = list(range(300, 10001, 50))

        peaks_by_resolution = {}

        for res in z_resolutions:
            # Initialize the presets with updated redshift resolution
            temp_presets = Presets(redshift_max=self.presets.redshift_max,
                                   redshift_resolution=res,
                                   forecast_year=self.presets.forecast_year)

            # Reinitialize DataVectors with updated presets and redshift range
            dv_temp = DataVectors(temp_presets)
            redshift_range = dv_temp.redshift_range

            # Get kernel peaks for both WL and NC kernels
            wl_kernel_peaks = None
            nc_kernel_peaks = None

            # Calculate WL kernel peaks if include_ia is True
            if include_ia:
                wl_kernel = dv_temp.get_wl_kernel(include_ia=True, return_chi=False)
                wl_kernel_peaks = self.get_kernel_peaks(wl_kernel, redshift_range)

            # Calculate NC kernel peaks if include_gbias is True
            if include_gbias:
                nc_kernel = dv_temp.get_nc_kernel(include_gbias=True, return_chi=False)
                nc_kernel_peaks = self.get_kernel_peaks(nc_kernel, redshift_range)

            # Store peaks by resolution in a nested dictionary format
            peaks_by_resolution[res] = {
                "wl": wl_kernel_peaks,
                "nc": nc_kernel_peaks
            }

        data_path = "data_output/kernels/"
        filename = f"kernel_peaks_zres_sweep_y{self.presets.forecast_year}.npy"
        np.save(f"{data_path}{filename}", peaks_by_resolution)

        return peaks_by_resolution

    def kernel_peaks_z_resolution_and_zmax_sweep(self,
                                                 kernel_type,
                                                 zmax_start=3.0,
                                                 zmax_end=4.0,
                                                 zmax_step=0.1,
                                                 res_start=300,
                                                 res_end=10000,
                                                 res_step=50,
                                                 include_ia=True,
                                                 include_gbias=True):
        """
        Perform a parametric sweep of redshift resolutions and zmax values, calculating kernel peaks for each
        combination of zmax and resolution.

        Parameters:
            kernel_type (str): Type of kernel to calculate ("wl" for weak lensing, "nc" for number counts).
            zmax_start (float): Starting value for zmax.
            zmax_end (float): Ending value for zmax.
            zmax_step (float): Step increment for zmax.
            res_start (int): Starting resolution for redshift range.
            res_end (int): Ending resolution for resolution range.
            res_step (int): Step increment for resolution.
            include_ia (bool): Include intrinsic alignment for WL kernels.
            include_gbias (bool): Include galaxy bias for NC kernels.

        Returns:
            dict: Nested dictionary with kernel peaks for each zmax and resolution.
                  Format: {str(zmax): {resolution: [(z_peak, value_peak), ...]}}
        """
        # Raise an error if kernel_type is not valid
        valid_kernel_types = ["wl", "nc"]
        if kernel_type not in valid_kernel_types:
            raise ValueError(f"Invalid kernel_type '{kernel_type}'. Must be one of {valid_kernel_types}.")

        peaks_by_zmax_and_resolution = {}

        for zmax in np.arange(zmax_start, zmax_end + zmax_step, zmax_step):
            # Round zmax to 1 decimal place for dictionary key
            # otherwise, floating point errors may cause issues
            # when calling the dictionary keys
            zmax_key = round(zmax, 1)
            peaks_by_zmax_and_resolution[zmax_key] = {}

            for resolution in range(res_start, res_end + 1, res_step):
                # Initialize Presets with the current resolution and zmax
                temp_presets = Presets(
                    cosmology=self.presets.cosmology,
                    redshift_max=round(zmax, 1),  # Use rounded zmax here
                    redshift_resolution=resolution,
                    forecast_year=self.presets.forecast_year
                )

                # Reinitialize DataVectors with updated presets and redshift range
                dv_temp = DataVectors(temp_presets)
                redshift_range = dv_temp.redshift_range

                # Calculate kernel peaks based on kernel_type
                if kernel_type == "wl" and include_ia:
                    wl_kernel = dv_temp.get_wl_kernel(include_ia, return_chi=False)
                    kernel_peaks = self.get_kernel_peaks(wl_kernel, redshift_range)
                elif kernel_type == "nc" and include_gbias:
                    nc_kernel = dv_temp.get_nc_kernel(include_gbias, return_chi=False)
                    kernel_peaks = self.get_kernel_peaks(nc_kernel, redshift_range)
                else:
                    kernel_peaks = None

                # Store peaks in a nested dictionary format
                peaks_by_zmax_and_resolution[zmax_key][resolution] = kernel_peaks

        # Save the full nested dictionary to a file
        self.save_data(f"{kernel_type}_kernel_peaks_zmax_and_zres_sweep",
                       peaks_by_zmax_and_resolution,
                       "kernels",
                       include_ccl_version=True)

        return peaks_by_zmax_and_resolution
