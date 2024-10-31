import numpy as np
from scripts.data_vectors import DataVectors
from scripts.presets import Presets


class DataVectorMetrics:

    def __init__(self, presets):
        if not isinstance(presets, Presets):
            raise TypeError(f"Expected a Presets object, but received {type(presets).__name__}.")
        # Initialize presets and data vectors with initial redshift range from presets
        self.presets = presets
        self.dv = DataVectors(presets)  # Store an instance of DataVectors for access to its methods
        self.redshift_range = self.dv.redshift_range

    def get_kernel_peaks(self, kernel_array, redshift_range):
        """
        Calculate the peak redshifts and values for each kernel in the given kernel array.
        """
        peaks = []
        for kernel in kernel_array:
            peak_index = np.argmax(kernel)
            peak_redshift = redshift_range[peak_index]
            peak_value = kernel[peak_index]
            peaks.append((peak_redshift, peak_value))
        return peaks

    def kernel_peaks_z_resolution_sweep(self, z_resolutions=None, include_ia=True, include_gbias=True, kernel_type="wl"):
        """
        Perform a parametric sweep of redshift resolutions, calculating kernel peaks for each resolution.

        Parameters:
            z_resolutions (list, optional): List of redshift resolutions to iterate over. If None, uses default range.
            include_ia (bool): Include intrinsic alignment for WL kernels.
            include_gbias (bool): Include galaxy bias for NC kernels.
            kernel_type (str): Either "wl" for weak lensing or "nc" for number counts.

        Returns:
            dict: A dictionary with resolutions as keys and kernel peaks (z, value) as values.
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

            # Get kernel peaks based on kernel type
            if kernel_type == "wl":
                wl_kernel = dv_temp.get_wl_kernel(include_ia, return_chi=False)
                kernel_peaks = self.get_kernel_peaks(wl_kernel, redshift_range)
            elif kernel_type == "nc":
                nc_kernel = dv_temp.get_nc_kernel(include_gbias, return_chi=False)
                kernel_peaks = self.get_kernel_peaks(nc_kernel, redshift_range)
            else:
                raise ValueError("Invalid kernel type specified. Use 'wl' for weak lensing or 'nc' for number counts.")

            peaks_by_resolution[res] = kernel_peaks

        return peaks_by_resolution
