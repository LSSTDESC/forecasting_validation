import numpy as np
from scripts.data_vectors import DataVectors
from scripts.presets import Presets
import matplotlib.pyplot as plt

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

    def get_delta_chi2(self, cls_gc=None, cls_ggl=None, cls_cs=None, cls_gc_ref=None, cls_ggl_ref=None,
                       cls_cs_ref=None):
        """
        Calculate the chi-squared difference between two sets of power spectra. The input power spectra should be
        3D arrays with dimensions (num_ells, num_lens_bins, num_lens_bins) for galaxy clustering, (num_ells, num_lens_bins,
        num_source_bins) for galaxy-galaxy lensing, and (num_ells, num_source_bins, num_source_bins) for cosmic shear.

        Parameters:
            cls_gc (np.ndarray): Galaxy clustering power spectra.
            cls_ggl (np.ndarray): Galaxy-galaxy lensing power spectra.
            cls_cs (np.ndarray): Cosmic shear power spectra.
            cls_gc_ref (np.ndarray): Reference galaxy clustering power spectra.
            cls_ggl_ref (np.ndarray): Reference galaxy-galaxy lensing power spectra.
            cls_cs_ref (np.ndarray): Reference cosmic shear power spectra.

        Returns:
            float: The chi-squared difference between the two sets of power spectra.
        """
        # Reshape the matrices if they are 2D
        num_ells = len(self.presets.ells)
        num_lens_bins = len(self.dv.lens_bins)
        num_source_bins = len(self.dv.source_bins)

        if cls_gc is not None and cls_gc.ndim == 2:
            cls_gc = cls_gc.reshape(num_ells, num_lens_bins, num_lens_bins)
            cls_gc_ref = cls_gc_ref.reshape(num_ells, num_lens_bins, num_lens_bins)

        if cls_ggl is not None and cls_ggl.ndim == 2:
            cls_ggl = cls_ggl.reshape(num_ells, num_lens_bins, num_source_bins)
            cls_ggl_ref = cls_ggl_ref.reshape(num_ells, num_lens_bins, num_source_bins)

        if cls_cs is not None and cls_cs.ndim == 2:
            cls_cs = cls_cs.reshape(num_ells, num_source_bins, num_source_bins)
            cls_cs_ref = cls_cs_ref.reshape(num_ells, num_source_bins, num_source_bins)

        # Now proceed with the chi-squared calculation as before
        has_gc = cls_gc is not None
        has_ggl = cls_ggl is not None
        has_cs = cls_cs is not None

        delta_ell = self.presets.ells[1:] - self.presets.ells[:-1]
        chi2_values = np.zeros_like(delta_ell)

        for i_ell, delta in enumerate(delta_ell):
            ell_mid = self.presets.ells[i_ell] + delta / 2

            # Galaxy Clustering component
            if has_gc:
                signal_matrix = np.copy(cls_gc[i_ell])
                noise_matrix = signal_matrix + np.eye(len(cls_gc[i_ell])) / (
                        self.presets.lens_parameters['number_density'] * 180 ** 2 / np.pi ** 2)
                signal_matrix -= cls_gc_ref[i_ell]

                if has_cs:
                    ggl_matrix = np.zeros((len(cls_gc[i_ell]), len(cls_cs[i_ell])))
                    ggl_matrix_ref = np.zeros_like(ggl_matrix)
                    if has_ggl:
                        ggl_matrix = cls_ggl[i_ell]
                        ggl_matrix_ref = cls_ggl_ref[i_ell]

                    # Construct the full signal and noise matrices with cross terms
                    signal_matrix = np.block([[cls_gc[i_ell], ggl_matrix], [ggl_matrix.T, cls_cs[i_ell]]])
                    lens_noise = np.eye(len(cls_gc[i_ell])) / (
                            self.presets.lens_parameters['number_density'] * 180 ** 2 / np.pi ** 2)
                    shear_noise = (np.eye(len(cls_cs[i_ell])) / (
                            self.presets.source_parameters['number_density'] * 180 ** 2 / np.pi ** 2)
                                   / 2 * self.presets.source_parameters['sigma_eps'] ** 2)

                    noise_matrix = np.copy(signal_matrix)
                    noise_matrix[:len(cls_gc[i_ell]), :len(cls_gc[i_ell])] += lens_noise
                    noise_matrix[len(cls_gc[i_ell]):, len(cls_gc[i_ell]):] += shear_noise

                    # Subtract the reference signal matrix
                    signal_matrix -= np.block(
                        [[cls_gc_ref[i_ell], ggl_matrix_ref], [ggl_matrix_ref.T, cls_cs_ref[i_ell]]])

            # Cosmic Shear only
            else:
                signal_matrix = np.copy(cls_cs[i_ell])
                noise_matrix = signal_matrix + (np.eye(len(cls_cs[i_ell])) / (
                        self.presets.source_parameters['number_density'] * 180 ** 2 / np.pi ** 2)
                                                / 2 * self.presets.source_parameters['sigma_eps'] ** 2)
                signal_matrix -= cls_cs_ref

            # Invert noise matrix and compute chi-squared for this ell bin
            inv_noise_matrix = np.linalg.inv(noise_matrix)
            chi2_values[i_ell] = delta * (2 * ell_mid + 1) / 2 * self.presets.f_sky * np.trace(
                (signal_matrix @ inv_noise_matrix) @ (signal_matrix @ inv_noise_matrix))

        return np.sum(chi2_values)

    def get_delta_chi2robert(self, cl_gc=None, cl_ggl=None, cl_cs=None, cl_gc_1=None, cl_ggl_1=None, cl_cs_1=None):
        # Reshape the matrices if they are 2D
        num_ells = len(self.presets.ells)
        num_lens_bins = len(self.dv.lens_bins)
        num_source_bins = len(self.dv.source_bins)

        if cl_gc is not None and cl_gc.ndim == 2:
            cl_gc = cl_gc.reshape(num_ells, num_lens_bins, num_lens_bins)
            cl_gc_1 = cl_gc_1.reshape(num_ells, num_lens_bins, num_lens_bins)

        if cl_ggl is not None and cl_ggl.ndim == 2:
            cl_ggl = cl_ggl.reshape(num_ells, num_lens_bins, num_source_bins)
            cl_ggl_1 = cl_ggl_1.reshape(num_ells, num_lens_bins, num_source_bins)

        if cl_cs is not None and cl_cs.ndim == 2:
            cl_cs = cl_cs.reshape(num_ells, num_source_bins, num_source_bins)
            cl_cs_1 = cl_cs_1.reshape(num_ells, num_source_bins, num_source_bins)

        # Determine which spectra are provided
        has_gc = cl_gc is not None
        has_ggl = cl_ggl is not None
        has_cs = cl_cs is not None

        delta_ell = self.presets.ells[1:] - self.presets.ells[:-1]
        chi2_at_ell = np.zeros_like(delta_ell)

        for i_ell in range(len(delta_ell)):
            ell = self.presets.ells[i_ell] + delta_ell[i_ell] / 2
            if has_gc:
                signal = np.copy(cl_gc[i_ell, :, :])
                noise = signal + np.eye(len(cl_gc[i_ell, :, 0])) / (
                        self.presets.lens_parameters['number_density'] * 180 ** 2 / np.pi ** 2)
                signal -= cl_gc_1[i_ell, :, :]

                if has_cs:
                    ggl = np.zeros((len(cl_gc[i_ell, :, 0]), len(cl_cs[i_ell, :, 0])))
                    ggl_1 = np.zeros_like(ggl)
                    if has_ggl:
                        ggl = cl_ggl[i_ell, :, :]
                        ggl_1 = cl_ggl_1[i_ell, :, :]

                    # Construct the full signal and noise matrices with cross terms
                    signal = np.block([[cl_gc[i_ell, :, :], ggl], [ggl.T, cl_cs[i_ell, :, :]]])
                    lens_noise = np.eye(len(cl_gc[i_ell, :, 0])) / (
                            self.presets.lens_parameters['number_density'] * 180 ** 2 / np.pi ** 2)
                    shear_noise = (np.eye(len(cl_cs[i_ell, :, 0])) / (
                            self.presets.source_parameters['number_density'] * 180 ** 2 / np.pi ** 2)
                                   / 2 * self.presets.source_parameters['sigma_eps'] ** 2)

                    noise = np.copy(signal)
                    noise[:len(cl_gc[i_ell, :, 0]), :len(cl_gc[i_ell, :, 0])] += lens_noise
                    noise[len(cl_gc[i_ell, :, 0]):, len(cl_gc[i_ell, :, 0]):] += shear_noise

                    # Subtract the reference signal matrix
                    signal -= np.block([[cl_gc_1[i_ell, :, :], ggl_1], [ggl_1.T, cl_cs_1[i_ell, :, :]]])
            else:
                signal = np.copy(cl_cs[i_ell, :, :])
                noise = signal + np.eye(len(cl_cs[i_ell, :, 0])) / (
                        self.presets.source_parameters['number_density'] * 180 ** 2 / np.pi ** 2) / 2 * \
                        self.presets.source_parameters['sigma_eps'] ** 2
                signal -= cl_cs_1[i_ell, :, :]

            noise = np.linalg.inv(noise)
            chi2_at_ell[i_ell] = delta_ell[i_ell] * (2 * ell + 1) / 2 * self.presets.f_sky * np.trace(
                (signal @ noise) @ (signal @ noise))

        return np.sum(chi2_at_ell)


