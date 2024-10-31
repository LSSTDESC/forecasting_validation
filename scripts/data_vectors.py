import numpy as np
import pyccl as ccl
from .srd_redshift_distributions import SRDRedshiftDistributions
from .tomographic_binning import TomographicBinning
from .presets import Presets


class DataVectors:

    def __init__(self, presets: Presets):
        if not isinstance(presets, Presets):
            raise TypeError(f"Expected a Presets object, but received {type(presets).__name__}.")
        # Set the cosmology to the user-defined value or a default if not provided
        self.cosmology = presets.cosmology
        self.redshift_range = presets.redshift_range
        self.ells = presets.ells
        self.perform_binning = presets.perform_binning
        self.forecast_year = presets.forecast_year
        self.redshift_max = presets.redshift_max
        self.redshift_resolution = presets.redshift_resolution

        self.save_data = presets.save_data

        self.nz = SRDRedshiftDistributions(presets)
        self.source_nz = np.array(self.nz.source_sample(save_file=False), dtype=np.float64)
        self.lens_nz = np.array(self.nz.lens_sample(save_file=False), dtype=np.float64)

        self.bin = TomographicBinning(presets)
        self.lens_bins = self.bin.lens_bins(save_file=False)
        self.source_bins = self.bin.source_bins(save_file=False)

    def get_ia_bias(self):
        # For now just simple constant IA bias
        ia_bias = (self.redshift_range, np.full_like(self.redshift_range, 1.0))
        return ia_bias

    def get_gbias(self):
        # For now just simple constant galaxy bias
        gbias = (self.redshift_range, np.full_like(self.redshift_range, 1.0))
        return gbias

    def cosmic_shear_cls(self, include_ia=True):
        ia_bias = self.get_ia_bias() if include_ia else None
        correlations = self.get_correlation_pairs()["cosmic_shear"]
        self.save_data("cosmic_shear_correlations",
                       correlations,
                       dir="angular_power_spectra",
                       include_ccl_version=False)

        cls_list = []
        for idx_1, idx_2 in correlations:
            tracer1 = ccl.WeakLensingTracer(self.cosmology,
                                            dndz=(self.redshift_range, self.source_bins[idx_1]),
                                            ia_bias=ia_bias)
            tracer2 = ccl.WeakLensingTracer(self.cosmology,
                                            dndz=(self.redshift_range, self.source_bins[idx_2]),
                                            ia_bias=ia_bias)

            # Compute Cl and append it to the list
            cls_list.append(ccl.angular_cl(self.cosmology, tracer1, tracer2, self.ells))

        # Stack into a numpy array of shape (num_ells, num_cls)
        cls_array = np.column_stack(cls_list)
        # Save the data with version information
        self.save_data("cosmic_shear_cls",
                       cls_array,
                       dir="angular_power_spectra",
                       include_ccl_version=True,
                       extra_info=self.get_extra_info())
        return cls_array

    def galaxy_clustering_cls(self, include_gbias=True):
        gbias = self.get_gbias() if include_gbias else None

        correlations = self.get_correlation_pairs()["galaxy_clustering"]
        self.save_data("galaxy_clustering_correlations",
                       correlations,
                       dir="angular_power_spectra",
                       include_ccl_version=False)

        cls_list = []

        for idx_1, idx_2 in correlations:
            tracer1 = ccl.NumberCountsTracer(self.cosmology,
                                             has_rsd=False,
                                             dndz=(self.redshift_range, self.lens_bins[idx_1]),
                                             bias=gbias)
            tracer2 = ccl.NumberCountsTracer(self.cosmology,
                                             has_rsd=False,
                                             dndz=(self.redshift_range, self.lens_bins[idx_2]),
                                             bias=gbias)

            cls_list.append(ccl.angular_cl(self.cosmology, tracer1, tracer2, self.ells))

        cls_array = np.column_stack(cls_list)
        # Save the data with version information
        self.save_data("galaxy_clustering_cls",
                       cls_array,
                       dir="angular_power_spectra",
                       include_ccl_version=True,
                       extra_info=self.get_extra_info())
        return cls_array

    def get_wl_kernel(self, include_ia=True, return_chi=False):
        """
        Compute the weak lensing (WL) kernel for each source bin.

        Parameters:
            include_ia (bool): Include intrinsic alignment bias if True.
            return_chi (bool): Return the comoving radial distance array if True.

        Returns:
            tuple: Arrays of WL kernels for each source bin. If return_chi is True,
                   also returns the comoving radial distance array (chi).
        """
        ia_bias = self.get_ia_bias() if include_ia else None
        # Prepare lists to store kernels and chi values
        kernel_list = []
        chi_list = []

        # Compute comoving radial distance once for the redshift range
        chi = ccl.comoving_radial_distance(self.cosmology, 1 / (1 + self.redshift_range))

        # Loop over each source bin
        for idx in self.source_bins.keys():
            # Initialize the WeakLensingTracer for each source bin
            tracer = ccl.WeakLensingTracer(self.cosmology,
                                           dndz=(self.redshift_range, self.source_bins[idx]),
                                           ia_bias=ia_bias)
            # Get kernel for the current source bin
            k, chi_bin = tracer.get_kernel(chi)

            # Store the kernel and chi for each bin
            kernel_list.append(k)
            chi_list.append(chi_bin)

        # Convert lists to numpy arrays for easy handling
        kernel_array = np.array(kernel_list)
        chi_array = np.array(chi_list)

        # Return kernels and optionally chi
        if return_chi:
            return kernel_array, chi_array
        else:
            return kernel_array

    def get_nc_kernel(self, include_gbias=True, return_chi=False):
        """
        Compute the number counts (NC) kernel for each lens bin.

        Parameters:
            include_gbias (bool): Include galaxy bias if True.
            return_chi (bool): Return the comoving radial distance array if True.

        Returns:
            tuple: Arrays of NC kernels for each lens bin. If return_chi is True,
                   also returns the comoving radial distance array (chi).
        """
        gbias = self.get_gbias() if include_gbias else None
        # Prepare lists to store kernels and chi values
        kernel_list = []
        chi_list = []

        # Compute comoving radial distance once for the redshift range
        chi = ccl.comoving_radial_distance(self.cosmology, 1 / (1 + self.redshift_range))

        # Loop over each lens bin
        for idx in self.lens_bins.keys():
            # Initialize the NumberCountsTracer for each lens bin
            tracer = ccl.NumberCountsTracer(self.cosmology,
                                            dndz=(self.redshift_range, self.lens_bins[idx]),
                                            bias=gbias,
                                            has_rsd=False)
            # Get kernel for the current lens bin
            k = tracer.get_kernel(chi)

            # Store the kernel and chi for each bin
            kernel_list.append(k)
            chi_list.append(chi)

        # Convert lists to numpy arrays for easy handling
        kernel_array = np.array(kernel_list)
        chi_array = np.array(chi_list)

        # Return kernels and optionally chi
        if return_chi:
            return kernel_array, chi_array
        else:
            return kernel_array

    def comoving_radial_distance(self):

        scale_factor = 1 / (1 + self.redshift_range)
        return ccl.comoving_radial_distance(self.cosmology, scale_factor)

    def cosmic_shear_correlations(self):
        """
        Calculates the source-source bin pairs for cosmic shear.

        Returns:
            list: List of all possible source-source bin pairs.
        """

        sources = self.source_bins
        selected_pairs = []
        source_keys = list(sources.keys())
        for i in range(len(source_keys)):
            for j in range(i, len(source_keys)):
                selected_pairs.append((source_keys[j], source_keys[i]))
        return selected_pairs

    def get_correlation_pairs(self):

        pairings = {
            "cosmic_shear": self.cosmic_shear_correlations(),
            "galaxy_galaxy_lensing": self.galaxy_galaxy_lensing_correlations(),
            "galaxy_clustering": self.galaxy_clustering_correlations()
        }

        return pairings

    def galaxy_clustering_correlations(self):
        """
        Calculates the lens-lens bin pairs for galaxy clustering.

        Returns:
            list: List of lens-lens bin pairs.
         """

        lenses = self.lens_bins

        selected_pairs = [(i, i) for i in lenses.keys()]

        return selected_pairs

    def galaxy_galaxy_lensing_correlations(self):
        """
        Calculates galaxy-galaxy lensing correlations by selecting lens-source bin pairs
        based on their overlap distributions and the allowed overlap threshold.

        Returns:
            selected_pairs (list): A list of selected lens-source bin pairs.
        """
        redshift_range = self.redshift_range
        lenses = self.lens_bins
        sources = self.source_bins
        allowed_overlap = 0.1 if self.forecast_year == "1" else 0.25
        selected_pairs = []  # Initialize an empty list to store selected lens-source distance pairs

        for lens_index, lens_distribution in lenses.items():
            # Calculate the center (peak) of the lens redshift distribution
            lens_center = redshift_range[np.argmax(lens_distribution)]
            for source_index, source_distribution in sources.items():
                # Calculate the center (peak) of the source redshift distribution
                source_center = redshift_range[np.argmax(source_distribution)]
                if source_center > lens_center:
                    # Calculate the overlap distribution by taking the element-wise minimum of the lens and
                    # source distributions at the given distances
                    overlap_distribution = np.minimum(lens_distribution, source_distribution)

                    # Integrate the source and lens distributions over the redshift range and
                    # store the result in 'a' and 'b'
                    a = np.trapz(source_distribution, redshift_range)
                    b = np.trapz(lens_distribution, redshift_range)

                    # Integrate the overlap distribution over the redshift range, divide it by 'a' and 'b',
                    # and store the result in 'overlap'
                    overlap = np.trapz(overlap_distribution, redshift_range) / a / b

                    # Check if the overlap is less than or equal to the allowed overlap threshold
                    # If the overlap is within the allowed range,
                    # add the lens-source distance pair to the 'selected_pairs' list
                    if overlap <= allowed_overlap:
                        selected_pairs.append((source_index, lens_index))

                return selected_pairs

    def get_extra_info(self):

        return f"zmax{self.redshift_max}_zres{self.redshift_resolution}"
