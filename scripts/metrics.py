import numpy as np
import pyccl as ccl
from .srd_redshift_distributions import SRDRedshiftDistributions
from .tomographic_binning import TomographicBinning
import yaml
import os


class Metrics:

    def __init__(self, cosmology, redshift_range, ells, forecast_year="1"):
        self.cosmology = cosmology
        self.redshift_range = redshift_range
        self.ells = ells

        supported_forecast_years = {"1", "10"}
        if forecast_year in supported_forecast_years:
            self.forecast_year = forecast_year
        else:
            raise ValueError(f"forecast_year must be one of {supported_forecast_years}.")

        current_dir = os.path.dirname(__file__)
        yaml_path = os.path.join(current_dir, "lsst_desc_parameters.yaml")

        # Load the YAML file
        with open(yaml_path, "r") as f:
            lsst_desc_parameters = yaml.load(f, Loader=yaml.FullLoader)

        self.lens_parameters = lsst_desc_parameters["lens_sample"][self.forecast_year]
        self.source_parameters = lsst_desc_parameters["source_sample"][self.forecast_year]

        self.lens_nz = SRDRedshiftDistributions(self.redshift_range,
                                                self.forecast_year).lens_sample()
        self.source_nz = SRDRedshiftDistributions(self.redshift_range,
                                                  self.forecast_year).source_sample()
        self.lens_bins = TomographicBinning(self.redshift_range,
                                            self.forecast_year).lens_bins()
        self.source_bins = TomographicBinning(self.redshift_range,
                                              self.forecast_year).source_bins()

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
        return cls_array

    def galaxy_clustering_cls(self, include_gbias=True):
        gbias = self.get_gbias() if include_gbias else None

        correlations = self.get_correlation_pairs()["galaxy_clustering"]

        cls_list = []

        for idx_1, idx_2 in correlations:
            tracer1 = ccl.NumberCountsTracer(self.cosmology,
                                             has_rsd=False,
                                             dndz=(self.redshift_range, self.lens_bins[idx_1]),
                                             bias=(self.redshift_range, np.full_like(self.redshift_range, 1.0)))
            tracer2 = ccl.NumberCountsTracer(self.cosmology,
                                             has_rsd=False,
                                             dndz=(self.redshift_range, self.lens_bins[idx_2]),
                                             bias=(self.redshift_range, np.full_like(self.redshift_range, 1.0)))

            cls_list.append(ccl.angular_cl(self.cosmology, tracer1, tracer2, self.ells))

        cls_array = np.column_stack(cls_list)
        return cls_array

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