import numpy as np
from numpy import exp
from .presets import Presets


# noinspection PyDefaultArgument,PyMethodMayBeStatic
class SRDRedshiftDistributions:
    """
        Generate the LSST DESC type redshift distributions
        for lens and source sample for year 1 and year 10.
        See the LSST DESC Science Requirements Document (SRD)
        https://arxiv.org/abs/1809.01669. The model used here
        is the Smail type redshift distribution. This class
        reads the parameters automatically from a yaml file
        included in this repository (lsst_desc_parameters.yaml).
        ...
        Attributes
        ----------
        redshift_range: array
        forecast_year: string
            year that corresponds to the SRD forecast. Accepted values
            are "1" and "10"
         """

    def __init__(self, presets: Presets):
        if not isinstance(presets, Presets):
            raise TypeError(f"Expected a Presets object, but received {type(presets).__name__}.")
        self.forecast_year = presets.forecast_year
        self.redshift_range = presets.redshift_range

        self.lens_parameters = presets.lens_parameters
        self.source_parameters = presets.source_parameters

        self.save_data = presets.save_data

    def smail_type_distribution(self,
                                redshift_range,
                                pivot_redshift,
                                alpha,
                                beta):

        """
        Generate the LSST DESC SRD parametric redshift distribution (Smail-type).
        For details check LSST DESC SRD paper https://arxiv.org/abs/1809.01669, equation 5.
        The redshift distribution parametrisation is a smail type of the form
        N(z) = (z / z0) ^ beta * exp[- (z / z0) ^ alpha],
        where z is redshift, z0 is pivot redshift, and alpha and beta are power law indices.
        ----------
        Arguments:
            redshift_range: array
                redshift range
            pivot_redshift: float
                pivot redshift
            alpha: float
                power law index in the exponent
            beta: float
                power law index in the prefactor
        Returns:
            redshift_distribution: array
                A Smail-type redshift distribution over a range of redshifts.
                """

        redshift_distribution = [(z / pivot_redshift) ** beta * exp(-(z / pivot_redshift) ** alpha) for z in
                                 redshift_range]

        return redshift_distribution

    def source_sample(self, normalized=True, save_file=True):

        redshift_distribution = self.smail_type_distribution(self.redshift_range,
                                                             self.source_parameters["z_0"],
                                                             self.source_parameters["alpha"],
                                                             self.source_parameters["beta"])
        if normalized:
            normalisation = np.trapz(redshift_distribution, x=self.redshift_range)
            redshift_distribution /= normalisation

        combined_data = {"redshift": self.redshift_range, "dndz": redshift_distribution}

        if save_file:
            self.save_data("source_sample",
                           combined_data,
                           directory="redshift_distributions",
                           include_ccl_version=False)

        return redshift_distribution

    def lens_sample(self, normalized=True, save_file=True):

        redshift_distribution = self.smail_type_distribution(self.redshift_range,
                                                             self.lens_parameters["z_0"],
                                                             self.lens_parameters["alpha"],
                                                             self.lens_parameters["beta"])
        if normalized:
            normalisation = np.trapz(redshift_distribution, x=self.redshift_range)
            redshift_distribution /= normalisation

        combined_data = {"redshift": self.redshift_range, "dndz": redshift_distribution}

        if save_file:
            self.save_data("lens_sample",
                           combined_data,
                           directory="redshift_distributions",
                           include_ccl_version=False)

        return redshift_distribution
