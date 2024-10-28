
#  Niko Sarcevic
#  nikolina.sarcevic@gmail.com
#  github/nikosarcevic
#  ----------

import numpy as np
from numpy import exp
import pandas
from scipy.integrate import simpson
import yaml
import os


# noinspection PyDefaultArgument
class SRDRedshiftDistributions(object):
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

    def __init__(self,
                 redshift_range,
                 forecast_year="1"):

        self.redshift_range = redshift_range

        supported_forecast_years = {"1", "10"}
        if forecast_year in supported_forecast_years:
            self.forecast_year = forecast_year
        else:
            raise ValueError(f"forecast_year must be one of {supported_forecast_years}.")

        current_dir = os.path.dirname(__file__)
        yaml_path = os.path.join(current_dir, "lsst_desc_parameters.yaml")

        # Load the YAML file
        with open(yaml_path, "r") as f:
            self.lsst_desc_parameters = yaml.load(f, Loader=yaml.FullLoader)

        self.source_parameters = self.lsst_desc_parameters["source_sample"][self.forecast_year]
        self.lens_parameters = self.lsst_desc_parameters["lens_sample"][self.forecast_year]

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

    def source_sample(self, normalised=True, save_file=True, file_format="npy"):
        alpha = self.source_parameters["alpha"]
        beta = self.source_parameters["beta"]
        pivot_redshift = self.source_parameters["z_0"]
        redshift_distribution = self.smail_type_distribution(self.redshift_range,
                                                             pivot_redshift,
                                                             alpha,
                                                             beta)
        if normalised:
            normalisation = simpson(redshift_distribution, self.redshift_range)
            redshift_distribution /= normalisation

        combined_data = {"redshift": self.redshift_range, "dndz": redshift_distribution}

        if save_file:
            self.save_to_file(combined_data, "source_sample", file_format)

        return redshift_distribution

    def lens_sample(self, normalised=True, save_file=True, file_format="npy"):
        alpha = self.lens_parameters["alpha"]
        beta = self.lens_parameters["beta"]
        pivot_redshift = self.lens_parameters["z_0"]
        redshift_distribution = self.smail_type_distribution(self.redshift_range,
                                                             pivot_redshift,
                                                             alpha,
                                                             beta)
        if normalised:
            normalisation = simpson(redshift_distribution, self.redshift_range)
            redshift_distribution /= normalisation

        combined_data = {"redshift": self.redshift_range, "dndz": redshift_distribution}

        if save_file:
            self.save_to_file(combined_data, "lens_sample", file_format)

        return redshift_distribution

    def save_to_file(self, data, name, file_format="npy"):

        if file_format == "npy":
            np.save(f"data_output/srd_{name}_dndz_year_{self.forecast_year}.npy", data)
        elif file_format == "csv":
            dndz_df = pandas.DataFrame(data)
            dndz_df.to_csv(f"data_output/srd_{name}_dndz_year_{self.forecast_year}.csv", index=False)