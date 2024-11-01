import numpy as np
import pyccl as ccl
from .presets import Presets
import matplotlib.pyplot as plt

class ChiRedshift:    

    def __init__(self, presets):
        if not isinstance(presets, Presets):
            raise TypeError(f"Expected a Presets object, but received {type(presets).__name__}.")
        # Set attributes from presets
        self.cosmology = presets.cosmology
        self.forecast_year = presets.forecast_year
        self.redshift_max = presets.redshift_max
        self.redshift_resolution = presets.redshift_resolution
        self.perform_binning = presets.perform_binning
        self.save_data = presets.save_data
    

    def chi_redshift_resolution(self, res_start=300, res_end=10000, step=50, decimal_places=4):

            chi_resolution = {}

            for resolution in range(res_start, res_end + 1, step):
                # Update Presets with the current resolution
                presets = Presets(
                    cosmology=self.cosmology,
                    redshift_max=self.redshift_max,
                    redshift_resolution=resolution,
                    forecast_year=self.forecast_year,
                    perform_binning=self.perform_binning
                )
                redshift_space = np.linspace(0., self.redshift_max, resolution) 
                scale_factor = 1 / (1 + redshift_space)
                chi = ccl.comoving_radial_distance(self.cosmology, scale_factor)
                chi_resolution[resolution] = {"chi": chi, "redshift": redshift_space}
            return chi_resolution

def plot_compare_chi(chi):

    keys = list(chi.keys())

    for i in keys:
        plt.plot(chi[i]['redshift'], chi[i]['chi'])

    plt.xlabel("Redshift", fontsize=18)
    plt.ylabel("Comoving distance", fontsize=18)
    
    plt.show()
