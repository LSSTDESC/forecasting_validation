import numpy as np
import yaml
import pyccl as ccl
import os


class Presets:
    def __init__(self,
                 cosmology=None,
                 redshift_max=5,
                 redshift_resolution=6000,
                 ell_min=20,
                 ell_max=2000,
                 ell_num=20,
                 forecast_year="1",
                 perform_binning=True,
                 should_save_data=True):
        # Set the cosmology to the user-defined value or a default if not provided
        if cosmology:
            self.cosmology = cosmology
        else:
            self.cosmology = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.83, n_s=0.96)

        self.redshift_max = redshift_max
        self.redshift_resolution = redshift_resolution
        redshift_space = np.linspace(0., self.redshift_max, self.redshift_resolution)
        self.redshift_range = np.array(redshift_space, dtype=np.float64)
        self.ell_min = ell_min
        self.ell_max = ell_max
        self.ell_num = ell_num
        ell_range = np.geomspace(self.ell_min, self.ell_max, self.ell_num)
        self.ells = np.array(ell_range, dtype=np.float64)
        self.perform_binning = perform_binning

        self.ccl_version = ccl.__version__

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
        self.f_sky = lsst_desc_parameters["sky"]["frac_sky"] 
        self.should_save_data = should_save_data

    def save_data(self, name, data, dir=None, extra_info=None, include_ccl_version=True):
        """
        Save data as a .npy file with the pyccl version included in the filename.

        Parameters:
            data (numpy.ndarray): The data to save.
            name (str): The base name for the file (without .npy extension).
            dir (str): The subdirectory to save the file in (default: None).
            extra_info (str): Additional information to include in the filename (default: None).
            include_version (bool): Whether to include the pyccl version in the filename (default: True).
        """
        # Get the pyccl version
        ccl_version = ccl.__version__
        path = "data_output/"

        # Create the full subdirectory path if provided, otherwise use base path
        subdir = os.path.join(path, dir) if dir else path
        version = f"_ccl_v{ccl_version}" if include_ccl_version else ""
        info = f"_{extra_info}" if extra_info else ""


        # Create the directory if it doesn't exist
        os.makedirs(subdir, exist_ok=True)

        # Create the filename with pyccl version included
        filename = os.path.join(subdir, f"{name}_y{self.forecast_year}{info}{version}.npy")

        # Save the data to a .npy file
        np.save(filename, data)
        print(f"Data saved to {filename}")