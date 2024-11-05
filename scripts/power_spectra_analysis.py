import numpy as np
import os
from scripts.presets import Presets
from scripts.data_vectors import DataVectors
from scripts.data_vector_metrics import DataVectorMetrics
import time
import gc


class PowerSpectraAnalysis:
    def __init__(self, presets):
        if not isinstance(presets, Presets):
            raise TypeError(f"Expected a Presets object, but received {type(presets).__name__}.")
        self.forecast_year = presets.forecast_year
        self.ell_max = presets.ell_max

        # Set up base directories for spectra and distance metric calculations
        base_dir = os.path.join("data_output", "spectra_sweep", f"spectra_y{self.forecast_year}")
        self.spectra_dir = os.path.join(base_dir, "cls")
        self.zmax_zres_dir = os.path.join(self.spectra_dir, "zscan")
        self.ell_dir = os.path.join(self.spectra_dir, "ellscan")

        # Fiducial directory for baseline spectra
        self.fiducial_dir = os.path.join(base_dir, "fiducial")

        # Directories for delta metric results
        self.metric_dir = os.path.join("data_output", "metric")
        self.metric_zmax_zres_dir = os.path.join(self.metric_dir, "zscan")
        self.metric_ell_dir = os.path.join(self.metric_dir, "ellscan")

        # Create directories
        for directory in [self.spectra_dir, self.zmax_zres_dir, self.ell_dir, self.fiducial_dir,
                          self.metric_dir, self.metric_zmax_zres_dir, self.metric_ell_dir]:
            os.makedirs(directory, exist_ok=True)

    def compute_fiducial_spectra(self):
        """
        Compute and save the fiducial power spectra (Cl) for galaxy clustering, galaxy-galaxy lensing, and cosmic shear.
        """
        presets = Presets(forecast_year=self.forecast_year, should_save_data=False)
        data = DataVectors(presets)

        # Compute fiducial power spectra
        cl_gc = data.galaxy_clustering_cls(include_all_correlations=True)
        cl_ggl = data.galaxy_galaxy_lensing_cls(include_all_correlations=True)
        cl_cs = data.cosmic_shear_cls(include_all_correlations=True)

        # Save each fiducial spectrum as its own .npy file in the fiducial directory
        np.save(os.path.join(self.fiducial_dir, f"cl_gc_fiducial_y{self.forecast_year}.npy"), cl_gc)
        np.save(os.path.join(self.fiducial_dir, f"cl_ggl_fiducial_y{self.forecast_year}.npy"), cl_ggl)
        np.save(os.path.join(self.fiducial_dir, f"cl_cs_fiducial_y{self.forecast_year}.npy"), cl_cs)
        print("Fiducial spectra saved.")


    def compute_cls_zres_and_zmax(self, zet_max_range=np.arange(3, 4.1, 0.1),
                                  zet_res_range=np.arange(300, 10050, 50).astype(int)):
        """
        Precompute and save power spectra (Cl) for each combination of zet_max and zet_res.
        """
        for val_max in zet_max_range:
            # Round `zmax` to 1 decimal place to avoid floating-point issues
            val_max_rounded = round(val_max, 1)
            val_max_str = f"{val_max_rounded:.1f}"

            for res in zet_res_range:
                # Ensure `zres` is an integer
                val_res = int(res)
                gc.collect()
                try:
                    presets = Presets(redshift_max=val_max, redshift_resolution=val_res,
                                      forecast_year=self.forecast_year, should_save_data=False)
                    data = DataVectors(presets)

                    cl_gc = data.galaxy_clustering_cls(include_all_correlations=True)
                    cl_ggl = data.galaxy_galaxy_lensing_cls(include_all_correlations=True)
                    cl_cs = data.cosmic_shear_cls(include_all_correlations=True)

                    # Save spectra with formatted filenames
                    np.save(os.path.join(self.zmax_zres_dir,
                                         f"cl_gc_y{self.forecast_year}_zmax{val_max_str}_zres{val_res}.npy"), cl_gc)
                    np.save(os.path.join(self.zmax_zres_dir,
                                         f"cl_ggl_y{self.forecast_year}_zmax{val_max_str}_zres{val_res}.npy"), cl_ggl)
                    np.save(os.path.join(self.zmax_zres_dir,
                                         f"cl_cs_y{self.forecast_year}_zmax{val_max_str}_zres{val_res}.npy"), cl_cs)
                    print(f"Saved spectra for zmax={val_max_str}, zres={val_res}")

                except Exception as e:
                    print(f"Error while saving spectra for zmax={val_max_str}, zres={val_res}: {e}")

    def compute_cls_ell_sweep(self, num_ell_values=50):
        """
        Precompute and save power spectra (Cl) as a function of varying ell values.
        Parameters:
            num_ell_values (int): Number of ell values to compute in the sweep (default: 50)
        """
        # Generate integer ell values in a geometrically spaced range
        ell_values = np.unique(np.geomspace(20, self.ell_max, num_ell_values).astype(int))

        for val_ell in ell_values:
            # Round and convert to integer
            ell_int = round(val_ell, 1)
            print(f"Attempting to compute and save spectra for ell={ell_int}")

            try:
                presets_ell = Presets(forecast_year=self.forecast_year, ell_num=ell_int, should_save_data=False)
                data_ell = DataVectors(presets_ell)

                # Compute power spectra for the current ell value
                cl_gc = data_ell.galaxy_clustering_cls(include_all_correlations=True)
                cl_ggl = data_ell.galaxy_galaxy_lensing_cls(include_all_correlations=True)
                cl_cs = data_ell.cosmic_shear_cls(include_all_correlations=True)

                # Save each spectrum in the ellscan directory
                np.save(os.path.join(self.ell_dir, f"cl_gc_y{self.forecast_year}_ell{ell_int}.npy"), cl_gc)
                np.save(os.path.join(self.ell_dir, f"cl_ggl_y{self.forecast_year}_ell{ell_int}.npy"), cl_ggl)
                np.save(os.path.join(self.ell_dir, f"cl_cs_y{self.forecast_year}_ell{ell_int}.npy"), cl_cs)
                print(f"Saved spectra for ell={ell_int}")

            except Exception as e:
                print(f"Error during ell sweep for ell={ell_int}: {e}")

    def compute_distance_metric_redshift_sweep(self,
                                               zet_max_range=np.arange(3, 4.1, 0.1),
                                               zet_res_range=np.arange(300, 10050, 50).astype(int)):
        """
        Compute the distance metric for precomputed zmax and zres sweep.
        """
        # Paths for fiducial spectra
        cl_gc_path = os.path.join(self.fiducial_dir, f"cl_gc_fiducial_y{self.forecast_year}.npy")
        cl_ggl_path = os.path.join(self.fiducial_dir, f"cl_ggl_fiducial_y{self.forecast_year}.npy")
        cl_cs_path = os.path.join(self.fiducial_dir, f"cl_cs_fiducial_y{self.forecast_year}.npy")

        # Confirm existence and load fiducial spectra
        if not all(os.path.isfile(path) for path in [cl_gc_path, cl_ggl_path, cl_cs_path]):
            print("Error: One or more fiducial spectra files are missing.")
            print(f"Expected at:\n{cl_gc_path}\n{cl_ggl_path}\n{cl_cs_path}")
            return

        cl_gc_fid = np.load(cl_gc_path)
        cl_ggl_fid = np.load(cl_ggl_path)
        cl_cs_fid = np.load(cl_cs_path)
        print(f"Loaded fiducial spectra from:\n{cl_gc_path}\n{cl_ggl_path}\n{cl_cs_path}")

        # Initialize the metric calculation
        presets = Presets(forecast_year=self.forecast_year, should_save_data=False)
        metric = DataVectorMetrics(presets)
        delta = np.zeros((len(zet_max_range), len(zet_res_range)))

        for i_max, val_max in enumerate(zet_max_range):
            val_max_str = f"{val_max:.1f}"

            for i_res, val_res in enumerate(zet_res_range):
                try:
                    cl_gc = np.load(os.path.join(self.zmax_zres_dir,
                                                 f"cl_gc_y{self.forecast_year}_zmax{val_max_str}_zres{val_res}.npy"))
                    cl_ggl = np.load(os.path.join(self.zmax_zres_dir,
                                                  f"cl_ggl_y{self.forecast_year}_zmax{val_max_str}_zres{val_res}.npy"))
                    cl_cs = np.load(os.path.join(self.zmax_zres_dir,
                                                 f"cl_cs_y{self.forecast_year}_zmax{val_max_str}_zres{val_res}.npy"))

                    delta[i_max, i_res] = metric.get_delta_chi2(cl_gc_fid, cl_ggl_fid, cl_cs_fid, cl_gc, cl_ggl, cl_cs)
                except Exception as e:
                    print(f"Error during metric computation for zmax={val_max_str}, zres={val_res}: {e}")

        delta_filename = os.path.join(self.metric_zmax_zres_dir, f"delta_sweep_zres_zmax_y{self.forecast_year}.npy")
        np.save(delta_filename, delta)
        print(f"Saved delta results for zres and zmax sweep: {delta_filename}")
        return delta

    def compute_distance_metric_ell_sweep(self, num_ell_values=50):
        """
        Compute the distance metric for precomputed ell sweep, directly loading and using precomputed files.
        Parameters:
            num_ell_values (int): Number of ell values to compute in the sweep (default: 50)
        """
        # Load fiducial spectra
        cl_gc_fid = np.load(os.path.join(self.fiducial_dir, f"cl_gc_fiducial_y{self.forecast_year}.npy"))
        cl_ggl_fid = np.load(os.path.join(self.fiducial_dir, f"cl_ggl_fiducial_y{self.forecast_year}.npy"))
        cl_cs_fid = np.load(os.path.join(self.fiducial_dir, f"cl_cs_fiducial_y{self.forecast_year}.npy"))
        print("Loaded fiducial spectra for chi-squared reference")

        # Initialize the metric calculation
        presets = Presets(forecast_year=self.forecast_year, should_save_data=False)
        metric = DataVectorMetrics(presets)

        # Set up a geometrically spaced array of ell values
        ell_values = np.unique(np.geomspace(20, self.ell_max, num_ell_values).astype(int))
        chi2_y1 = np.zeros(len(ell_values))

        for i, val_ell in enumerate(ell_values):
            try:
                # Load the precomputed spectra for this ell value
                cl_gc = np.load(os.path.join(self.ell_dir, f"cl_gc_y{self.forecast_year}_ell{val_ell}.npy"))
                cl_ggl = np.load(os.path.join(self.ell_dir, f"cl_ggl_y{self.forecast_year}_ell{val_ell}.npy"))
                cl_cs = np.load(os.path.join(self.ell_dir, f"cl_cs_y{self.forecast_year}_ell{val_ell}.npy"))

                # Ensure all loaded spectra match the fiducial spectra's shape
                cl_gc = self.match_shape(cl_gc, cl_gc_fid)
                cl_ggl = self.match_shape(cl_ggl, cl_ggl_fid)
                cl_cs = self.match_shape(cl_cs, cl_cs_fid)

                # Perform the chi-squared calculation
                delta_diff = metric.get_delta_chi2(cl_gc_fid, cl_ggl_fid, cl_cs_fid, cl_gc, cl_ggl, cl_cs)
                chi2_y1[i] = np.abs(delta_diff)  # Store the chi-squared result for this ell value

                print(f"Processed ell={val_ell} with delta chi-squared: {chi2_y1[i]}")

            except Exception as e:
                print(f"Error loading or computing chi-squared for ell={val_ell}: {e}")
                chi2_y1[i] = np.nan  # Indicate error in computation for this ell value

        # Save the results
        ell_filename = os.path.join(self.metric_ell_dir, f"delta_ell_sweep_y{self.forecast_year}.npy")
        np.save(ell_filename, chi2_y1)
        print(f"Saved delta chi-squared results for ell sweep: {ell_filename}")
        return chi2_y1

    def match_shape(self, arr, target_shape):
        """
        Adjust the shape of `arr` to match `target_shape` by trimming or interpolating.
        """
        target_len = target_shape.shape[0]
        arr_len = arr.shape[0]

        if arr_len == target_len:
            return arr  # No adjustment needed
        elif arr_len > target_len:
            # Trim array to target length
            return arr[:target_len]
        else:
            # Interpolate to increase length if needed
            x_old = np.linspace(0, 1, arr_len)
            x_new = np.linspace(0, 1, target_len)
            return np.interp(x_new, x_old, arr, axis=0)
