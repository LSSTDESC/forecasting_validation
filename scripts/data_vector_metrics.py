import numpy as np
from scripts.data_vectors import DataVectors
from scripts.presets import Presets


class DataVectorMetrics:

    def __init__(self, presets):
        if not isinstance(presets, Presets):
            raise TypeError(f"Expected a Presets object, but received {type(presets).__name__}.")
        # Set attributes from presets
        self.presets = presets

    def investigate_data_vector_stability(self,
                                          cls_type="shear",
                                          difference_type="relative"):
        """
        General function to investigate stability of Cls across different redshift resolutions.

        Parameters:
            cls_type: str, either "shear" for cosmic shear Cls or "clustering" for galaxy clustering Cls.
            difference_type: str, one of "absolute", "relative", or "fractional" to specify the type of difference.

        Returns:
            stability_metrics: dict with average differences in Cls between consecutive resolutions.
            results: dict containing Cl values for each redshift resolution.
        """

        # Define a list of redshift resolutions to explore, from 300 to 5000 with steps of 50
        redshift_resolutions = np.arange(300, 5001, 50)
        results = {}

        # Compute Cls for each redshift resolution
        for res in redshift_resolutions:
            # Initialize DataVectors with the current redshift resolution
            dv = DataVectors(self.presets)

            # Choose Cls calculation based on cls_type
            if cls_type == "shear":
                cls = dv.cosmic_shear_cls()
            elif cls_type == "clustering":
                cls = dv.galaxy_clustering_cls()
            else:
                raise ValueError("Invalid cls_type. Choose either 'shear' or 'clustering'.")

            results[res] = cls

        # Calculate stability metrics by comparing Cls at consecutive resolutions
        stability_metrics = {}
        for i in range(1, len(redshift_resolutions)):
            res1 = redshift_resolutions[i - 1]
            res2 = redshift_resolutions[i]

            # Compute differences based on the selected difference type
            diff_dict = {
                "absolute": results[res2] - results[res1],
                "relative": (results[res2] / results[res1]) - 1,
                "fractional": (results[res2] - results[res1]) / results[res1]
            }
            diff = diff_dict[difference_type]
            avg_diff = np.mean(np.abs(diff))  # Use np.abs to ensure positive values for averaging

            stability_metrics[(res1, res2)] = avg_diff

        return stability_metrics, results
