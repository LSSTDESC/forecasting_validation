import numpy as np
from scripts.metrics import Metrics



def compute_bin_centers(bin_distributions, redshift_range, decimal_places=2):
    """
    Compute bin centers as the redshift value where the redshift distribution is maximized.

    Parameters:
        bin_distributions (dict): A dictionary where keys are bin indices and values are the bin distributions.
        redshift_range (numpy.ndarray): The range of redshift values.
        decimal_places (int): Number of decimal places to round the bin centers.

    Returns:
        dict: A dictionary where keys are bin indices and values are the computed bin centers.
    """
    # Extract the bin indices and number of bins
    bin_indices = list(bin_distributions.keys())
    num_bins = len(bin_indices)

    # Pre-allocate an array for bin centers
    bin_centers_array = np.empty(num_bins)

    # Loop over the redshift bins and compute the center for each
    for i, (index, bin_distribution) in enumerate(bin_distributions.items()):
        max_index = np.argmax(bin_distribution)  # Find the index of the maximum value
        bin_centers_array[i] = redshift_range[max_index]  # Store the center in the array

    # Vectorized rounding of the bin centers
    bin_centers_rounded = np.round(bin_centers_array, decimal_places)

    # Convert back to dictionary with bin indices as keys
    bin_centers_dict = dict(zip(bin_indices, bin_centers_rounded))

    return bin_centers_dict


def investigate_datavector_stability(cosmology, ells, forecast_year="1", cls_type="shear", difference_type="relative"):
    """
    General function to investigate stability of Cls across different redshift resolutions.

    Parameters:
        cosmology: ccl.Cosmology object for computing Cls.
        ells: array of ell values for angular power spectrum calculation.
        forecast_year: str, "1" or "10", indicating the LSST DESC forecast year.
        cls_type: str, either "shear" for cosmic shear Cls or "clustering" for galaxy clustering Cls.
        difference_type: str, one of "absolute", "relative", or "fractional" to specify the type of difference.

    Returns:
        stability_metrics: dict with average differences in Cls between consecutive resolutions.
        results: dict containing Cl values for each redshift resolution.
    """
    # Expanded list of redshift resolutions to explore
    # Sample redshift resolutions from 300 to 5000 with very small increments
    redshift_resolutions = np.arange(300, 5001, 50)  # Generates resolutions from 300 to 5000 with steps of 50s
    results = {}

    # Compute Cls for each redshift resolution
    for res in redshift_resolutions:
        redshift_range = np.linspace(0, 3.5, res)
        mtx = Metrics(cosmology, redshift_range, ells, forecast_year=forecast_year)

        if cls_type == "shear":
            cls = mtx.cosmic_shear_cls()
        elif cls_type == "clustering":
            cls = mtx.galaxy_clustering_cls()
        else:
            raise ValueError("Invalid cls_type. Choose either 'shear' or 'clustering'.")

        results[res] = cls

    # Calculate stability metrics by comparing consecutive resolutions
    stability_metrics = {}
    for i in range(1, len(redshift_resolutions)):
        res1 = redshift_resolutions[i - 1]
        res2 = redshift_resolutions[i]

        # Compute differences based on the selected type
        diff_dict = {
            "absolute": results[res2] - results[res1],
            "relative": (results[res2] / results[res1]) - 1,
            "fractional": (results[res2] - results[res1]) / results[res1]
        }
        diff = diff_dict[difference_type]
        avg_diff = np.mean(np.abs(diff))  # Use np.abs to ensure positive values for averaging

        stability_metrics[(res1, res2)] = avg_diff

    return stability_metrics, results


def find_bin_center(bin_distribution, redshift_range, decimal_places=2):
    """
    Helper method to calculate and round the bin center based on the maximum of the bin distribution.

    Parameters:
        bin_distribution (numpy.ndarray): The redshift distribution for a specific bin.
        redshift_range (numpy.ndarray): The range of redshift values.
        decimal_places (int): Number of decimal places to round the bin centers.

    Returns:
        float: The rounded bin center.
    """
    max_index = np.argmax(bin_distribution)
    return round(redshift_range[max_index], decimal_places)


def get_bin_centers(bins_dict, redshift_range, decimal_places=2):
    """
    Calculate and round the bin centers for all bins in a dictionary based on the maximum of each bin's distribution.

    Parameters:
        bins_dict (dict): A dictionary where each key represents a bin and each value is the redshift distribution for that bin.
        redshift_range (numpy.ndarray): The range of redshift values.
        decimal_places (int): Number of decimal places to round the bin centers.

    Returns:
        dict: A dictionary with bin keys and their corresponding rounded bin centers.
    """
    bin_centers = {}
    for bin_key, bin_distribution in bins_dict.items():
        max_index = np.argmax(bin_distribution)
        bin_centers[bin_key] = round(redshift_range[max_index], decimal_places)
    return bin_centers


def get_galaxy_fraction_in_bin(bin_distribution, redshift_range):
    """
    Calculate the fraction of the total galaxies within each bin's redshift range.

    Parameters:
        bin_distribution (np.ndarray or dict): The redshift distribution for each bin.
            If a dict, the function will compute for each bin.
        redshift_range (np.ndarray): The array of redshift values corresponding to the distribution.

    Returns:
        fraction_in_bin (np.ndarray): Array of fractions of galaxies in each bin.
    """

    def compute_fraction(distribution, total_distribution):
        # Integrate the bin distribution over the redshift range
        bin_integral = np.trapz(distribution, redshift_range)

        # Calculate the fraction for this bin relative to the total distribution
        bin_fraction = bin_integral / np.trapz(total_distribution, redshift_range)
        return bin_fraction

    # Infer the number of bins based on the input bin_distribution
    if isinstance(bin_distribution, dict):
        bin_distributions = np.array(list(bin_distribution.values()))  # Convert dict to array of bins
    else:
        bin_distributions = bin_distribution  # If already an array, use directly

    # Calculate the total distribution by summing over all inferred bins
    total_distribution = np.sum(bin_distributions, axis=0)

    # Compute the fraction for each bin
    fractions = [compute_fraction(distr, total_distribution) for distr in bin_distributions]

    return np.array(fractions)


def calculate_average_galaxies_per_bin(binned_distribution, redshift_range, number_density):
    """
    Calculate the average number of galaxies for each bin using the binned distribution.

    Parameters:
        binned_distribution (dict or np.ndarray): Redshift distribution for each bin.
            If a dict, the function will compute for each bin.
        redshift_range (np.ndarray): The array of redshift values (redshift range).
        number_density (float): The number density of galaxies per arcminute squared.

    Returns:
        avg_galaxies_per_bin (np.ndarray): Array of average number of galaxies for each bin.
    """

    def compute_nz_avg(distribution):
        # Integrate the distribution over the redshift range
        total_integral = np.trapz(distribution, redshift_range)

        # Calculate fraction and total number of galaxies in the bin
        fraction_in_bin = total_integral
        fraction_in_bin /= np.trapz(np.concatenate(list(binned_distribution.values())), redshift_range)
        total_number_of_galaxies = total_integral * number_density

        # Calculate average number of galaxies in this bin
        return fraction_in_bin * total_number_of_galaxies

    # If binned_distribution is a dictionary (multiple bins), calculate for each bin
    if isinstance(binned_distribution, dict):
        avg_galaxies_per_bin = [compute_nz_avg(distr) for distr in binned_distribution.values()]
        return np.array(avg_galaxies_per_bin)

    # If a single bin is passed (as an array), compute directly
    return np.array([compute_nz_avg(binned_distribution)])


def compare_bin_centers_over_resolutions(cosmo,
                                         ells,
                                         res_start=300,
                                         res_end=6000,
                                         step=50,
                                         decimal_places=2):
    """
    Compare tomographic bin centers across a range of redshift resolutions.

    Parameters:
        cosmo_params (dict): Cosmological parameters for initializing the Cosmology.
        ells (numpy.ndarray): Array of ell values for power spectrum calculations.
        bins_dict (dict): Dictionary of bins with redshift distributions.
        res_start (int): Starting resolution for redshift range (default: 300).
        res_end (int): Ending resolution for redshift range (default: 5000).
        step (int): Increment in redshift resolution (default: 50).
        decimal_places (int): Number of decimal places to round the bin centers.

    Returns:
        dict: Nested dictionary containing bin centers for each redshift resolution.
    """
    bin_centers_resolutions = {}

    for resolution in range(res_start, res_end + 1, step):
        redshift_range = np.linspace(0, 3.5, resolution)

        # Initialize Metrics class (assuming it takes cosmo, redshift_range, and ells as input)
        mtx = Metrics(cosmo, redshift_range, ells)

        # Calculate bin centers for source and lens bins
        source_bin_centers = get_bin_centers(mtx.source_bins, redshift_range, decimal_places)
        lens_bin_centers = get_bin_centers(mtx.lens_bins, redshift_range, decimal_places)

        # Store results in the dictionary
        bin_centers_resolutions[f"{resolution}"] = {
            "source_bin_centers": source_bin_centers,
            "lens_bin_centers": lens_bin_centers
        }

    return bin_centers_resolutions
