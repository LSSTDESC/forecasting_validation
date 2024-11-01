# Benchmarking Stability and Metrics for Robust Cosmological Forecasting in LSST DESC

Welcome to the **Forecasting Validation Project** repository. This repository provides scripts, analyses, and validation tools as part of the **LSST DESC Forecasting Topical Team (TT)**. The primary focus is on validating angular power spectrum calculations in CCL to ensure robust cosmological forecasting.

## Overview

This repository supports the validation of **[CCL (Core Cosmology Library)](https://github.com/LSSTDESC/CCL)**, a DESC tool used for theoretical modelling. We might extend and apply these validation tools to other DESC software packages, including:
- **[Firecrown](https://github.com/LSSTDESC/firecrown)**: Likelihood generation and parameter estimation.
- **[Augur](https://github.com/LSSTDESC/augur)**: Forecasting framework for parameter inference.
- **[TJPCov](https://github.com/LSSTDESC/TJPCov)**: Computation of covariance matrices for cosmological analyses.

Together, these tools form a powerful suite for DESC’s cosmological forecasting and validation efforts.

## Topics

The primary goal is to validate the computation of **angular power spectra** in CCL. We define a set of metrics and analyses, including:
- **Redshift Distribution and Binning Stability**: Confirm the stability of redshift distributions and tomographic binning configurations across survey settings.
- **Lensing Kernel**: Analyze changes in the lensing kernel with varying redshift and \(\ell\)-binning to confirm consistency.
- **Angular Power Spectra**: Compare angular power spectra computed by CCL with other tools, examining variations under different redshift and \(\ell\)-binning schemes.
- **CCL Version Comparison**: Identify any differences in angular power spectra and lensing kernels between CCL versions 2.8 and 3.0.

These metrics guide us in validating CCL's performance, exploring optimal binning for source and lens samples, and understanding the impact of software updates.

## Tomographic Binning Metrics Analysis

The `NZMetrics` class evaluates the stability of tomographic bin centers across redshift resolutions and `zmax` values. This is essential for ensuring robust binning configurations in cosmological analyses.

### Key Functions
1. **`compare_bin_centers_over_zresolutions`**: 
   - Compares bin center values across various redshift resolutions.
   - Iteratively increases resolution to identify stable bin centers for both source and lens bins.
   - Outputs a dictionary of bin centers at each resolution, with an option to save results.

2. **`compare_bin_centers_over_zresolutions_and_zmax`**: 
   - Extends the analysis by examining bin stability across both `zmax` and redshift resolution ranges.
   - Creates a nested dictionary with bin centers at each combination of `zmax` and resolution values.

### Analysis Objectives
These metrics aim to:
- **Verify Bin Stability**: Ensure that bin centers stabilize as redshift resolution increases, reducing sensitivity to small variations.
- **Determine Optimal Resolution**: Identify the resolution beyond which bin centers remain stable, enabling efficient calculations.
- **Assess `zmax` Sensitivity**: Understand the impact of `zmax` on bin stability to optimize parameter choices for survey setups.

## Data Vector Metrics Analysis

The `DataVectorMetrics` class supports the analysis of cosmological data vectors, including the calculation of kernel peaks and chi-squared differences. This is essential for verifying the accuracy of weak lensing and number counts (NC) kernels in different resolutions.

### Key Functions

1. **`get_kernel_peaks`**:
   - Identifies peak redshifts and values within a given kernel array.
   - Checks kernel length against redshift range and returns peaks as `(redshift, value)` pairs.

2. **`kernel_peaks_z_resolution_sweep`**:
   - Conducts a parametric sweep of kernel peaks across redshift resolutions, for both WL and NC kernels.
   - Outputs a dictionary of peak values for each resolution, aiding in stability analysis.

3. **`get_delta_chi2`**:
   - Calculates the chi-squared difference between two sets of power spectra (galaxy clustering, galaxy-galaxy lensing, and cosmic shear).
   - Utilizes `cls_gc`, `cls_ggl`, and `cls_cs` arrays, reshaping them if necessary, to compute chi-squared values across bins and \(\ell\)-bins.

4. **`kernel_peaks_z_resolution_and_zmax_sweep`**:
   - Performs a parametric sweep across both `zmax` and resolution, storing kernel peaks for both WL and NC.
   - Saves kernel peaks in nested dictionaries with flexibility for intrinsic alignment and galaxy bias.

### Analysis Objectives
This suite helps to:
- **Evaluate Kernel Consistency**: Ensure weak lensing and number count kernels stabilize across various resolutions.
- **Compute Chi-Squared Differences**: Calculate chi-squared differences to quantify variations between model predictions and reference spectra.
- **Test `zmax` and Resolution Sensitivity**: Assess the impact of varying both `zmax` and resolution on kernel peaks for stability.

## Repository Structure

- **`scripts/`**: Computational scripts for binning, kernel validation, and other tasks.
- **`analysis/`**: Documentation of analysis results and validation metrics.
- **`config/`**: Sample `.ini` and `.yaml` configuration files for reproducing validation setups.

## Getting Started

Clone this repository and install the required packages:

```bash
git clone <repository_url>
cd ForecastingValidationProject
# Install required packages (e.g., CCL, Firecrown)
