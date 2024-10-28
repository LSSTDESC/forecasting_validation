# Forecasting Validation Project

Welcome to the Forecasting Validation Project repository, which contains scripts and analyses dedicated to validating tools and methodologies within the Forecasting Topical Team (TT) of the LSST DESC.

## Overview

This repository provides resources for validating key DESC tools, including:
- **CCL** (Core Cosmology Library): Cosmology calculations and modeling
- **Firecrown**: Likelihood and cosmological parameter estimation
- **Augur**: Forecasting and parameter inference framework
- **TJPCov**: Covariance computation library for cosmological analyses

Together, these tools are used to forecast and validate cosmological models and statistical inferences 
as part of DESC's broader objectives. 


## Topics

Here we want to primarily validate CCL in terms of angular power spectra.
For that, we will define certain metrics such as:
- **Angular Power Spectra**: Comparing the angular power spectra computed by CCL with those from other tools. 
We will also perform a parametric study where we vary redshift and ell binning to see
how the angular power spectra change.
- **Lensing Kernel**: Comparing the lensing kernel computed by CCL with those from other tools.
As above, we will perform a parametric study where we vary redshift and ell binning to see
how the lensing kernel changes.
- **Comoving Distance**: Comparing the comoving distance computed by CCL with those from other tools.
Parametric study applies here as well.
- **Differences between CCL versions 2.8 and 3.0**: We will compare the angular power spectra
and lensing kernel between CCL versions 2.8 and 3.0.
- ** Validation nz and binning**: We will validate the redshift distribution and binning again just to make sure that it is stable.
## Contents

- **Scripts**: Automation and computation scripts for forecasting workflows.
- **Analysis**: Detailed analyses and validation results for the Forecasting TT project.
- **Configuration**: Configurations and sample `.ini` and `.yaml` files to reproduce forecasting setups.

## Getting Started

To get started, clone this repository and install the required packages for each tool. Example commands and more detailed usage instructions can be found in the corresponding directories.

```bash
git clone <repository_url>
