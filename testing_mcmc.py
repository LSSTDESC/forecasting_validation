import pyccl as ccl
print(f"I am using pyccl version {ccl.__version__}")
import numpy as np
import matplotlib.pyplot as plt
from scripts.presets import Presets
from scripts.data_vectors import DataVectors
from scripts.data_vector_metrics import DataVectorMetrics
import emcee

# This is the fiducial
forecast_year = "1"
presets = Presets(forecast_year=forecast_year, should_save_data=False)
data = DataVectors(presets)
cl_gc = np.array(data.galaxy_clustering_cls(include_all_correlations=True))
cl_ggl = np.array(data.galaxy_galaxy_lensing_cls(include_all_correlations=True))
cl_cs = np.array(data.cosmic_shear_cls(include_all_correlations=True))
metric = DataVectorMetrics(presets)
cls_noise = metric.get_matrix(cl_gc, cl_ggl, cl_cs, True)

def likelihood(params):
    Om = float(params['Om'])
    Ob = float(params['Ob'])
    h = float(params['h'])
    sigma8 = float(params['sigma8'])
    n_s = float(params['ns'])
    w0 = float(params['w0'])
    wa = float(params['wa'])
    if Ob >= Om or Ob <= 0 or Om <= 0 or sigma8 <= 0:
        return -np.inf
    try:
        cosmo = ccl.Cosmology(Omega_c=(Om-Ob),
                                Omega_b=Ob,
                                h=h,
                                sigma8=sigma8,
                                n_s=n_s,
                                w0=w0,
                                wa=wa)
        presets_mc = Presets(cosmology=cosmo, forecast_year=forecast_year, should_save_data=False)
        
        #presets_mc = Presets(cosmology=cosmo,redshift_resolution=600, forecast_year=forecast_year, should_save_data=False)
        data = DataVectors(presets_mc)
        cl_gc_mc = np.array(data.galaxy_clustering_cls(include_all_correlations=True))
        cl_ggl_mc = np.array(data.galaxy_galaxy_lensing_cls(include_all_correlations=True))
        cl_cs_mc = np.array(data.cosmic_shear_cls(include_all_correlations=True))
        metric = DataVectorMetrics(presets_mc)
        cls = metric.get_matrix(cl_gc_mc, cl_ggl_mc, cl_cs_mc, False)
        return metric.get_loglike(cls, cls_noise)
    except:
        return -np.inf

from nautilus import Prior

prior = Prior()
prior.add_parameter('Om', dist=(0,1))
prior.add_parameter('Ob', dist=(0,1))
prior.add_parameter('h', dist=(0.5,1))
prior.add_parameter('sigma8', dist=(0.5,1))
prior.add_parameter('ns', dist=(0.5,1))
prior.add_parameter('w0', dist=(-5,0))
prior.add_parameter('wa', dist=(-10,10))


from nautilus import Sampler
filename = "benchmark_y" + forecast_year + ".h5"
sampler = Sampler(prior, likelihood, n_live=5000, filepath=filename, pool = 50)
sampler.run(verbose=True)
