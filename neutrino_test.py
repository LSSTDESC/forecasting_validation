import sys, platform, os
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import camb
from camb import model, initialpower
import pyccl as ccl

#-------------
#input parameters
h = 0.67556
Omega_c=0.26377065934278865
Omega_b=0.0482754208891869
ns = 0.9667
redshift = 0.9
sigma8=0.8225
#-------------

H0 = h*100
ombh2 = Omega_b*h*h
omch2 = Omega_c*h*h
default_As = 2e-9

def Pk_camb(mode = 'sigma8', mnu = 0):
    '''
    The code ensures that low_k region (k<0.02) is the same with different neutrino mass
    It also ensures that the sigma8 is fixed
    '''
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, neutrino_hierarchy='normal')
    pars.InitPower.set_params(ns=ns, As  = default_As)
    pars.set_matter_power(redshifts=[redshift], kmax=2.0)
    pars.NonLinear = model.NonLinear_both
    results = camb.get_results(pars)
    results.calc_power_spectra(pars)
    omch2_no_nu = (Omega_c-results.get_Omega('nu'))*h*h  
        
    if mode == "sigma8":
        
        pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2_no_nu, mnu=mnu, neutrino_hierarchy='normal')
        pars.InitPower.set_params(ns=ns, As  = default_As)
        pars.set_matter_power(redshifts=[redshift], kmax=2.0)
        pars.NonLinear = model.NonLinear_both
        results.calc_power_spectra(pars)
        results = camb.get_results(pars)

        #print(results.get_sigma8())
        default_sigma8 = results.get_sigma8()
        
    #As is computed from re-scaling sigma8
    
    if mode == "sigma8":
        scale = (sigma8/default_sigma8)**2
        As = scale*default_As
    elif mode == "As":
        As = 2e-9
    else:
        raise ValueError("mode is either sigma8 or As")


    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2_no_nu, mnu=mnu, neutrino_hierarchy='normal')
    pars.InitPower.set_params(ns=ns, As  = As)
    pars.set_matter_power(redshifts=[redshift], kmax=2.0)
    pars.NonLinear = model.NonLinear_both
    results.calc_power_spectra(pars)
    results = camb.get_results(pars)
    #results.get_Omega('nu'), results.get_Omega('de'), results.get_Omega('cdm'), results.get_Omega('photon')
    
    
    print("sigma8:")
    print(results.get_sigma8())

    kh, z, pk_camb = results.get_matter_power_spectrum(minkh=1e-4, maxkh=2, npoints = 200)
    return kh, pk_camb



def Pk_ccl(mode = 'sigma8', mnu = 0):
    if mode == 'sigma8':
        cosmo_A = ccl.Cosmology(Omega_c=Omega_c, Omega_b=Omega_b, h=h, n_s=ns,\
                          sigma8=sigma8, transfer_function = "boltzmann_camb", m_nu=mnu, mass_split='normal')
        omega_m_A = cosmo_A['Omega_m']
        
        cosmo_B = ccl.Cosmology(Omega_c=Omega_c, Omega_b=Omega_b, h=h, n_s=ns,\
                          sigma8=sigma8, transfer_function = "boltzmann_camb", m_nu=0, mass_split='normal')
        omega_m_B = cosmo_B['Omega_m']
        dm = omega_m_A-omega_m_B
        
        
        cosmo1 = ccl.Cosmology(Omega_c=Omega_c-dm, Omega_b=Omega_b, h=h, n_s=ns,\
                          sigma8=sigma8, transfer_function = "boltzmann_camb", m_nu=mnu, mass_split='normal')
        
    elif mode == "As":
        cosmo_A = ccl.Cosmology(Omega_c=Omega_c, Omega_b=Omega_b, h=h, n_s=ns,\
                          A_s=default_As, transfer_function = "boltzmann_camb", m_nu=mnu, mass_split='normal')
        omega_m_A = cosmo_A['Omega_m']
        
        cosmo_B = ccl.Cosmology(Omega_c=Omega_c, Omega_b=Omega_b, h=h, n_s=ns,\
                          A_s=default_As, transfer_function = "boltzmann_camb", m_nu=0, mass_split='normal')
        omega_m_B = cosmo_B['Omega_m']
        dm = omega_m_A-omega_m_B
        
        cosmo1 = ccl.Cosmology(Omega_c=Omega_c-dm, Omega_b=Omega_b, h=h, n_s=ns,\
                          A_s=default_As, transfer_function = "boltzmann_camb", m_nu=mnu, mass_split='normal')


    wavenumber = np.linspace(0.0001, 1, 1000)
    scalefactor = 1/(1+redshift)

    pk_nlin1 = ccl.nonlin_matter_power(cosmo1, wavenumber, scalefactor)
    
    return wavenumber/h, pk_nlin1*h**3
    