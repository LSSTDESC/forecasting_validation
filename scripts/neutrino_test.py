import numpy as np
import camb
from camb import model
import pyccl as ccl


def parameters():
    # Input parameters
    h = 0.67556
    Omega_c = 0.26377065934278865
    Omega_b = 0.0482754208891869
    ns = 0.9667
    redshift = 0.9
    sigma8 = 0.8225

    # Return dictionary with input and derived parameters
    return {
        "h": h,
        "Omega_c": Omega_c,
        "Omega_b": Omega_b,
        "ns": ns,
        "redshift": redshift,
        "sigma8": sigma8,
        "H0": h * 100,
        "ombh2": Omega_b * h**2,
        "omch2": Omega_c * h**2,
        "As": 2e-9
    }


def camb_power_spectrum(mode='sigma8', mnu=0, min_k=1e-4, max_k=2, num_k=200):
    """
    Compute the matter power spectrum using CAMB.

    Parameters:
    - mode (str): Either 'sigma8' or 'As' to specify normalization mode.
    - mnu (float): Sum of neutrino masses.

    Returns:
    - kh (array): Wavenumbers (h/Mpc).
    - pk_camb (array): Matter power spectrum values.
    """
    # Fetch parameters
    params = parameters()
    h, H0, ombh2, omch2, ns, redshift, sigma8, default_As = (
        params["h"], params["H0"], params["ombh2"], params["omch2"],
        params["ns"], params["redshift"], params["sigma8"], params["As"]
    )

    # Helper function to configure CAMB parameters
    def configure_camb(ombh2, omch2, As):
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, neutrino_hierarchy='normal')
        pars.InitPower.set_params(ns=ns, As=As)
        pars.set_matter_power(redshifts=[redshift], kmax=2.0)
        pars.NonLinear = camb.model.NonLinear_both
        return pars

    # Step 1: Calculate neutrino-adjusted omch2
    pars = configure_camb(ombh2, omch2, default_As)
    results = camb.get_results(pars)
    results.calc_power_spectra(pars)
    omch2_no_nu = (params["Omega_c"] - results.get_Omega('nu')) * h ** 2

    # Step 2: Adjust As based on the normalization mode
    if mode == "sigma8":
        # Calculate default sigma8
        pars = configure_camb(ombh2, omch2_no_nu, default_As)
        results = camb.get_results(pars)
        default_sigma8 = results.get_sigma8()

        # Rescale As to match the target sigma8
        As = (sigma8 / default_sigma8) ** 2 * default_As
    elif mode == "As":
        As = default_As
    else:
        raise ValueError("Invalid mode: choose either 'sigma8' or 'As'.")

    # Step 3: Final CAMB run with adjusted As
    pars = configure_camb(ombh2, omch2_no_nu, As)
    results = camb.get_results(pars)
    results.calc_power_spectra(pars)

    # Output sigma8 for verification
    print("sigma8:", results.get_sigma8())

    # Step 4: Extract matter power spectrum
    kh, _, pk_camb = results.get_matter_power_spectrum(minkh=min_k, maxkh=max_k, npoints=num_k)
    return kh, pk_camb[0].T


def ccl_power_spectrum(mode='sigma8', mnu=0, min_k=1e-4, max_k=2, num_k=200):
    """
    Compute the non-linear matter power spectrum using CCL.

    Parameters:
    - mode (str): Either 'sigma8' or 'As' to specify normalization mode.
    - mnu (float): Sum of neutrino masses.

    Returns:
    - wavenumber (array): Wavenumbers (1/Mpc).
    - pk_nlin1 (array): Non-linear matter power spectrum (Mpc^3).
    """
    # Fetch parameters
    params = parameters()
    h, Omega_c, Omega_b, ns, redshift, sigma8, default_As = (
        params["h"], params["Omega_c"], params["Omega_b"], params["ns"],
        params["redshift"], params["sigma8"], params["As"]
    )

    # Helper function to create a CCL Cosmology instance
    def create_cosmology(Omega_c, Omega_b, h, ns, norm, mnu):
        return ccl.Cosmology(
            Omega_c=Omega_c,
            Omega_b=Omega_b,
            h=h,
            n_s=ns,
            transfer_function="boltzmann_camb",
            m_nu=mnu,
            mass_split="normal",
            **norm
        )

    # Determine normalization
    if mode == 'sigma8':
        norm = {"sigma8": sigma8}
    elif mode == 'As':
        norm = {"A_s": default_As}
    else:
        raise ValueError("Invalid mode: choose either 'sigma8' or 'As'.")

    # Calculate the difference due to neutrinos
    cosmo_with_mnu = create_cosmology(Omega_c, Omega_b, h, ns, norm, mnu)
    cosmo_no_mnu = create_cosmology(Omega_c, Omega_b, h, ns, norm, mnu=0)
    Omega_c_diff = cosmo_with_mnu['Omega_m'] - cosmo_no_mnu['Omega_m']

    # Adjust Omega_c to account for neutrinos
    cosmo = create_cosmology(Omega_c - Omega_c_diff, Omega_b, h, ns, norm, mnu)

    # Compute the power spectrum
    wavenumber = np.linspace(min_k, max_k, num_k)  # in units of h/Mpc
    scalefactor = 1 / (1 + redshift)
    pk_nlin1 = ccl.nonlin_matter_power(cosmo, wavenumber, scalefactor)

    # Convert to physical units
    k = wavenumber / h
    pk = pk_nlin1 * h ** 3

    return k, pk


def compute_power_spectra(mnu=0.3, min_k=1e-4, max_k=2, num_k=200):
    """
    Compute power spectra for both CAMB and CCL models and for neutrino mass
    values mnu=0 and a non-zero neutrino mass (default set to 0.3),
    and for normalization modes 'sigma8' and 'As'.
    This is aa wrapper function that calls the respective functions defined above
    to compute power spectra.

    Parameters:
        - mnu (float): Sum of neutrino masses. Default is 0.3.
        - min_k (float): Minimum wavenumber. Default is 1e-4.
        - max_k (float): Maximum wavenumber. Default is 2.
        - num_k (int): Number of wavenumber points. Default is 200.

    Returns:
    - power_spectra (dict): Nested dictionary containing k and pk values.
    """
    models = ["camb", "ccl"]  # Thoeretical predictions
    mnu_values = [0, mnu]  # Neutrino mass values
    modes = ["sigma8", "As"]  # Normalization modes

    # Initialize the nested dictionary
    power_spectra = {}

    for model in models:
        power_spectra[model] = {}
        for mode in modes:
            power_spectra[model][mode] = {}
            for mnu_val in mnu_values:
                if model == "camb":
                    k, pk = camb_power_spectrum(mode=mode, mnu=mnu_val, min_k=min_k, max_k=max_k, num_k=num_k)
                elif model == "ccl":
                    k, pk = ccl_power_spectrum(mode=mode, mnu=mnu_val, min_k=min_k, max_k=max_k, num_k=num_k)
                else:
                    raise ValueError(f"Unsupported model: {model}")

                power_spectra[model][mode][f"mnu={mnu_val}"] = {"k": k, "pk": pk}

    return power_spectra


# Define color schemes for plots
colors = {
    "camb":
        {"sigma8": {"mnu=0": "darkorange", "mnu=0.3": "teal"},
        "As": {"mnu=0": "orangered", "mnu=0.3": "darkslategrey"}},
    "ccl":
        {"sigma8": {"mnu=0": "darkorange", "mnu=0.3": "teal"},
        "As": {"mnu=0": "orangered", "mnu=0.3": "darkslategrey"}},
}

# Define line styles for plots
line_styles = {
    "camb": {
        "sigma8": {"mnu=0": "-", "mnu=0.3": "-"},
        "As": {"mnu=0": "-", "mnu=0.3": "-"}
    },
    "ccl": {
        "sigma8": {"mnu=0": "--", "mnu=0.3": "--"},
        "As": {"mnu=0": "--", "mnu=0.3": "--"}}
}

# Define line widths for plots
line_widths = {
    "camb": { "sigma8": {"mnu=0": 3, "mnu=0.3": 3},
              "As": {"mnu=0": 3, "mnu=0.3": 3}},
    "ccl": {"sigma8": {"mnu=0": 3,"mnu=0.3": 3},
            "As": {"mnu=0": 3, "mnu=0.3": 3}}
}


def Pk_camb(mode='sigma8', mnu=0):
    '''
    The code ensures that low_k region (k<0.02) is the same with different neutrino mass
    It also ensures that the sigma8 is fixed
    '''
    params = parameters()
    h, H0, ombh2, omch2, ns, redshift, sigma8, default_As, Omega_c = (
        params["h"], params["H0"], params["ombh2"], params["omch2"],
        params["ns"], params["redshift"], params["sigma8"], params["As"], params["Omega_c"]
    )

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
    pars.InitPower.set_params(ns=ns, As=As)
    pars.set_matter_power(redshifts=[redshift], kmax=2.0)
    pars.NonLinear = model.NonLinear_both
    results.calc_power_spectra(pars)
    results = camb.get_results(pars)
    #results.get_Omega('nu'), results.get_Omega('de'), results.get_Omega('cdm'), results.get_Omega('photon')
    
    print("sigma8:")
    print(results.get_sigma8())

    kh, z, pk_camb = results.get_matter_power_spectrum(minkh=1e-4, maxkh=2, npoints=200)
    return kh, pk_camb


def Pk_ccl(mode='sigma8', mnu=0):
    params = parameters()
    h, H0, ombh2, omch2, ns, redshift, sigma8, default_As, Omega_c, Omega_b = (
        params["h"], params["H0"], params["ombh2"], params["omch2"],
        params["ns"], params["redshift"], params["sigma8"], params["As"], params["Omega_c"], params["Omega_b"]
    )

    if mode == 'sigma8':
        cosmo_A = ccl.Cosmology(Omega_c=Omega_c, Omega_b=Omega_b, h=h, n_s=ns,
                                sigma8=sigma8, transfer_function = "boltzmann_camb", m_nu=mnu, mass_split='normal')
        omega_m_A = cosmo_A['Omega_m']
        
        cosmo_B = ccl.Cosmology(Omega_c=Omega_c, Omega_b=Omega_b, h=h, n_s=ns,
                                sigma8=sigma8, transfer_function = "boltzmann_camb", m_nu=0, mass_split='normal')
        omega_m_B = cosmo_B['Omega_m']
        dm = omega_m_A - omega_m_B

        
        cosmo = ccl.Cosmology(Omega_c=Omega_c-dm, Omega_b=Omega_b, h=h, n_s=ns,
                               sigma8=sigma8, transfer_function = "boltzmann_camb", m_nu=mnu, mass_split='normal')
        
    elif mode == "As":
        cosmo_A = ccl.Cosmology(Omega_c=Omega_c, Omega_b=Omega_b, h=h, n_s=ns,\
                          A_s=default_As, transfer_function = "boltzmann_camb", m_nu=mnu, mass_split='normal')
        omega_m_A = cosmo_A['Omega_m']
        
        cosmo_B = ccl.Cosmology(Omega_c=Omega_c, Omega_b=Omega_b, h=h, n_s=ns,\
                          A_s=default_As, transfer_function = "boltzmann_camb", m_nu=0, mass_split='normal')
        omega_m_B = cosmo_B['Omega_m']
        dm = omega_m_A - omega_m_B
        
        cosmo = ccl.Cosmology(Omega_c=Omega_c-dm, Omega_b=Omega_b, h=h, n_s=ns,\
                          A_s=default_As, transfer_function = "boltzmann_camb", m_nu=mnu, mass_split='normal')


    wavenumber = np.linspace(0.0001, 1, 1000)
    scalefactor = 1/(1+redshift)

    pk_nlin1 = ccl.nonlin_matter_power(cosmo, wavenumber, scalefactor)
    
    return wavenumber/h, pk_nlin1*h**3
    