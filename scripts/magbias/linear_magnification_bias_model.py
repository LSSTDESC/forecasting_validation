import numpy as np
from astropy.table import Table, vstack
import h5py as hpy

class LinearMagnificationBiasModel:
    def __init__(self, catalog_path, magstep=0.1, magdiff=1.0):
        """
        Initialize the LinearMagnificationBias class.

        Parameters:
        -----------
        catalog_path : str
            path to the catalog of galaxies
        magstep : float
            step size between limiting magnitudes for alpha calculation (see self.N)
        magdiff : float
            difference between limiting magnitude and begining of "faint-end" (see self.N)
        """
        
        f = hpy.File(catalog_path)
        df = f['df']['table'] ; dflen = df.len()
        data_all = df.fields('values_block_0')[:dflen]
        columns = ['spec_z', 'photo_z', 'i_mag', 'mag_rest']
        t = Table(names=columns, data=data_all)
        self.full_cat = t.group_by('photo_z')
        
        self.magstep = magstep
        self.magdiff = magdiff
    
    def make_tomobin_sample(self, binmin, binmax):
        """
        Creates a sample of galaxies with mean redshifts between a given minimum and maximum. 
        
        Parameters:
        -----------
        binmax : float
            maximum mean redshift
        binmin : float
            minimum mean redshift
            
        Returns:
        --------
        Galaxies that belong in the tomographic binning as specified. 
        """
        
        samples = []
        for group in self.full_cat.groups:
            photz = group['photo_z'][0]
            if photz > binmax: break
            elif photz > binmin: samples += [group]
            
        return vstack(samples)
    
    def N(self, sample, maglim):
        """
        Calculates number density of galaxies as a function of i-band limiting magnitude.
        
        Parameters:
        -----------
        sample: astropy.table
            sample of galaxies in given tomographic bin
        maglims: np.ndarray
            numpy array of the limiting magnitudes over which to calculate N (x-axis)
            
        Returns:
        --------
        N : np.ndarray
            array of cumulative number densities
        mags : np.ndarray
            array of corresponding limiting magnitudes for N
        """
        
        limiting_magnitudes = np.arange(maglim-self.magdiff, maglim, self.magstep)
        
        N = []
        for maglim in limiting_magnitudes: 
            N_maglim = np.sum(sample['i_mag'].data < maglim)
            N += [N_maglim]
            
        return np.array(N), limiting_magnitudes
    
    def alpha(self, N, maglims):
        """
        Calculates the average slope of a sample from its faint-end N(<i_mag).
        
        Parameters:
        -----------
        N : np.ndarray
            array of cumulative number densities
        mags : np.ndarray
            array of corresponding limiting magnitudes for N
            
        Returns:
        --------
        alpha : float
            faint end slope of a sample of galaxies
        """
        
        # Fit a linear model to the log of the cumulative counts
        p, cov = np.polyfit(maglims,
                            np.log10(N),
                            1,
                            cov=True)
        alpha_fit = 2.5 * p[0]
        errors = 2.5 * np.sqrt(cov[0][0])
        
        return alpha_fit, errors
        
    def sample_i_magbias(self, samplei, maglim):
        """
        Calculates the magnification bias for a sample of galaxies.
        
        Parameters:
        -----------
        samplei : astropy.table
            sample of galaxies (e.g. galaxies within a certain tomographic bin)
        maglims : tuple
            lower and upper magnitude limits for the magnification bias calculation
            
        Returns:
        --------
        alpha : float
            value for the magnification bias for the given galaxy sample
        alpha_err : float
            error of the calculated value for the magnification bias for the given galaxy sample
        """
        N_mags, mags = self.N(samplei, maglim)
        return self.alpha(N_mags, mags)
    
     
    def magbias(self, tomo_binnings, limiting_magnitude=25.5):
        """
        Calculates the magnification bias given tomographic binnings and limiting magnitudes. 
        
        Parameters:
        -----------
        tomo_binnings : np.ndarray
            the bin edges for the tomogrpahic bins (must start with 0)
        limiting_magnitudes : tuple
            the upper and lower magnitude limits to use when calculating alpha
        """
        
        samples = [self.make_tomobin_sample(t0, t1) for t0,t1 in zip(tomo_binnings, tomo_binnings[1:])]
        alphas = [self.sample_i_magbias(sampi, limiting_magnitude) for sampi in samples]
        
        return alphas
    
    
    