"""
.. module:: Pantheon_Plus
    :synopsis: Pantheon_Plus likelihood from   Pantheon+ arXiv:2202.04077

.. moduleauthor:: Vivian Poulin <vivian.poulin@umontpellier.fr>, with help from Dillon Brout and Dan Scolnic

Based on the previous Pantheon lkl from Rodrigo von Marttens and Antonella Cid, which was based on JLA likelihood writted by Benjamin Audren

.. code::

    C00 = mag_covmat_file

.. note::

    Since there are a lot of file manipulation involved, the "pandas" library
    has to be installed -- it is an 8-fold improvement in speed over numpy, and
    a 2-fold improvement over a fast Python implementation. The "numexpr"
    library is also needed for doing the fast array manipulations, done with
    blas daxpy function in the original c++ code. Both can be installed with
    pip (Python package manager) easily.

"""
import numpy as np
import scipy.linalg as la
import montepython.io_mp as io_mp
try:
    import numexpr as ne
except ImportError:
    raise io_mp.MissingLibraryError(
        "This likelihood has intensive array manipulations. You "
        "have to install the numexpr Python package. Please type:\n"
        "(sudo) pip install numexpr --user")
from montepython.likelihood_class import Likelihood_sn


class Pantheon_Plus(Likelihood_sn):

    def __init__(self, path, data, command_line):

	#Read the data and covariance matrix
	##For Pantheon+ alone we have to remove the very low-z (z<0.01) SN1a that are used as SH0ES calibrators.
	##This requires manipulating the covariance matrix.
        try:
            Likelihood_sn.__init__(self, path, data, command_line)
        except IOError:
            raise io_mp.LikelihoodError(
                "The Pantheon_Plus data files were not found. Please check if "
                "the following files are in the data/Pantheon_Plus directory: "
                "\n-> Pantheon_Plus.dataset"
                "\n-> lcparam_full_long.txt"
                "\n-> sys_full_long.dat")

        # are there conflicting experiments?
        conflicting_experiments = [
            'Pantheon', 'Pantheon_Plus_SH0ES',
            'hst', 'sh0es']
        for experiment in conflicting_experiments:
            if experiment in data.experiments:
                raise io_mp.LikelihoodError(
                    'Pantheon_Plus reports conflicting SN or H0 measurments from: %s' %(experiment))

        # Load matrices from text files, whose names were read in the
        # configuration file
        self.C00 = self.read_matrix(self.mag_covmat_file)
        # Reading light-curve parameters from self.data_file (Pantheon+SH0ES.dat)
        self.light_curve_params = self.read_light_curve_parameters()


        # Reordering by J. Renk. The following steps can be computed in the
        # initialisation step as they do not depend on the point in parameter-space
        #   -> likelihood evaluation is 30% faster

        # Compute the covariance matrix
        # The module numexpr is used for doing quickly the long multiplication
        # of arrays (factor of 3 improvements over numpy). It is used as a
        # replacement of blas routines cblas_dcopy and cblas_daxpy
        # For numexpr to work, we need (seems like a bug, but anyway) to create
        # local variables holding the arrays. This cost no time (it is a simple
        # pointer assignment)
        C00 = self.C00
        covm = ne.evaluate("C00")

	#VP: This routine isolates the part of the data that are at z>self.z_min = 0.01 by default
	#The data are ordered by increasing z, which simplifies things.
        sn = self.light_curve_params
        true_size=0
        ignored = 0
        for ii in range(len(self.light_curve_params.zHD)):
                if self.light_curve_params.zHD[ii]>self.z_min:
                	true_size+=1
                else:
                	ignored+=1
        #print(true_size,ignored)
        self.true_size = true_size
        newcovm = np.zeros((true_size,true_size), 'float64')
        newcovm=covm[ignored:,ignored:]


        # Update the diagonal terms of the covariance matrix with the
        # statistical error
	## VP: statistical errors are already included! we ignore this step, leave it for comparison with former lkl.
#        covm += np.diag(sn.m_b_corr_err**2)

        # Whiten the residuals, in two steps.
        # Step 1) Compute the Cholesky decomposition of the covariance matrix, in
        # place. This is a time expensive (0.015 seconds) part, which is why it is
        # now done in init. Note that this is different to JLA, where it needed to
        # be done inside the loglkl function.
        self.cov = la.cholesky(newcovm, lower=True, overwrite_a=True)
        # Step 2) depends on point in parameter space -> done in loglkl calculation


    def loglkl(self, cosmo, data):
        """
        Compute negative log-likelihood (eq.15 Betoule et al. 2014)

        """
        # Recover the distance moduli from CLASS (a size N vector of double
        # containing the predicted distance modulus for each SN in the JLA
        # sample, given the redshift of the supernova.)


        redshifts = self.light_curve_params.zHD
        size = redshifts.size

	#VP: true size is the number of data points after removing those with z < self.z_min
        moduli = np.empty((self.true_size, ))
        Mb_obs = np.empty((self.true_size, ))
        good_z = 0

        for index, row in self.light_curve_params.iterrows():
            z_cmb = row['zHD']
            z_hel = row['zHEL']
            Mb_corr = row['m_b_corr']
	    #this condition allows to extract the data with "good z", i.e. z>z_min
            if z_cmb > self.z_min:
            	moduli[good_z] = 5 * np.log10((1+z_cmb)*(1+z_hel)*cosmo.angular_distance(z_cmb)) + 25
            	Mb_obs[good_z] = Mb_corr
            	good_z+=1

        # Convenience variables: store the nuisance parameters in short named
        # variables
        M = (data.mcmc_parameters['M']['current'] *
             data.mcmc_parameters['M']['scale'])

        # Compute the residuals (estimate of distance moduli - exact moduli)
        residuals = np.empty((self.true_size,))
        sn = self.light_curve_params
        # This operation loops over all supernovae!
        # Compute the approximate moduli
        residuals = Mb_obs - M

        # Remove from the approximate moduli the one computed from CLASS
        residuals -= moduli

        # Step 2) (Step 1 is done in the init) Solve the triangular system, also time expensive (0.02 seconds)
        residuals = la.solve_triangular(self.cov, residuals, lower=True, check_finite=False)

        # Finally, compute the chi2 as the sum of the squared residuals
        chi2 = (residuals**2).sum()

        return -0.5 * chi2
