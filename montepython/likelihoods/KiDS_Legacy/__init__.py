######################################################################################################################
# Likelihood for KiDS-Legacy COSEBIs                                                                                 #
######################################################################################################################
#                                                                                                                    #
# Based on the fiducial KiDS-Legacy cosmic shear analysis pipeline by Wright et al. 2025 (arXiv:2503.19441)          #
# and earlier KiDS MontePython likelihoods.                                                                         #
# Written by Benjamin Stoelzner.                                                                                     #
#                                                                                                                    #
# This likelihood should reproduce the results from the fiducial CosmoSIS pipeline. It uses COSEBIs as summary       #
# statistic, the fiducial mass-dependent intrinsic alignment model, and the fiducial scale cuts (2'<theta<300').     #
# The code for calculating COSEBIs from shear Cls was adopted from:                                                  #
# https://github.com/KiDS-WL/kcap/blob/v2/utils/bandpower_cosebis.py                                                 #
#                                                                                                                    #
# If you are interested in using KiDS-Legacy data products other than COSEBIs:                                       #
# See the data products on the KiDS-Website: https://kids.strw.leidenuniv.nl/sciencedata.php                         #
# These includes chains, data files, and CosmoSIS ini files for COSEBIs, band powers, and 2PCFs,                     #
# including a joint analysis with external probes and featuring a pre-trained cosmopower emulator                    #
# and the scale-cuts module for selecting a different number of COSEBI modes, band power bins, and 2PCF bins         #
#                                                                                                                    #
# To run the full KiDS-Legacy analysis pipeline see: https://github.com/AngusWright/CosmoPipe                        #
#                                                                                                                    #
# Please cite the following papers when KiDS-Legacy data:                                                            #
#   - Wright et al. 2025 (A&A, 686, A170)    [KiDS DR5 data release paper]                                           #
#   - Wright et al. 2025 (A&A, 703, A158)    [KiDS-Legacy cosmic shear analysis]                                     #
#   - Stölzner et al. 2025 (A&A, 702, A169)  [KiDS-Legacy consistency and joint constraints with external probes]    #
#   - Reischke et al. 2025 (A&A, 699, A124)  [KiDS-Legacy covariance]                                                #
#   - Wright et al. 2025 (A&A, 703, A144)    [KiDS Legacy redshift calibration]                                      #
# and include the following acknowledgement in your paper:                                                           #
#   "Based on observations made with ESO Telescopes at the La Silla Paranal Observatory under programme              #
#    IDs 179.A-2004, 177.A-3016, 177.A-3017, 177.A-3018, 298.A-5015."                                                #
#                                                                                                                    #
# See also our follow-up analyses:                                                                                   #
#   - Reischke et al 2025 (arXiv: 2512.11041) [Constraints on dark energy, neutrino mass, and curvature]             #
#   - Stölzner et al 2025 (arXiv: 2512.11039) [Constraints on Horndeski gravity]                                     #
#                                                                                                                    #
# ATTENTION:                                                                                                         #   
# This likelihood only produces valid results for \Omega_k = 0,                                                      #
# i.e. flat cosmologies!                                                                                             #
######################################################################################################################

from montepython.likelihood_class import Likelihood

import os
import sys
import numpy as np
from scipy import interpolate as itp
from scipy import integrate as itg
from scipy.linalg import cholesky, solve_triangular
from astropy.io import fits

class KiDS_Legacy(Likelihood):

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)

        # Force the cosmological module to store Pk for redshifts up to
        # max(self.z) and for k up to k_max
        self.need_cosmo_arguments(data, {'output': 'mPk'})
        self.need_cosmo_arguments(data, {'P_k_max_h/Mpc': self.k_max_h_by_Mpc})
        self.need_cosmo_arguments(data, {'nonlinear_min_k_max': self.nonlinear_min_k_max})

        ## Compute non-linear power spectrum if requested
        # it seems like HMcode needs the full argument to work...
        if self.method_non_linear_Pk in ['hmcode', 'Hmcode', 'HMcode', 'HMCODE']:
            self.need_cosmo_arguments(data, {'non linear': self.method_non_linear_Pk})
            print('Using {:} 2020 to obtain the non-linear P(k, z)! \n'.format(self.method_non_linear_Pk))
            self.need_cosmo_arguments(data, {'hmcode_version': '2020_baryonic_feedback'})
        else:
            print('Only using the linear P(k, z) for ALL calculations \n (check keywords for "method_non_linear_Pk"). \n')

        # set up array of ells for Cl integrations:
        self.ells = np.logspace(np.log10(self.ell_min), np.log10(self.ell_max), self.nells)

        # Calculate number of tomographic bin combinations
        self.nzcorrs = int(self.nzbins * (self.nzbins + 1) / 2)

        # Read data
        # Data vector format: 1-1, 1-2, ..., 1-6, 2-2, ..., 6-6
        self.cosebis_data, self.cosebis_cov, self.cosebis_inv_cov, self.z_samples, self.hist_samples = self.load_data_file()
        self.cholesky_transform = cholesky(self.cosebis_cov, lower=True)

        self.WnLog = []
        # Read WnLog window functions (assuming that the data directory contains the weight functions up to the chosen nmaxcosebis for the given theta range and that log_ell is the same in all files!)
        for i in range(self.nmaxcosebis):
            log_ell, Wn = np.loadtxt(os.path.join(self.data_directory,'WnLog/WnLog%d-%.2f-%.2f.table'%(i+1,self.theta_min, self.theta_max))).T
            ell = np.exp(log_ell)
            # theory prediction requires ell*Wn/2/pi
            self.WnLog.append(ell*Wn/(2*np.pi))
        self.WnLog = np.array(self.WnLog)
        self.ell_window = ell

        # Read covariance matrix of IA parameters in the massdep model
        if self.IA_model == 'massdep':
            self.massdep_means = np.loadtxt(os.path.join(self.data_directory, self.filename_massdep_means))
            self.massdep_cov = np.loadtxt(os.path.join(self.data_directory, self.filename_massdep_cov))
            self.massdep_cholesky = np.linalg.cholesky(self.massdep_cov)
            self.M_piv = 10 ** self.log10_M_piv

        # Check if any of the n(z) needs to be shifted in loglkl by D_z{1...n}:
        self.shift_n_z_by_D_z = np.zeros(self.nzbins, 'bool')
        for zbin in range(self.nzbins):
            param_name = 'D_z{:}'.format(zbin + 1)
            if param_name in data.mcmc_parameters:
                self.shift_n_z_by_D_z[zbin] = True

        if self.shift_n_z_by_D_z.any():
            # load the correlation matrix of the D_z shifts:
            try:
                fname = os.path.join(self.data_directory, self.filename_corrmat_D_z)
                corrmat_D_z = np.loadtxt(fname)
                print('Loaded correlation matrix for D_z shifts from: \n {:} \n'.format(fname))
                self.L_matrix_D_z = np.linalg.cholesky(corrmat_D_z)
            except:
                print('Could not load correlation matrix of D_z shifts, hence treating them as independent! \n')
                self.L_matrix_D_z = np.eye(self.nzbins)
 
        # prevent undersampling of histograms!
        if self.nzmax < len(self.z_samples) - 1:
            print("You're trying to integrate at lower resolution than supplied by the n(z) histograms. \n Increase nzmax>={:}! Aborting now...".format(len(self.z_samples) - 1))
            exit()
        # if that's the case, we want to integrate at histogram resolution and need to account for
        # the extra zero entry added
        elif self.nzmax == len(self.z_samples) - 1:
            self.nzmax = self.z_samples.shape[0]
            # requires that z-spacing is always the same for all bins...
            self.z_p = self.z_samples
            print('Integrations performed at resolution of histogram! \n')
        # if we interpolate anyway at arbitrary resolution the extra 0 doesn't matter
        else:
            self.nzmax += 1
            self.z_p = np.linspace(0.0, self.z_samples.max(), self.nzmax)
            print('Integration performed at set nzmax={:} resolution! \n'.format(self.nzmax - 1))
        if self.z_p[0] == 0:
            self.z_p[0] = 1e-4
        self.pz = np.zeros((self.nzmax, self.nzbins))
        self.pz_norm = np.zeros(self.nzbins, 'float64')
        self.splines_pz = []

        for zbin in range(self.nzbins):
            # we assume that the z-spacing is the same for each histogram
            spline_pz = itp.interp1d(self.z_samples, self.hist_samples[zbin, :], kind=self.type_redshift_interp, fill_value='extrapolate')
            self.splines_pz.append(spline_pz)
            mask_min = self.z_p >= self.z_samples.min()
            mask_max = self.z_p <= self.z_samples.max()
            mask = mask_min & mask_max
            # points outside the z-range of the histograms are set to 0!
            self.pz[mask, zbin] = spline_pz(self.z_p[mask])
            # Normalize selection functions
            dz = self.z_p[1:] - self.z_p[:-1]
            self.pz_norm[zbin] = np.sum(0.5 * (self.pz[1:, zbin] + self.pz[:-1, zbin]) * dz)
            
        self.zmax = self.z_p.max()
        self.need_cosmo_arguments(data, {'z_max_pk': self.zmax})

        return

    def loglkl(self, cosmo, data):
        
        cosebis_theory = self.cosmo_calculations(cosmo, data)

        if self.write_out_theory:
            fname = os.path.join(self.data_directory, self.theory_file)
            # for now we just dump the theory vector with no further details,
            # ASCII style...
            np.savetxt(fname, cosebis_theory)
            print('Saved theory vector to: \n {:} \n'.format(fname))
            print('Aborting run now. \n Set flag "write_out_theory = False" for likelihood evaluations \n and double-check your data vector!')
            exit()

        diff_vec = self.cosebis_data - cosebis_theory

        # this is for running smoothly with MultiNest
        # (in initial checking of prior space, there might occur weird solutions)
        if np.isinf(diff_vec).any() or np.isnan(diff_vec).any():
            chi2 = 2e12
        else:
            # don't invert that matrix...
            # use the Cholesky decomposition instead:
            yt = solve_triangular(self.cholesky_transform, diff_vec, lower=True)
            chi2 = yt.dot(yt)

        # enforce Gaussian priors on NUISANCE parameters if requested:
        if self.use_gaussian_prior_for_nuisance:
            for idx_nuisance, nuisance_name in enumerate(self.gaussian_prior_name):
                scale = data.mcmc_parameters[nuisance_name]['scale']
                chi2 += (data.mcmc_parameters[nuisance_name]['current'] * scale - self.gaussian_prior_center[idx_nuisance])**2 / self.gaussian_prior_sigma[idx_nuisance]**2

        return -0.5 * chi2

    def load_data_file(self):
        """
        Function to load and process the data FITS file
        """

        with fits.open(os.path.join(self.data_directory, self.data_file)) as f:
            # Read COSEBIS data, mode, covariance
            cosebis_data = f['En'].data['VALUE']
            cosebis_mode = f['En'].data['ANGBIN']
            cosebis_cov = f['COVMAT'].data
            # Check if dimensionalities match
            assert 2*cosebis_data.shape[0] == cosebis_cov.shape[0], 'Error: dimensionalities of data and covariance matrix do not match!'
            # Only keep E mode covariance
            cosebis_cov = cosebis_cov[:cosebis_data.shape[0],:][:,:cosebis_data.shape[0]]
            cosebis_inv_cov = np.linalg.inv(cosebis_cov)
            # If mask data if required
            if self.nmaxcosebis < np.max(cosebis_mode):
                mask = cosebis_mode <= self.nmaxcosebis
                cosebis_data = cosebis_data[mask]
                cosebis_cov = cosebis_cov[mask,:][:,mask]
            elif self.nmaxcosebis > np.max(cosebis_mode):
                raise Exception('Error: data file only contains %d COSEBIs modes!'%(np.max(cosebis_mode)))

            # Read redshift distributions
            z_mid = f['NZ_SOURCE'].data['Z_MID']
            nz = len(z_mid)
            z_hist = np.zeros((self.nzbins,nz))
            for i in range(self.nzbins):
                z_hist[i] = f['NZ_SOURCE'].data['BIN%d'%(i+1)]

        return(cosebis_data, cosebis_cov, cosebis_inv_cov, z_mid, z_hist)


    def get_IA_factor(self, z, linear_growth_rate, rho_crit, Omega_m, small_h, amplitude, exponent):

        const = 5e-14 / small_h**2 # Mpc^3 / M_sol

        # arbitrary convention
        z0 = 0.3
        factor = -1. * amplitude * const * rho_crit * Omega_m / linear_growth_rate * ((1. + z) / (1. + z0))**exponent

        return factor

    def get_critical_density(self, small_h):
        """
        The critical density of the Universe at redshift 0.

        Returns
        -------
        rho_crit in solar masses per cubic Megaparsec.

        """

        # yay, constants...
        Mpc_cm = 3.08568025e24 # cm
        M_sun_g = 1.98892e33 # g
        G_const_Mpc_Msun_s = M_sun_g * (6.673e-8) / Mpc_cm**3.
        H100_s = 100. / (Mpc_cm * 1.0e-5) # s^-1

        rho_crit_0 = 3. * (small_h * H100_s)**2. / (8. * np.pi * G_const_Mpc_Msun_s)

        return rho_crit_0

    def get_matter_power_spectrum(self, r, z, cosmo, data):

        # Get power spectrum P(k=l/r,z(r)) from cosmological module
        # Linear extrapolation in log-space of P(k) for k>k_max
        pk = np.zeros((self.nells, self.nzmax), 'float64')
        pk_lin = np.zeros((self.nells, self.nzmax), 'float64')
        k_z = np.zeros((self.nells, self.nzmax), 'float64')

        k_max_in_inv_Mpc = self.k_max_h_by_Mpc * cosmo.h()

        for idx_z in range(self.nzmax):
            all_k_in_inv_Mpc = (self.ells + 0.5) / r[idx_z]
            # For k values larger than k_max_in_inv_Mpc we use an interpolation of the matter power spectrum to larger values
            idx_larger_k_max_in_inv_Mpc = all_k_in_inv_Mpc>k_max_in_inv_Mpc
            if any(idx_larger_k_max_in_inv_Mpc):
                itp_start = np.where(idx_larger_k_max_in_inv_Mpc)[0][0]
                itp_indices = np.arange(itp_start-3,itp_start)
                p_dm = np.polyfit(np.log(all_k_in_inv_Mpc[itp_indices]), [np.log(cosmo.pk(all_k_in_inv_Mpc[i], z[idx_z])) for i in itp_indices], 1)
                p_lin_dm = np.polyfit(np.log(all_k_in_inv_Mpc[itp_indices]), [np.log(cosmo.pk_lin(all_k_in_inv_Mpc[i], z[idx_z])) for i in itp_indices], 1)
            for idx_ell in range(self.nells):
                # standard Limber approximation:
                #k = ells[idx_ell] / r[idx_z]
                # extended Limber approximation (cf. LoVerde & Afshordi 2008):
                k_in_inv_Mpc = (self.ells[idx_ell] + 0.5) / r[idx_z]
                if k_in_inv_Mpc > k_max_in_inv_Mpc:
                    pk_dm = np.exp(np.polyval(p_dm, np.log(k_in_inv_Mpc)))
                    pk_lin_dm = np.exp(np.polyval(p_lin_dm, np.log(k_in_inv_Mpc)))
                else:
                    pk_dm = cosmo.pk(k_in_inv_Mpc, z[idx_z])
                    pk_lin_dm = cosmo.pk_lin(k_in_inv_Mpc, z[idx_z])

                pk[idx_ell, idx_z] = pk_dm
                pk_lin[idx_ell, idx_z] = pk_lin_dm
                k_z[idx_ell, idx_z] = k_in_inv_Mpc

        return pk, pk_lin

    def get_lensing_kernel(self, r, pr):
        """
        Compute function g_i(r), that depends on r and the bin
        g_i(r) = 2r(1+z(r)) int_r^+\infty drs p_r(rs) (rs-r)/rs
        """

        g = np.zeros((self.nzmax, self.nzbins), 'float64')
        for Bin in range(self.nzbins):
            # shift only necessary if z[0] = 0
            for nr in range(1, self.nzmax - 1):
            #for nr in range(self.nzmax - 1):
                fun = pr[nr:, Bin] * (r[nr:] - r[nr]) / r[nr:]
                g[nr, Bin] = np.sum(0.5*(fun[1:] + fun[:-1]) * (r[nr+1:] - r[nr:-1]))
                g[nr, Bin] *= 2. * r[nr] * (1. + self.z_p[nr])

        return g

    def get_shear_power_spectrum(self, cosmo, data):
        """
        Function to calculate angular shear-shear power spectra, Cls.
        """

        # Omega_m contains all species!
        Omega_m = cosmo.Omega_m()
        small_h = cosmo.h()

        # needed for IA modelling:
        if self.IA_model == 'NLA':
            if ('A_IA' in data.mcmc_parameters) and ('exp_IA' in data.mcmc_parameters):
                amp_IA = data.mcmc_parameters['A_IA']['current'] * data.mcmc_parameters['A_IA']['scale']
                exp_IA = data.mcmc_parameters['exp_IA']['current'] * data.mcmc_parameters['exp_IA']['scale']
            elif ('A_IA' in data.mcmc_parameters) and ('exp_IA' not in data.mcmc_parameters):
                amp_IA = data.mcmc_parameters['A_IA']['current'] * data.mcmc_parameters['A_IA']['scale']
                # redshift-scaling is turned off:
                exp_IA = 0.
            intrinsic_alignment = True
        elif self.IA_model == 'massdep':
            amp_IA_uncorr = data.mcmc_parameters['A_IA']['current'] * data.mcmc_parameters['A_IA']['scale']
            beta_uncorr = data.mcmc_parameters['beta']['current'] * data.mcmc_parameters['beta']['scale']
            # Build a vector of IA nuisance parameters: [A_IA, beta, log10_M_mean_{1-6}]
            IA_parameters_uncorr = np.concatenate(([amp_IA_uncorr, beta_uncorr],[data.mcmc_parameters['log10M_mean_%d'%i]['current'] * data.mcmc_parameters['log10M_mean_%d'%i]['scale'] for i in range(1,self.nzbins+1)]))
            # Calculate correlated IA parameters
            IA_parameters = self.massdep_cholesky.dot(IA_parameters_uncorr)
            amp_IA = IA_parameters[0]
            beta = IA_parameters[1]
            M_mean = [10 ** IA_parameters[2+i] for i in range(self.nzbins)]
            # Do not use any additional redshift scaling
            exp_IA = 0
            intrinsic_alignment = True
        else:
            intrinsic_alignment = False
    
        # One wants to obtain here the relation between z and r, this is done
        # by asking the cosmological module with the function z_of_r
        r, dzdr = cosmo.z_of_r(self.z_p)

        # Compute now the selection function p(r) = p(z) dz/dr normalized
        # to one. The np.newaxis helps to broadcast the one-dimensional array
        # dzdr to the proper shape. Note that p_norm is also broadcasted as
        # an array of the same shape as p_z
        if (self.shift_n_z_by_D_z.any()):

            # correlate D_z shifts:
            D_z = np.zeros(self.nzbins)
            for zbin in range(self.nzbins):

                param_name = 'D_z{:}'.format(zbin + 1)
                if param_name in data.mcmc_parameters:
                    D_z[zbin] = data.mcmc_parameters[param_name]['current'] * data.mcmc_parameters[param_name]['scale']
        
            D_z_corr = self.L_matrix_D_z.dot(D_z)
            
            pz = np.zeros((self.nzmax, self.nzbins), 'float64')
            pz_norm = np.zeros(self.nzbins, 'float64')
            for zbin in range(self.nzbins):
                # Changed sign w.r.t. KiDS-1000 likelihood to make it consistent with cosmosis pipeline
                z_mod = self.z_p - D_z_corr[zbin]
                spline_pz = self.splines_pz[zbin]
                # check for z<0
                mask_min = z_mod >= 0
                mask_max = z_mod <= self.z_samples.max()
                mask = mask_min & mask_max
                # points outside the z-range of the histograms are set to 0!
                pz[mask, zbin] = spline_pz(z_mod[mask])
                # Normalize selection functions
                dz = self.z_p[1:] - self.z_p[:-1]
                pz_norm[zbin] = np.sum(0.5 * (pz[1:, zbin] + pz[:-1, zbin]) * dz)

            pr = pz * (dzdr[:, np.newaxis] / pz_norm)

        else:
            # use fiducial dn/dz loaded in the __init__:
            pr = self.pz * (dzdr[:, np.newaxis] / self.pz_norm)

        # get linear growth rate if IA are modelled:
        if intrinsic_alignment:
            rho_crit = self.get_critical_density(small_h)
            # derive the linear growth factor D(z)
            linear_growth_rate = np.zeros_like(self.z_p)
            #print self.redshifts
            for idx_z, z in enumerate(self.z_p):
                linear_growth_rate[idx_z] = cosmo.scale_independent_growth_factor(z)
            # normalize to unity at z=0:
            linear_growth_rate /= cosmo.scale_independent_growth_factor(0.)

        g = self.get_lensing_kernel(r, pr)
        pk, pk_lin = self.get_matter_power_spectrum(r, self.z_p, cosmo, data)
        Cl_integrand = np.zeros((self.nzmax, self.nzcorrs), 'float64')
        Cl = np.zeros((self.nzcorrs, self.nells), 'float64')

        Cl_GG_integrand = np.zeros_like(Cl_integrand)
        Cl_GG = np.zeros_like(Cl)

        if intrinsic_alignment:
            Cl_II_integrand = np.zeros_like(Cl_integrand)
            Cl_II = np.zeros_like(Cl)

            Cl_GI_integrand = np.zeros_like(Cl_integrand)
            Cl_GI = np.zeros_like(Cl)

        list_cl_keys = []
        dr = r[1:] - r[:-1]
        # Start loop over l for computation of C_l^shear
        for il in range(self.nells):
            # find Cl_integrand = (g(r) / r)**2 * P(l/r,z(r))
            for Bin1 in range(self.nzbins):
                for Bin2 in range(Bin1, self.nzbins):
                    if il == 0:
                        list_cl_keys += ['bin_{:}_{:}'.format(Bin2 + 1, Bin1 + 1)]
                    Cl_GG_integrand[1:, self.__one_dim_index(Bin1,Bin2)] = g[1:, Bin1] * g[1:, Bin2] / r[1:]**2 * pk[il, 1:]
                    if intrinsic_alignment:
                        if self.IA_model == 'NLA':
                            factor_IA = self.get_IA_factor(self.z_p, linear_growth_rate, rho_crit, Omega_m, small_h, amp_IA, exp_IA) 
                            Cl_II_integrand[1:, self.__one_dim_index(Bin1, Bin2)] = pr[1:, Bin1] * pr[1:, Bin2] * factor_IA[1:]**2 / r[1:]**2 * pk[il, 1:]
                            Cl_GI_integrand[1:, self.__one_dim_index(Bin1, Bin2)] = (g[1:, Bin1] * pr[1:, Bin2] + g[1:, Bin2] * pr[1:, Bin1]) * factor_IA[1:] / r[1:]**2 * pk[il, 1:]
                        elif self.IA_model == 'massdep':
                            factor_IA = self.get_IA_factor(self.z_p, linear_growth_rate, rho_crit, Omega_m, small_h, amp_IA, exp_IA) 
                            coef_1 = self.f_r[Bin1] * np.power((M_mean[Bin1] / self.M_piv), beta)
                            coef_2 = self.f_r[Bin2] * np.power((M_mean[Bin2] / self.M_piv), beta)
                            Cl_II_integrand[1:, self.__one_dim_index(Bin1, Bin2)] = coef_1 * coef_2 * pr[1:, Bin1] * pr[1:, Bin2] * factor_IA[1:]**2 / r[1:]**2 * pk[il, 1:]
                            if Bin1 != Bin2:
                                Cl_GI_integrand[1:, self.__one_dim_index(Bin1, Bin2)] = (coef_2 * g[1:, Bin1] * pr[1:, Bin2] + coef_1 * g[1:, Bin2] * pr[1:, Bin1]) * factor_IA[1:] / r[1:]**2 * pk[il, 1:]
                            else:
                                Cl_GI_integrand[1:, self.__one_dim_index(Bin1, Bin2)] = (coef_2**2 * g[1:, Bin1] * pr[1:, Bin2] + coef_1**2 * g[1:, Bin2] * pr[1:, Bin1]) * factor_IA[1:] / r[1:]**2 * pk[il, 1:]

            # Integrate over r to get C_l^shear_ij = P_ij(l)
            # C_l^shear_ij = 9/16 Omega0_m^2 H_0^4 \sum_0^rmax dr (g_i(r)
            # g_j(r) /r**2) P(k=l/r,z(r)) dr
            # It is then multiplied by 9/16*Omega_m**2
            # and then by (h/2997.9)**4 to be dimensionless
            # (since P(k)*dr is in units of Mpc**4)
            for Bin in range(self.nzcorrs):
                Cl_GG[Bin, il] = np.sum(0.5 * (Cl_GG_integrand[1:, Bin] + Cl_GG_integrand[:-1, Bin]) * dr)
                Cl_GG[Bin, il] *= 9. / 16. * Omega_m**2
                Cl_GG[Bin, il] *= (small_h / 2997.9)**4

                if intrinsic_alignment:
                    Cl_II[Bin, il] = np.sum(0.5 * (Cl_II_integrand[1:, Bin] + Cl_II_integrand[:-1, Bin]) * dr)

                    Cl_GI[Bin, il] = np.sum(0.5 * (Cl_GI_integrand[1:, Bin] + Cl_GI_integrand[:-1, Bin]) * dr)
                    # here we divide by 4, because we get a 2 from g(r)!
                    Cl_GI[Bin, il] *= 3. / 4. * Omega_m
                    Cl_GI[Bin, il] *= (small_h / 2997.9)**2

        if intrinsic_alignment:
            Cl = Cl_GG + Cl_GI + Cl_II
        else:
            Cl = Cl_GG

        if self.write_out_Cls:
            Cls_out = self.ells
            fname = os.path.join(self.data_directory, 'Cls_tot.txt')
            header = 'ells, '
            for idx in range(self.nzcorrs):
                header += list_cl_keys[idx] + ', '
                Cls_out = np.column_stack((Cls_out, Cl[idx, :]))
            header = header[:-2]
            np.savetxt(fname, Cls_out, header=header)
            print('Saved Cls to: \n {:} \n'.format(fname))

        return Cl

    def cosmo_calculations(self, cosmo, data):

        Cls = self.get_shear_power_spectrum(cosmo, data)
        theory_vec = self.Cl_to_cosebis(Cls)
        
        return theory_vec

    def __one_dim_index(self, Bin1, Bin2):
        """
        This function is used to convert 2D sums over the two indices (Bin1, Bin2)
        of an N*N symmetric matrix into 1D sums over one index with N(N+1)/2
        possible values.
        """

        if Bin1 <= Bin2:
            return Bin2 + self.nzbins * Bin1 - (Bin1 * (Bin1 + 1)) // 2
        else:
            return Bin1 + self.nzbins * Bin2 - (Bin2 * (Bin2 + 1)) // 2

    def log_interpolate(self, x_arr, y_arr, x):
        """
        Adopted from: https://github.com/KiDS-WL/kcap/blob/v2/utils/bandpower_cosebis.py
        """
        log_x_arr = np.log(x_arr)
        if np.any(y_arr <= 0):
            intp = itp.InterpolatedUnivariateSpline(
                        log_x_arr, y_arr, ext=2)
        else:
            log_intp = itp.InterpolatedUnivariateSpline(
                            log_x_arr, np.log(y_arr), ext=2)

            def intp(log_x):
                return np.exp(log_intp(log_x))

        def low_extrap(log_x):
            dx = log_x_arr[1] - log_x_arr[0]
            dy = np.log(y_arr[1]) - np.log(y_arr[0])
            return y_arr[0] * np.exp(dy/dx * (log_x - log_x_arr[0]))

        def high_extrap(log_x):
            dx = log_x_arr[-1] - log_x_arr[-2]
            dy = np.log(y_arr[-1]) - np.log(y_arr[-2])
            return y_arr[-1] * np.exp(dy/dx * (log_x - log_x_arr[-1]))

        log_x = np.log(x)
        y = np.piecewise(log_x,
                        [log_x < log_x_arr[0],
                        (log_x >= log_x_arr[0]) & (log_x <= log_x_arr[-1])],
                        [low_extrap, intp, high_extrap])
        return y
    
    def Cl_to_cosebis(self,Cls):
        """
        Adopted from: https://github.com/KiDS-WL/kcap/blob/v2/utils/bandpower_cosebis.py
        """
        # number of bin combinations
        n_corrs = int(self.nzbins*(self.nzbins+1)/2)
        n_data = n_corrs * self.nmaxcosebis
        # Data vector format: 1-1, 1-2, ..., 1-6, 2-2, ..., 6-6
        # Shear Cls follow the same bin order
        cosebis_theory = np.zeros(n_data)
        for i in range(n_corrs):
            Cl_EE = np.zeros(len(self.ell_window))
            m = np.ones(len(self.ell_window), dtype=bool)
            m &= (self.ell_window >= self.ells.min())
            m &= (self.ell_window <= self.ells.max())
            Cl_EE[m] = self.log_interpolate(self.ells, Cls[i], self.ell_window[m])
            EE = itg.simpson(self.WnLog * Cl_EE, self.ell_window, axis=1)
            cosebis_theory[i*self.nmaxcosebis:(i+1)*self.nmaxcosebis] = EE
            
        return(cosebis_theory)
        
        
