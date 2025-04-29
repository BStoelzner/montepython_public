import os
import numpy as np
import warnings
import montepython.io_mp as io_mp
from montepython.likelihood_class import Likelihood
import scipy.constants as conts
from scipy import interpolate as itp
from scipy.interpolate import RectBivariateSpline

class bao_eBOSS_DR16_Lya_auto(Likelihood):

    # initialization routine

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)

        # are there conflicting experiments?
        conflicting_experiments = [
            'BOSS_DR11_Lya_auto', 'BOSS_DR11_Lya_cross',
            'BOSS_DR12_Lya_auto', 'BOSS_DR12_Lya_combined',
            'BOSS_DR12_Lya_cross', 'eBOSS_DR14_Lya_auto',
            'eBOSS_DR14_Lya_combined', 'eBOSS_DR14_Lya_cross']
        for experiment in conflicting_experiments:
            if experiment in data.experiments:
                raise io_mp.LikelihoodError(
                    'bao_eBOSS_DR16_Lya_auto reports conflicting BAO measurments from: %s' %(experiment))

        # Read the datafile.
        print('Including eBOSS Lya-auto.')
        self.lya_data = np.loadtxt(os.path.join(self.data_directory, self.data_file))
        self.lya_DM = np.unique(self.lya_data[:, 0])
        self.lya_DH = np.unique(self.lya_data[:, 1])
        self.lya_lkl = np.reshape(self.lya_data[:, 2], [self.lya_DM.shape[0], self.lya_DH.shape[0]])
        # Create 3rd degree spline to interpolate later.
        self.lya_Interp = RectBivariateSpline(self.lya_DM, self.lya_DH, self.lya_lkl, kx=3, ky=3)
        # end of initialization

    # compute likelihood

    def loglkl(self, cosmo, data):
        loglkl = 0.0

        DM_at_z = cosmo.angular_distance(self.lya_dr16_z_eff) * (1. + self.lya_dr16_z_eff)
        H_at_z = cosmo.Hubble(self.lya_dr16_z_eff)
        DH_at_z = 1.0/H_at_z
        rd = cosmo.rs_drag() * self.rs_rescale
        # Compute the theoretical value of the observable
        theo_DM_at_z = DM_at_z / rd
        theo_DH_at_z_in_Mpc_inv = DH_at_z / rd

        # Interpolate the value within the spline and take the log.
        # Note that the data contains a normalized likelihood, which
        # is fine for MCMC, but is a problem for evidence calculations
        loglkl = np.log(float(self.lya_Interp(theo_DM_at_z, theo_DH_at_z_in_Mpc_inv)[0]))

        return loglkl
