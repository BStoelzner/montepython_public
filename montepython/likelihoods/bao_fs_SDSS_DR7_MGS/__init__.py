import os
import numpy as np
import warnings
import montepython.io_mp as io_mp
from montepython.likelihood_class import Likelihood
import scipy.constants as conts

class bao_fs_SDSS_DR7_MGS(Likelihood):

    # initialization routine

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)

        # needed arguments in order to get sigma_8(z) up to z=1 with correct precision
        self.need_cosmo_arguments(data, {'output': 'mPk'})
        self.need_cosmo_arguments(data, {'P_k_max_h/Mpc': 1.})
        self.need_cosmo_arguments(data, {'z_max_pk': 1.})

        # are there conflicting experiments?
        conflicting_experiments = [
            'bao_boss', 'bao_smallz_2014']
        for experiment in conflicting_experiments:
            if experiment in data.experiments:
                raise io_mp.LikelihoodError(
                    'bao_fs_SDSS_DR7_MGS reports conflicting BAO measurments from: %s' %(experiment))

        # define arrays for values of z and data points
        self.z = np.array([], 'float64')
        self.DV_by_rd = np.array([], 'float64')
        self.fsig8 = np.array([], 'float64')

        # read redshifts and data points
        with open(os.path.join(self.data_directory, self.data_file), 'r') as filein:
            for i, line in enumerate(filein):
                if line.strip() and line.find('#') == -1:
                    this_line = line.split()
                    # load redshifts and D_V / r_s
                    if this_line[2] == 'DV_over_rs':
                        self.z = np.append(self.z, float(this_line[0]))
                        self.DV_by_rd = np.append(
                            self.DV_by_rd, float(this_line[1]))
                    # load f * sigma8
                    elif this_line[2] == 'f_sigma8':
                        self.fsig8 = np.append(self.fsig8, float(this_line[1]))

        # read covariance matrix
        self.cov_data = np.loadtxt(os.path.join(self.data_directory, self.cov_file))

        # number of bins
        self.num_bins = np.shape(self.z)[0]

        # number of data points
        self.num_points = np.shape(self.cov_data)[0]

        # end of initialization

    # compute likelihood

    def loglkl(self, cosmo, data):

        # define array for  values of D_M_diff = D_M^th - D_M^obs and H_diff = H^th - H^obs,
        # ordered by redshift bin (z=[0.38, 0.51, 0.61]) as following:
        # data_array = [DM_diff(z=0.38), H_diff(z=0.38), DM_diff(z=0.51), .., .., ..]
        data_array = np.array([], 'float64')

        # for each point, compute comoving angular diameter distance D_M = (1 + z) * D_A,
        # sound horizon at baryon drag rs_d, theoretical prediction
        for i in range(self.num_bins):
            da = cosmo.angular_distance(self.z[i])
            dr = self.z[i] / cosmo.Hubble(self.z[i])
            dv = pow(da * da * (1 + self.z[i]) * (1 + self.z[i]) * dr, 1. / 3.)
            rd = cosmo.rs_drag()

            theo = dv / rd
            theo_fsig8 = cosmo.scale_independent_growth_factor_f(self.z[i])*cosmo.sigma(8./cosmo.h(),self.z[i])

            # calculate difference between the sampled point and observations
            DV_diff = theo - self.DV_by_rd[i]
            fsig8_diff = theo_fsig8 - self.fsig8[i]

	    # save to data array
            data_array = np.append(data_array, DV_diff)
            data_array = np.append(data_array, fsig8_diff)

        # compute chi squared
        inv_cov_data = np.linalg.inv(self.cov_data)

        chi2 = np.dot(np.dot(data_array,inv_cov_data),data_array)

        # return ln(L)
        loglkl = - 0.5 * chi2
        return loglkl
