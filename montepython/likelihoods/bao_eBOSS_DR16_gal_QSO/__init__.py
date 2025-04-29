import os
import numpy as np
import warnings
import montepython.io_mp as io_mp
from montepython.likelihood_class import Likelihood
import scipy.constants as conts
from scipy import interpolate as itp
from scipy.interpolate import RectBivariateSpline

class bao_eBOSS_DR16_gal_QSO(Likelihood):

    # initialization routine

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)

        # are there conflicting experiments?
        conflicting_experiments = [
            'bao', 'bao_boss', 'bao_known_rs', 'bao_angular',
            'bao_boss_aniso', 'bao_boss_aniso_gauss_approx',
            'bao_boss_dr12', 'bao_fs_boss_dr12', 'bao_fs_boss_dr12_twobins',
            'bao_fs_eBOSS_DR16_QSO', 'bao_fs_eBOSS_DR16_LRG']
        for experiment in conflicting_experiments:
            if experiment in data.experiments:
                raise io_mp.LikelihoodError(
                    'bao_eBOSS_DR16_gal_QSO reports conflicting BAO measurments from: %s' %(experiment))

        # define arrays for values of z and data points DM/r_s and DH/r_s.
        self.z = np.array([], 'float64')
        self.DM_over_rs = np.array([], 'float64')
        self.DH_over_rs = np.array([], 'float64')

        # Counting the number of data point
        # in LRG dr16 and QSO.
        lrg_dr16_points = 0
        qso_dr16_points = 0

        # Loadind the data depending on the datasets asked in the param file.
        # By defualt all the datasets are loaded.
        if self.lrg_dr12:
            print("BOSS LRG DR12 = " + str(self.lrg_dr12))
            with open(os.path.join(self.data_directory, self.lrg_dr12_data_file), 'r') as filein:
                for i, line in enumerate(filein):
                    if line.strip() and line.find('#') == -1:
                        this_line = line.split()
                        # load redshifts
                        self.z = np.append(self.z, float(this_line[0]))
                        # load D_M / rs
                        if this_line[2] == 'DM_over_rs':
                            self.DM_over_rs = np.append(
                                self.DM_over_rs, float(this_line[1]))
                        # load D_H/rs
                        elif this_line[2] == 'DH_over_rs':
                            self.DH_over_rs = np.append(
                                self.DH_over_rs, float(this_line[1]))

            lrg_dr12_points = i
        if self.lrg_dr16:
            print("eBOSS LRG DR16 = " + str(self.lrg_dr16))
            with open(os.path.join(self.data_directory, self.lrg_dr16_data_file), 'r') as filein:
                for i, line in enumerate(filein):
                    if line.strip() and line.find('#') == -1:
                        this_line = line.split()
                        # load redshifts
                        self.z = np.append(self.z, float(this_line[0]))
                        # load and D_M / rs
                        if this_line[2] == 'DM_over_rs':
                            self.DM_over_rs = np.append(
                                self.DM_over_rs, float(this_line[1]))
                        # load D_H/rs
                        elif this_line[2] == 'DH_over_rs':
                            self.DH_over_rs = np.append(
                                self.DH_over_rs, float(this_line[1]))
            # Counting the total number of datasets in LRG DR16 dataset. Important to set
            # the covariance matrix in the right order.
            lrg_dr16_points = i
        if self.qso_dr16:
            print("eBOSS QSO = " + str(self.qso_dr16))
            if  ('sdssdr16_lyauto') in data.experiments:
                if  not 'sdssdr16_lyaxqso' in data.experiments:
                    print("If you include QSO dataset in the 'sdssdr16_gal_qso' AND ly-alpha autocorrelation," +
                    "we highly recommend to take into account the crosscorrelation of the two dataset i.e. " +
                    "include also the likelihood sdssdr16_lyaxqso likelihood.")

            with open(os.path.join(self.data_directory, self.qso_data_file), 'r') as filein:
                for i, line in enumerate(filein):
                    if line.strip() and line.find('#') == -1:
                        this_line = line.split()
                        # load redshift
                        self.z = np.append(self.z, float(this_line[0]))
                        # load D_M / rs
                        if this_line[2] == 'DM_over_rs':
                            self.DM_over_rs = np.append(
                                self.DM_over_rs, float(this_line[1]))
                        # load D_H/rs
                        elif this_line[2] == 'DH_over_rs':
                            self.DH_over_rs = np.append(
                                self.DH_over_rs, float(this_line[1]))
            # Counting the total number of datasets in QSO datasets. Important to set
            # the covariance matrix in the right order.
            qso_dr16_points = i
        # Pick out the unique redshifts from the datafiles.
        self.z = np.unique(self.z)
        # Number of bins
        self.num_bins = np.shape(self.z)[0]
        # Number of data points
        self.num_points = 2*self.num_bins

        # Read covariance matrix and set them according to the chosen dataset
        self.cov_data = np.zeros((self.num_points,self.num_points), 'float64')
        if self.lrg_dr12:
            self.cov_data = np.loadtxt(os.path.join(self.data_directory, self.lrg_dr12_cov_file))
        if self.lrg_dr16:
            # Check if LRG DR12 covariance has already been loaded in the covaraince matrix...
            if np.all(self.cov_data == 0):
                # If not, load the LRG DR16 dataset.
                self.cov_data = np.loadtxt(os.path.join(self.data_directory, self.lrg_dr16_cov_file))
            else:
                # If LRG DR12 covariance matrix alreadt exists then change the size of the covariance matrix
                # and then add the LRG DR16 coavariance matrix along the digonal with LRG DR12.
                cov_data_Temp = self.cov_data
                self.cov_data = np.zeros((np.shape(cov_data_Temp)[0] +
                                          (lrg_dr16_points + 1), np.shape(cov_data_Temp)[0]
                                          + (lrg_dr16_points + 1)))
                self.cov_data[0:np.shape(cov_data_Temp)[0],0:np.shape(cov_data_Temp)[0]] = cov_data_Temp
                self.cov_data[np.shape(cov_data_Temp)[0]:np.shape(cov_data_Temp)[0] +
                              (lrg_dr16_points + 1),np.shape(cov_data_Temp)[0]:np.shape(cov_data_Temp)[0] +
                              (lrg_dr16_points + 1)] = np.loadtxt(os.path.join(self.data_directory, self.lrg_dr16_cov_file))
        # Same as above but for QSO covariance matrix. Check, read and adjust the covariance matrix.
        if self.qso_dr16:
            if np.all(self.cov_data == 0):
                self.cov_data = np.loadtxt(os.path.join(self.data_directory, self.qso_cov_file))
            else:
                cov_data_Temp = self.cov_data
                self.cov_data = np.zeros((np.shape(cov_data_Temp)[0] +
                                          (qso_dr16_points + 1), np.shape(cov_data_Temp)[0]
                                          + (qso_dr16_points + 1)))
                self.cov_data[0:np.shape(cov_data_Temp)[0],0:np.shape(cov_data_Temp)[0]] = cov_data_Temp
                self.cov_data[np.shape(cov_data_Temp)[0]:np.shape(cov_data_Temp)[0] +
                              (qso_dr16_points + 1),np.shape(cov_data_Temp)[0]:np.shape(cov_data_Temp)[0] +
                              (qso_dr16_points + 1)] = np.loadtxt(os.path.join(self.data_directory, self.qso_cov_file))

    # compute likelihood
    def loglkl(self, cosmo, data):
        loglkl = 0.0

        # define array for  values of D_M_diff = D_M^th - D_M^obs and H_diff = H^th - H^obs,
        # ordered by redshift bin (z=[0.38, 0.51, 0.61]) as following:
        # data_array = [DM_diff(z=0.38), H_diff(z=0.38), DM_diff(z=0.51), .., .., ..]
        data_array = np.array([], 'float64')
        # for each point, compute comoving angular diameter distance D_M = (1 + z) * D_A,
        # sound horizon at baryon drag rs_d, theoretical prediction
        for i in range(self.num_bins):
            DM_at_z = cosmo.angular_distance(self.z[i]) * (1. + self.z[i])
            H_at_z = cosmo.Hubble(self.z[i])
            DH_at_z = 1.0/H_at_z
            rd = cosmo.rs_drag() * self.rs_rescale

            theo_DM_at_z = DM_at_z / rd
            theo_DH_at_z_in_Mpc_inv = DH_at_z / rd

            # calculate difference between the sampled point and observations
            DM_diff = theo_DM_at_z - self.DM_over_rs[i]
            H_diff = theo_DH_at_z_in_Mpc_inv - self.DH_over_rs[i]

            # save to data array
            data_array = np.append(data_array, DM_diff)
            data_array = np.append(data_array, H_diff)

        # compute chi squared
        inv_cov_data = np.linalg.inv(self.cov_data)
        chi2 = np.dot(np.dot(data_array,inv_cov_data),data_array)

        # return ln(L)
        loglkl = - 0.5 * chi2

        return loglkl
