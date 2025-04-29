import os
import numpy as np
from montepython import io_mp
from montepython.likelihood_class import Likelihood
import scipy.constants as const
import sys
sys.path.append('.')
from . import scan_util as util

class bao_eBOSS_DR14_Lya_combined(Likelihood):

    # initialization routine

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)

        # define array for values of z and data points
        self.corr_types = []
        self.z = np.array([], 'float64')
        self.types = []

        scan_locations = {}
        scan_locations['comb'] = self.data_directory + '/' + self.cf_scan

        # read redshifts and data points
        for line in open(os.path.join(
                self.data_directory, self.file), 'r'):
            if (line.strip().find('#') == -1) and (len(line.strip())>0) and (line.split()[0] == 'comb'):
                self.corr_types += [line.split()[0]]
                self.z = np.append(self.z, float(line.split()[1]))
                self.types += [set([int(line.split()[2]),int(line.split()[3])])]

        # number of data points
        self.num_points = np.shape(self.z)[0]

        #Make our interpolators
        self.chi2_interpolators = util.chi2_interpolators(scan_locations,self.transverse_fid,self.parallel_fid)

        # end of initialization

    # compute log likelihood
    def loglkl(self, cosmo, data):

        chi2 = 0.

        # for each point, compute angular distance da, radial distance dr,
        # volume distance dv, sound horizon at baryon drag rs_d,
        # theoretical prediction and chi2 contribution
        # classes: (D_V/rs=3, Dv/Mpc=4, DA/rs=5, c/Hrs=6, rs/D_v=7, D_M/rs=8, H rs/rs_fid=9, D_M rs_fid/rs=10)
        for i in range(self.num_points):

            da = cosmo.angular_distance(self.z[i])
            dr = self.z[i] / cosmo.Hubble(self.z[i])
            H  = cosmo.Hubble(self.z[i]) * const.c / 1000.

            dv = pow(da * da * (1 + self.z[i]) * (1 + self.z[i]) * dr, 1. / 3.)
            dm = da * (1 + self.z[i])

            rd = cosmo.rs_drag() * self.rd_rescale

            if (self.types[i] == set([5,6])):
                transverse = da / rd
                parallel = (const.c / 1000.) / (H * rd)
                chi2 += self.chi2_interpolators.get_Dchi2_from_distances(transverse,parallel,corr_type=self.corr_types[i])
            elif (self.types[i] == set([8,6])):
                transverse = dm / rd
                parallel = (const.c / 1000.) / (H * rd)
                chi2 += self.chi2_interpolators.get_Dchi2_from_distances(transverse,parallel,corr_type=self.corr_types[i])
            else:
                raise io_mp.LikelihoodError(
                    "In likelihood %s. " % self.name +
                    "BAO data types %s " % self.types[i] +
                    "in %d-th line not appropriately chosen." % i)

        # return ln(L)
        lkl = - 0.5 * chi2

        return lkl
