# scan_util.py
#
# This is a module containing subfunctions to read the chi2 tables from BOSS and
# eBOSS Lya auto and Lya-QSO cross correlation results
#
# class:chi2_interpolators
#       - __init__
#       - get_chi2_distances
#

import numpy as np
from scipy.interpolate import RectBivariateSpline

#####################################################################

#Class to read alpha_t by alpha_p chi2 scans e.g. from BOSS and interpolate.
class chi2_interpolators():
    def __init__(self,scan_locations,transverse_fid,parallel_fid):
        """
        Arguments:
        scan_locations: dictionary of filepaths to the different scans, with
                        keys as scan types.
        transverse_fid: fiducial value of transverse separation used to
                        calculate alpha_t.
        parallel_fid:   fiducial value of parallel separation used to calculate
                        alpha_p.
        """

        #Create a dictionary containing an interpolator for each scan.
        interpolators = {}
        for corr_type in scan_locations:
            scan = np.loadtxt(scan_locations[corr_type])

            #Column numbers in scan for data points.
            ap_index = 0
            at_index = 1
            chi2_index = 2

            #Get the alphas and make the scan grid.
            ap = np.array(sorted(set(scan[:,ap_index])))
            at = np.array(sorted(set(scan[:,at_index])))
            N_ap = ap.shape[0]
            N_at = at.shape[0]
            grid = np.zeros((N_at,N_ap))

            for i in range(N_ap):
                #Filter the data to only those corresponding to the ap value.
                indices = (scan[:,ap_index]==ap[i])
                scan_chunk = scan[indices,:]
                #Ensure that they're sorted by at value.
                scan_chunk = scan_chunk[scan_chunk[:,at_index].argsort()]
                #Add the chi2 column to the grid.
                #Note that the grid is of shape (N_at,N_ap)
                grid[:,i] = scan_chunk[:,chi2_index]

            #Make the interpolator (x refers to at, y refers to ap).
            interpolators[corr_type] = RectBivariateSpline(at,ap,grid,kx=1,ky=1)

        #Add the dictionary to the object.
        self.interpolators = interpolators
        self.transverse_fid = transverse_fid
        self.parallel_fid = parallel_fid

        return

    #Function to return the interpolated value of chi2 given distance measures.
    def get_Dchi2_from_distances(self,transverse,parallel,corr_type='cf'):
        """
        Arguments:
        transverse: value of transverse separation to evaluate chi2 for.
        parallel:   value of parallel separation to evaluate chi2 for.
        corr_type:  which scan to interpolate.

        Returns:
        Dchi2:       value of delta chi2
        """

        #Convert distances to alphas.
        at = transverse/self.transverse_fid
        ap = parallel/self.parallel_fid

        #With the new alphas, get the log likelihood.
        Dchi2 = self.interpolators[corr_type](at,ap)

        return Dchi2
