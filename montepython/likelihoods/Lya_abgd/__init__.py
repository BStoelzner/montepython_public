from montepython.likelihood_class import Likelihood
import io_mp
import re  # Module to handle regular expressions
import sys
import os
import numpy as np
from scipy import interpolate
from scipy.linalg import block_diag
import pickle
import matplotlib.pyplot as plt

# Backwards-compatibility fix for older python versions
try:
  xrange(5)
  dictitems = lambda x: x.iteritems()
except:
  dictitems = lambda x: x.items()


class Lya_abgd(Likelihood):

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)

        if self.verbose > 0:
          print("Initializing Lya_abgd likelihood")

        self.need_cosmo_arguments(data, {"output": "mPk"})
        self.need_cosmo_arguments(data, {"P_k_max_h/Mpc": 1.5*self.kmax})

        # Derived_lkl is a new type of derived parameter calculated in the likelihood, and not known to class.
        # This first initialising avoids problems in the case of an error in the first point of the MCMC.
        data.derived_lkl = {"lya_neff":0,"area_criterion":0,"self_distance_weighted":0.,"weight_largest":0.,"weightsum_2":0.,"weightsum_4":0}

        # Set up the bin file for the points that do not pass the sanity checks
        self.bin_file_path = os.path.join(command_line.folder,self.bin_file_name)
        if not os.path.exists(self.bin_file_path):
          with open(self.bin_file_path, "w") as bin_file:
            bin_file.write("#")
            for name in data.get_mcmc_parameters(["varying"]):
              name = re.sub("[$*&]", "", name)
              bin_file.write(" %s\t" % name)
            for name in data.get_mcmc_parameters(["derived"]):
              name = re.sub("[$*&]", "", name)
              bin_file.write(" %s\t" % name)
            for name in data.get_mcmc_parameters(["derived_lkl"]):
              name = re.sub("[$*&]", "", name)
              bin_file.write(" %s\t" % name)
            bin_file.write("\n")
            bin_file.close()
        if "sigma8" not in data.get_mcmc_parameters(["derived"]) or "z_reio" not in data.get_mcmc_parameters(["derived"]):
          raise io_mp.ConfigurationError("Error: Lya_abgd likelihood need sigma8 and  z_reio as derived parameters")

        # Initialize the grids
        self.grid_size = [self.abg_grid_size,self.abd_grid_size,self.thermal_grid_size,1]

        # Number of non-astro params (i.e. alpha, beta, and gamma)
        self.params_numbers = 3

        alphas_abg = np.zeros(self.abg_grid_size, "float64")
        betas_abg = np.zeros(self.abg_grid_size, "float64")
        gammas_abg = np.zeros(self.abg_grid_size, "float64")

        alphas_abd = np.zeros(self.abd_grid_size, "float64")
        betas_abd = np.zeros(self.abd_grid_size, "float64")
        deltas_abd = np.zeros(self.abd_grid_size, "float64")

        alphas_thermal = np.zeros(self.thermal_grid_size, "float64")
        betas_thermal = np.zeros(self.thermal_grid_size, "float64")
        gammas_thermal = np.zeros(self.thermal_grid_size, "float64")

        # Read the abg grid
        file_path = os.path.join(self.data_directory, self.abg_grid_file)
        if os.path.exists(file_path):
          with open(file_path, "r") as grid_file:
            line = grid_file.readline()
            while line.find("#") != -1:
              line = grid_file.readline()
            while line.find("\n") != -1 and len(line) == 3:
              line = grid_file.readline()
            for index in range(self.abg_grid_size):
              alphas_abg[index] = float(line.split()[0])
              betas_abg[index] = float(line.split()[1])
              gammas_abg[index] = float(line.split()[2])
              line = grid_file.readline()
            grid_file.close()
        else:
          raise io_mp.ConfigurationError("Error: ABG grid file is missing")

        # Read the abd grid
        file_path = os.path.join(self.data_directory, self.abd_grid_file)
        if os.path.exists(file_path):
          with open(file_path, "r") as grid_file:
            line = grid_file.readline()
            while line.find("#") != -1:
              line = grid_file.readline()
            while line.find("\n") != -1 and len(line) == 3:
              line = grid_file.readline()
            for index in range(self.abd_grid_size):
              alphas_abd[index] = float(line.split()[0])
              betas_abd[index] = float(line.split()[1])
              deltas_abd[index] = float(line.split()[2])
              line = grid_file.readline()
            grid_file.close()
        else:
          raise io_mp.ConfigurationError("Error: ABD grid file is missing")

        # Read the thermal grid
        file_path = os.path.join(self.data_directory, self.thermal_grid_file)
        if os.path.exists(file_path):
          with open(file_path, "r") as grid_file:
            line = grid_file.readline()
            while line.find("#") != -1:
              line = grid_file.readline()
            while line.find("\n") != -1 and len(line) == 3:
              line = grid_file.readline()
            for index in range(self.thermal_grid_size):
              alphas_thermal[index] = float(line.split()[0])
              betas_thermal[index] = float(line.split()[1])
              gammas_thermal[index] = float(line.split()[2])
              line = grid_file.readline()
            grid_file.close()
        else:
          raise io_mp.ConfigurationError("Error: Thermal ABG grid file is missing")

        # Set redshift independent parameters -- params order: z_reio, sigma_8, n_eff, f_UV
        self.zind_param_size = [3, 5, 5, 3] # The number of grid values for each distinct constant parameter
        self.zind_param_min = np.array([7., 0.5, -2.6, 0.]) # Minimum value for each parameter
        self.zind_param_max = np.array([15., 1.5, -2.0, 1.]) # Maximum value for each parameter
        zind_param_ref = np.array([9., 0.829, -2.3074, 0.]) # Fiducial value for each parameter
        self.zreio_range = self.zind_param_max[0]-self.zind_param_min[0] # Range of z_reio
        self.neff_range = self.zind_param_max[2]-self.zind_param_min[2] # Range of n_eff

        # Set redshift dependent parameters -- params order: mean_f , t0, slope.
        # The number of grid values for each distinct redshift dependent parameter
        zdep_params_size = [9, 3, 3]

        # Set mean flux values
        self.flux_ref_old = (np.array([0.669181, 0.617042, 0.564612, 0.512514, 0.461362, 0.411733, 0.364155, 0.253828, 0.146033, 0.0712724]))

        # Manage the data sets
        # FIRST (NOT USED) DATASET (19 wavenumbers) ***XQ-100***
        self.zeta_range_XQ = [3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2]  # List of redshifts corresponding to the 19 wavenumbers (k)
        self.k_XQ = [0.003,0.006,0.009,0.012,0.015,0.018,0.021,0.024,0.027,0.03,0.033,0.036,0.039,0.042,0.045,0.048,0.051,0.054,0.057]


        # SECOND DATASET (7 wavenumbers) ***HIRES/MIKE***
        self.zeta_range_mh = [4.2, 4.6, 5.0, 5.4] # List of redshifts corresponding to the 7 wavenumbers (k)
        self.k_mh = [0.00501187,0.00794328,0.0125893,0.0199526,0.0316228,0.0501187,0.0794328] # Note that k is in s/km

        self.zeta_full_length = (len(self.zeta_range_XQ) + len(self.zeta_range_mh))
        self.kappa_full_length = (len(self.k_XQ) + len(self.k_mh))
        # Which snapshots we use (first 7 for first dataset, last 4 for second one)
        self.redshift = [3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.2, 4.6, 5.0, 5.4]

        # Set T0 and slope values for the given snapshots
        self.t0_ref_old = np.array([11251.5, 11293.6, 11229.0, 10944.6, 10421.8, 9934.49, 9227.31, 8270.68, 7890.68, 7959.4])
        self.slope_ref_old = np.array([1.53919, 1.52894, 1.51756, 1.50382, 1.48922, 1.47706, 1.46909, 1.48025, 1.50814, 1.52578])

        t0_values_old = np.zeros(( 10, zdep_params_size[1] ),"float64")
        t0_values_old[:,0] = np.array([7522.4, 7512.0, 7428.1, 7193.32, 6815.25, 6480.96, 6029.94, 5501.17, 5343.59, 5423.34])
        t0_values_old[:,1] = self.t0_ref_old[:]
        t0_values_old[:,2] = np.array([14990.1, 15089.6, 15063.4, 14759.3, 14136.3, 13526.2, 12581.2, 11164.9, 10479.4, 10462.6])

        slope_values_old = np.zeros(( 10, zdep_params_size[2] ),"float64")
        slope_values_old[:,0] = np.array([0.996715, 0.979594, 0.960804, 0.938975, 0.915208, 0.89345, 0.877893, 0.8884, 0.937664, 0.970259])
        slope_values_old[:,1] = [1.32706, 1.31447, 1.30014, 1.28335, 1.26545, 1.24965, 1.2392, 1.25092, 1.28657, 1.30854]
        slope_values_old[:,2] = self.slope_ref_old[:]

        self.t0_min = t0_values_old[:,0]*0.1
        self.t0_max = t0_values_old[:,2]*1.4
        self.slope_min = slope_values_old[:,0]*0.8
        self.slope_max = slope_values_old[:,2]*1.15

        ALL_zdep_params = len(self.flux_ref_old) + len(self.t0_ref_old) + len(self.slope_ref_old)
        astroparams_number_KRIG = len(self.zind_param_size) + ALL_zdep_params

        # Import the grids for Kriging
        file_path = os.path.join(self.data_directory, self.astro_spectra_file)
        if os.path.exists(file_path):
          self.input_full_matrix_interpolated_ASTRO = self.loadpickle(file_path)
        else:
          raise io_mp.ConfigurationError("Error: astro spectra file is missing")

        file_path = os.path.join(self.data_directory, self.abg_spectra_file)
        if os.path.exists(file_path):
          self.input_full_matrix_interpolated_ABG = self.loadpickle(file_path)
        else:
          raise io_mp.ConfigurationError("Error: abg spectra file is missing")

        file_path = os.path.join(self.data_directory, self.abd_spectra_file)
        if os.path.exists(file_path):
          self.input_full_matrix_interpolated_ABD = self.loadpickle(file_path)
        else:
          raise io_mp.ConfigurationError("Error: abd spectra file is missing")

        file_path = os.path.join(self.data_directory, self.thermal_spectra_file)
        if os.path.exists(file_path):
          self.input_full_matrix_interpolated_THERMAL = self.loadpickle(file_path)
        else:
          raise io_mp.ConfigurationError("Error: thermal spectra file is missing")

        file_path = os.path.join(self.data_directory, self.lcdm_spectra_file)
        if os.path.exists(file_path):
          self.input_full_matrix_interpolated_LCDM = self.loadpickle(file_path)
        else:
          raise io_mp.ConfigurationError("Error: lcdm spectra file is missing")

        self.input_full_matrix_interpolated = [self.input_full_matrix_interpolated_ABG, self.input_full_matrix_interpolated_ABD, self.input_full_matrix_interpolated_THERMAL, self.input_full_matrix_interpolated_LCDM]
        # Set the corresponding grid lengths
        grid_length_ASTRO = len(self.input_full_matrix_interpolated_ASTRO[0,0,:])
        grid_length_ABG = len(self.input_full_matrix_interpolated_ABG[0,0,:])
        grid_length_ABD = len(self.input_full_matrix_interpolated_ABD[0,0,:])
        grid_length_THERMAL = len(self.input_full_matrix_interpolated_THERMAL[0,0,:])
        grid_length_LCDM = len(self.input_full_matrix_interpolated_LCDM[0,0,:])

        # Import the ASTRO GRID (ordering of params: z_reio, sigma_8, n_eff, f_UV, mean_f(z), t0(z), slope(z))
        file_path = os.path.join(self.data_directory, self.astro_grid_file)
        if os.path.exists(file_path):
          self.X = np.zeros((grid_length_ASTRO,astroparams_number_KRIG), "float64")
          for param_index in range(astroparams_number_KRIG):
            self.X[:,param_index] = np.genfromtxt(file_path, usecols=[param_index], skip_header=1)
        else:
          raise io_mp.ConfigurationError("Error: abd+astro grid file is missing")

        # Interpolation in the astro params space
        self.redshift_list = np.array([3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.6, 5.0, 5.4]) # This corresponds to the combined dataset (MIKE/HIRES + XQ-100)
        # Set minimum and maximum values for the kriging normalisation
        self.F_prior_min = np.array([0.535345,0.493634,0.44921,0.392273,0.338578,0.28871,0.218493,0.146675,0.0676442,0.0247793])
        self.F_prior_max = np.array([0.803017,0.748495,0.709659,0.669613,0.628673,0.587177,0.545471,0.439262,0.315261,0.204999])

        # Interpolation in the cosmo params space
        self.k = np.logspace(np.log10(self.kmin), np.log10(self.kmax), num=self.k_size)
        self.Tks_grid_abg = np.empty((self.abg_grid_size,self.k_size))
        self.Tks_grid_abd = np.empty((self.abd_grid_size,self.k_size))
        self.Tks_grid_thermal = np.empty((self.thermal_grid_size,self.k_size))
        self.Tks_grid_lcdm = np.empty((1,self.k_size))
        for k in range(self.abg_grid_size):
          self.Tks_grid_abg[k] = self.T_abg(self.k,alphas_abg[k],betas_abg[k],gammas_abg[k])
        for k in range(self.thermal_grid_size):
          self.Tks_grid_thermal[k] = self.T_abg(self.k,alphas_thermal[k],betas_thermal[k],gammas_thermal[k])
        for k in range(self.abd_grid_size):
          self.Tks_grid_abd[k] = self.T_abd(self.k,alphas_abd[k],betas_abd[k],deltas_abd[k])
        self.Tks_grid_lcdm[0] = self.k/self.k # An array of length k of the value 1.0, since it is the T(k) of LCDM
        self.Tks_grid = [self.Tks_grid_abg, self.Tks_grid_abd, self.Tks_grid_thermal, self.Tks_grid_lcdm]
        self.Tks_grid_concat = np.concatenate(self.Tks_grid)

        # Load the data and covariance matrices
        if not self.DATASET == "mike-hires":
          raise io_mp.LikelihoodError("Error: for the time being, only the mike - hires dataset is available")

        file_path = os.path.join(self.data_directory, self.MIKE_spectra_file)
        if os.path.exists(file_path):
          y_M_reshaped = self.loadpickle(file_path)
        else:
          raise io_mp.ConfigurationError("Error: MIKE spectra file is missing")

        file_path = os.path.join(self.data_directory, self.HIRES_spectra_file)
        if os.path.exists(file_path):
          y_H_reshaped = self.loadpickle(file_path)
        else:
          raise io_mp.ConfigurationError("Error: HIRES spectra file is missing")

        file_path = os.path.join(self.data_directory, self.MIKE_cov_file)
        if os.path.exists(file_path):
          cov_M_inverted = self.loadpickle(file_path)
        else:
          raise io_mp.ConfigurationError("Error: MIKE covariance matrix file is missing")

        file_path = os.path.join(self.data_directory, self.HIRES_cov_file)
        if os.path.exists(file_path):
          cov_H_inverted = self.loadpickle(file_path)
        else:
          raise io_mp.ConfigurationError("Error: HIRES covariance matrix file is missing")

        file_path = os.path.join(self.data_directory, self.PF_noPRACE_file)
        if os.path.exists(file_path):
          self.PF_noPRACE = self.loadpickle(file_path)
        else:
          raise io_mp.ConfigurationError("Error: PF_noPRACE file (i.e. the reference flux power spectrum) is missing")

        self.cov_MH_inverted = block_diag(cov_H_inverted,cov_M_inverted)
        self.y_MH_reshaped = np.concatenate((y_H_reshaped, y_M_reshaped))

        print("Initialization of Lya_abgd likelihood done")

    # The following functions are used elsewhere in the code

    # Model functions, T^2=P_model/P_ref
    def T_abd(self,k,alpha,beta,delta):
      return (1. - delta)*(1. + (alpha*k)**(beta))**(-(3./2.)*beta) + delta
    def T_abg(self,k,alpha,beta,gamma):
      return (1. + (alpha*k)**(beta))**(gamma)
    def T_abgd(self,k,alpha,beta,gamma,delta):
      return (1. - delta)*(1. + (alpha*k)**(beta))**(gamma) + delta
    def inverse_T_abgd(self,Tk,alpha,beta,gamma,delta):
      return (((Tk-delta)/(1. - delta))**(1./gamma)-1.)**(1./beta)/alpha

    # Analytical function for the redshift dependence of t0 and slope
    def z_dep_func(self,parA, parS, z):
      return parA*(( (1.+z)/(1.+self.zp) )**parS)

    # Functions for the Kriging interpolation
    def ordkrig_distance_astro(self,p1, p2, p3, p4, p5, p6, p7, v1, v2, v3, v4, v5, v6, v7):
      return (((p1 - v1)**2 + (p2 - v2)**2 + (p3 - v3)**2 + (p4 - v4)**2 + (p5 - v5)**2 + (p6 - v6)**2 + (p7 - v7)**2)**(0.5) + self.epsilon_astro)**self.exponent_astro

    def ordkrig_norm_astro(self,p1, p2, p3, p4, p5, p6, p7, v1, v2, v3, v4, v5, v6, v7):
      return np.sum(1./self.ordkrig_distance_astro(p1, p2, p3, p4, p5, p6, p7, v1, v2, v3, v4, v5, v6, v7))

    def ordkrig_lambda_astro(self,p1, p2, p3, p4, p5, p6, p7, v1, v2, v3, v4, v5, v6, v7):
      return (1./self.ordkrig_distance_astro(p1, p2, p3, p4, p5, p6, p7, v1, v2, v3, v4, v5, v6, v7))/self.ordkrig_norm_astro(p1, p2, p3, p4, p5, p6, p7, v1, v2, v3, v4, v5, v6, v7)

    def ordkrig_estimator_astro(self,p21, z):
      pa10 = (self.z_dep_func(p21[8], p21[9], z[:])*1e4)/(self.t0_max[:]-self.t0_min[:])
      pb10 = self.z_dep_func(p21[10], p21[11], z[:])/(self.slope_max[:]-self.slope_min[:])
      if self.TEST_set_astro_to_fiducial:
        self.flux_ref_old = (np.array([0.669181, 0.617042, 0.564612, 0.512514, 0.461362, 0.411733, 0.364155, 0.253828, 0.146033, 0.0712724]))
        self.t0_ref_old = np.array([11251.5, 11293.6, 11229.0, 10944.6, 10421.8, 9934.49, 9227.31, 8270.68, 7890.68, 7959.4])
        self.slope_ref_old = np.array([1.53919, 1.52894, 1.51756, 1.50382, 1.48922, 1.47706, 1.46909, 1.48025, 1.50814, 1.52578])
        pa10 = self.t0_ref_old/(self.t0_max[:]-self.t0_min[:])
        pb10 = self.slope_ref_old/(self.slope_max[:]-self.slope_min[:])
        p21[4:8] = self.flux_ref_old[-4:]
      p37 = np.concatenate((p21[0:8], pa10[6:], pb10[6:]))
      astrokrig_result = np.zeros((self.zeta_full_length, self.kappa_full_length ), "float64")
      for index in range(self.num_z_XQ,len(self.redshift)):
        astrokrig_result[index,:] = np.sum(np.multiply(self.ordkrig_lambda_astro(p37[0]/self.zreio_range, p37[1], p37[2]/self.neff_range, p37[3], p37[4+index-self.num_z_XQ]/(self.F_prior_max[index-self.num_z_overlap]-self.F_prior_min[index-self.num_z_overlap]), p37[8+index-self.num_z_XQ], p37[12+index-self.num_z_XQ], self.X[:,0], self.X[:,1], self.X[:,2], self.X[:,3], self.X[:,4+index-self.num_z_overlap], self.X[:,14+index-self.num_z_overlap], self.X[:,24+index-self.num_z_overlap]), self.input_full_matrix_interpolated_ASTRO[index,:,:]),axis=1)
      return astrokrig_result

    def ordkrig(self, params, z_list, weights):
      pksum = np.zeros((self.zeta_full_length, self.kappa_full_length))
      for typ in range(4):
        matrix_new = np.zeros(( self.zeta_full_length, self.kappa_full_length, self.grid_size[typ]), "float64")
        NEW_matrix = np.zeros(( self.grid_size[typ], self.zeta_full_length, self.kappa_full_length), "float64")
        full_matrix_interpolated = np.zeros(( self.zeta_full_length, self.kappa_full_length, self.grid_size[typ]), "float64")
        for i in range(self.zeta_full_length):
          for j in range(self.kappa_full_length):
            NEW_matrix[:,i,j] = self.input_full_matrix_interpolated[typ][i,j,:]
        matrix_new = NEW_matrix + self.ordkrig_estimator_astro(params,z_list) - 1.
        matrix_new = np.clip(matrix_new, 0. , None)
        for i in range(self.zeta_full_length):
          for j in range(self.kappa_full_length):
            full_matrix_interpolated[i,j,:] = matrix_new[:,i,j]
        pksum += np.sum(np.multiply(weights[typ],full_matrix_interpolated[:,:,:]),axis=2)
      return pksum

    # Function to write to bin file
    def write_error_to_binfile(self,data,errorname):
      with open(self.bin_file_path, "a") as bin_file:
        bin_file.write("#"+errorname+"\t")
        for elem in data.get_mcmc_parameters(["varying"]):
          bin_file.write(" %.6e\t" % data.mcmc_parameters[elem]["current"])
        for elem in data.get_mcmc_parameters(["derived"]):
          bin_file.write(" %.6e\t" % data.mcmc_parameters[elem]["current"])
        bin_file.write("\n")
        bin_file.close()
      sys.stderr.write("#"+errorname+"\n")
      sys.stderr.flush()
      self.TEST_NODATA_lkl = data.boundary_loglike
      return data.boundary_loglike

    # Function to deal with pickles created in older python versions
    def loadpickle(self,file_path):
      try:
        with open(file_path,"rb") as pkl:
          return pickle.load(pkl,encoding="bytes")
      except Exception as e:
        with open(file_path,"r") as pkl:
          return pickle.load(pkl)


    # Start of the actual likelihood computation function
    def loglkl(self, cosmo, data):

        k = self.k

        # Initialise the bin file
        if not os.path.exists(self.bin_file_path):
          with open(self.bin_file_path, "w") as bin_file:
            bin_file.write("#")
            for name in data.get_mcmc_parameters(["varying"]):
              name = re.sub("[$*&]", "", name)
              bin_file.write(" %s\t" % name)
            for name in data.get_mcmc_parameters(["derived"]):
              name = re.sub("[$*&]", "", name)
              bin_file.write(" %s\t" % name)
            for name in data.get_mcmc_parameters(["derived_lkl"]):
              name = re.sub("[$*&]", "", name)
              bin_file.write(" %s\t" % name)
            bin_file.write("\n")
            bin_file.close()

        astro_pars = {'T0a':0.74,'T0s':-4.38,'gamma_a':1.45,'gamma_s':1.93,'Fz1':0.35,'Fz2':0.26,'Fz3':0.18,'Fz4':0.07,'F_UV':0.0}
        for par in astro_pars:
          if par in data.mcmc_parameters:
            astro_pars[par] = data.mcmc_parameters[par]["current"]*data.mcmc_parameters[par]["scale"]
          else:
            print("WARNING :: In likelihood 'Lya_abgd' you are missing the parameter {}, it has been set to the default value of '{}'".format(par,astro_pars[par]))

        # Get P(k) from CLASS
        h=cosmo.h()
        Plin = np.zeros(len(k), "float64")
        for index_k in range(len(k)):
          Plin[index_k] = cosmo.pk_lin(k[index_k]*h, 0.0)
        Plin *= h**3

        # Compute the Lya k scale
        k_neff=self.k_s_over_km*299792.458*cosmo.Hubble(self.z)/cosmo.h()/(1.+self.z)

        derived = cosmo.get_current_derived_parameters(data.get_mcmc_parameters(["derived"]))
        for name, value in dictitems(derived):
          data.mcmc_parameters[name]["current"] = value
          data.mcmc_parameters[name]["current"] /= data.mcmc_parameters[name]["scale"]

        # Obtain current z_reio, sigma_8, and neff from CLASS
        z_reio=data.mcmc_parameters["z_reio"]["current"]
        # Check that z_reio is in the correct range
        if z_reio<self.zind_param_min[0]:
          z_reio = self.zind_param_min[0]
        if z_reio>self.zind_param_max[0]:
          z_reio=self.zind_param_max[0]
        sigma8=data.mcmc_parameters["sigma8"]["current"]
        neff=cosmo.pk_tilt(k_neff*h,self.z)

        # Store neff as a derived_lkl parameter
        data.derived_lkl["lya_neff"] = neff

        # First sanity check, to make sure the cosmological parameters are in the correct range
        if (sigma8<self.zind_param_min[1] or sigma8>self.zind_param_max[1]) or (neff<self.zind_param_min[2] or neff>self.zind_param_max[2]):
          self.err_lkl = self.write_error_to_binfile(data,"Error_cosmo")
          if not self.TEST_nodata_run:
            if self.verbose > 0:
              print("FINISHED abgd LOGLKL CALC on Error_cosmo")
            return self.err_lkl

        # Here Neff is the standard N_eff (effective d.o.f.)
        classNeff=cosmo.Neff()

        # Store the current CLASS values for later
        from initialise import recover_cosmological_module
        cosmo_lcdm_equiv = recover_cosmological_module(data)
        lcdm_equiv_params = data.cosmo_arguments.copy()

        # Make sure to give a warning if nothing was removed
        flag_something_removed = False
        # To calculate the LCDM-equivalent, we need to remap the non-LCDM parameters
        # First we deal with cases where we have extra relativistic d.o.f. (following 1412.6763)
        if "xi_idr" in lcdm_equiv_params or "N_ur" in lcdm_equiv_params or "N_ncdm" in lcdm_equiv_params or "N_dg" in lcdm_equiv_params:
          eta2=(1.+0.2271*classNeff)/(1.+0.2271*3.046)
          eta=np.sqrt(eta2)

          # Set the relativistic d.o.f to match the simulations
          if "N_ur" in lcdm_equiv_params:
            lcdm_equiv_params["N_ur"] = 3.046
          if "N_ncdm" in lcdm_equiv_params:
            del lcdm_equiv_params["N_ncdm"]

          # Adjust lcdm params to account for extra d.o.f.
          if "omega_b" in lcdm_equiv_params:
            lcdm_equiv_params["omega_b"] *= 1./eta2
          if "omega_cdm" in lcdm_equiv_params:
            lcdm_equiv_params["omega_cdm"] *= 1./eta2
          if "H0" in lcdm_equiv_params:
            lcdm_equiv_params["H0"] *= 1./eta
          if "100*theta_s" in lcdm_equiv_params:
            raise io_mp.ConfigurationError("Error: run with H0 instead of 100*theta_s")

          # Deal with Interacting Dark Matter with Dark Radiation (ETHOS-like models)
          if "xi_idr" in lcdm_equiv_params or "N_idr" in lcdm_equiv_params or "N_dg" in lcdm_equiv_params:
            flag_something_removed = True
            # Class can take Omega_idm, omega_idm, or f_idm, so the following lines are needed to compute the lcdm equivalent
            if "Omega_idm" in lcdm_equiv_params:
              if "Omega_cdm" in lcdm_equiv_params:
                lcdm_equiv_params["Omega_cdm"] +=lcdm_equiv_params["Omega_idm"]
              if "omega_cdm" in lcdm_equiv_params:
                lcdm_equiv_params["omega_cdm"] +=lcdm_equiv_params["Omega_idm"]*h*h/eta/eta
              del lcdm_equiv_params["Omega_idm"]
            if "omega_idm" in lcdm_equiv_params:
              if "Omega_cdm" in lcdm_equiv_params:
                lcdm_equiv_params["Omega_cdm"] +=lcdm_equiv_params["omega_idm"]/h/h
              if "omega_cdm" in lcdm_equiv_params:
                lcdm_equiv_params["omega_cdm"] +=lcdm_equiv_params["omega_idm"]/eta2
              del lcdm_equiv_params["omega_idm"]
              # Now we clean up variable no longer needed. We cover all possible cases of notation
              to_remove = ["xi_idr","N_dg","N_idr","a_idm_dr","a_dark","Gamma_0_nadm","nindex_dark","n_index_idm_dr","nindex_idm_dr","f_idm_dr","f_idm","m_idm"]
              for bad in to_remove:
                if bad in lcdm_equiv_params:
                  del lcdm_equiv_params[bad]

          # Deal with Hot Dark Matter
          if "m_ncdm" in lcdm_equiv_params and not "omega_ncdm" in lcdm_equiv_params and not "Omega_ncdm" in lcdm_equiv_params:
            flag_something_removed = True
            lcdm_equiv_params["m_ncdm"] *= 1./eta2
          if "omega_ncdm" in lcdm_equiv_params and not "Omega_ncdm" in lcdm_equiv_params and not "m_ncdm" in lcdm_equiv_params:
            flag_something_removed = True
            lcdm_equiv_params["omega_ncdm"] *= 1./eta2

          # Deal with Warm Dark Matter
          if "m_ncdm" in lcdm_equiv_params and ("omega_ncdm" in lcdm_equiv_params or "Omega_ncdm" in lcdm_equiv_params):
            flag_something_removed = True
            if "Omega_ncdm" in lcdm_equiv_params:
              if "Omega_cdm" in lcdm_equiv_params:
                lcdm_equiv_params["Omega_cdm"] +=lcdm_equiv_params["Omega_ncdm"]
              if "omega_cdm" in lcdm_equiv_params:
                lcdm_equiv_params["omega_cdm"] +=lcdm_equiv_params["Omega_ncdm"]*h*h/eta/eta
              del lcdm_equiv_params["Omega_ncdm"]
            if "omega_ncdm" in lcdm_equiv_params:
              if "Omega_cdm" in lcdm_equiv_params:
                lcdm_equiv_params["Omega_cdm"] +=lcdm_equiv_params["omega_ncdm"]/h/h
              if "omega_cdm" in lcdm_equiv_params:
                lcdm_equiv_params["omega_cdm"] +=lcdm_equiv_params["omega_ncdm"]/eta2
              del lcdm_equiv_params["omega_ncdm"]
              # Now we clean up variable no longer needed. We cover all possible cases of notation
              to_remove = ["m_ncdm","T_ncdm","ncdm_fluid_approximation","l_max_ncdm","Number of momentum bins","Maximum q","Quadrature strategy"]
              for bad in to_remove:
                if bad in lcdm_equiv_params:
                  del lcdm_equiv_params[bad]

        # Now we deal with models that do not affect the relativistic d.o.f.
        # Deal with interacting DM - baryons
        if "cross_idm_b" or "log_cross_idmb" in lcdm_equiv_params:
          flag_something_removed = True
          # The fraction of idm needs to be added to cdm (note that if f_idm has been passed, this is not needed
          if "Omega_idm" in lcdm_equiv_params:
            if "Omega_cdm" in lcdm_equiv_params:
              lcdm_equiv_params["Omega_cdm"] +=lcdm_equiv_params["Omega_idm"]
            if "omega_cdm" in lcdm_equiv_params:
              lcdm_equiv_params["omega_cdm"] +=lcdm_equiv_params["Omega_idm"]*h*h
            del lcdm_equiv_params["Omega_idm"]
          if "omega_idm" in lcdm_equiv_params:
            if "Omega_cdm" in lcdm_equiv_params:
              lcdm_equiv_params["Omega_cdm"] +=lcdm_equiv_params["omega_idm"]/h/h
            if "omega_cdm" in lcdm_equiv_params:
              lcdm_equiv_params["omega_cdm"] +=lcdm_equiv_params["omega_idm"]
            del lcdm_equiv_params["omega_idm"]
          # Now we clean up variable no longer needed. We cover all possible cases of notation
          to_remove = ["log_cross_idmb","cross_idm_b", "f_idm", "m_idm", "n_index_idm_b"]
          for bad in to_remove:
            if bad in lcdm_equiv_params:
              del lcdm_equiv_params[bad]
        if not flag_something_removed and not self.isLCDM:
          raise io_mp.LikelihoodError("In likelihood 'Lya_abgd', there was nothing removed from the parameter list, this means something has most likely gone wrong if not running in LambdaCDM. In this case, your run is probably not as intended. Feel free to add behavior defining the LCDM equivalent model for your specific case within the likelihood.")

        # Set up the lcdm equivalent model for CLASS
        if cosmo_lcdm_equiv.state:
          cosmo_lcdm_equiv.struct_cleanup()
        cosmo_lcdm_equiv.empty()
        cosmo_lcdm_equiv.set(lcdm_equiv_params)
        cosmo_lcdm_equiv.compute()
        # Call CLASS again to get the lcdm equivalent
        Plin_equiv = np.zeros(len(k), "float64")
        h = cosmo_lcdm_equiv.h()
        for index_k in range(len(k)):
          Plin_equiv[index_k] = cosmo_lcdm_equiv.pk_lin(k[index_k]*h, 0.0)
        Plin_equiv *= h**3
        # Erase new parameters
        cosmo_lcdm_equiv.struct_cleanup()
        cosmo_lcdm_equiv.empty()

        # Calculate area criterion
        from scipy.interpolate import InterpolatedUnivariateSpline
        kPlcdm_interp = InterpolatedUnivariateSpline(k,k*Plin_equiv,k=5)
        kP_interp = InterpolatedUnivariateSpline(k,k*Plin,k=5)
        P1Dlcdm = np.empty_like(k[:-1])
        P1D = np.empty_like(k[:-1])
        for i, kval in enumerate(k[:-1]):
          P1Dlcdm[i] = 1./(2.*np.pi)*kPlcdm_interp.integral(kval,k[-1])
          P1D[i] = 1./(2.*np.pi)*kP_interp.integral(kval,k[-1])
        ratio = InterpolatedUnivariateSpline(k[:-1],P1D/P1Dlcdm)
        area = ratio.integral(self.area_criterion_kmin,self.area_criterion_kmax)
        area_lcdm = self.area_criterion_kmax-self.area_criterion_kmin
        self.area_criterion = (area_lcdm-area)/area_lcdm
        data.derived_lkl["area_criterion"] = self.area_criterion

        # Calculate T(k) (ratio between model and lcdm equivalent
        Tk = np.zeros(len(k), "float64")
        Tk = np.sqrt(abs(Plin)/abs(Plin_equiv))


        # Second sanity check: check that for small values of k (below k_eq), the T(k) asymptotes to 1 within 1%
        k_eq_der=cosmo.get_current_derived_parameters(["k_eq"])
        k_eq=k_eq_der["k_eq"]/h
        self.err_lkl = 0.
        if any(abs(Tk[k<np.maximum(k_eq,k[0])]**2-1.0)>0.01):
          self.err_lkl = self.write_error_to_binfile(data,"Error_equiv")
          if not self.TEST_nodata_run:
            if self.verbose > 0:
              print("FINISHED abgd LOGLKL CALC on Error_equiv")
            return self.err_lkl


        # Different smoothing options (this is mainly to deal with possible oscillations at high k)
        k_fit = k.copy()
        Tk_fit = Tk.copy()

        # First smoothing method: increase smoothing with log(k)
        if self.smoothing_method == "increasing_smoothing_logk":
          mask = (1.-Tk_fit)>0.5*(1-Tk_fit[-1])
          delta_logk = np.log10(k_fit)[:,None] - np.log10(k_fit)
          sigmas = np.zeros_like(k_fit)+1e-10
          k_first = k_fit[mask][0]
          width = self.smoothing_scale_logk
          sigmas = 1e-10+(0.5+0.5*np.tanh((np.log(k_fit)-np.log(k_first))/width))*self.smoothing_scale_logk
          weights_smoothing = np.array([np.exp(-delta_logk[i]*delta_logk[i]/(2*sigmas[i]*sigmas[i])) for i in range(len(delta_logk))])
          weights_smoothing/=np.sum(weights_smoothing,axis=1,keepdims=True)

        # Second smoothing method: continuous smoothing with log(k)
        if self.smoothing_method == "logk_smoothing":
          delta_logk = np.log10(k_fit)[:,None] - np.log10(k_fit)
          sigma = self.smoothing_scale_logk
          weights_smoothing = np.exp(-delta_logk*delta_logk/(2*sigma*sigma))
          weights_smoothing/=np.sum(weights_smoothing,axis=1,keepdims=True)
        if self.smoothing_method == "logk_smoothing" or self.smoothing_method == "increasing_smoothing_logk":
          Tk_fit_smoothed = np.sqrt(np.dot(weights_smoothing,Tk_fit*Tk_fit))

        # Third smoothing method: fit an alpha-beta-gamma-delta curve to model
        if self.smoothing_method == "fit_abgd":
          logk_fit = np.log(k_fit)
          try:
            from ManyPoly import ManyPoly
          except:
            from .ManyPoly import ManyPoly
          mnyply = ManyPoly(logk_fit,Tk_fit,self.logk_scale_abgd_fit,degree=3)
          ddTk = mnyply.derivative(2).eval()
          idx_slope_bend = np.argmax(ddTk>self.d2_Tk_dlnk2_abgd_limit)
          if idx_slope_bend == 0:
            # Model has no suppression before the end of the k range: it is a model with very small alpha
            guess_alpha = 0.1/k_fit[-1]
            guess_delta = 0.
            guess_beta = 0.5
            guess_gamma = -0.5
          else:
            # Model has suppression before the end of the k range: alpha ~ 1/k_50
            guess_k_50 = k_fit[idx_slope_bend]
            guess_delta = Tk_fit[-1]
            guess_alpha = 1./guess_k_50
            guess_beta = 2.0
            guess_gamma = -2.0
          guess_abgd=[guess_alpha,guess_beta,guess_gamma,guess_delta]

          import scipy.optimize
          try:
            params,pcov = scipy.optimize.curve_fit(self.T_abgd,k_fit,Tk_fit,p0=guess_abgd,maxfev=len(k_fit)*200)
          except RuntimeError:
            if self.verbose > 0:
              print("FINISHED abgd LOGLKL CALC on Error_abgd_fit")
            return self.write_error_to_binfile(data,"Error_abgd_fit")
          Tk_fit_smoothed = self.T_abgd(k_fit,*params)

        if self.smoothing_method == None or self.smoothing_method == "None":
          Tk_fit_smoothed = Tk_fit
        if np.any(np.abs(Tk_fit**2-Tk_fit_smoothed**2)>0.1):
          self.err_lkl = self.write_error_to_binfile(data,"Error_smooth")
          if not self.TEST_nodata_run:
            if self.verbose > 0:
              print("FINISHED abgd LOGLKL CALC on Error_smooth")
            return self.err_lkl

        Tk_fit = Tk_fit_smoothed


        # Different interpolation options (this is mainly to deal with interpolating in the sparse grid),
        # see 2206.08188 for more details.
        # First fitting option: calculate distance weights
        if not self.use_least_square == "bounded":
          diffs_ABG = np.empty((self.abg_grid_size,))
          diffs_ABD = np.empty((self.abd_grid_size,))
          diffs_THERMAL = np.empty((self.thermal_grid_size,))
          diffs_LCDM = np.empty((1,))
          for i in range(self.abg_grid_size):
            diffs_ABG[i] = np.sum((self.Tks_grid_abg[i]-Tk_fit)**2)**0.5
          for i in range(self.abd_grid_size):
            diffs_ABD[i] = np.sum((self.Tks_grid_abd[i]-Tk_fit)**2)**0.5
          for i in range(self.thermal_grid_size):
            diffs_THERMAL[i] = np.sum((self.Tks_grid_thermal[i]-Tk_fit)**2)**0.5
          for i in range(1):
            diffs_LCDM[i] = np.sum((self.Tks_grid_lcdm[i]-Tk_fit)**2)**0.5
          diffs_ALL = np.concatenate([diffs_ABG,diffs_ABD,diffs_THERMAL,diffs_LCDM])
          distance_weights = 1./(diffs_ALL+self.epsilon_tk)**self.exponent_tk
          distance_weights = distance_weights/np.sum(distance_weights)
        if self.use_least_square == None or self.use_least_square == "None":
          weights = distance_weights

        # Second fitting option: calculate least squares
        elif "hybrid" in self.use_least_square:
          from scipy.optimize import lsq_linear
          matrix = (self.Tks_grid_concat).T
          goal = Tk_fit
          weightweights = (diffs_ALL+self.epsilon_tk)**self.exponent_tk
          weightweights = weightweights/np.sum(weightweights)
          avgweightweight = np.average(weightweights)
          weightweights = np.diag(weightweights)
          # This takes a number: 0 = use leastsq, infty = use nearest
          use_lstsq_vs_nearest = self.use_lstsq_vs_nearest
          lambd = len(matrix)**2*1./avgweightweight**2*use_lstsq_vs_nearest
          try:
            if self.use_least_square == "hybrid":
              lstsq_output = lsq_linear(np.dot(matrix.T,matrix)+lambd*np.dot(weightweights.T,weightweights),np.dot(matrix.T,goal))
            elif self.use_least_square == "hybrid_bounded":
              lstsq_output = lsq_linear(np.dot(matrix.T,matrix)+lambd*np.dot(weightweights.T,weightweights),np.dot(matrix.T,goal),bounds=(0,1))
          except ValueError:
            if self.verbose > 1:
              print(matrix,goal,weightweights,lambd,avgweightweight)
            if self.verbose > 0:
              print("FINISHED abgd LOGLKL CALC on error (hybrid)")
            return self.write_error_to_binfile(data,("Error_hybrid_bounded_failed" if "bounded" in self.use_least_square else "Error_hybrid_failed"))
          weights = lstsq_output["x"]
          weights = weights/np.sum(weights)
        elif "regularized" in self.use_least_square:
          from scipy.optimize import lsq_linear
          matrix = (self.Tks_grid_concat).T
          goal = Tk_fit
          lambd = self.regularized_lambda
          try:
            if self.use_least_square == "regularized":
              lstsq_output = lsq_linear(np.dot(matrix.T,matrix)+lambd*np.diag(np.ones(len(matrix[0]))),np.dot(matrix.T,goal))
            elif self.use_least_square == "regularized_bounded":
              lstsq_output = lsq_linear(np.dot(matrix.T,matrix)+lambd*np.diag(np.ones(len(matrix[0]))),np.dot(matrix.T,goal),bounds=(0,1))
          except ValueError:
            if self.verbose > 1:
              print(matrix,goal,lambd)
            if self.verbose > 0:
              print("FINISHED abgd LOGLKL CALC on error (regularized)")
            return self.write_error_to_binfile(data,("Error_regularized_bounded_failed" if "bounded" in self.use_least_square else "Error_regularized_failed"))
          weights = lstsq_output["x"]
          weights = weights/np.sum(weights)
        elif self.use_least_square == "bounded":
          from scipy.optimize import lsq_linear
          try:
            lstsq_output = lsq_linear((self.Tks_grid_concat).T,Tk_fit,bounds=(0,1))
          except ValueError:
            if self.verbose > 0:
              print("FINISHED abgd LOGLKL on error (bounded lst_sq)")
            return self.write_error_to_binfile(data,"Error_bounded_failed")
          weights = lstsq_output["x"]
          weights = weights/np.sum(weights)
        else:
          raise io_mp.LikelihoodError("use_least_square method not understood!")
        if self.TEST_nodata_run:
          self.N_close = np.count_nonzero(weights>=self.weight_min)
          self.closest_sims = np.argsort(weights)[::-1][:self.N_close]

        # Store un-seperated weights
        self.weights = weights
        weights = np.split(weights,[self.abg_grid_size,self.abg_grid_size+self.abd_grid_size,self.abg_grid_size+self.abd_grid_size+self.thermal_grid_size])

        mask = [weights[typ]>=self.weight_min for typ in range(len(weights))]
        weighted_Tk_tot = np.zeros(len(self.k))
        for typ in range(len(weights)):
          weighted_Tk_tot+=np.sum(weights[typ][:,np.newaxis]*self.Tks_grid[typ],axis=0)

        # Store different weight sums as derived parameters, for possible checks
        data.derived_lkl["weight_largest"] = np.max(self.weights)
        data.derived_lkl["weightsum_2"] = np.sum(np.sort(self.weights)[::-1][:2])
        data.derived_lkl["weightsum_4"] = np.sum(np.sort(self.weights)[::-1][:4])
        self_distance_weighted = np.max(np.abs(weighted_Tk_tot**2-Tk**2))
        data.derived_lkl["self_distance_weighted"] = self_distance_weighted

        # In no_data mode, return some exit messages
        if self.TEST_nodata_run:
          if self.verbose > 0:
            if self.err_lkl != 0.0:
              print("FINISHED abgd LOGLKL CALC on some error")
            else:
              print("FINISHED abgd LOGLKL CALC in nodata mode")
          return self.err_lkl

        # If the model has passed all the sanity checks, do the final chi2 computation
        chi2=0

        # Calculate the actual model for given astro params, cosmo params, and weights
        model_H = np.zeros (( len(self.zeta_range_mh), len(self.k_mh) ), "float64")
        model_M = np.zeros (( len(self.zeta_range_mh)-1, len(self.k_mh) ), "float64")
        theta=np.array([z_reio,sigma8,neff,astro_pars['F_UV'],astro_pars['Fz1'],astro_pars['Fz2'],astro_pars['Fz3'],astro_pars['Fz4'],astro_pars['T0a'],astro_pars['T0s'],astro_pars['gamma_a'],astro_pars['gamma_s']])
        model = self.PF_noPRACE*self.ordkrig(theta, self.redshift_list, weights)
        upper_block = np.vsplit(model, [7,11])[0]
        lower_block = np.vsplit(model, [7,11])[1]
        model_H[:,:] = lower_block[:,19:]
        model_H_reshaped = np.reshape(model_H, -1, order="C")
        model_M[:,:] = lower_block[:3,19:]
        model_M_reshaped = np.reshape(model_M, -1, order="C")
        model_MH_reshaped = np.concatenate((model_H_reshaped,model_M_reshaped))

        chi2 = np.dot((self.y_MH_reshaped - model_MH_reshaped),np.dot(self.cov_MH_inverted,(self.y_MH_reshaped - model_MH_reshaped)))

        loglkl = - 0.5 * chi2

        return loglkl
