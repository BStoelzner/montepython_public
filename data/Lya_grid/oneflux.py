import numpy as np
import pickle
import os

epsilon_astro = 1e-8
exponent_astro = 6.
pkl = open("full_matrix_interpolated_ABG.pkl", 'r')
astromat = pickle.load(pkl)
#redshift, wavenumber, simnumber
print(astromat.shape)
grid_length_ASTRO = astromat.shape[-1]
np.set_printoptions(threshold=np.nan,linewidth=800)
np.savetxt("ABG81_flux.txt",astromat[:,:,81])


