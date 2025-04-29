BAO+ likelihoods for eBOSS DR16

These are what were used for the headline results from the Alam et al. DR16 cosmology paper. We recommend people use the same for testing cosmology.

***IMPORTANT***

If you are using BOSS DR12 results (as you should) *only* use the likelihood for first two redshift bins. The eBOSS results combine with BOSS results for z > 0.6. The first two BOSS DR12 results span 0.2 < z < 0.6 (split into 0.2 < z < 0.5 and 0.4 < z < 0.6). For user convenience, the BOSS DR12 results corresponding to these two redshift bins have been extracted and are provided bundled with this DR16 release. The fsigma8(z) values are slightly different that those from the BOSS release, as we have applied a rescaling factor that accounts for the maximum likelihood for the geometrical factors, relative to the template used to extract the fsigma8(z) values. See the appendix of Alam et al. (2020) for more details.


 * Format for likelihoods with "DMDHfs8" in the name *

These are Gaussian approximations to the joint likelihood of D_M(zeff)/r_d, D_H(zeff)/r_d, and f(zeff)sigma_8(zeff) (definitions below) as determined from combining BAO and full-shape likelihoods. For quasars, this matches the full-shape only likelihood, as the quasars do not use reconstruction. For LRGs, BAO measurements are obtained from the post-reconstruction clustering data and the full-shape measurements are obtained from the pre-reconstruction data. The two methods are combined statistically, factoring in their correlations. There is one file for the data vector and a separate one for the covariance matrix.

r_d is the sound horizon at the drag epoch (e.g., as determined by camb.) It is 147.8 Mpc in the fiducial cosmology used by eBOSS.

D_M(zeff) is the co-moving angular diameter distance (which is the same as the co-moving distance in a flat cosmology).

D_H(zeff) = c/H(zeff), where c is the speed of light and H(z) is the expansion rate.

f(z)sigma_8(z), is the amplitude of the velocity power spectrum, given sigma_8(z) is the normalization of the linear power spectrum on scales of 8Mpc/h at redshift z and f is the linear growth rate of structure, f = d D /d ln(a) (with a the scale factor, i.e. a = 1/[1+z]).

 * Format for likelihoods with "DVfs8" in the name *

Same as above, except D_V(zeff)/r_d is used. D_V(zeff) is the spherically averaged distance defined as

D_V(zeff) = [zD_H(z)D^2_M(z)]^(1/3)

 * Format for ELG likelihood *

The ELG likelihood is not well-approximated as Gaussian. Therefore, a grid of the relative probability as function of D_M(zeff)/r_d, D_H(zeff)/r_d, and f(zeff)sigma_8(zeff) is used.

The result is based on a simultaneous fit to the isotropic BAO position in the post-reconstruction monopole and the full-shape pre-reconstruction monopole, quadrupole and hexadecapole .

 * Lyman-alpha likelihood *

The DR16 Lyman-alpha BAO likelihoods are provided separately for the auto (LYAUTO) and cross correlation (LYXQSO). They can be treated as independent. The format for each is:

Column 1:  (DM/rd) at z=2.334
Column 2:  (DH/rd) at z=2.334
Column 3:  likelihood (relative to the best point on the grid)

The Lyman-alpha likelihoods thus contain only the geometrical information (D_M(zeff)/r_d, D_H(zeff)/r_d). The files in this folder match those in the BAO only folder.

 * Previous SDSS results *

The BOSS DR12 likelihoods bundled in this release correspond to the two lower redshift bins of the originally published likelihood. These are independent of the DR16 results provided here; the DR16 LRG results supersede the original DR12 results for the 0.5 < z < 0.75 redshift bin.One can compare the BOSS results we have included to the published BOSS DR12 likelihoods available from https://data.sdss.org/sas/dr12/boss/papers/clustering/ALAM_ET_AL_2016_consensus_and_individual_Gaussian_constraints.tar.gz . . As mentioned above, the fsigma8(z) results are slightly different due to the application of a rescaling factor.

The BAO+full shape likelihood for the DR7 Main Galaxy Sample (MGS), designed to be independent from BOSS (z < 0.2), is based on the from Howlett et al. (2015) and was provided to us by Cullan Howlett via private communication. It combines the post-reconstruction estimate of D_V/r_d with fs8.
