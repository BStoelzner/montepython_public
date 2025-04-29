BAO likelihoods to be used for cosmology likelihood codes for eBOSS DR16

***IMPORTANT I***

If you are using BOSS DR12 results (as you should) *only* use the likelihood for first two redshift bins. The eBOSS results combine with BOSS results for z > 0.6. The first two BOSS DR12 results span 0.2 < z < 0.6 (split into 0.2 < z < 0.5 and 0.4 < z < 0.6). For user convenience, the BOSS DR12 results corresponding to these two redshift bins have been extracted and are provided bundled with this DR16 release.


 * Format for likelihoods with "DMDH" in the name *

These are Gaussian approximations to the joint likelihood of D_M(zeff)/r_d and D_H(zeff)/r_d (definitions below) as determined from BAO measurements. There is one file for the data vector and a separate one for the covariance matrix.

r_d is the sound horizon at the drag epoch (e.g., as determined by camb.) It is 147.8 Mpc in the fiducial cosmology used by eBOSS.

D_M(zeff) is the co-moving angular diameter distance (which is the same as the co-moving distance in a flat cosmology).

D_H(zeff) = c/H(zeff), where c is the speed of light and H(z) is the expansion rate.


 * Format for likelihoods with "DVtable" in the name *

For the ELG sample, we provide the likelihood in terms of D_V(zeff)/r_d at zeff = 0.845. The likelihood is normalized to have a maximum of 1.

D_V(zeff) is the spherically averaged distance defined as

[zD_H(zeff)D_M(zeff)^2]^(1/3)

 * Lyman-alpha likelihood *

The DR16 Lyman-alpha BAO likelihoods are provided separately for the auto (LYAUTO) and cross correlation (LYXQSO). They can be treated as independent. The format for each is:

Column 1:  (DM/rd) at z=2.334
Column 2:  (DH/rd) at z=2.334
Column 3:  likelihood (relative to the best point on the grid)


 * Previous SDSS results *

The BOSS DR12 likelihoods bundled in this release correspond to the two lower redshift bins of the originally published likelihood. These are independent of the DR16 results provided here; the DR16 LRG results supersede the original DR12 results for the 0.5 < z < 0.75 redshift bin.

One can compare the BOSS results we have included to the published BOSS DR12 likelihoods available from https://data.sdss.org/sas/dr12/boss/papers/clustering/ALAM_ET_AL_2016_consensus_and_individual_Gaussian_constraints.tar.gz

The BAO likelihood for the DR7 Main Galaxy Sample (MGS), designed to be independent from BOSS (z < 0.2), is available within the source files here: https://arxiv.org/e-print/1409.3242 and also from the publisher here: https://academic.oup.com/mnras/article/449/1/835/1298372#supplementary-data
