This directory contains the likelihood module for a KiDS-Legacy cosmic shear analysis with COSEBIs measurements from [Wright et al. 2025](http://arxiv.org/abs/2503.19441).

Requires Class >=3.3.0 for the calculation of non-linear corrections with hmcode2020.

This likelihood was adapted from earlier KiDS MontePython likelihoods and should run out of the box without installing external packages. It uses COSEBIs as summary statistic, the 
fiducial mass-dependent intrinsic alignment model, and the fiducial scale cuts (2'<theta<300'). 
The code for calculating COSEBIs from shear Cls was adopted from: https://github.com/KiDS-WL/kcap/blob/v2/utils/bandpower_cosebis.py.

This likelihood should reproduce the results from the fiducial CosmoSIS pipeline, available on the [KiDS-Website](https://kids.strw.leidenuniv.nl/sciencedata.php) and the [CosmoSIS standard library](https://github.com/cosmosis-developers/cosmosis-standard-library/blob/main/examples/KiDS-Legacy.ini).

To run the full KiDS-Legacy analysis pipeline see: https://github.com/AngusWright/CosmoPipe

If you are interested in using KiDS-Legacy data products other than COSEBIs:
See the data products on the [KiDS-Website](https://kids.strw.leidenuniv.nl/sciencedata.php)
These include chains, data files, and cosmosis ini files for COSEBIs, band powers, and 2PCFs, including a joint analysis with external probes
and featuring a pre-trained cosmopower emulator and the scale-cuts module for selecting a different number of COSEBI modes, band power bins, and 2PCF bins

Please cite the following papers when using this data:
- [Wright et al. 2025](https://arxiv.org/abs/2503.19439) (A&A, 686, A170)    [KiDS DR5 data release paper]
- [Wright et al. 2025](http://arxiv.org/abs/2503.19441) (A&A, 703, A158)    [KiDS-Legacy cosmic shear analysis]
- [Stölzner et al. 2025](http://arxiv.org/abs/2503.19442) (A&A, 702, A169)  [KiDS-Legacy consistency and joint constraints with external probes]
- [Reischke et al. 2025](https://arxiv.org/abs/2410.06962) (A&A, 699, A124)  [KiDS-Legacy covariance]
- [Wright et al. 2025](https://arxiv.org/abs/2503.19440) (A&A, 703, A144)    [KiDS Legacy redshift calibration]

and include the following acknowledgement in your paper:
- "Based on observations made with ESO Telescopes at the La Silla Paranal Observatory under programme
   IDs 179.A-2004, 177.A-3016, 177.A-3017, 177.A-3018, 298.A-5015."

See also our follow-up analyses:
- [Reischke et al 2025](https://arxiv.org/abs/2512.11041) (arXiv: 2512.11041) [Constraints on dark energy, neutrino mass, and curvature]
- [Stölzner et al 2025](https://arxiv.org/abs/2512.11039) (arXiv: 2512.11039) [Constraints on Horndeski gravity]

WARNING: This likelihood only produces valid results for `\Omega_k = 0`, i.e. flat cosmologies!
