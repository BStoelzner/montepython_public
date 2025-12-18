This directory contains data for a KiDS-Legacy cosmic shear analysis with COSEBIs measurements from [Wright et al. 2025](http://arxiv.org/abs/2503.19441).

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

This folder contains a fits file with the following extensions:
- En, Bn: COSEBIs E- and B modes
- COVMAT: Covariance matrix of COSEBIs E- and B modes
- NZ_SOURCE: Redshift distribution per tomographic bin

Additionally, we provide the dz covariance matrix in ascii format, the covariance matrix of IA parameters (see Sect. 2.4.4 and Appendix B in [Wright et al. 2025](https://arxiv.org/abs/2503.19441)), and COSEBIs window functions.


WARNING: This likelihood only produces valid results for `\Omega_k = 0`, i.e. flat cosmologies!
