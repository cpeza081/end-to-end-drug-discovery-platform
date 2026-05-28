"""
dd_prep — Deep Docking library preparation pipeline.
 
Automates the sequential preparation of a SMILES chemical library
for use with the Deep Docking (DD) virtual screening platform
(Gentile et al., Nature Protocols 2022).
 
Pipeline stages (each independently toggleable):
  1. filter      — RDKit property-based pre-filtering
  2. split       — chunk the library into manageable files
  3. flipper     — OpenEye stereoisomer enumeration
  4. tautomer    — OpenEye dominant tautomer / protonation state
  5. organize    — rename and collect into library_prepared/
  6. fingerprint — RDKit Morgan FP calculation → library_prepared_fp/
  7. omega       — OpenEye 3-D conformer generation (optional)
"""
 
__version__ = "0.1.0"
__author__ = "dd_prep contributors"