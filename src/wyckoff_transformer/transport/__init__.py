"""Li-ion transport descriptors and labels for the family-blind SE search.

The plan is ``docs/se_new_families_campaign_plan.md``; the campaign driver is
``scripts/se_new_families_campaign.py``.

* :mod:`.sources` -- the labelled sets: OBELiX (experimental) and Kahle 2020 (AIMD);
* :mod:`.frameworks` -- the Li-free, anonymised framework of a structure and the
  matcher that groups structures by it;
* :mod:`.pes` -- frozen-framework and empty-framework PES scans with a universal
  potential, and the descriptors read from them (Maevskiy et al. 2025's Xi, MPE and
  free volumes; the ICML pipeline's empty-framework percolation barrier);
* :mod:`.descriptors` -- the cheap structure descriptors (BVSE barriers, anion
  framework, corner-sharing, Sendek's model, Li polyhedra);
* :mod:`.scores` -- partial ROC AUC with label weights, bootstrap intervals and the
  rediscovery percentiles the gates are read from.
"""
