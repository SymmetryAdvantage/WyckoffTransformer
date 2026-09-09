CRySPR experiments/improvements
0. Allocate relaxation budget per DoF (confirmed via a study, TBD)
1. Rattling to break symmetry (confirmed via a study, TBD)
2. Fix volume inflation (confusing - cryspr won't converge without factor=1.3)
3. Conditioning audit - new robust protocol on converged model
4. Template-matched starts instead of random PyXtal draws - implemented as
   `wyformer-protocol --stage template`, off by default. 81.5% recovery from one
   start on the 400-gene oracle cohort against 60.8% for ten random ones, but on
   e9ywwsie only MetaSUN 0.281 -> 0.293 for 17% more compute: what it recovers is
   mostly already in LeMat-Bulk (docs/cryspr_template_starts.md)

Backbone
1. Train the models longer with the low LR
2. Increase model capacity (NSCC, in progress)
3. Predict lattice
4. Better SS representation (NSCC, in progress)
5. Wyckoff Regressor - censored min(E|gene) likelihood implemented, untrained (docs/csp_mode.md)
6. Architecture improvement - pooling, log-z, conditioning
7. Enrich internal representation - technical and physical
8. Decoder (Beam search/energy selection/etc) - composition-constrained decoder implemented (docs/csp_mode.md)
9. Inverse cascade - sample elements, then Wyckoffs
10. Condition on E - min(E, polymorphs); structure-level label training as wjwmgjag (docs/dirty_data_conditioning.md), gene-level label implemented but untrained
11. Condition on the composition - implemented, CPU pilot only (yamls/models/lemat_bulk_ehull/ehull_composition.yaml)
12. CSP decoder picks z itself - union-over-z decoding, 3.6x less relaxation budget

Screening (composition level, docs/composition_screening_results.md)
1. Formula table and MP experimental provenance - done
2. Censored composition -> f*(X) model, two heads, deep ensemble - done; does not
   clearly beat plain MSE, and the label-noise constant matters more than the loss
3. Shallow-hull answer key - done, but it partly rewards predicting where
   Alexandria looked; use the generated-structure test instead
4. Comparison against g_C, g_D, g_D - g_C, Magpie+GBDT - done; g_D - g_C is the
   weakest thing tested
5. Prefilter on a real generation run - done; MetaSUN 2.23x in the top decile of
   novel formulas, SUN unmeasurable below ~10^4 genes
6. Neighbourhood density as a provenance feature - done, null in three arms
7. Displacement lower bound L(X) - implemented, not run at scale
8. Formula enumeration (smact) and shrinkage across the ranked list - not built
9. SUN at 10^4 genes, and composing with the gene-level critic - not started

Dataset
1. LeMat-bulk preprocessing cuts on force - drops nice ICSD structures. Replaced by
   lemat_bulk_fmax1: cut at 1 eV/A and condition on (E_hull, Delta_E_polymorph, f_max)
   instead of filtering (docs/dirty_data_conditioning.md)
2. Dataset selection: full, deduplicated Wyckoff genes, E_hull <= 0.1 eV

Postprocessing
1. Built-in novelty filter
