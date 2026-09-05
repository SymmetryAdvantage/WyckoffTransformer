CRySPR experiments/improvements
0. Allocate relaxation budget per DoF (confirmed via a study, TBD)
1. Rattling to break symmetry (iapetus running)
2. Fix volume inflation (pending zeus study, likely combined with the full lattice prediction)
3. Conditioning audit - new robust protocol on converged model

Backbone
1. Train the models longer with the low LR
2. Increase model capacity (NSCC, in progress)
3. Predict lattice
4. Better SS representation (NSCC, in progress)
5. Wyckoff Regressor - censored min(E|gene) likelihood implemented, untrained (docs/csp_mode.md)
6. Architecture improvement - pooling, log-z, conditioning
7. Enrich internal representation - technical and physical
8. Beam search decoder - composition-constrained decoder implemented (docs/csp_mode.md)
9. Inverse cascade - sample elements, then Wyckoffs
10. Condition on E - min(E, polymorphs); gene-level label implemented, backbone untrained

Dataset
1. Dataset selection: full, deduplicated Wyckoff genes, E_hull <= 0.1 eV

Postprocessing
1. Built-in novelty filter
