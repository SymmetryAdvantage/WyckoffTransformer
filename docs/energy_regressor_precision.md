# Energy regressor precision on archive and generated structures

Measured 2026-10-28 at git `66a4b3b592e343dacf76cddb7db37a2b89084a60`. [W&B analysis run](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/rycc5sdi). Model: `SymmetryAdvantage/WyFormer-LeMat-Energy` at `1efdb3b730d14d424679a612a8edccbb08725b3b`; training run [`min_energy_adamw_wsd_h7x_xl-20261015-185731`](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/min_energy_adamw_wsd_h7x_xl-20261015-185731). The analysis code and all derived tables are preserved in the run's `energy_regressor_evaluation` artifact. No new ORB or DFT calculation was run.

Archive test MAE is **24.4 meV/atom** against the intended DFT gene-minimum target and **83.5 meV/atom** against ORB. On generated DFT structures, MAE is **22.8 meV/atom** for structures known to LeMat-Bulk-fixed and **123.8 meV/atom** for novel structures. Precision measured on the archive therefore does not transfer directly to novel generated structures.

The model predicts the lowest observed **DFT formation energy of a gene**, rather than the energy of every geometry realising it. Errors against individual structures therefore include the energy gap between that structure and the gene's observed minimum. ORB errors additionally include the difference between ORB's OMat24 scale and the model's MP PBE/PBE+U scale.

## Energy normalization and coverage

All energies and errors below are in **eV/atom**. Raw total energies are divided by the atom count of the saved structure, then referenced to the same elemental minima as the model: LeMat-Bulk-fixed `compatible_pbe@bf8f0b31`. This common subtraction preserves the residual against raw total energy per atom exactly. No MP2020 correction, fitted offset, hull-distance substitution or clipping is applied. Bias is prediction minus observation. ORB comparisons are explicitly cross-scale; `energy_fields.check_compatible` records the differences in `provenance.json`. Predictions average 16 random equivalent descriptions.

The W&B inventory contains 80 collections; 79 provide the required protocol files. One collection (`se-families-hera-20261022-protocol`) lacks `manifest.json` and is listed as unavailable. Every source artifact is pinned by version, ID and digest in `protocol_inventory.json`. Tables retain failed rows as exclusions. Stale CIFs attached to rows without a successful saved structure and finite energy are ignored. Successful pairs are checked for equal composition and atom count.

Composition audit: 0 mismatched successful pairs; 0 excluded cohorts. Unsupported model vocabulary and unresolved symmetry remain explicit missing predictions; denominators below count finite prediction–target pairs.

## A Archive ORB energies

Source: [LeMat-Bulk-MLIP-Hull-All](https://huggingface.co/datasets/LeMaterial/LeMat-Bulk-MLIP-Hull-All), pinned in `archive_provenance.json`. Uniform samples of 5,000 IDs from each fixed-dataset train/validation/test split, seed 20261028, were selected before filtering labels. Training membership here means membership in the regressor's split; ORB's own training membership has not been established. The gene-minimum labels were built over all three splits, as recorded by the model, so a held-out row is not necessarily a held-out gene.

| Split | N | ORB MAE | ORB RMSE | ORB bias | DFT gene minimum MAE | DFT structure MAE | DFT–ORB MAE baseline |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | 5000 | 0.0752 | 0.1704 | -0.0491 | 0.0085 | 0.0154 | 0.0730 |
| val | 5000 | 0.0821 | 0.1787 | -0.0514 | 0.0251 | 0.0309 | 0.0711 |
| test | 5000 | 0.0835 | 0.1859 | -0.0516 | 0.0244 | 0.0316 | 0.0722 |

## B Saved protocol ORB energies

The unit is the saved best structure for each unique initial gene in each track, rather than each relaxation trial or sampled duplicate. Fixed-symmetry and free tracks may choose different trials. Initial-gene predictions measure screening precision before relaxation; final-gene predictions measure inference from the gene recovered from the saved geometry. `same_gene` uses the augmentation-invariant gene identity. Final genes are novel only if absent from all splits of the protocol's recorded reference.

Structure novelty uses the saved protocol's `novel_structure` label. Initial and final gene novelty are recomputed against that protocol reference. 150 initial-gene labels differ from the saved labels under the current augmented identity. The sources use upstream LeMat-Bulk (76 cohorts) or LeMat-Bulk-fixed (3 cohorts); reference-specific and per-source results are included in `summary.csv`. The pooled tables describe the available evaluations and are not an estimate over independent draws: some arms share cohorts or geometries. Deduplicated summaries are supplied separately.

| Track | Predictor gene | N | Excluded | MAE | RMSE | Bias |
| --- | --- | --- | --- | --- | --- | --- |
| fixed_symmetry | initial | 77586 | 246 | 0.1710 | 0.3922 | -0.1260 |
| fixed_symmetry | final | 77586 | 246 | 0.1675 | 0.3901 | -0.1233 |
| free | initial | 77586 | 246 | 0.1314 | 0.3560 | -0.0271 |
| free | final | 77585 | 247 | 0.1196 | 0.3399 | -0.0536 |

### Marginal splits

| Track | Axis | Value | N | MAE | Bias |
| --- | --- | --- | --- | --- | --- |
| fixed_symmetry | initial gene novel reference | False | 27785 | 0.1212 | -0.1009 |
| fixed_symmetry | initial gene novel reference | True | 49801 | 0.1988 | -0.1400 |
| fixed_symmetry | novel structure | False | 21582 | 0.0793 | -0.0545 |
| fixed_symmetry | novel structure | nan | 6003 | 0.1942 | -0.1436 |
| fixed_symmetry | novel structure | True | 50001 | 0.2078 | -0.1547 |
| free | initial gene novel reference | False | 27785 | 0.1098 | -0.0458 |
| free | initial gene novel reference | True | 49801 | 0.1434 | -0.0167 |
| free | novel structure | False | 20144 | 0.0837 | -0.0447 |
| free | novel structure | nan | 6010 | 0.1454 | -0.0148 |
| free | novel structure | True | 51432 | 0.1484 | -0.0217 |
| free | final gene novel reference | False | 23114 | 0.0925 | -0.0533 |
| free | final gene novel reference | True | 54472 | 0.1479 | -0.0160 |
| free | final gene novel reference | nan | 0 | — | — |
| free | same gene | False | 38372 | 0.1621 | 0.0068 |
| free | same gene | True | 39214 | 0.1013 | -0.0603 |
| free | same gene | nan | 0 | — | — |

`True` means novel for the novelty axes and unchanged for `same_gene`; `nan` means unavailable. The 6,003 fixed-symmetry and 6,010 free finite-energy rows without a structure-novelty label fail the saved structural-validity check; they remain a separate category. Restricting to valid structures gives initial-gene MAEs 0.1691 (fixed) and 0.1302 (free), and final-gene free MAE 0.1190. All joint intersections of initial gene novelty, structure novelty, final gene novelty and gene preservation, with track and reference, are in `summary.csv` (`b: joint split` and `b: per source joint split`). Both initial and final predictions are tabulated for each cell, including counts, MAE, RMSE, bias, median and 90th-percentile absolute error, correlations, and fractions within 50 and 100 meV/atom. `rows.csv.gz` preserves all row labels.

## C Figshare DFT energies

Source: [Figshare 29145101 version 6](https://figshare.com/articles/dataset/Generated_crystals_for_WyFormer_DiffCSP_DiffCSP_WyCryst_SymmCD_CrystalFormer_MiAD/29145101/6). All 24 DFT cohorts are inventoried. Raw `dft_e_uncorrected` or `energy_dft` is used with the saved final structure. The 94-row `DFT-PBE-54` cohort lacks raw total energies and a recoverable formation-energy convention, so it has no numeric target here. Other rows without readable model vocabulary are also excluded explicitly.

Structure novelty is computed both against LeMat-Bulk-fixed (all regressor-dataset splits, augmented fingerprint then default `StructureMatcher`) and against MP-20 (all benchmark splits, composition then default `StructureMatcher`). The former asks whether a structure is known to the regressor's corpus; the latter follows the generated-data benchmark. Symmetry recovery uses the training tolerances 0.1 Å and 5°. Fingerprint-gated novelty inherits the protocol's candidate-gating limitation: matches with inconsistent detected fingerprints may be missed.

| Novelty reference | Novel | N | Excluded | MAE | RMSE | Bias |
| --- | --- | --- | --- | --- | --- | --- |
| fixed | False | 17506 | 22 | 0.0228 | 0.1235 | -0.0055 |
| fixed | True | 38614 | 79 | 0.1238 | 0.2514 | -0.0372 |
| mp20 | False | 12198 | 3 | 0.0328 | 0.1071 | 0.0081 |
| mp20 | True | 43922 | 98 | 0.1088 | 0.2418 | -0.0371 |

### DFT cohorts split by novelty against MP-20

| Cohort | Known N | Known MAE | Novel N | Novel MAE |
| --- | --- | --- | --- | --- |
| mp_20/Compositional/DFT | 145 | 0.0784 | 848 | 0.1794 |
| mp_20/Crys-JEPA/DFT | 149 | 0.0480 | 849 | 0.0641 |
| mp_20/CrystalFormer/DFT | 7 | 0.0096 | 82 | 0.1477 |
| mp_20/DiffCSP++/DFT | 3 | 0.0320 | 100 | 0.1218 |
| mp_20/DiffCSP/1k-sample/CHGNet_free/DFT | 144 | 0.0185 | 848 | 0.0896 |
| mp_20/DiffCSP/1k-sample/DFT | 134 | 0.0275 | 858 | 0.0925 |
| mp_20/DiffCSP/1k-sample/eq-V2_free/DFT | 147 | 0.0138 | 840 | 0.0996 |
| mp_20/DiffCSP/DFT | 7 | 0.0109 | 97 | 0.0801 |
| mp_20/MatterGen/MatterGen_10k/CHGNet_fix/DFT | 1 | 0.0247 | 102 | 0.1375 |
| mp_20/MatterGen/MatterGen_10k/DFT-10k | 1235 | 0.0206 | 8562 | 0.0906 |
| mp_20/MatterGen/MatterGen_10k/DFT | 3 | 0.0106 | 101 | 0.1078 |
| mp_20/MiAD/CHGNet_free/DFT-GGA-relax-1 | 3790 | 0.0407 | 6097 | 0.0785 |
| mp_20/MiAD/CHGNet_free/DFT | 3779 | 0.0345 | 6073 | 0.0737 |
| mp_20/SymmCD/CHGNet_free/DFT | 149 | 0.0230 | 813 | 0.1653 |
| mp_20/SymmCD/DFT | 5 | 0.0150 | 91 | 0.1917 |
| mp_20/WyckoffLLM-naive/DiffCSP++/DFT | 4 | 0.0087 | 98 | 0.1416 |
| mp_20/WyckoffTransformer-harmonic/DiffCSP++/DFT | 8 | 0.0125 | 93 | 0.1632 |
| mp_20/WyckoffTransformer-letters/DiffCSP++/DFT | 3 | 0.0064 | 101 | 0.1854 |
| mp_20/WyckoffTransformer/CrySPR/CHGNet_fix/DFT | 4 | 0.0148 | 100 | 0.1321 |
| mp_20/WyckoffTransformer/DFT-PBE-54 | 0 | — | 0 | — |
| mp_20/WyckoffTransformer/DiffCSP++/DFT | 6 | 0.0131 | 98 | 0.1279 |
| mp_20/WyckoffTransformer/DiffCSP++10k/CHGNet_free/DFT-GGA-relax-1 | 1245 | 0.0247 | 8575 | 0.1399 |
| mp_20/WyckoffTransformer/DiffCSP++10k/CHGNet_free/DFT | 1227 | 0.0233 | 8396 | 0.1346 |
| mp_20_biternary/WyCryst/CrySPR/CHGNet_fix/DFT | 3 | 0.0191 | 100 | 0.2111 |

## Reproduction and saved outputs

Use the platform's documented interpreter from the checkout root. The stages are `prepare`, `sample`, `symmetry`, `infer`, `novelty`, and `report` in `scripts/evaluate_energy_regressor.py`; finish `sample` and `symmetry` before the final `infer`. Pin the model and ORB dataset revisions from the saved provenance. `prepare --source-inventory protocol_inventory.json` selects the saved exact artifact versions; `prepare --reuse-inventory` resumes local assembly. No stage loads an MLIP calculator or launches DFT.

The W&B artifact contains the source inventory, archive sample IDs and labels, Figshare metadata, common elemental references, model field provenance, gene registry, individual predictions, recovered symmetry, `rows.csv.gz`, `summary.csv`, the error-distribution figure (PNG/PDF), verification record, this report and code. The local run directory is a working copy; W&B holds the recoverable results. Large JSON files are gzip-compressed in the artifact.
