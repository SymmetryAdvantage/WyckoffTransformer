# Energy regressor precision on archive and generated structures

Measured 2026-10-28 at git `66a4b3b592e343dacf76cddb7db37a2b89084a60`. [W&B analysis run](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/rycc5sdi). Model: `SymmetryAdvantage/WyFormer-LeMat-Energy` at `1efdb3b730d14d424679a612a8edccbb08725b3b`; training run [`min_energy_adamw_wsd_h7x_xl-20261015-185731`](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/min_energy_adamw_wsd_h7x_xl-20261015-185731). The analysis code and all derived tables are preserved in the run's `energy_regressor_evaluation` artifact. No new ORB or DFT calculation was run.

Archive test MAE is **24.4 meV/atom** against the intended DFT gene-minimum target and **83.5 meV/atom** against ORB. On generated DFT structures, MAE is **22.8 meV/atom** for structures known to LeMat-Bulk-fixed and **123.8 meV/atom** for novel structures. Precision measured on the archive therefore does not transfer directly to novel generated structures.

The model predicts the lowest observed **DFT formation energy of a gene**, rather than the energy of every geometry realising it. Errors against individual structures therefore include the energy gap between that structure and the gene's observed minimum. ORB errors additionally include the difference between ORB's OMat24 scale and the model's MP PBE/PBE+U scale.

The diagnostic in §D separates these terms for known genes. On novel DFT structures
with known genes, the regressor's MAE against the archive minimum is only
**9.0 meV/atom**; the structure-to-minimum gap is **131.0 meV/atom**. The large
structure-energy error in that subgroup therefore comes primarily from the
difference in targets. The corresponding decomposition for novel genes remains
unmeasured because their minima are unknown.

The repeat retaining one lowest-energy observed structure per gene is in §E.

## Energy normalization and coverage

All energies and errors below are in **eV/atom**. Raw total energies are divided by the atom count of the saved structure, then referenced to the same elemental minima as the model: LeMat-Bulk-fixed `compatible_pbe@bf8f0b31`. This common subtraction preserves the residual against raw total energy per atom exactly. No MP2020 correction, fitted offset, hull-distance substitution or clipping is applied. Bias is prediction minus observation. ORB comparisons are explicitly cross-scale; `energy_fields.check_compatible` records the differences in `provenance.json`. Predictions average 16 random equivalent descriptions.

The W&B inventory contains 80 collections; 79 provide the required protocol files. One collection (`se-families-hera-20261022-protocol`) lacks `manifest.json` and is listed as unavailable. Every source artifact is pinned by version, ID and digest in `protocol_inventory.json`. Tables retain failed rows as exclusions. Stale CIFs attached to rows without a successful saved structure and finite energy are ignored. Successful pairs are checked for equal composition and atom count.

Original protocol composition audit (B): 0 mismatched successful initial/final pairs; 0 excluded cohorts. The additional C gene-to-geometry audit in §F finds nine mismatches in recovered gene composition and excludes them from the feature analysis. Unsupported model vocabulary and unresolved symmetry remain explicit missing predictions; denominators below count finite prediction–target pairs.

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

## D Diagnostic: regression error versus the archive-to-structure gap

Measured 2026-10-28 at git `b9e0689e34d58d524b3fcd997e7dcf7eb4b184c8`
with working-tree additions preserved as source in the
[diagnostic W&B run](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/53xvvvq7).
The model and training run are unchanged. Inputs are pinned to the original analysis
artifact `symmetry-advantage/WyckoffTransformer/energy_regressor_precision_rycc5sdi:v1`.
No new inference, relaxation, MLIP calculation or DFT calculation was run.

For a final gene present in LeMat-Bulk-fixed, let `m_archive` be its minimum DFT
formation energy across all three dataset splits, exactly as used in training.
The structure-energy residual decomposes as

```text
regressor_error = prediction - m_archive
structure_gap   = observed_structure_energy - m_archive
structure_error = regressor_error - structure_gap
```

The lookup was reconstructed from all 5,326,830 archive rows. Its build record
matches the model's recorded training cache exactly. Every reconstructed minimum
equals the stored training target; gene membership and the original archive-sample
targets are unchanged. The diagnostic saves all 75,399 matching reference rows,
covering 50,272 requested genes, with their IDs and split membership. It checks
the residual identity row by row and the squared-error identity, including its
cross term, in every pooled summary. Component MAEs do **not** add.

### DFT structures with known final genes

This table uses the same finite pairs for all three errors. **Errors are
meV/atom**, unlike the eV/atom tables above.

| Structure known to LeMat-fixed? | N | Distinct genes | Regressor versus archive minimum MAE | Structure versus archive minimum MAE | Regressor versus structure MAE |
| --- | --- | --- | --- | --- | --- |
| Yes | 17,506 | 9,340 | 7.2 | 20.0 | 22.8 |
| No | 1,028 | 672 | **9.0** | **131.0** | **130.0** |

For novel structures with known genes, the regressor still predicts its training
target accurately. An oracle that predicts each archive minimum exactly would
score **131.0 meV/atom**, essentially the same as the regressor's 130.0. Thus the
large structure-energy error in this subgroup is dominated by the difference
between the archive target and the generated structure's energy. It does not
demonstrate a comparable deterioration in prediction of the archive minimum.

Removing duplicate geometries leaves the novel-structure result unchanged.
Giving each gene equal weight gives regressor, structure-gap and total MAEs of
9.5, 143.9 and 142.9 meV/atom, respectively. Of the 1,028 novel structures, 1,006
have genes present in training: their regressor MAE is 8.3 meV/atom. The other
22 structures have genes confined to validation/test and regressor MAE
44.0 meV/atom; that small subgroup does not establish generalisation to new genes.

The structure gap is not uniformly positive. **23.2%** of the novel structures
lie more than 50 meV/atom below their archive gene minimum, **28.0%** lie more
than 50 meV/atom above it, and 48.8% lie within that interval. Its signed mean
is only -8.1 meV/atom despite its 131.0 meV/atom MAE. These observations rule out
a one-sided failure to reach an existing minimum as the sole explanation.
The gap includes differences in geometry and in the underlying calculations;
this diagnostic does not separate them. Additional sampling can find structures
below an archive's observed minimum, and calculation differences can also shift
the energy. The observed minimum must not be treated as a universal physical floor.

### Archive controls and limits of the diagnosis

In the original 5,000-row archive test sample, **696 rows (13.9%)** share a gene
with training. Their gene-minimum MAE is 7.6 meV/atom; the remaining 4,304 rows
have MAE 27.1 meV/atom, compared with 24.4 pooled. Shared genes improve the pooled
number, but removing them does not produce the approximately 124 meV/atom error
seen against novel generated DFT structures. These controls use saved DFT row
energies for the structure gap; the parent report's ORB target is retained
separately in the diagnostic rows.

The **37,586 finite DFT pairs with novel genes** have no archive minimum. Their
123.6 meV/atom total MAE therefore remains undecomposed. The known-gene result
cannot establish how much of that error comes from predicting a new gene's
minimum. Independent minimum searches with consistent DFT settings would be
needed to answer that question.

The artifact contains row-level residuals and exclusions, matched reference rows,
gene minima, per-cohort and per-gene results, duplicate controls, the error figure,
provenance and source code. ORB protocol decompositions are supplementary and
explicitly cross-scale: their structure gaps also contain the OMat24/MP scale
difference. Reproduce with `scripts/diagnose_energy_regressor.py`, using the
platform's documented interpreter. It reads the pinned W&B artifact and the
model's original dataset cache, and logs its outputs to W&B.

## E Repeat: one lowest-energy observed structure per gene

Measured 2026-10-28 at git `b9e0689e34d58d524b3fcd997e7dcf7eb4b184c8` with working-tree source preserved in the [W&B analysis run](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/bjfoqicx). The model and training run at the top of this report are unchanged. Inputs are pinned to `symmetry-advantage/WyckoffTransformer/energy_regressor_diagnostic_53xvvvq7:v2`; its lineage points to the original precision study.

The minimum is chosen by observed **energy per atom**, before filtering predictions or splitting novelty categories. Gene identity is the augmentation-invariant final gene recovered from the saved structure. Ties use the geometry identifier, then the pinned source-row ID. Every pool contains exactly one retained structure per resolved gene with a finite energy. Missing predictions remain exclusions. Rows without recoverable genes or energies are retained in the selection audit. Novelty labels belong to the selected structure; a gene is never selected separately inside known and novel structure categories.

ORB and DFT minima are computed separately. Protocol minima are pooled across all available cohorts within each relaxation track. Archive ORB minima are chosen across the original 15,000 sampled rows, then reported by the selected row's split; they are not minima over the full ORB archive. DFT minima are chosen across all Figshare cohorts. These are observed minima at the available search budgets, not estimates of global physical minima. Energy normalization, exclusions and the cross-scale ORB comparison follow the original study. The inherited pooling of raw DFT energies assumes the same nominal MP PBE/PBE+U scale; differences between calculations remain unresolved. Separate minima within each DFT cohort are also saved.

**Errors below are meV/atom.** Bias is prediction minus the retained observation. No new inference, relaxation, MLIP calculation or DFT calculation was run.

### Archive ORB minima

| Split | N | ORB MAE | ORB RMSE | ORB bias | DFT gene-min MAE | DFT structure MAE | DFT–ORB MAE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | 4998 | 75.2 | 170.5 | -49.1 | 8.5 | 15.4 | 73.0 |
| val | 4994 | 81.6 | 175.7 | -51.8 | 25.2 | 30.9 | 70.7 |
| test | 4997 | 83.5 | 185.9 | -51.6 | 24.4 | 31.6 | 72.2 |

### Protocol ORB minima

| Track | Predictor gene | N | Excluded | MAE | RMSE | Bias |
| --- | --- | --- | --- | --- | --- | --- |
| fixed_symmetry | initial | 73892 | 0 | 171.0 | 366.9 | -127.0 |
| fixed_symmetry | final | 73892 | 0 | 167.6 | 365.0 | -124.0 |
| free | initial | 74377 | 0 | 131.1 | 327.4 | -26.8 |
| free | final | 74376 | 1 | 119.5 | 310.2 | -53.4 |

Both predictor columns score the same structures selected by final gene. Initial-gene scores therefore include any gene change during relaxation. A companion selection by initial gene is in summary.csv.

### Generated DFT minima

| Novelty reference | Novel structure | N | Excluded | MAE | RMSE | Bias |
| --- | --- | --- | --- | --- | --- | --- |
| LeMat-fixed | False | 9328 | 0 | 21.7 | 119.7 | -3.9 |
| LeMat-fixed | True | 25503 | 1 | 122.2 | 262.3 | -29.9 |
| MP-20 | False | 6221 | 0 | 35.6 | 117.6 | 14.4 |
| MP-20 | True | 28610 | 1 | 108.3 | 250.9 | -31.1 |

### Including the archive minimum as an eligible structure

The generated-only pool retains the best generated observation for each gene. The second pool also offers that gene's lowest DFT structure from all LeMat-Bulk-fixed training-dataset splits. It contains the same eligible genes; archive-only genes are not added. No ORB energy enters this union. Archive IDs and splits are saved, and all such archive structures are known to LeMat-fixed. MP-20 novelty is left unavailable for archive candidates rather than inferred.

| Eligible observations | Novel gene | N | Excluded | MAE | RMSE | Bias |
| --- | --- | --- | --- | --- | --- | --- |
| Generated only | False | 9981 | 0 | 29.6 | 142.2 | -3.3 |
| Generated only | True | 24850 | 1 | 121.7 | 260.5 | -30.9 |
| Generated + archive | False | 9981 | 0 | 16.8 | 110.6 | 11.9 |
| Generated + archive | True | 24850 | 1 | 121.7 | 260.5 | -30.9 |

### Known-gene diagnostic after minimum selection

| Eligible observations | Novel selected structure | N | Prediction vs archive minimum MAE | Structure vs archive minimum MAE | Total MAE |
| --- | --- | --- | --- | --- | --- |
| Generated only | False | 9328 | 7.4 | 18.8 | 21.7 |
| Generated only | True | 653 | 9.6 | 144.0 | 142.8 |
| Generated + archive | False | 9690 | 7.5 | 7.1 | 12.4 |
| Generated + archive | True | 291 | 9.9 | 167.3 | 164.7 |

Generated-only selection retains 34,832 genes (34,831 finite prediction–target pairs) and has 95.3 meV/atom overall MAE. Novel genes have 121.7 meV/atom MAE. Novel selected structures with known genes have 142.8 meV/atom MAE. Selecting one observed minimum per gene removes repeated weighting, but does not guarantee that the generated minimum coincides with the archive minimum. The gap between them still mixes geometry and calculation differences.

For novel genes in the generated-only minimum-per-gene DFT pool, residuals (prediction minus observation) have mean -30.9 and median -19.2 meV/atom. 59.3% are negative; 24.4% are below -100 meV/atom and 12.5% above +100 meV/atom. The 10th–90th percentile interval is [-211.5, 122.1] meV/atom. Subtracting the sample median, the best constant shift for these same rows, reduces MAE only from 121.7 to 119.9 meV/atom. The errors have an optimistic tilt and substantial scatter in both directions. This offset check is descriptive and is not a held-out calibration result.

The full artifact contains the selected rows and their source IDs, every candidate's selection status, all pooled and marginal summaries, per-cohort DFT minima, the archive-union candidates, exact input digests, verification records, figure and source. Reproduce with `scripts/evaluate_energy_regressor_gene_minima.py` using the platform's documented interpreter. `--update-doc` replaces this section in the repository report.

## F Error drivers beyond novelty

Measured 2026-10-29 at git `b9e0689e34d58d524b3fcd997e7dcf7eb4b184c8` with exact working-tree source in the [W&B analysis run](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/fhi7gdnh). Inputs are the pinned one-minimum-per-final-gene generated DFT pool in §E. Geometry bytes and saved final genes are verified against the original source digests. No new energy calculation, relaxation or regressor inference was run.

Of 34,832 selected structures, 9 fail the new check that the recovered gene preserves the geometry's reduced composition; they are explicit exclusions, retained with their source IDs in feature_exclusions.csv. The prior protocol composition audit concerned initial-to-final pairs in B and did not check this C gene-to-geometry identity. A missing prediction is also excluded from error models. Previously reported C numbers remain the pinned historical results; the diagnostic denominators below reflect this audit.

The feature table includes cell size and volume, volume per atom, density, covalent-radius packing/contact proxies, space group, atomic and lattice degrees of freedom, elemental fractions, chemistry descriptors, predicted formation energy, training-formula/chemical-system coverage, corpus template coverage, observation count, and recorded DFT Hubbard-U/pseudopotential metadata. Composition coverage uses only the original training split. Anonymous-template coverage uses all three corpus splits and is labelled accordingly. Missing metadata is not treated as GGA.

All marginal effects are associations. Cohort/novelty-adjusted rank correlations and within-cohort block permutations supplement raw bins. Diagnostic models predict signed residual and log(1 + absolute residual / 0.025 eV/atom), never using observed target energy as a feature. The controls contain cohort and novelty. Grouped cross-validation holds out entire reduced formulas. A second check holds out each source cohort with at least 300 rows; smaller cohorts remain available for training but are not its test denominator. Hyperparameters are fixed; no model selection is performed on held-out scores. Error-model predictions diagnose reproducible patterns and are not a validated replacement energy model.

### Main findings

The strongest signals beyond cohort and gene/structure novelty are **packing and
short-contact geometry, chemistry, and coverage of the chemical composition or
anonymous Wyckoff template**. Raw volume, atom count and space-group identity are
less informative. The following associations use the 24,842 usable novel-gene
rows; they do not identify a physical cause by themselves.

| Feature | Observed association |
| --- | --- |
| Closest contact relative to the sum of covalent radii | Lowest fifth (ratio ≤0.842): MAE **168.7 meV/atom**, bias -80.8. Highest fifth (ratio >0.970): MAE **82.2**, bias -6.5. This association has the same sign in all **11** source cohorts with at least 300 novel-gene rows. Covalent radii are a proxy; the ratio is not a binary validity test. |
| Anonymous template coverage in the model corpus | Unseen templates: **231.6 meV/atom**, n=1,484. Known templates: **114.6**, n=23,358. This is distinct from exact-gene novelty: every row here has a novel gene. |
| Chemistry | Gd-containing novel genes: **695.5 meV/atom**, n=321, bias **+482.4**; carbon-containing: **241.8**, n=824. These are overlapping populations and chemistry/cohort associations, not isolated element effects. Gd cases contribute a conspicuous positive-error tail. |
| Positional degrees of freedom | Zero DoF: median absolute error **30.0 meV/atom**, MAE 106.9. More than 11 DoF: median **73.4**, MAE 114.7. Typical error rises, but MAE is not monotonic in DoF, and zero atomic DoF still leaves a free lattice. |
| Number of atoms | More than 19 atoms: MAE **140.1 meV/atom**; at most 8: **120.7**. Cohort/novelty-adjusted rank correlation with absolute error is only **0.048**. |
| Volume per atom | No monotonic trend: the lowest fifth has MAE **109.5**, a middle fifth **132.4**, and the highest fifth **111.8 meV/atom**. Chemistry-normalized contact/packing descriptors are more useful than raw volume. |
| Training-formula coverage | At most one training row for the formula: MAE **126.9 meV/atom**; more than five: **114.2**. The lookup excludes validation/test rows. |

Space group remains a modest predictor when used as a categorical variable, but
its within-cohort block-permutation importance in the full model is much smaller
than packing, chemistry or corpus support. Correlated groups such as space group,
DoF and orbit count can substitute for one another; these scores do not assign
independent causal effects.

The recorded Hubbard-U metadata adds almost no predictive information beyond
cohort/novelty controls. In rows with recorded settings, the transition-metal
oxide/fluoride MP-U chemistry proxy agrees with U being enabled. Each element has
only one recorded POTCAR label in this selected data, so POTCAR labels are
excluded as standalone predictors: they would merely encode which elements are
present. This does **not** establish agreement with every archive calculation's
pseudopotential convention. The chemical outliers' physical cause remains open.

A separate audit of all **34,706** rows with saved DFT entries finds **zero**
entry-to-geometry composition or atom-count mismatches. The other 126 entries
have no such saved metadata. This rules out a simple cell-size normalization
mistake in the audited targets; it does not validate their electronic convergence.

For novel genes, the full diagnostic model explains **27.7%** of the variance of
log absolute error on formula-held-out rows, versus **4.9%** for cohort/novelty
controls. On cohort-held-out rows it explains **19.7%**. Predicting the signed
residual from all features reduces MAE from **121.6 to 104.2 meV/atom** under
formula holdout, and **120.3 to 111.9** under cohort holdout. These models use
features of the final DFT geometry; this is an error diagnostic, not a measured
improvement to pre-relaxation gene screening. Most residual variation remains
unexplained. The results support a role for structural realization and for
chemistry/coverage, and do not establish that either mechanism dominates the
error against the unknown true gene minimum.

### Open data issue: anomalous Gd energies

**Status: open; suspected data problem, investigation deferred.** Recorded
2026-10-29 at base commit `b9e0689e34d58d524b3fcd997e7dcf7eb4b184c8`.
The working hypothesis is an inconsistency in the data or energy conventions;
this has not been established, and model error remains an alternative.

In the usable **novel-gene, generated-only DFT minimum-per-gene** population,
the **321 Gd-containing structures** have:

| Metric | meV/atom |
| --- | ---: |
| MAE | **695.5** |
| Mean signed error, prediction minus observation | **+482.4** |
| Median absolute error | **267.4** |

The positive bias means the regressor predicts energies that are too high.
This differs from the overall novel-gene population's modest negative bias.
The anomaly occurs in multiple cohorts, including MatterGen and MiAD. Examples
below use the original common formation-energy reference; values are **eV/atom**.

| Formula | Selected DFT cohort | Prediction | Observed DFT | Signed error |
| --- | --- | ---: | ---: | ---: |
| Gd6FeB3 | MatterGen/MatterGen_10k/DFT-10k | +4.9114 | +0.3936 | +4.5178 |
| Gd4B2Br3 | MatterGen/MatterGen_10k/DFT-10k | +3.3749 | -0.8666 | +4.2416 |
| Gd4C3 | MiAD/CHGNet_free/DFT | +3.9677 | -0.0571 | +4.0248 |

The existing entry/geometry audit found no atom-count or composition mismatch
in the saved DFT entries. That check does not establish agreement of electronic
settings or energy definitions with LeMat. The generated Gd entries expose the
same `Gd` POTCAR label; a shared label alone does not verify the full POTCAR
hash/version or compatibility with archive calculations.

For the later investigation:

1. Trace the largest Gd residuals back to the raw generated DFT entries and the
   model's LeMat gene-minimum labels; stratify the archive records by upstream
   database and calculation provenance.
2. Compare exact Gd POTCAR hashes/versions, 4f core/valence treatment, spin
   settings and Hubbard-U conventions on both sides, including the elemental
   reference calculations.
3. Audit raw total energies, units, corrections and elemental references using
   the recorded energy-field definitions and `energy_fields.check_compatible`.
4. If the records cannot settle the issue, compare selected geometries with
   calculations using one explicitly consistent DFT setup.

The diagnostic evidence is pinned in
`symmetry-advantage/WyckoffTransformer/energy_error_drivers_fhi7gdnh:v1`
([analysis run](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/fhi7gdnh)),
especially `features.csv.gz`, `element_errors.csv`, `largest_errors.csv` and
`entry_geometry_audit.csv.gz`. The documentation update additionally saves
`gd_issue_rows.csv.gz`, containing all 321 cases with final gene keys, geometry
digests, cohorts, predictions, targets and descriptors. The model revision and
training run remain those stated at the top of this report. Interpretation of
the Gd-specific errors remains provisional until this issue is resolved.

### Held-out predictability

| Population | Holdout | Outcome | Model | N | R² | Residual MAE after error prediction (meV/atom) |
|---|---|---|---|---:|---:|---:|
| all | heldout_formula | error | controls | 34822 | 0.038 | 93.4 |
| all | heldout_formula | error | full | 34822 | 0.300 | 83.4 |
| all | heldout_formula | log_absolute_error | controls | 34822 | 0.285 | — |
| all | heldout_formula | log_absolute_error | full | 34822 | 0.452 | — |
| all | heldout_cohort | error | controls | 33893 | -0.062 | 108.9 |
| all | heldout_cohort | error | full | 33893 | 0.254 | 88.7 |
| all | heldout_cohort | log_absolute_error | controls | 33893 | 0.211 | — |
| all | heldout_cohort | log_absolute_error | full | 33893 | 0.398 | — |
| novel_genes | heldout_formula | error | controls | 24842 | 0.036 | 118.4 |
| novel_genes | heldout_formula | error | full | 24842 | 0.287 | 104.2 |
| novel_genes | heldout_formula | log_absolute_error | controls | 24842 | 0.049 | — |
| novel_genes | heldout_formula | log_absolute_error | full | 24842 | 0.277 | — |
| novel_genes | heldout_cohort | error | controls | 24120 | -0.115 | 151.4 |
| novel_genes | heldout_cohort | error | full | 24120 | 0.234 | 111.9 |
| novel_genes | heldout_cohort | log_absolute_error | controls | 24120 | -0.089 | — |
| novel_genes | heldout_cohort | log_absolute_error | full | 24120 | 0.197 | — |

### Feature blocks beyond cohort and novelty

Single-block additions below use novel genes and formula-held-out predictions. The outcome is log absolute error; R² quantifies predictability of error severity, not energy accuracy. Correlated blocks can substitute for one another.

| Added block | R² |
|---|---:|
| full | 0.277 |
| controls+packing | 0.159 |
| controls+chemistry | 0.154 |
| controls+support | 0.124 |
| controls+dof | 0.104 |
| controls+prediction | 0.103 |
| controls+symmetry | 0.093 |
| controls+size | 0.087 |
| controls+calculation | 0.050 |
| controls | 0.049 |

### Raw feature bins for novel genes

Errors below are meV/atom. These bins do not by themselves control for chemistry or other structural variables. Their counts and signed errors are included to distinguish bias from scatter.

| Feature | Bin | N | MAE | Bias | Median absolute error |
|---|---|---:|---:|---:|---:|
| volume_per_atom | (13.432, 16.71] | 4968 | 124.3 | -32.7 | 74.9 |
| volume_per_atom | (16.71, 20.796] | 4968 | 132.4 | -38.3 | 73.9 |
| volume_per_atom | (20.796, 26.88] | 4968 | 130.0 | -26.0 | 68.1 |
| volume_per_atom | (26.88, 2402.416] | 4969 | 111.8 | -34.4 | 54.6 |
| volume_per_atom | (5.2909999999999995, 13.432] | 4969 | 109.5 | -23.3 | 69.5 |
| n_atoms | (0.999, 8.0] | 6140 | 120.7 | -0.5 | 56.6 |
| n_atoms | (11.0, 14.0] | 4816 | 120.2 | -32.2 | 67.8 |
| n_atoms | (14.0, 19.0] | 4939 | 113.8 | -40.0 | 71.8 |
| n_atoms | (19.0, 96.0] | 4770 | 140.1 | -73.9 | 87.5 |
| n_atoms | (8.0, 11.0] | 4177 | 112.6 | -14.5 | 60.8 |
| positional_dof | (-0.5, 0.5] | 1257 | 106.9 | -22.6 | 30.0 |
| positional_dof | (0.5, 2.5] | 3448 | 140.0 | -42.4 | 54.8 |
| positional_dof | (11.5, inf] | 12332 | 114.7 | -34.4 | 73.4 |
| positional_dof | (2.5, 5.5] | 3232 | 133.1 | -36.5 | 70.2 |
| positional_dof | (5.5, 11.5] | 4573 | 122.2 | -11.4 | 69.3 |
| min_covalent_distance_ratio | (0.337, 0.842] | 4969 | 168.7 | -80.8 | 98.5 |
| min_covalent_distance_ratio | (0.842, 0.885] | 4968 | 133.6 | -32.9 | 74.7 |
| min_covalent_distance_ratio | (0.885, 0.923] | 4968 | 122.0 | -19.5 | 67.2 |
| min_covalent_distance_ratio | (0.923, 0.97] | 4968 | 101.0 | -14.6 | 61.7 |
| min_covalent_distance_ratio | (0.97, 4.424] | 4968 | 82.2 | -6.5 | 47.4 |
| train_formula_count | (-0.001, 1.0] | 15570 | 126.9 | -47.5 | 75.1 |
| train_formula_count | (1.0, 5.0] | 4603 | 111.2 | -10.9 | 57.0 |
| train_formula_count | (5.0, 419.0] | 4669 | 114.2 | 4.6 | 54.5 |
| mp_u_chemistry_proxy | 0.0 | 21105 | 122.8 | -28.2 | 66.8 |
| mp_u_chemistry_proxy | 1.0 | 3737 | 114.8 | -46.8 | 71.0 |
| dft_uses_hubbard | 0.0 | 21025 | 122.9 | -28.2 | 66.8 |
| dft_uses_hubbard | 1.0 | 3707 | 115.2 | -46.9 | 71.4 |
| dft_uses_hubbard | missing | 110 | 87.3 | -23.6 | 53.2 |

The artifact preserves the feature matrix, selected geometries, source settings and coverage provenance, all bins/correlations, element and cohort summaries, out-of-fold predictions, model scores, within-cohort permutation importance, figures and exact source. Reproduce with `scripts/diagnose_energy_error_drivers.py` and the platform's documented interpreter. The stages are prepare, models and report.

## G Earlier versus final checkpoints

Measured 2026-10-29 at base git `b9e0689e34d58d524b3fcd997e7dcf7eb4b184c8`; exact working-tree source and inputs are in the [W&B comparison run](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/77z0yc9r). All 22 retained `best_model` versions of training run `min_energy_adamw_wsd_h7x_xl-20261015-185731` are evaluated. The final weight file is byte-identical to the previously studied published model.

The fixed populations are the archive gene-minimum targets from the retained archive sample in §E, and the generated-only DFT per-final-gene minima from §E. The nine gene/geometry composition mismatches and one unsupported generated gene are excluded identically for every checkpoint. Gd is retained in the main results and separately excluded in the named sensitivity analysis; its data investigation remains deferred. No energies, geometries, novelty labels or minimum selections change with checkpoint. ORB is not used as the main target in this overfitting comparison.

Sixteen equivalent descriptions are sampled once per gene in CPU batches and saved as exact numeric input tensors. Every checkpoint sees those same inputs. Evaluation uses FP32, no dropout, no compilation and no conditioning. The augmentation-vectorized calculation is checked against single-draw evaluation for the first batch of every checkpoint. Training history is read from logged history events because the run-summary fields are stale.

Errors below are **meV/atom**. Archive training/test columns use the archive's observed DFT gene minimum; generated columns use the fixed observed generated DFT minimum. These remain different target definitions; the comparison tests transfer as training proceeds, not accuracy against an unknown physical gene minimum.

### Interpretation

**Late training improves archive accuracy while degrading transfer to novel generated genes.** Between epoch 5,000 (`v8`) and the final checkpoint, archive training MAE falls from 26.2 to 8.5 meV/atom, and archive test MAE on genes absent from training falls from 31.6 to 27.1. Meanwhile, novel generated-gene MAE rises from 113.6 to 121.6.

The paired deterioration is **8.02 meV/atom**, with formula-cluster bootstrap 95% interval [6.68, 9.41]. Without Gd, novel-gene MAE still rises from 107.1 to 114.1; its paired increase is 6.98 [5.94, 8.12]. The effect therefore persists independently of the open Gd anomaly.

Generated structures with known genes improve from 36.8 to 29.6 meV/atom. Thus the deterioration is concentrated in transfer to novel genes, rather than a general numerical or inference failure.

This supports overfitting to archive-specific patterns as a contributor to the observed transfer error, even while archive validation improves. It does not establish that predictions of the true physical gene minimum became worse: increasingly accurate archive-minimum predictions can also disagree with incomplete generated realizations. Using the epoch-5,000 checkpoint recovers about 8.0 meV/atom on these data, but still leaves 113.6 meV/atom novel-gene MAE. This effect does not explain the full archive-to-generated accuracy gap. The checkpoint comparison alone is not a validated policy for selecting deployment checkpoints.

### All retained checkpoints

| Version | Epoch | Archive train MAE | Archive test unseen-gene MAE | Generated novel-gene MAE | Novel genes excluding Gd MAE | Novel genes with unseen templates MAE |
|---|---:|---:|---:|---:|---:|---:|
| v0 | 0 | 336.34 | 324.53 | 367.82 | 359.86 | 408.21 |
| v1 | 250 | 43.96 | 46.50 | 125.81 | 118.61 | 241.02 |
| v2 | 750 | 34.70 | 38.66 | 116.90 | 109.32 | 225.71 |
| v3 | 1000 | 29.98 | 35.67 | 119.44 | 111.87 | 230.27 |
| v4 | 1250 | 29.86 | 34.73 | 118.46 | 110.81 | 232.80 |
| v5 | 2500 | 27.85 | 32.34 | 114.28 | 107.59 | 216.29 |
| v6 | 2500 | 28.51 | 33.47 | 115.77 | 108.68 | 219.19 |
| v7 | 5000 | 26.64 | 32.18 | 115.56 | 108.85 | 223.44 |
| v8 | 5000 | 26.18 | 31.60 | 113.58 | 107.11 | 221.43 |
| v9 | 6000 | 26.15 | 31.87 | 114.88 | 107.50 | 222.72 |
| v10 | 16500 | 21.43 | 29.46 | 114.11 | 106.82 | 220.25 |
| v11 | 17000 | 18.10 | 27.80 | 115.23 | 108.16 | 220.67 |
| v12 | 17750 | 15.56 | 27.65 | 117.10 | 109.70 | 223.36 |
| v13 | 18000 | 14.72 | 27.23 | 116.62 | 109.38 | 223.96 |
| v14 | 18250 | 12.95 | 26.81 | 117.69 | 110.61 | 223.98 |
| v15 | 18500 | 13.00 | 26.65 | 116.60 | 109.45 | 222.90 |
| v16 | 18750 | 12.45 | 27.16 | 118.79 | 111.45 | 226.18 |
| v17 | 19000 | 11.79 | 26.76 | 118.88 | 111.56 | 227.95 |
| v18 | 19250 | 10.45 | 26.69 | 119.08 | 111.73 | 228.76 |
| v19 | 19500 | 9.76 | 26.97 | 120.61 | 113.17 | 230.59 |
| v20 | 19750 | 9.06 | 27.04 | 121.03 | 113.48 | 230.98 |
| v21 | 19999 | 8.54 | 27.06 | 121.60 | 114.09 | 231.64 |

### Paired changes

Differences are final minus earlier MAE; **positive means the final model is worse**. The 95% intervals resample entire reduced formulas, retaining all their genes, using the same rows and augmentation draws on both sides. The three earlier checkpoints were specified before inference. Comparisons among all other checkpoints are descriptive, not an independently validated checkpoint-selection policy.

| Earlier | Population | N | Final minus earlier MAE | 95% interval |
|---|---|---:|---:|---|
| v8 | archive_test_unseen_genes | 4304 | -4.53 | [-6.12, -2.92] |
| v8 | dft_novel_genes | 24842 | +8.02 | [+6.68, +9.41] |
| v8 | dft_novel_no_gd | 24521 | +6.98 | [+5.94, +8.12] |
| v8 | dft_novel_unseen_templates | 1484 | +10.21 | [+4.04, +16.61] |
| v8 | dft_known_genes | 9980 | -7.17 | [-7.69, -6.65] |
| v10 | archive_test_unseen_genes | 4304 | -2.40 | [-3.83, -0.87] |
| v10 | dft_novel_genes | 24842 | +7.50 | [+6.19, +8.79] |
| v10 | dft_novel_no_gd | 24521 | +7.27 | [+6.35, +8.17] |
| v10 | dft_novel_unseen_templates | 1484 | +11.38 | [+5.75, +16.48] |
| v10 | dft_known_genes | 9980 | -5.09 | [-5.53, -4.66] |
| v13 | archive_test_unseen_genes | 4304 | -0.17 | [-1.14, +0.94] |
| v13 | dft_novel_genes | 24842 | +4.99 | [+4.15, +5.88] |
| v13 | dft_novel_no_gd | 24521 | +4.71 | [+4.01, +5.42] |
| v13 | dft_novel_unseen_templates | 1484 | +7.68 | [+3.65, +11.89] |
| v13 | dft_known_genes | 9980 | -2.69 | [-2.92, -2.45] |

The re-evaluated final predictions differ from the original stored 16-draw estimates by 0.284 meV/atom on average because the sampled descriptions/order differ. This difference is shared neither as noise nor as a changing selection between the new checkpoint evaluations: their input tensors are identical.

The artifact preserves the input tensors, fixed target rows, every checkpoint's per-augmentation predictions, weight artifact IDs/digests and field provenance, summary statistics, paired bootstrap intervals, training history, figures and exact source. Weights remain recoverable from their original W&B versions. Reproduce with `scripts/compare_energy_regressor_checkpoints.py`; stages are prepare, infer and report. Select the inference GPU according to the platform instructions. No new training, relaxation or energy calculation was run.

## Reproduction and saved outputs

Use the platform's documented interpreter from the checkout root. The stages are `prepare`, `sample`, `symmetry`, `infer`, `novelty`, and `report` in `scripts/evaluate_energy_regressor.py`; finish `sample` and `symmetry` before the final `infer`. Pin the model and ORB dataset revisions from the saved provenance. `prepare --source-inventory protocol_inventory.json` selects the saved exact artifact versions; `prepare --reuse-inventory` resumes local assembly. No stage loads an MLIP calculator or launches DFT.

The W&B artifact contains the source inventory, archive sample IDs and labels, Figshare metadata, common elemental references, model field provenance, gene registry, individual predictions, recovered symmetry, `rows.csv.gz`, `summary.csv`, the error-distribution figure (PNG/PDF), verification record, this report and code. The local run directory is a working copy; W&B holds the recoverable results. Large JSON files are gzip-compressed in the artifact.
