# The `lemat_bulk_fmax1_stress_ehull01` Dataset

## 1. Overview & Motivation

`lemat_bulk_fmax1_stress_ehull01` is a thermodynamically filtered subset of `lemat_bulk_fmax1_stress`, retaining only crystal structures with energy above the convex hull:

$$E_{\text{hull}} \le 0.1 \text{ eV/atom}$$

### Purpose
While the full `lemat_bulk_fmax1_stress` dataset covers a broad distribution of configurations across high-energy metastable phases (including battery delithiation intermediates, high-temperature configurations, and strained polymorphs), generative tasks targeting synthesizable, near-ground-state materials benefit from a prior concentrated on the thermodynamically stable domain ($E_{\text{hull}} \le 0.1\text{ eV/atom}$).

By filtering at 0.1 eV/atom, the dataset retains **1,637,402 structures (~30.73% of LeMat-Bulk)**, providing a massive, diverse training set while discarding the unstable tail.

---

## 2. Provenance & Filtering Pipeline

### Parent Dataset
- **Source**: `cache/lemat_bulk_fmax1_stress`
- **Underlying Archive**: LeMat-Bulk (`compatible_pbe` calculations from Materials Project, Alexandria, and OQMD).
- **Quality Filters**:
  - Maximum residual atomic force: $f_{\max} \le 1.0\text{ eV/\AA}$ (excluding pathological unrelaxed geometry).
  - Maximum sequence length: $\le 61$ Wyckoff sites.
  - Valid convex hull distance (`energy_above_hull` is not NaN).
  - Formation energy window: $|e_{\text{form}}| \le 5\text{ eV/atom}$.
  - Conditioning channels: `energy_above_hull`, `delta_e_polymorph`, `max_force`, `stress_trace`, `stress_vons_mises`.

### Energy Definition
The energy criterion uses the `energy_above_hull` field, computed against the self-referential LeMat-Bulk PBE convex hull via `wyckoff_transformer.formula_energy.hull_table` (`PDEntry(full_formula, energy_corrected)`) and clipped at zero (`max(0, e_hull)`). See [`docs/e_hull_definitions.md`](e_hull_definitions.md) for full context.

### Slicing Strategy
Rather than re-running raw extraction, PyXtal spacegroup analysis, and tokenization from scratch (which would take ~20+ CPU hours and risk stochastic site ordering differences), `lemat_bulk_fmax1_stress_ehull01` is generated deterministically by **slicing the pre-tokenized tensors and DataFrames** of `lemat_bulk_fmax1_stress`:
1. Slices every tensor and ragged list in `tensors/lemat_bulk_fmax1_sg_multiplicity.safetensors` along the example dimension where `energy_above_hull <= 0.1`.
2. Slices split DataFrames in `data.pkl.gz`.
3. Copies `tokenisers/lemat_bulk_fmax1_sg_multiplicity.json` untouched, ensuring 100% identical vocabulary, site symmetries, multiplicity mappings, and start token definitions.
4. Generates updated `split_ids.json` matching the filtered `val` and `test` row indices.

---

## 3. Split Statistics

| Split | Parent Rows (`lemat_bulk_fmax1_stress`) | Sliced Rows ($E_{\text{hull}} \le 0.1\text{ eV/atom}$) | Retention Rate |
| :--- | :---: | :---: | :---: |
| **train** | 5,127,342 | **1,576,041** | 30.74% |
| **val** | 100,000 | **30,543** | 30.54% |
| **test** | 100,000 | **30,818** | 30.82% |
| **Total** | 5,327,342 | **1,637,402** | 30.73% |

The near-identical retention rate across splits (~30.7%) confirms that the original random partitioning was unbiased with respect to convex hull distance.

---

## 4. Cache Directory Structure

The artifacts reside at `cache/lemat_bulk_fmax1_stress_ehull01/`:

```
cache/lemat_bulk_fmax1_stress_ehull01/
├── tensors/
│   └── lemat_bulk_fmax1_sg_multiplicity.safetensors  (1,996 MB)
├── tokenisers/
│   └── lemat_bulk_fmax1_sg_multiplicity.json        (3.9 MB)
├── data.pkl.gz                                       (128.7 MB)
└── split_ids.json                                    (520 KB)
```

---

## 5. Reproducibility Guide

The dataset creation is completely automated and verified by unit tests.

### Running the Creation Script
From the repository root:

```bash
python scripts/slice_dataset_by_ehull.py \
    --source-dataset lemat_bulk_fmax1_stress \
    --target-dataset lemat_bulk_fmax1_stress_ehull01 \
    --tokeniser-name lemat_bulk_fmax1_sg_multiplicity \
    --ehull-cutoff 0.1
```

If the cache directory is in a non-default location, set `WYCKOFF_CACHE_DIR`:

```bash
WYCKOFF_CACHE_DIR=/path/to/cache python scripts/slice_dataset_by_ehull.py \
    --source-dataset lemat_bulk_fmax1_stress \
    --target-dataset lemat_bulk_fmax1_stress_ehull01 \
    --tokeniser-name lemat_bulk_fmax1_sg_multiplicity \
    --ehull-cutoff 0.1
```

Execution takes ~6 minutes on standard cluster storage.

### Automated Tests
The slicing logic is covered by unit tests in [`src/wyckoff_transformer/tests/test_slice_dataset.py`](../src/wyckoff_transformer/tests/test_slice_dataset.py):

```bash
pytest src/wyckoff_transformer/tests/test_slice_dataset.py
```

---

## 6. Training Configuration & Verification

### Model Configs
- Standard mini-batched training: [`yamls/models/lemat/unconditional_5x.yaml`](../yamls/models/lemat/unconditional_5x.yaml) with `dataset: lemat_bulk_fmax1_stress_ehull01`.
- Exact split-size batching: [`yamls/models/lemat/unconditional_5x_ehull01.yaml`](../yamls/models/lemat/unconditional_5x_ehull01.yaml)
  - `train_batch_size: 1576041` (or 50,000 for standard GPU VRAM constraints)
  - `val_batch_size: 30543`
  - `test_batch_size: 30818`

### Pilot Training Run
The dataset was verified end-to-end on an NVIDIA A100 SXM4 40GB GPU:
- **W&B Run Name**: `revived-sea-1876`
- **Run ID**: [`y7bymnm4`](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/y7bymnm4)
- **Validation Loss**: descended from 57.46 (epoch 0) to 52.27 (epoch 2).
- **Structure Validity (SMACT)**:
  - Validation test set: 83.79%
  - Generated structures: **94.62%**
