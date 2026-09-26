# General notes
See inside the folders for the description of the data files. The symlinks are used to create
virtual copies of a dataset, to be preprocessed with different settings.
# CDVAE datasets
- [Perov-5](perov_5) (Castelli et al., 2012): contains 19k perovksite materials, which share similar structure, but has different composition.

- [Carbon-24](carbon_24) (Pickard, 2020): contains 10k carbon materials, which share the same composition, but have different structures.

- [MP-20](mp_20) (Jain et al., 2013): contains 45k general inorganic materials, including most experimentally known materials with no more than 20 atoms in unit cell.

# matbench-discovery MP-2022 & MPTrj
As orginally desgined, models in matbench-discovery train on MPTrj and predict WBM.
We experimented with different training datasets:
1. `mp_2022` is MP 2022 – relaxed structures from Materials Project. Downloaded by this [notebook](../scripts/data_preprocesssing/mp_2022.ipynb).
2. `mp_trj_full` is MPTrj – the full dataset, including both relaxed and unrelaxed structures. Downloaded by this  [notbook](../scripts/data_preprocesssing/mptrj_extract_all.ipynb). Note that
symmetry changes only slighly during relaxation, meaning that after preprocessing the data a large number of
structures with the same Wyckoff representation have the same energy; [analysis](../research_notebooks/mptrj_duplicates.ipynb).

If you modify the training data, be extremely careful that the target is _formation energy per atom_ and it's computed with same reference energies as WBM. Train / val split is entirely our choice, and can be modified freely.

The different symlinks in `data` allow to define variants of the datasets to be processed with different tolerances. The tolerance is `wyformer-cache-dataset --symmetry-precision` / `--symmetry-a-tol` and _is not set automatically_.

Tolerance didn't (2024) have a significant impact. Hence, for further experiments, just `mp_2022` seems to be a reasonable choice.

# LeMat-Bulk

Rebuilding any LeMat variant: follow
[docs/lemat_bulk_pipeline.md](../docs/lemat_bulk_pipeline.md), which is the canonical
order and also records which variant currently on disk was built with which rules. The
step that is easy to miss and silently wrong to skip is `scripts/recover_mp_forces.py`:
30,679 rows (22.1% of Materials Project) reach us with an empty `forces` array, and a
build without it either drops them or substitutes a fabricated `max_force` at half the
true value. `lemat_bulk_fmax1` and the whole `lemat_bulk_ehull` family predate it.

# Provenance

Where each entry came from, for the case where every copy is lost. This is
documentation, not how a machine gets its data -- see
[docs/data_store.md](../docs/data_store.md): the stores are replicated by
`scripts/store_sync.sh`, because transferring beats both re-downloading (the
internet is the slow link) and rebuilding (CPU is scarce on iapetus, and
ASPIRE2a needs PBS for preprocessing).

Tracked in git, so a fresh clone has them:

| Entry | Origin |
| --- | --- |
| `mp_20`, `perov_5`, `carbon_24` | CDVAE; cited below |
| `mpts_52` | [DiffCSP](https://github.com/jiaor17/DiffCSP/tree/main/data/mpts_52) |
| `alex_mp_20` | [MatterGen](https://github.com/microsoft/mattergen/tree/main/data-release/alex-mp) (git-LFS) |
| `mp_20_biternary` | `scripts/select_from_mp_20.py`, from `mp_20` |
| `wbm` | matbench-discovery; test only |
| `matbench_discovery_mp_2022` | `scripts/data_preprocesssing/mp_2022.ipynb` |
| `matbench_discovery_mp_trj_full` | `scripts/data_preprocesssing/mptrj_extract_all.ipynb` |
| `mp_provenance.csv.gz` | `scripts/pull_mp_provenance.py` |
| `mp_2026_gga_gap` | `scripts/build_mp_gga_band_gap.py`; see its own README |

Not tracked -- too large for git, and replicated by `store_sync.sh` instead:

| Entry | Origin |
| --- | --- |
| `lemat-bulk/raw/data.parquet` | HuggingFace `LeMaterial/LeMat-Bulk`, config `compatible_pbe` |
| `lemat-bulk/convergence_labels.parquet` | `scripts/build_lemat_bulk_fmax.py`; per-row `max_force` and stress, MP gaps filled by `scripts/recover_mp_forces.py` |
| `lemat-bulk/cif_prepared` | `scripts/prepare_cif.py`, from `raw` |
| `lemat-bulk/lemat_pbe.csv.gz` | `scripts/process_lemat.py`, from `cif_prepared` |
| `lemat-bulk/lemat_pbe_ehull.csv.gz` | `wyckoff_transformer.formula_energy.hull_table` |
| `lemat-bulk/labels.parquet` | `scripts/build_lemat_bulk_fmax.py --labels-only` |
| `lemat-bulk/20_wyckoffs` | `scripts/pipeline_lemat_20wyckoffs.py` |
| `lemat_bulk_fmax1` | `scripts/build_lemat_bulk_fmax.py --recovered-forces none`; superseded, see below |
| `lemat_bulk_fmax1_stress` | `scripts/build_lemat_bulk_fmax.py` |
| `formula_energy/` | `wyckoff_transformer.formula_energy.dataset` |
| `unique_fingerprints.parquet` | BAWL novelty reference, copied from LeMat-GenBench; see `wyckoff_transformer.evaluation.bawl_reference` |
| every `cache/<dataset>` | `wyformer-cache-dataset`, then `scripts/tokenise_a_dataset.py` |

Whether each dataset is current or obsolete, and what its energy columns mean
(DFT settings, correction, reference hull), is recorded in
`yamls/datasets/<name>.yaml`, not here; see `docs/energy_fields.md`. Only
`lemat_bulk_fmax1_stress` (with its `_ehull01` slice), `mp_20`, `mp_2026_gga_gap`
and `formula_energy` are current.

Two entries have **no** recorded provenance and no producer in this repository.
Establish it before relying on either, and do not assume they can be rebuilt:

- `all_compositions.npz` (24 MB) -- nothing in `src/` or `scripts/` reads it; the
  only mention of the name is a comment in
  `wyckoff_transformer/evaluation/hull_energy.py` about LeMat-GenBench. It may
  simply be an orphan.
- `lemat-bulk/train.csv.gz` (926 MB) -- no code writes it.

# Citation
Perov_5:

```
@article{castelli2012new,
  title={New cubic perovskites for one-and two-photon water splitting using the computational materials repository},
  author={Castelli, Ivano E and Landis, David D and Thygesen, Kristian S and Dahl, S{\o}ren and Chorkendorff, Ib and Jaramillo, Thomas F and Jacobsen, Karsten W},
  journal={Energy \& Environmental Science},
  volume={5},
  number={10},
  pages={9034--9043},
  year={2012},
  publisher={Royal Society of Chemistry}
}
```

```
@article{castelli2012computational,
  title={Computational screening of perovskite metal oxides for optimal solar light capture},
  author={Castelli, Ivano E and Olsen, Thomas and Datta, Soumendu and Landis, David D and Dahl, S{\o}ren and Thygesen, Kristian S and Jacobsen, Karsten W},
  journal={Energy \& Environmental Science},
  volume={5},
  number={2},
  pages={5814--5819},
  year={2012},
  publisher={Royal Society of Chemistry}
```

Carbon_24:

```
@misc{carbon2020data,
  doi = {10.24435/MATERIALSCLOUD:2020.0026/V1},
  url = {https://archive.materialscloud.org/record/2020.0026/v1},
  author = {Pickard,  Chris J.},
  keywords = {DFT,  ab initio random structure searching,  carbon},
  language = {en},
  title = {AIRSS data for carbon at 10GPa and the C+N+H+O system at 1GPa},
  publisher = {Materials Cloud},
  year = {2020},
  copyright = {info:eu-repo/semantics/openAccess}
}
```

MP_20:

```
@article{xie2021crystal,
  title={Crystal Diffusion Variational Autoencoder for Periodic Material Generation},
  author={Xie, Tian and Fu, Xiang and Ganea, Octavian-Eugen and Barzilay, Regina and Jaakkola, Tommi},
  journal={arXiv preprint arXiv:2110.06197},
  year={2021}
}
```

```
@article{jain2013commentary,
  title={Commentary: The Materials Project: A materials genome approach to accelerating materials innovation},
  author={Jain, Anubhav and Ong, Shyue Ping and Hautier, Geoffroy and Chen, Wei and Richards, William Davidson and Dacek, Stephen and Cholia, Shreyas and Gunter, Dan and Skinner, David and Ceder, Gerbrand and others},
  journal={APL materials},
  volume={1},
  number={1},
  pages={011002},
  year={2013},
  publisher={American Institute of PhysicsAIP}
}
```
