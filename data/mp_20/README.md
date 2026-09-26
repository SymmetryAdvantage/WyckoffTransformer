# MP-20

MP-20 contains 45231 general inorganic materials that differ in both structure and composition. There are 89 elements and the materials have 1 - 20 atoms in the unit cells. MP-20 includes most experimentally known materials with no more than 20 atoms in unit cell.

## What is in the dataset

MP-20 includes almost all experimentally stable materials from the Materials Project (Jain et al., 2013) with unit cells including at most 20 atoms. We only include materials that are originally from ICSD (Belsky et al., 2002) to ensure the experimental stability, and these materials represent the majority of experimentally known materials with at most 20 atoms in unit cells.

## Stability of curated materials

All 45231 are experimentally synthesizable. All materials are at local energy minimum after DFT relaxation. In addition, we only select materials with energy above the hull smaller than 0.08 eV/atom and formation energy smaller than 2 eV/atom.

## Visualization of structures

<p align="center">
  <img src="../../assets/mp_20.png" />
</p>

## Citation

Please consider citing the following paper:

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




## Energy corrections (added 2026-09-26, not part of the upstream CDVAE README)

`formation_energy_per_atom` and `e_above_hull` are Materials Project GGA/GGA+U values
**with the `MaterialsProject2020Compatibility` (MP2020) anion and +U corrections applied**,
not the legacy scheme and not raw energies. No uncorrected energy is included, so the
corrections cannot be undone from these files. Recomputing from MP's raw entry energies, MP2020
reproduces 89.1% of rows within 1 meV/atom, against 41.7% for the legacy scheme.

MP-20 energies are therefore on the same scale as the MP2020-corrected DFT energies of the
WyFormer DFT samples, but **not** as LeMat-Bulk's raw PBE(+U) energies; see
`docs/e_hull_definitions.md`. The legacy MP API now serves legacy-scheme values that do not
match MP-20, so it is no reference for these numbers. Evidence and method are in
`docs/archive/mp20_energy_corrections.md`, and the check is in
`scripts/check_mp20_energy_corrections.py`.
