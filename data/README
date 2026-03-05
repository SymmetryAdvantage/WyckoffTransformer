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

The different symlinks in `data` allow to define variants of the datasets to be processed with different tolerances. The tolerance is set in the `cache_a_dataset.py` script and _is not done automatically_.

Tolerance didn't (2024) have a significant impact. Hence, for further experiments, just `mp_2022` seems to be a reasonable choice.

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
