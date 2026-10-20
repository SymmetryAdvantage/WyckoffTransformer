# Generating and relaxing structures

The [README](../README.md#inference) has the one-liners for the published models. This page
covers the rest: other model sources, generation options, and the 3D relaxation stage in
detail.

## Wyckoff representations
Wyckoff representations are produced and stored in WanDB during model training. The `wyformer-generate` CLI (installed with the package) generates them from a trained model. Using a HuggingFace model:
```bash
wyformer-generate <output-file> --hf-model SymmetryAdvantage/<model-name>
```
Using a W&B run:
```bash
wyformer-generate <output-file> --wandb-run <wandb-id> --use-cached-tensors
```
Using a local model directory (must contain `best_model_params.pt`, `config.yaml`, and `wyckoff_processor.json`):
```bash
wyformer-generate <output-file> --model-path runs/<run-id>
```
Note that the code does not automatically download Wandb artifacts, you need to do it manually, and place them in the `runs` folder when restoring via run ID.

To constrain generation to specific elements:
```bash
wyformer-generate <output-file> --hf-model SymmetryAdvantage/<model-name> --required-elements Li-S --allowed-elements Li-S-P-O
```
To override the space group distribution from a cached dataset:
```bash
wyformer-generate <output-file> --hf-model SymmetryAdvantage/<model-name> --sg-dist mp_20
```

See this [repository](https://github.com/SymmetryAdvantage/WyFormer-inference-demo) for a demo of standalone inference using the PyPI package as a library.

## The chemical-system pipeline
The README's [chemical-system pipeline](../README.md#advanced-generation-chemical-system-conditioning) runs in three steps:
1. [WyFormer-LeMat-Chemsys](https://huggingface.co/SymmetryAdvantage/WyFormer-LeMat-Chemsys)
   generates Wyckoff genes that contain only the requested elements.
2. [WyFormer-LeMat-Energy](https://huggingface.co/SymmetryAdvantage/WyFormer-LeMat-Energy)
   predicts each gene's formation energy. The genes are ranked by their distance to the PBE
   convex hull of [LeMat-Bulk-fixed](https://huggingface.co/datasets/kazeevn/LeMat-Bulk-fixed).
3. The best distinct genes are relaxed with ORB v3 (`orb_conserv_inf`), the potential
   LeMat-Bulk publishes its MLIP convex hull for.

- **Devices:** `--device cuda` speeds up the first two steps. They take about a minute on a
  GPU for 1,000 genes.
- **Hull reference:** the first `--hf-reference` run builds the hull from about 170 MB of
  the dataset. That takes about two minutes, and the result is cached in
  `~/.cache/wyckoff_transformer/hull_references/`.
- **Why `--allow-incompatible-energy`:** the regressor was trained on upstream LeMat-Bulk
  energies, and the hull comes from the fixed release. Without the flag,
  `wyformer-gene-screen` refuses to compare them. A retrain on LeMat-Bulk-fixed will replace
  it, and the flag will no longer be needed.
- **Outputs:** `batio_screen.csv` holds the predicted formation energy, the hull energy and
  their difference (`score`) for each kept gene. The relaxed structures and the energy of
  each are under `batio_relaxed/`, laid out as in [PyXtal placement and MLIP relaxation](#pyxtal-placement-and-mlip-relaxation). A gene
  that PyXtal cannot place in any trial, usually a large cell, gets an empty row.
- **Chemical system:** `--chemical-system` asks for compounds that contain every listed
  element. See [chemical-system mode](chemical_system_mode.md) for what that means
  for subsystems such as TiO2.

## 3D structures
There are two ways to generate 3D structures from Wyckoff representations: DiffCSP++, and PyXtal placement followed by MLIP relaxation (the CrySPR scheme). They later can be relaxed with CHGNet and/or DFT.
### DiffCSP++
Wyckoffs can be relaxed with modified [DiffCSP++ code](https://github.com/kazeevn/DiffCSPNew/tree/master)
### PyXtal placement and MLIP relaxation
[CrySPR](https://chemrxiv.org/engage/chemrxiv/article-details/66b308a501103d79c5fd9b91) scheme using [`pyxtal`](https://pyxtal.readthedocs.io/en/latest/index.html) and an ML force field is integrated directly into the package. Install the optional extra first:
```bash
pip install "wyckoff-transformer[relax]"
```
The extra brings MACE and ORB. `--mlip` selects a registered potential; the other backends in
[`cryspr/mlips.py`](../src/wyckoff_transformer/cryspr/mlips.py) need their own dependency
sets. `--mlip orb_conserv_inf` (and the other LeMat-Bulk hull names) loads exactly the
checkpoint that LeMat-Bulk's hull was built with, so the relaxed energies can be compared
with that hull.
```bash
wyformer-cryspr genes.json.gz --mlip orb_conserv_inf --output-dir results/ --workers 8
```
Without `--mlip`, `--model` is loaded as MACE:
```bash
wyformer-cryspr WyckoffTransformer_mp_20.json \
    --model https://github.com/ACEsuit/mace-foundations/releases/download/mace_mp_0/2023-12-10-mace-128-L0_energy_epoch-249.model \
    --output-dir results/ --start 0 --end 1000
# model_name defaults to the model file stem
head results/2023-12-10-mace-128-L0_energy_epoch-249_results.csv
model,id,formula,energy,energy_per_atom
...
2023-12-10-mace-128-L0_energy_epoch-249,35,H6O8Si2,-97.98,...
```
URL-based MACE models are downloaded once and cached in `~/.cache/wyckoff_transformer/mace_models/`. A local path is accepted too: `--model /path/to/model.model`.

With `--workers > 1`, the CLI uses spawned worker processes. Each worker constructs its own calculator, and the process environment is forced to single-threaded BLAS/OpenMP settings (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`) to avoid CPU oversubscription. This is usually the right mode for CPU-bound batch relaxation. On GPU, start conservatively and scale `--workers` only if the device has enough memory for multiple concurrent calculators.

Key options:
- `--n-trials N` — number of random PyXtal trials per structure (default 6)
- `--fmax F` — force convergence criterion in eV/Å (default 0.01)
- `--model-name NAME` — label for the results CSV (default: `--mlip`, or the model file stem)
- `--device auto|cpu|cuda` — PyTorch device selection (default: auto)
- `--workers W` — number of parallel worker processes; each worker builds its own calculator (default 1)

Each PyXtal trial is relaxed in stages: first with a `FixSymmetry` constraint (a fix-cell warm-up, then cell + positions), then without it, and finally a *rattle* — positions and cell perturbed by a finite amount and relaxed again unconstrained, kept only if it wins 1 meV/atom. The rattle is what lets a structure leave a symmetric stationary point at all: the forces along symmetry-breaking modes vanish identically there, so gradient descent alone cannot, and the unconstrained stage took zero steps in 78% of trials. Per-trial CIFs are numbered by stage; `${full_formula}_kept.cif` is the structure the trial kept, `rattle.json` records what the rattle did, and `min_e_strc.cif` links to the kept CIF of the lowest-energy trial.

### CHGNet relaxation
The structures from all models can be optionally relaxed with CHGNet.
```bash
$ cp scripts/cryspr_chgnet.py mp_20/WyckoffLLM-naive/DiffCSP++/parsed_materials_10000_pyxtal.json_structures.json.gz /your/working/dir/
$ cd /your/working/dir/
$ gzip -d parsed_materials_10000_pyxtal.json_structures.json.gz
$ python ./cryspr_chgnet.py 0 -1 ./parsed_materials_10000_pyxtal.json_structures.json wylm-dcpp
$ head wylm-dcpp_id_formula_energy.csv
model,id,formula,energy
wylm-dcpp,0,Y4Ho4Ir8,-128.26178
wylm-dcpp,1,La4Cu4Si8,-89.13443
wylm-dcpp,2,Rb3Br3,-20.12497
wylm-dcpp,3,Gd1Ni2,-26.83080
wylm-dcpp,4,Pb3O18,-96.37995
wylm-dcpp,5,Tl2Au2S4Br4,-41.40260
wylm-dcpp,6,Ru1As2In3Ce3,-49.81073
wylm-dcpp,7,Pd4As4Ni4Se4,-73.34074
wylm-dcpp,8,Na4Lu4F16,-149.18192
```
- Output files are in the same format as [PyXtal placement and MLIP relaxation](#pyxtal-placement-and-mlip-relaxation).
- `${reduced_formula}_${full_formula}_cell+pos.cif` is the CHGNet relaxed structure.
### DFT relaxation
We followed the [Materials Project protocol](https://docs.materialsproject.org/methodology/materials-methodology/calculation-details), [`atomate2.vasp.flows.mp.MPGGADoubleRelaxStaticMaker`](https://materialsproject.github.io/atomate2/reference/atomate2.vasp.flows.mp.MPGGADoubleRelaxStaticMaker.html). There isn't much to add, as the rest of the details of running DFT, unfortunately, depend on the HPC setup, and VASP is not open source. [Here](https://github.com/kazeevn/NSCC-VASP-computer) is the code to run at ASPIRE2.
