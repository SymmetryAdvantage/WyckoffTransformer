# Wyckoff Transformer: Generation of Symmetric Crystals [ICML 2025]
Crystal symmetry plays a fundamental role in determining its physical, chemical, and electronic properties such as electrical and thermal conductivity, optical and polarization behavior, and mechanical strength. Almost all known crystalline materials have internal symmetry. However, this is often inadequately addressed by existing generative models, making the consistent generation of stable and symmetrically valid crystal structures a significant challenge. We introduce WyFormer, a generative model that directly tackles this by formally conditioning on space group symmetry. It achieves this by using Wyckoff positions as the basis for an elegant, compressed, and discrete structure representation. To model the distribution, we develop a permutation-invariant autoregressive model based on the Transformer encoder and an absence of positional encoding. Extensive experimentation demonstrates WyFormer's compelling combination of attributes: it achieves best-in-class symmetry-conditioned generation, incorporates a physics-motivated inductive bias, produces structures with competitive stability, predicts material properties with competitive accuracy even without atomic coordinates, and exhibits unparalleled inference speed.

# WyFormer Generated Datasets
If you just need the generated datasets for benchmarking, they are available at Figshare, including both original and DFT: [WyFormer](https://figshare.com/articles/dataset/WyFormer_generated_structures/29094701); [WyFormer, DiffCSP(++), SymmCD, MiAD, WyCryst, CrystalFormer](https://figshare.com/articles/dataset/Generated_crystals_for_WyFormer_DiffCSP_DiffCSP_WyCryst_SymmCD_CrystalFormer_MiAD/29145101).

# Installation
1. Clone the repository
3. Copy `pyproject.toml.zeus` to `pyproject.toml`
4. Edit `pyproject.toml` as nesessary for your CUDA environment. The file won't work out-of-the-box. The simplest way is to remove the local torch wheels from `pyproject.toml`, and then use the instructions from the [pytorch website](https://pytorch.org/get-started/locally/) for your CUDA version. [Here](https://github.com/python-poetry/poetry/issues/6409) [are](https://github.com/lucaspar/poetry-torch?tab=readme-ov-file) poetry-specific instructions. Optionally, [pytorch-sparse](https://github.com/rusty1s/pytorch_sparse) and [pytorch-scatter](https://github.com/rusty1s/pytorch_scatter) are needed to run the CDVAE property prediction model, but otherwise can be skipped.
5. Run `poetry install`.
6. Log into Wandb, and configure your entity; or disable Wandb. Internally, we used `symmetry-advantage`. It can be configured in poetry:
```bash
poetry self add poetry-dotenv-plugin
echo "WANDB_ENTITY=symmetry-advantage" > .env
```
5. Preprocess the data on Wychoff positions:
```bash
python scripts/preprocess_wychoffs.py
```
# Running a pilot model
To veryfy that the installation is working, run a pilot model. Next token prediction:
```bash
python scripts/cache_a_dataset.py mp_20
python scripts/tokenise_a_dataset.py mp_20 yamls/tokenisers/mp_20_sg_multiplicity.yaml --new-tokenizer
python scripts/train.py yamls/models/NextToken/v6/base_sg.yaml mp_20 cuda --pilot
```
This will train a model, and save the results in the `runs` folder. The files are:
- `best_model_params.pt` - the model weights chosen by the validation loss
- `generated_wp_no_calibration.json.gz` Wyckoff representation of the generated structures
- `generated_wp_temperature_calibration.json.gz` Wyckoff representation of the generated structures, with the temperature calibration applied
- `token_engineers.pkl.gz` - preprocessing information, e. g. multiplicity and spherical harmonics
- `tokenizers.pkl.gz` - tokenizers
# Training Data Preprocessing
The available datasets correspond to the folders in `data` and `cdvae/data`. Dataset idetifiers are the folder names, they are used throught the project. Note that some of the folders are symlinks.
Available datasets (in GitHub): `mp_20`, `mp_20_biternary` (binary and ternary structures from MP-20), `mpts_52`, `carbon_24`, `perov_5`. It is also possible to download and use `matbench_discovery_mp_2022` [notebook](scripts/data_preprocesssing/mp_2022.ipynb) and `matbench_discovery_mp_trj_full` [notebook](scripts/data_preprocesssing/mptrj_extract_all.ipynb).

For any data to be used for training, we need to do two preprocessing steps.
## Compute and cache symmetry information
```bash
python scripts/cache_a_dataset.py <dataset-name>
```
This will create a pickled representaiton of the dataset in `cache/<dataset-name>/data.pkl.gz`. The script supports setting symmetry tolerance and _this is not done automatically_, the datasets which include tolerance in their name were obtained by manually using the command-line option.
## Tokenization
The tokenization script serves two purposes: it produces the mapping from the real data to token ids, and saves the resulting tensors. To produce a new tokenizer:
```bash
python scripts/tokenise_a_dataset.py <dataset-name> <path-to-tokenizer-yaml> --new-tokenizer
```
Tokenizer configs are stored in `yamls/tokenisers`. The tokeniser is saved to `cache/<dataset-names>/tokenisers/**.pkl.gz`, preserving the folder structure of the config.

Alternatively, you can use a cached tokeniser. This is important when a model that was trained on one dataset is  applied to a different dataset.
```bash
python scripts/tokenise_a_dataset.py <dataset-name> <path-to-tokenizer-yaml> --tokenizer-path cache/<dataset-names>/tokenisers/<tokenizer-name>.pkl.gz
```
# Training
```bash
python scripts/train.py <path-to-model-yaml> <dataset-name> <device>
```
The model weights are saved to `runs/<run-id>`, and to WanDB, along with the tokenizer. See [here](yamls/models/README.md) for the list of configs. Adding `--pilot` will run the model for a small number of epochs.

# Generating structures
## Wyckoff representations
Wyckoff representations are produced and stored in WanDB during model training. They can be generated by:
```python
scripts/generate.py --wandb-run <wandb-id> <output-file> --use-cached-tensors
```
Note that the code does not automatically download Wandb artifacts, you need to do it manually, and place them in the `runs` folder.

## 3D Structures
There are two ways to generate 3D structures from Wyckoff representations: DiffCSP++ and CHGNet. They later can be relaxed with CHGNet and/or DFT.
### DiffCSP++
Wyckoffs can be relaxed with modified [DiffCSP++ code](https://github.com/kazeevn/DiffCSPNew/tree/master)
### CrySPR + CHGNet
[CrySPR](https://chemrxiv.org/engage/chemrxiv/article-details/66b308a501103d79c5fd9b91) scheme that combines [`pyxtal`](https://pyxtal.readthedocs.io/en/latest/index.html) with CHGNet (or any other machine-learning interatomic potentials)
```bash
$ cp scripts/cryspr_pyxtal_chgnet.py mp_20/WyckoffTransformer/WyckoffTransformer_mp_20.json.gz /your/working/dir/
$ cd /your/working/dir/
$ gzip -d WyckoffTransformer_mp_20.json.gz
$ python ./cryspr_pyxtal_chgnet.py 0 -1 ./WyckoffTransformer_mp_20.json model_name
# model_name is optional, default is "model_name"
$ head -40 model_name_id_formula_energy_0-1000.csv
model,id,formula,energy
......
model_name,35,H6O8Si2,-97.98080444335938
model_name,36,CdDy,-6.170680522918701
model_name,37,Ba8Sr4As4O24,-264.228572845459
model_name,38,Co2Ni2F10O2,-81.85555267333984
......
$ ls 35/
min_e_strc.cif  trial-0  trial-1  trial-2  trial-3  trial-4  trial-5  trial-lowest
```
- There are `n_trial_each_wyckoff_gene=6` trial calculations under each id directory.
- `trial-lowest` and `min_e_strc.cif`
are the symbolic links to the one with the lowest energy out of all trials.
- Under each trial, `${reduced_formula}_${full_formula}_cell+pos.cif` is the CHGNet relaxed structure.
- `${reduced_formula}_${full_formula}_2_cell+pos_symmetrized.cif` is symmetrized structure by a certain symmetry precision (`symprec`), see each log file for details.
- This script can also be used as a module,
```python
from scripts.cryspr_pyxtal_chgnet import single_run, single_pyxtal
import json
# from chgnet.model import CHGNetCalculator
# calculator = CHGNetCalculator(use_device="cpu")
# you could use other calculator
# the following codes do not take effect to set OMP_NUM_THREADS,
# please set this env var in your terminal, like in Bash
# by `export OMP_NUM_THREADS=1` before running Python
# otherwise you will get much slower CHGNet run!
import os
os.environ["OMP_NUM_THREADS"] = "1"

with open("./WyckoffTransformer_mp_20.json") as f:
    data = json.load(f)

gene = data[99]
atoms_in = single_pyxtal(gene)
atoms_relaxed = single_run(atoms_in)
```
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
- Output files are in the same format as above ([CrySPR + CHGNet](#cryspr--chgnet)).
- `${reduced_formula}_${full_formula}_cell+pos.cif` is the CHGNet relaxed structure.
### DFT relaxation
We followed the [Materials Project protocol](https://docs.materialsproject.org/methodology/materials-methodology/calculation-details), [`atomate2.vasp.flows.mp.MPGGADoubleRelaxStaticMaker`](https://materialsproject.github.io/atomate2/reference/atomate2.vasp.flows.mp.MPGGADoubleRelaxStaticMaker.html). There isn't much to add, as the rest of the details of running DFT, unfortunately, depend on the HPC setup, and VASP is not open source. [Here](https://github.com/kazeevn/NSCC-VASP-computer) is the code to run at ASPIRE2.
# Generated Data Analysis
## Storage
### Public Figshare
Most analyzed datasets in a uniform format are available at [Figshare](https://figshare.com/articles/dataset/Generated_crystals_for_WyFormer_DiffCSP_DiffCSP_WyCryst_SymmCD_CrystalFormer_MiAD/29145101).
### Private Dropbox
The raw files are stored in a private Dropbox. To pull:
```bash
rclone copy "NUS_Dropbox:/Nikita Kazeev/Wyckoff Transformer data/generated.tar.gz" . --progress
tar -xvf generated.tar.gz
```
To push:
```bash
tar --use-compress-program=pigz -cvf generated.tar.gz generated
rclone copy generated.tar.gz "NUS_Dropbox:/Nikita Kazeev/Wyckoff Transformer data/" --progress
```
Tar is used to handle the large number of small files, and `pigz` is used to speed up the compression. The Dropbox folder is private, if you are a collaborator, please contact for access.

## Preprocessing
In order to be analyzed the data must be preprocessed and cached. To preprocess all generated datasets in `generated/datasets.yaml`:
```bash
poetry run python scripts/cache_generated_datasets.py
```
It supports filtering by dataset and transformations, e. g.:
```bash
poetry run python scripts/cache_generated_datasets.py --dataset mp_20 --transformations DiffCSP++ DFT
```
Completing this step will enable loading the data with `evaluation.generated_dataset.GeneratedDataset.from_cache`

## Metric computation
The ICML 2025 results were computed by the notebooks in [ICML_eval](ICML_eval). They include, but not limited to the following metrics:
1. S.U.N. - the fraction of stable, unique, and novel structures.
2. S.S.U.N. - the fraction of symmetric, stable, unique, and novel structures.
3. Space Group $\chi^2$ - the $\chi^2$ statistic of the space group distribution between the generated and the test set.
4. P1 - the fraction of generated structures that lack internal symmetries, i. e. belong to space group P1.
5. Property similarity and naive validity from [Xie et al.](https://arxiv.org/abs/2110.06197)
