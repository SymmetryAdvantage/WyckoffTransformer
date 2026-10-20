# Local development and training

## Installation
1. Clone the repository
2. Run `uv venv --python 3.12`
3. Install the dependencies, including torch. There are several options:
  - Manually install torch with your local flavour, e.g., `uv pip install torch --index-url https://download.pytorch.org/whl/cu130`, then run `uv pip install -e`
  - Configure `uv.toml` with your desired indices, see `scripts/platforms/zeus/uv.toml` and `scripts/platforms/cpu/uv.toml`

   Either route installs stock PyXtal from PyPI (pinned to 1.1.5). This project
   used to carry a PyXtal fork for a `check_wp` bug -- one species' like-like
   distance tolerance applied to every pair -- but our fix was adopted upstream,
   and the fork is gone.
4. `wandb` library is used extensively and must be installed. Logging can be disabled via `WANDB_MODE=disabled`. Otherwise, log into Wandb. Internally, we use `WANDB_ENTITY=symmetry-advantage`.

Machine-specific setups are in [platforms](platforms/README.md).

## Running a pilot model
To verify that the installation is working, run a pilot model. Next token prediction:
```bash
wyformer-cache-dataset mp_20
python scripts/tokenise_a_dataset.py mp_20 yamls/tokenisers/mp_20_sg_multiplicity.yaml --new-tokenizer
python scripts/train.py yamls/models/NextToken/v6/base_sg.yaml mp_20 cuda --pilot
```
This will train a model, and save the results in the `runs` folder. The files are:
- `best_model_params.pt` - the model weights chosen by the validation loss
- `config.yaml` - the configuration used for training
- `wyckoff_processor.json` - tokenizers and preprocessing metadata (token engineers)
- `generated_wp_no_calibration.json.gz` - Wyckoff representation of the generated structures (if configured to evaluate generation)
- `generated_wp_temperature_calibration.json.gz` - Wyckoff representation of the generated structures, with the temperature calibration applied (if configured to evaluate generation)

## Training Data Preprocessing
The available datasets correspond to the folders in `data` and `cdvae/data`. Dataset idetifiers are the folder names, they are used throught the project. Note that some of the folders are symlinks.
Available datasets (in GitHub): `alex_mp_20`, `mp_20`, `mp_20_biternary` (binary and ternary structures from MP-20), `mpts_52`, `carbon_24`, `perov_5`. It is also possible to download and use `matbench_discovery_mp_2022` [notebook](../scripts/data_preprocesssing/mp_2022.ipynb) and `matbench_discovery_mp_trj_full` [notebook](../scripts/data_preprocesssing/mptrj_extract_all.ipynb). LeMat-Bulk has its own [pipeline](lemat_bulk_pipeline.md).

For any data to be used for training, we need to do two preprocessing steps.
### Compute and cache symmetry information
```bash
wyformer-cache-dataset <dataset-name>
```
This runs pyxtal over every structure and writes the Wyckoff representation to `cache/<dataset-name>/{train,val,test}.parquet`, one file per split -- see [the data store](data_store.md#the-dataset-cache-one-parquet-file-per-split), which also says how to convert a cache built before 2026-09-22. Every numeric, boolean and string column of the split CSVs is carried through as a per-structure label; `--scalar-columns` narrows that to a chosen few.

`--max-sites N` **drops** structures with more than N Wyckoff sites. Nothing truncates a structure: keeping its first N sites would silently change its composition while leaving its energy labels attached. Symmetry tolerance is `--symmetry-precision` / `--symmetry-a-tol` and _is not set automatically_; the datasets whose name includes a tolerance were built by passing it.
### Tokenization
The tokenization script serves two purposes: it produces the mapping from the real data to token ids, and saves the resulting tensors. To produce a new tokenizer:
```bash
python scripts/tokenise_a_dataset.py <dataset-name> <path-to-tokenizer-yaml> --new-tokenizer
```
Tokenizer configs are stored in `yamls/tokenisers`. The processor is saved to `cache/<dataset-names>/tokenisers/**.json`, preserving the folder structure of the config.

Alternatively, you can use a cached tokeniser. This is important when a model that was trained on one dataset is  applied to a different dataset.
```bash
python scripts/tokenise_a_dataset.py <dataset-name> <path-to-tokenizer-yaml> --tokenizer-path cache/<dataset-names>/tokenisers/<tokenizer-name>.json
```
## Training
```bash
python scripts/train.py <path-to-model-yaml> <dataset-name> <device>
```
The model weights are saved to `runs/<run-id>`, and to WanDB, along with the processor metadata. See [here](../yamls/models/README.md) for the list of configs. Adding `--pilot` will run the model for a small number of epochs.
### Several GPUs of one node
```bash
torchrun --standalone --nproc-per-node <n-gpus> scripts/train.py <path-to-model-yaml> <dataset-name> cuda
```
`train_batch_size` stays the global batch, so a config trains the same run on any number of GPUs. See [distributed training](distributed_training.md).
### Resuming an interrupted run
Training writes `runs/<run-id>/last_checkpoint.pt` every `optimisation.checkpoint_period` epochs
(defaulting to `validation_period`) and once more when the loop ends. It holds the weights, the
optimiser and schedule state, the RNG and the loaders' shuffle position, so a run continued from
it takes the same steps it would have taken had it never stopped -- unlike
`best_model_params.pt`, which holds weights alone and restarts the optimiser cold.
```bash
python scripts/train.py <path-to-model-yaml> <dataset-name> <device> --resume <run-id>
```
This logs back into the same W&B run rather than opening a second one, and refuses to start if
the config given differs from the one that run recorded, since the schedule and the optimiser
are carried over from the checkpoint. A run that died before writing its first checkpoint has to
be started over.
## Preparing Representative Checkpoints
To train and prepare representative checkpoints for datasets like `alex_mp_20` or `mp_20`, you can follow this end-to-end pipeline. Please replace `<dataset-name>` with your target dataset (e.g., `alex_mp_20` or `mp_20`).

First, cache and tokenize the dataset:

```bash
wyformer-cache-dataset <dataset-name>
python scripts/tokenise_a_dataset.py <dataset-name> yamls/tokenisers/<dataset-name>_sg_multiplicity.yaml --new-tokenizer
```
Before training, ensure that your model configuration file points to the correct tokenizer. For example, in `yamls/models/NextToken/v6/base_sg_schedule_free.yaml`, update the tokenizer name to match your dataset:

```yaml
tokeniser:
  name: <dataset-name>_sg_multiplicity
```
Then, initiate the training run:

```bash
python scripts/train.py yamls/models/NextToken/v6/base_sg_schedule_free.yaml <dataset-name> <device>
```
