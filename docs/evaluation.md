# Screening, ranking and evaluating generated structures

## Screening generated genes
### DFT fixed-hull screening

`wyformer-dft-screen` ranks generated genes before an expensive DFT campaign. It
combines the provenance-aware composition-floor ensemble with the gene attainable-
energy critic, comparing both independently to the same immutable LeMat-Bulk PBE
hull:

```bash
uv run wyformer-dft-screen generated/<run>/wyckoff_genes.json.gz \
    --formula-ensemble runs/formula_energy/ensemble.pt \
    --regressor-path runs/<gene-energy-run> \
    --top 1000 --out generated/<run>/dft_screen.csv
```

This command uses no MLIP energies and performs no relaxation or active learning;
the shortlist feeds the external DFT workflow. See the
[DFT fixed-hull adversarial screening design](dft_fixed_hull_attack.md) for
the objective, estimator boundaries, ablations, and reporting protocol.

### Generative novelty screening

The energy screen trades novelty for stability, so it pairs with a second lever
read off the generator itself. `wyformer-gene-novelty` scores each gene by
`-log p(gene)` under the model that produced it -- a continuous novelty estimator
that needs no reference set (AUC 0.93 against the fingerprint lookup, 0.81
against post-relaxation structure novelty):

```bash
uv run wyformer-gene-novelty generated/<run>/wyckoff_genes.json.gz \
    --model-path runs/<run> --condition energy_above_hull=0 \
    --permutation-samples 64 --device cuda \
    --out generated/<run>/gene_novelty.csv
```

Score the pool at the condition it was generated at. See
[generative novelty screening](generative_novelty_screen.md) for what the
density is, how the two levers combine, and what the combination is worth.

## Ranking model variants (`wyformer-protocol`)

For comparing WyFormer variants during development, `wyformer-protocol` runs the
whole funnel from sampled Wyckoff genes to MetaSUN in one command:

```bash
uv run wyformer-protocol generated/<run>/wyckoff_genes.json.gz \
    --output-dir generated/<run>/protocol \
    --cores 20            # or --devices cuda:0,cuda:1
```

Filters that need no potential — validity, uniqueness, novelty against
LeMat-Bulk — run first, so only gene-novel structures are relaxed, and the
relaxation itself uses one PyXtal trial and two stages. That is about 3.5×
cheaper than the full CrySPR protocol at equal statistical power. The headline
number is `metasun_per_sampled_gene` in `<output-dir>/funnel.json`.

`--mlip` is restricted to potentials LeMat-Bulk publishes a convex hull for
(ORB by default), because `e_above_hull` is only meaningful when the energy and
the hull come from the same model. Structure validity, BAWL fingerprinting and
the hull energy are implemented in `wyckoff_transformer.evaluation` and pinned
to LeMat-GenBench's own implementations by tests, so nothing imports that
package at runtime.

See [the de novo ranking protocol](de_novo_ranking_protocol.md) for the
measurement rationale, the sample sizes it can resolve, required reference data,
and known limitations, and [every `e_hull` in this
repository](e_hull_definitions.md) before comparing a hull energy from one
part of the codebase with one from another — six things carry that name, and
they differ in reference set, energy source and sign convention.

## Generated Data Analysis
### Storage
#### Public Figshare
Most analyzed datasets in a uniform format are available at [Figshare](https://figshare.com/articles/dataset/Generated_crystals_for_WyFormer_DiffCSP_DiffCSP_WyCryst_SymmCD_CrystalFormer_MiAD/29145101). The WyFormer structures alone, original and DFT-relaxed, are in a [separate record](https://figshare.com/articles/dataset/WyFormer_generated_structures/29094701).
#### Private Dropbox
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

### Preprocessing
In order to be analyzed the data must be preprocessed and cached. To preprocess all generated datasets in `generated/datasets.yaml`:
```bash
uv run python scripts/cache_generated_datasets.py
```
It supports filtering by dataset and transformations, e. g.:
```bash
uv run python scripts/cache_generated_datasets.py --dataset mp_20 --transformations DiffCSP++ DFT
```
Completing this step will enable loading the data with `evaluation.generated_dataset.GeneratedDataset.from_cache`

### Metric computation
The ICML 2025 results were computed by the notebooks in [ICML_eval](../ICML_eval). They include, but not limited to the following metrics:
1. S.U.N. - the fraction of stable, unique, and novel structures.
2. S.S.U.N. - the fraction of symmetric, stable, unique, and novel structures.
3. Space Group $\chi^2$ - the $\chi^2$ statistic of the space group distribution between the generated and the test set.
4. P1 - the fraction of generated structures that lack internal symmetries, i. e. belong to space group P1.
5. Property similarity and naive validity from [Xie et al.](https://arxiv.org/abs/2110.06197)
