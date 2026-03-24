# ICML 2025 Experiments
## Main paper
1. Structure generation `NextToken/v6/base_sg.yaml`
2. MP-20 band gap `yamls/models/mp_20/band_gap/icml_2025.yaml`
3. MP-20 energy `yamls/models/mp_20/formation_energy_per_site/icml_2025.yaml`
4. MPTS-52 generation `yamls/models/NextToken/v6/base_sg_no_compile.yaml`
## Ablation
1. Spheric harmonic inputs, predicting harmonic clusters: `yamls/models/mp_20/NextToken/augmented/cluster_harmonic_schedule_free.yaml`
2. Wyckoff positions encoded with letters: `NextToken/v6/sg_letters.yaml`
# General notes
1. Configs are not maintained, so old configs might be incompatible with the current code.
2. Configs are referenced by the WanDB runs, and are not supposed to be changed.
# matbench-discovery
[This WanDB workspace](https://wandb.ai/symmetry-advantage/WyckoffTransformer?nw=wrbkiq2xgjk) is a good place to find some models and their performance.