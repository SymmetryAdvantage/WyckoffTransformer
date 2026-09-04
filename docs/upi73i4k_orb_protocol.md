# `upi73i4k` under the ORB ranking protocol

Run 2026-09-04. Genes: `generated/upi73i4k/wyckoff_genes_ehull0_n2500.json.gz`
(2500 genes, `--condition-value 0.0`). Outputs in
`generated/upi73i4k/protocol/`.

Settings, from `manifest.json`: ORB v3 conservative-inf
(`orb-v3-conservative-inf-omat-20250404`) against its own LeMat-Bulk hull,
1 PyXtal trial × 2 CrySPR stages, `fmax 0.05`, 20 workers on one RTX 6000 Ada.
52 min wall for the 1873 gene-novel genes, 3 min for the remaining 627, 8.6
worker-hours in total.

## The funnel

| step | count | per sampled gene |
|---|---:|---:|
| sampled genes | 2500 | 1.0000 |
| valid gene | 2500 | 1.0000 |
| unique gene | 2500 | 1.0000 |
| produced a structure | 2490 | 0.9960 |
| valid structure | 2271 | 0.9084 |
| unique structure | 2271 | 0.9084 |
| novel structure | 1806 | 0.7224 |
| `e_hull ≤ 0.1` — **MetaSUN** | 290 | **0.1160** |
| `e_hull ≤ 0` — **SUN** | 7 | **0.0028** |

`e_above_hull` over the 2271 valid structures: mean 0.364, median 0.241, sd
0.406, min −0.030, max 3.379 eV/atom; 25.9% ≤ 0.1, 15.2% ≤ 0.05, 1.5% ≤ 0.

Uniqueness is exact and unremarkable: `StructureMatcher` found no duplicate
among the 2271, so every valid structure is distinct.

## Relaxing the gene-known genes changes the reading

This run is the first under the revised protocol, which relaxes every unique
gene rather than discarding the 627 whose Wyckoff fingerprint already occurs in
LeMat-Bulk. Splitting the funnel by that screen shows why it matters.

| cohort | genes | structure | valid | novel | mean `e_hull` | median | ≤ 0.1 | ≤ 0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gene-novel | 1873 | 1863 | 1688 | 1688 | 0.417 | 0.307 | 0.161 | 0.0041 |
| gene-known | 627 | 627 | 583 | 118 | 0.212 | 0.085 | 0.542 | 0.0463 |
| **all** | 2500 | 2490 | 2271 | 1806 | **0.364** | **0.241** | **0.259** | **0.0150** |

- **The energy distribution was biased.** The gene-known cohort is far more
  stable than the novel one — 54.2% at or below 0.1 eV/atom against 16.1%, and
  a median three and a half times lower. Scoring only the gene-novel genes
  conditions `e_above_hull` on novelty and drops exactly the genes that
  reproduce real materials, so the old mean of 0.417 overstated the model's
  typical output; over all 2500 genes it is 0.364.
- **118 of the discarded genes were not duplicates at all.** They share a
  fingerprint with a LeMat-Bulk entry, but `StructureMatcher` rejects the
  match: same space group and same elements on the same Wyckoff orbits,
  different structure. 18 of them are metastable, which is the whole difference
  between MetaSUN 0.1088 and 0.1160.
- **CrySPR recovery, measured.** Of the 583 gene-known genes that gave a valid
  structure, **465 (79.8%)** relax back to a LeMat-Bulk structure the matcher
  accepts. These genes were extracted from real materials, so this is a direct
  control on the reconstruction stage on inputs whose answer is known, and it
  is consistent with the ceiling the [reconstruction
  study](cryspr_reconstruction_study.md) is designed to measure. Note also that
  100% of them produced a structure at all (627/627) against 99.5% for the
  novel cohort.

## Against the MACE/LeMat-GenBench run

`generated/upi73i4k/RESULTS.md` scored the same genes with MACE-MP through
LeMat-GenBench. MetaSUN per generated gene barely moves — 0.1160 here against
0.114 there — but the stability distribution is much harsher under ORB: mean
`e_hull` 0.364 vs 0.308, metastable fraction 0.259 vs 0.303, stable 0.015 vs
0.041.

Three things differ at once, so this is not a model-to-model comparison:

- the potential and its hull (ORB against ORB's, MACE against `mace_mp`'s,
  whose checkpoint is unidentified — see the [protocol
  doc](de_novo_ranking_protocol.md));
- the relaxation budget: 1 trial at `fmax 0.05` with two stages here, 3 trials
  there, and GenBench re-relaxed at `fmax 0.02` before scoring;
- novelty: `StructureMatcher` here, structure-matcher-based novelty in the
  `single_mlip` config there (0.740).

## What this run changed in the protocol

Two defects surfaced while running it.

**Structure novelty was vacuous.** The score stage hashed each structure with
BAWL and tested membership in `data/unique_fingerprints.parquet`. That file
holds no BAWL hashes: all 4,719,106 entries are LeMat-GenBench *augmented
Wyckoff* fingerprints (`AUG_12_('Ba', '4j', 1):1_...`), which a BAWL hash
(`<md5>_<formula>`) can never equal, so every structure came back novel — the
first ORB funnel reported novelty as exactly 1688/1688. BAWL is now gone;
novelty and uniqueness both use the two-stage filter in
`evaluation/novelty.py`. See the [protocol doc](de_novo_ranking_protocol.md).

**Workers left CUDA contexts on the wrong GPU.** Passing `device="cuda:1"` is
not enough: anything reaching for the *current* device initialises `cuda:0` and
leaves a ~200 MB primary context there, 4 GB across 20 workers, on a card
someone else was training on. `_pin_visible_device` now sets
`CUDA_VISIBLE_DEVICES` per worker before CUDA initialises.

Two environment notes, neither a code defect:

- ORB on CUDA needs `libomp.so.5`, which this machine does not have:
  `/usr/local/magma/lib/libmagma.so`, pulled in by `libtorch_cuda_linalg.so`,
  links it, so every `torch.linalg` call on GPU raises `Error in dlopen`. ORB's
  conservative regressor calls `torch.linalg.det` for the stress, so nothing
  runs. Worked around with a symlink to Intel's ABI-compatible `libiomp5.so`
  (GPU and CPU energies then agree to 0.3 meV on Si-diamond); the real fix is
  to install an LLVM `libomp5` package.
- Worker thread pinning has to happen in the parent. A spawned worker imports
  pandas, and MKL fixes its thread count when it is first loaded, so setting
  `OMP_NUM_THREADS` inside the pool initialiser is too late and every worker
  runs multi-threaded.

## See also

- [The de novo ranking protocol](de_novo_ranking_protocol.md)
- [`upi73i4k` e-hull-conditioning audit](upi73i4k_ehull_conditioning_audit.md)
- [How much does Wyckoff → structure cost us?](cryspr_reconstruction_study.md)
