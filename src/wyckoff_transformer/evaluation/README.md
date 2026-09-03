Various codes for evaluation of the generated data.

`generated_dataset.GeneratedDataset` provides a unified interface for no less
than 24 different data storage formats.

## The de novo ranking protocol

`protocol.py` and `wyckoff_transformer.cli.protocol` implement the cascade from
sampled Wyckoff genes to MetaSUN used to rank model variants. See
[docs/de_novo_ranking_protocol.md](../../../docs/de_novo_ranking_protocol.md).

The modules below exist so that scoring needs no LeMat-GenBench install —
it is not on PyPI and its pinned CUDA `torch-scatter` wheels hold torch at 2.6,
which cannot coexist with this package's `torch ==2.11.0`. Each is pinned to the
original by `tests/test_genbench_equivalence.py`, which is the only place
LeMat-GenBench is referenced and skips when it is absent.

| module | what it replaces |
|---|---|
| `hull_mlips.py` | the MLIP↔hull pairing; refuses potentials with no published hull |
| `hull_energy.py` | `lemat_genbench.preprocess.reference_energies.get_energy_above_hull` |
| `structure_validity.py` | `lemat_genbench.metrics.validity_metrics.OverallValidityMetric` |
| `oxidation_state.py` | `lemat_genbench.utils.oxidation_state` (vendored verbatim, Apache-2.0) |
| `bawl.py` | `material_hasher.hasher.bawl` (Apache-2.0) |
| `bawl_reference.py` | `lemat_genbench.data.reference_fingerprint_loader` |
| `data/` | the ICSD oxidation-state priors those need |

`hull_mlips.verify_hull_energies()` checks that a locally built calculator
reproduces the published per-structure energies. Use it whenever an MLIP
dependency changes: a checkpoint mismatch is otherwise silent, and shifts every
`e_above_hull` derived from that hull.
