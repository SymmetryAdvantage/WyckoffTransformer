"""Unfinished optimisations must not enter energy ranking as relaxed trials."""

import json
from unittest.mock import patch

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.filters import FrechetCellFilter

from wyckoff_transformer.cryspr import generator, relaxer


def test_step_limit_rejects_cell_stress_even_when_atomic_forces_vanish(tmp_path):
    atoms = bulk("Cu", "fcc", a=4.2)
    atoms.calc = EMT()
    assert np.abs(atoms.get_forces()).max() < 1e-10

    with pytest.raises(relaxer.RelaxationNotConverged, match="after 0 steps"):
        relaxer.run_ase_relaxer(
            atoms, EMT(), cell_filter=FrechetCellFilter, fix_symmetry=False,
            steps_limit=0, wdir=tmp_path, logfile=tmp_path / "cell.log",
        )
    assert list(tmp_path.glob("*.cif"))  # Keep the failed geometry for diagnosis.


def test_converged_cell_is_returned(tmp_path):
    atoms = relaxer.run_ase_relaxer(
        bulk("Cu", "fcc", a=4.2), EMT(), cell_filter=FrechetCellFilter,
        fix_symmetry=False, wdir=tmp_path, logfile=tmp_path / "cell.log",
    )
    forces = FrechetCellFilter(atoms).get_forces()
    assert np.linalg.norm(forces, axis=1).max() < 0.05


def test_fixed_cell_warmup_may_continue_but_final_stage_must_converge(tmp_path):
    atoms = bulk("Cu", "fcc", a=4.2, cubic=True)
    atoms.positions[0, 0] += 0.1
    # Both stages exhaust a zero-step budget. The warm-up may hand over its
    # geometry; the unconstrained cell stage must reject it.
    with pytest.raises(relaxer.RelaxationNotConverged, match="3_no-sym_cell\\+pos"):
        relaxer.stepwise_relax_stages(
            atoms, EMT(), fix_symmetry=False, rattle=False, steps_limit=0,
            wdir=tmp_path,
        )
    assert list(tmp_path.glob("*_1_fix-cell.cif"))


def test_unconverged_rattle_keeps_converged_baseline_and_records_rejection(tmp_path):
    atoms = relaxer.run_ase_relaxer(
        bulk("Cu", "fcc", a=4.2, cubic=True), EMT(),
        cell_filter=FrechetCellFilter, fix_symmetry=False,
        wdir=tmp_path, logfile=tmp_path / "baseline.log",
    )
    stages = relaxer._rattle_stage(
        atoms, rattle_stdev=0.2, strain_stdev=0.03, rattle_accept=0.001,
        seed=1, logfile=tmp_path / "rattle.log", calculator=EMT(),
        steps_limit=0, wdir=tmp_path,
    )
    assert stages.kept is atoms
    assert stages.rattled is None
    assert stages.rattle_accepted is False
    record = json.loads((tmp_path / relaxer.RATTLE_VERDICT_FILE).read_text())
    assert record["accepted"] is False
    assert record["converged"] is False
    assert record["delta_ev_per_atom"] is None
    assert "did not converge" in record["failure"]


def test_rattle_only_schedule_cannot_fall_back_to_unrelaxed_start(tmp_path):
    with pytest.raises(relaxer.RelaxationNotConverged, match="4_rattle_no-sym"):
        relaxer.stepwise_relax_stages(
            bulk("Cu", "fcc", a=4.2), EMT(), fix_symmetry=False,
            release_symmetry=False, warmup=False, seed=1, steps_limit=0,
            wdir=tmp_path,
        )


def test_rattle_only_schedule_keeps_converged_output_even_if_energy_is_higher(tmp_path):
    from ase.calculators.singlepoint import SinglePointCalculator

    start = bulk("Cu", "fcc", a=4.2)
    candidate = start.copy()
    candidate.calc = SinglePointCalculator(candidate, energy=1.0)
    start_calc = SinglePointCalculator(start, energy=0.0)
    with patch.object(relaxer, "run_ase_relaxer", return_value=candidate):
        stages = relaxer.stepwise_relax_stages(
            start, start_calc, fix_symmetry=False, release_symmetry=False,
            warmup=False, seed=1, wdir=tmp_path,
        )
    assert stages.kept is candidate
    assert stages.rattle_accepted is True


@pytest.mark.parametrize("all_failed", [False, True])
def test_func_run_retries_unconverged_trials(tmp_path, all_failed):
    atoms = bulk("Cu", "fcc", a=3.6)
    atoms.calc = EMT()
    gene = {"group": 225, "species": ["Cu"], "numIons": [4], "sites": [["4a"]]}
    last_trial = (
        relaxer.RelaxationNotConverged("step limit") if all_failed else
        (atoms, atoms.get_potential_energy(), (atoms, atoms.get_potential_energy()))
    )
    with patch.object(generator, "single_pyxtal", return_value=atoms), \
            patch.object(generator, "relax_trial", side_effect=[
                relaxer.RelaxationNotConverged("step limit"),
                last_trial,
            ]) as relax:
        result = generator.func_run(
            "cu", gene, EMT(), output_dir=tmp_path, n_trials=2,
        )
    if all_failed:
        assert result == (None, None, None, None, None)
    else:
        assert result[0] is atoms
    assert relax.call_count == 2


def test_seeded_pyxtal_draws_reproduce_cell_and_positions(tmp_path):
    gene = {"group": 136, "species": ["Ti", "O"], "numIons": [2, 4],
            "sites": [["2a"], ["4f"]]}
    first = generator.single_pyxtal(gene, wdir=tmp_path, seed=12)
    second = generator.single_pyxtal(gene, wdir=tmp_path, seed=12)
    other = generator.single_pyxtal(gene, wdir=tmp_path, seed=13)
    assert first is not None and second is not None and other is not None
    np.testing.assert_array_equal(first.cell.array, second.cell.array)
    np.testing.assert_array_equal(first.positions, second.positions)
    assert not np.array_equal(first.cell.array, other.cell.array)
