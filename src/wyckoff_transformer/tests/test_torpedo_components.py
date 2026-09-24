"""The pieces torpedo-run adds: the subsystem closure, the joint hull, the residuals.

None of these need the dataset cache. The closure is checked on the hand-sized
prior of ``test_system_prior``; the hull on a toy A-B-C reference whose answers
can be worked out on paper; the correction on synthetic residuals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wyckoff_transformer.tests.test_system_prior import FE, LI, MN, O, build_prior


# --------------------------------------------------------------------------- #
# Subsystem closure
# --------------------------------------------------------------------------- #
def _system_counts(plan):
    counts = {}
    for system in plan.systems():
        counts[system] = counts.get(system, 0) + 1
    return counts


def test_the_closure_requests_every_binary_of_a_ternary_target():
    prior = build_prior()
    plan = prior.closure_plan(["Li-O-Mn"], 120, rng=0)
    counts = _system_counts(plan)
    assert set(counts) == {"Li-O-Mn", "Li-O", "Li-Mn", "O-Mn"}, counts
    assert sum(counts.values()) == 120
    assert counts["Li-O-Mn"] == 60, "target_share=0.5 of the target's rows go to it"
    assert all(counts[s] == 20 for s in ("Li-O", "Li-Mn", "O-Mn"))


def test_a_subsystem_shared_by_two_targets_is_asked_for_by_both():
    prior = build_prior()
    plan = prior.closure_plan(["Li-O-Mn", "Li-O-Fe"], 1200, rng=0)
    counts = _system_counts(plan)
    assert sum(counts.values()) == 1200
    assert counts["Li-O"] == 2 * counts["Li-Mn"]
    assert "Li" not in counts, "unaries are left out by default"


def test_the_apportionment_sums_exactly_for_awkward_sizes():
    prior = build_prior()
    for n in (1, 7, 101):
        assert len(prior.closure_plan(["Li-O-Mn", "Li-O-Fe"], n, rng=1)) == n


def test_the_closure_is_reproducible_and_fresh_per_round():
    from wyckoff_transformer.roe.builtin import SubsystemClosureSampler

    prior = build_prior()
    first = prior.closure_plan(["Li-O-Mn"], 200, rng=3)
    again = prior.closure_plan(["Li-O-Mn"], 200, rng=3)
    assert first.systems() == again.systems()
    assert (first.space_groups == again.space_groups).all()

    sampler = SubsystemClosureSampler.__new__(SubsystemClosureSampler)
    sampler.prior = prior
    sampler.targets = ["Li-O-Mn"]
    sampler._rng = np.random.default_rng(0)
    sampler.query = {"min_arity": 2, "target_share": 0.5, "sg_temperature": 1.0}
    one, two = sampler.plan(200), sampler.plan(200)
    assert one.systems() != two.systems(), "a top-up round must not repeat the plan"


def test_targets_are_distinct_systems_of_the_asked_arity():
    prior = build_prior()
    targets = prior.sample_targets(2, arity=3, novel_fraction=0.0, rng=0)
    assert len(set(targets)) == 2
    assert all(len(t) == 3 for t in targets)
    assert set(targets) <= {(LI, O, MN), (LI, O, FE)}


def test_the_plan_survives_json_with_its_own_vocabulary(tmp_path):
    import json

    from wyckoff_transformer.roe.builtin import PlanFileSampler

    plan = build_prior().closure_plan(["Li-O-Mn"], 40, rng=0)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan.manifest()))
    replayed = PlanFileSampler(path).plan(40)
    assert sorted(replayed.systems()) == sorted(plan.systems())
    assert replayed.query["targets"] == ["Li-O-Mn"]


# --------------------------------------------------------------------------- #
# The joint hull
# --------------------------------------------------------------------------- #
def _reference():
    """A, B, C at zero; AB at -0.5 eV/atom formation energy. Total energies = e_form * n."""
    rows = [
        ("r1", "Li1", "Li", 0.0), ("r2", "O1", "O", 0.0), ("r3", "Mn1", "Mn", 0.0),
        ("r4", "Li1O1", "Li-O", -1.0),
    ]
    return pd.DataFrame(rows, columns=["immutable_id", "full_formula", "chemsys",
                                       "energy_corrected"]).set_index("immutable_id")


def _scores(candidates):
    from wyckoff_transformer.formula_energy.joint_hull import joint_hull_scores
    from wyckoff_transformer.formula_energy.screen import HullLookup

    frame = pd.DataFrame(candidates, columns=["formula", "energy"])
    return joint_hull_scores(frame, HullLookup(_reference()))["joint_e_hull"]


def test_a_candidate_alone_is_scored_against_the_reference():
    # H_ref at LiO2 (x_O = 2/3): the LiO-O tie line, -0.5 * (1 - 2/3) / (1/2) = -1/3.
    assert _scores([("Li1O2", 0.0)])[0] == pytest.approx(1 / 3)
    assert _scores([("Li1O2", -0.5)])[0] == pytest.approx(-0.5 + 1 / 3)


def test_polymorphs_are_scored_against_each_other():
    scores = _scores([("Li1O2", -0.6), ("Li1O2", -0.5)])
    assert scores[1] == pytest.approx(0.1), "the higher, against the lower"
    assert scores[0] == pytest.approx(-0.1), (
        "the lower, against the next best thing at its composition -- not against "
        "the reference, or two hundred polymorphs would all look like hits")


def test_a_ternary_is_measured_against_the_binaries_the_cohort_found():
    # Mn-O candidate deep enough that the ternary's hull point drops under it.
    alone = _scores([("Li1Mn1O2", -0.4)])[0]
    assert alone < 0, "below the DFT hull on its own"
    with_binary = _scores([("Li1Mn1O2", -0.4), ("Mn1O1", -1.5)])
    assert with_binary[0] > 0, "but above the hull once the new binary is on it"
    assert with_binary[1] < 0


def test_a_unary_candidate_is_not_put_on_the_hull():
    scores = _scores([("Li1", -0.3), ("Li1O2", -0.2)])
    assert scores[0] == pytest.approx(-0.3), "scored against the elemental reference"
    assert scores[1] == pytest.approx(-0.2 + 1 / 3), (
        "and a generated element predicted below its reference does not re-base "
        "the compounds")


def test_an_uncovered_system_is_undecided():
    scores = _scores([("Na1Cl1", -1.0), ("Li1O2", None)])
    assert np.isnan(scores[0]) and np.isnan(scores[1])


# --------------------------------------------------------------------------- #
# The residual correction
# --------------------------------------------------------------------------- #
def test_a_well_sampled_system_gets_its_own_mean():
    from wyckoff_transformer.gene_energy_residuals import ResidualCorrection

    chemsys = ["Li-O"] * 1000 + ["Fe-O"] * 1000
    residuals = [0.1] * 1000 + [-0.1] * 1000
    correction = ResidualCorrection(chemsys, residuals, kappa=3)
    assert correction.bias("Li-O") == pytest.approx(0.1, abs=1e-3)
    assert correction.bias("Fe-O") == pytest.approx(-0.1, abs=1e-3)


def test_an_unseen_system_backs_off_to_its_subsystems_then_the_global_mean():
    from wyckoff_transformer.gene_energy_residuals import ResidualCorrection

    correction = ResidualCorrection(["Li-O"] * 500 + ["Fe-S"] * 500,
                                    [0.2] * 500 + [0.0] * 500, kappa=1)
    assert correction.bias("Li-Mn-O") == pytest.approx(0.2, abs=1e-2)
    assert correction.bias("K-Na") == pytest.approx(0.1, abs=1e-6)


def test_heavy_shrinkage_is_the_global_offset():
    from wyckoff_transformer.gene_energy_residuals import ResidualCorrection

    correction = ResidualCorrection(["Li-O"] * 10 + ["Fe-S"] * 10,
                                    [0.2] * 10 + [0.0] * 10, kappa=1e9)
    assert correction.bias("Li-O") == pytest.approx(0.1, abs=1e-6)


def test_validation_prefers_the_correction_when_offsets_are_real():
    from wyckoff_transformer.gene_energy_residuals import validate

    rng = np.random.default_rng(0)
    rows = []
    for split in ("val", "test"):
        for system, offset in (("Li-O", 0.1), ("Fe-O", -0.1), ("Mn-O", 0.05)):
            for _ in range(200):
                rows.append((split, system, offset + rng.normal(0, 0.02), rng.uniform(0, 0.2)))
    frame = pd.DataFrame(rows, columns=["split", "chemsys", "residual", "dft_e_hull"])
    report = validate(frame)
    assert report["correction_helps"]
    best = report["kappa"][str(report["best_kappa"])]
    assert best["mae"] < report["raw"]["mae"] / 3


def test_known_gene_lookup_matches_both_words():
    from wyckoff_transformer.gene_energy_residuals import KnownGeneEnergies

    table = KnownGeneEnergies(np.array([5, 1, 5]), np.array([7, 2, 8]), np.array([-1.0, -2.0, -3.0]))
    found = table.lookup([(1, 2), (5, 8), (5, 9), (4, 2)])
    assert found[0] == -2.0 and found[1] == -3.0
    assert np.isnan(found[2]) and np.isnan(found[3])
