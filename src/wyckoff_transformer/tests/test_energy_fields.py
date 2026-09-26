"""Energy-field provenance: the vocabulary refuses what it does not know, and
:func:`check_compatible` refuses what does not mean the same thing."""
from __future__ import annotations

import json
import logging

import pytest
from omegaconf import OmegaConf

from wyckoff_transformer.energy_fields import (
    EnergyField,
    EnergyFieldError,
    EnergySource,
    IncompatibleEnergyFieldError,
    canonical_id,
    check_compatible,
    mlip_hull_reference,
)

LEMAT = EnergySource("dft", "PBE_MP", "none")
MP2020 = EnergySource("dft", "PBE_MP", "MaterialsProject2020Compatibility")


def hull(source=LEMAT, reference="lemat_bulk_pbe") -> EnergyField:
    return EnergyField("energy_above_hull", "per_atom", source, reference)


def test_round_trip_through_json_and_omegaconf():
    field = EnergyField("formation_energy", "per_atom", MP2020, "mp_cdvae_2021", "gene_min")
    assert EnergyField.from_dict(json.loads(json.dumps(field.to_dict()))) == field
    as_omegaconf = OmegaConf.to_container(OmegaConf.create(field.to_dict()))
    assert EnergyField.from_dict(as_omegaconf) == field


@pytest.mark.parametrize("bad", [
    {"quantity": "enthalpy", "extent": "per_atom", "source": LEMAT.to_dict()},
    {"quantity": "energy", "extent": "per_cell", "source": LEMAT.to_dict()},
    {"quantity": "energy", "extent": "total",
     "source": {"method": "dft", "dft_settings": "PBE_54"}},
    {"quantity": "energy", "extent": "total",
     "source": {"method": "dft", "dft_settings": "PBE_MP", "correction": "MP2020"}},
    # A distance to a hull without a hull is meaningless...
    {"quantity": "energy_above_hull", "extent": "per_atom", "source": LEMAT.to_dict()},
    # ...and a bare energy has no reference to name.
    {"quantity": "energy", "extent": "total", "source": LEMAT.to_dict(),
     "reference": "lemat_bulk_pbe"},
    {"quantity": "formation_energy", "extent": "per_atom", "source": LEMAT.to_dict(),
     "reference": "some_hull"},
])
def test_unregistered_labels_are_refused(bad):
    with pytest.raises(EnergyFieldError):
        EnergyField.from_dict(bad)


def test_an_mlip_must_repeat_its_training_scale():
    EnergySource("mlip", "PBE_OMat24", "none", "uma-s-1/omat").validate()
    with pytest.raises(EnergyFieldError, match="trained on PBE_OMat24"):
        EnergySource("mlip", "PBE_MP", "none", "uma-s-1/omat").validate()
    with pytest.raises(EnergyFieldError, match="exact checkpoint"):
        EnergySource("mlip", "PBE_MP").validate()


def test_canonical_names():
    assert canonical_id(hull()) == "energy_above_hull"
    assert canonical_id(EnergyField("energy", "total", LEMAT)) == "energy"
    assert canonical_id(EnergyField("energy", "per_atom", LEMAT), "uncorrected") == \
        "energy_per_atom_uncorrected"
    assert canonical_id(EnergyField("formation_energy", "per_atom", LEMAT, "lemat_bulk_pbe",
                                    "gene_min")) == "gene_min_formation_energy_per_atom"


def test_identical_fields_are_compatible():
    assert check_compatible(hull(), hull(), "test") == []


def test_aggregate_is_not_a_difference_of_scale():
    target = EnergyField("formation_energy", "per_atom", LEMAT, "lemat_bulk_pbe", "gene_min")
    reference = EnergyField("formation_energy", "per_atom", LEMAT, "lemat_bulk_pbe")
    assert check_compatible(target, reference, "test") == []


def test_mp2020_against_raw_is_refused_naming_both_parts():
    with pytest.raises(IncompatibleEnergyFieldError) as error:
        check_compatible(hull(MP2020, "mp_cdvae_2021"), hull(), "mp_20 model vs LeMat hull")
    message = str(error.value)
    assert "source.correction" in message and "reference" in message
    assert "--allow-incompatible-energy" in message


def test_a_different_hull_is_refused():
    mlip = EnergySource("mlip", "PBE_OMat24", "none", "orb-v3-conservative-inf-omat-20250404")
    with pytest.raises(IncompatibleEnergyFieldError):
        check_compatible(hull(), hull(mlip, mlip_hull_reference("orb_conserv_inf")), "test")


def test_unknown_provenance_matches_nothing():
    with pytest.raises(IncompatibleEnergyFieldError, match="unknown"):
        check_compatible(None, hull(), "test")
    with pytest.raises(IncompatibleEnergyFieldError):
        check_compatible(None, None, "test")


def test_allow_warns_and_returns_the_differences(caplog):
    with caplog.at_level(logging.WARNING):
        lines = check_compatible(hull(MP2020, "mp_cdvae_2021"), hull(), "test", allow=True)
    assert len(lines) == 2
    assert "anyway" in caplog.text


def test_every_published_hull_has_a_label():
    """The hull registry and the energy vocabulary must not drift apart."""
    from wyckoff_transformer.energy_fields import mlip_hull_energy_field
    from wyckoff_transformer.evaluation.hull_mlips import HULL_MLIPS

    for hull_type in HULL_MLIPS:
        field = mlip_hull_energy_field(hull_type)
        assert field.reference == mlip_hull_reference(hull_type)
        assert (field.source.method == "dft") == (not HULL_MLIPS[hull_type].is_runnable)
