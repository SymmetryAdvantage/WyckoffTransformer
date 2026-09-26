"""Dataset manifests: every one in the repository validates, fields resolve by name,
alias or quantity, and obsolete datasets are refused for new work."""
from __future__ import annotations

import logging

import pytest

from wyckoff_transformer import dataset_manifest as dm
from wyckoff_transformer.energy_fields import check_compatible

CURRENT = {"lemat_bulk_fmax1_stress", "lemat_bulk_fmax1_stress_ehull01", "mp_20",
           "mp_2026_gga_gap", "formula_energy"}


def test_every_manifest_validates_and_only_the_agreed_datasets_are_current():
    manifests = dm.all_manifests()
    assert {n for n, m in manifests.items() if m.status == "current"} == CURRENT
    for name, manifest in manifests.items():
        assert manifest.name == name


def test_a_slice_inherits_its_parents_labels():
    parent = dm.load_manifest("lemat_bulk_fmax1_stress")
    child = dm.load_manifest("lemat_bulk_fmax1_stress_ehull01")
    assert child.fields_record() == parent.fields_record()


def test_the_superseded_lemat_variant_shares_formation_energy_and_hull():
    """Same reference entry set; it only covers fewer rows."""
    old = dm.load_manifest("lemat_bulk_fmax1")
    new = dm.load_manifest("lemat_bulk_fmax1_stress")
    for name in ("formation_energy_per_atom", "energy_above_hull",
                 "gene_min_formation_energy_per_atom"):
        assert check_compatible(new.resolve(name).energy, old.resolve(name).energy, name) == []


def test_mp_20s_hull_resolves_by_its_old_name_and_is_not_lemats(caplog):
    mp_20 = dm.load_manifest("mp_20")
    with caplog.at_level(logging.WARNING):
        field = mp_20.resolve("e_above_hull")
    assert field.name == "energy_above_hull" and field.column == "e_above_hull"
    assert "superseded name" in caplog.text
    lemat = dm.load_manifest("lemat_bulk_fmax1_stress").resolve("energy_above_hull")
    with pytest.raises(ValueError):
        check_compatible(lemat.energy, field.energy, "test")


def test_a_bare_quantity_resolves_to_the_rows_own_value():
    stress = dm.load_manifest("lemat_bulk_fmax1_stress")
    assert stress.resolve("formation_energy").name == "formation_energy_per_atom"
    assert dm.load_manifest("mp_2026_gga_gap").resolve("energy").name == "energy_per_atom"


def test_an_undeclared_field_is_an_error_not_a_guess():
    with pytest.raises(dm.ManifestError, match="declares no field"):
        dm.load_manifest("mp_20").resolve("delta_e_polymorph")


def test_column_in_finds_old_and_new_spellings():
    field = dm.load_manifest("mp_20").resolve("energy_above_hull")
    assert field.column_in({"e_above_hull", "band_gap"}) == "e_above_hull"
    assert field.column_in({"energy_above_hull"}) == "energy_above_hull"
    with pytest.raises(dm.ManifestError):
        field.column_in({"band_gap"})


def test_misnamed_energy_fields_are_refused():
    raw = {"name": "x", "status": "current", "fields": {"e_hull": {"energy": {
        "quantity": "energy_above_hull", "extent": "per_atom", "reference": "lemat_bulk_pbe",
        "source": {"method": "dft", "dft_settings": "PBE_MP"}}}}}
    with pytest.raises(dm.ManifestError, match="canonical name is 'energy_above_hull'"):
        dm.parse_manifest(raw)


def test_an_energy_without_provenance_is_refused():
    raw = {"name": "x", "status": "current",
           "fields": {"formation_energy_per_atom": {"quantity": "formation_energy"}}}
    with pytest.raises(dm.ManifestError, match="without provenance"):
        dm.parse_manifest(raw)


def test_an_unregistered_dataset_is_obsolete():
    assert "no manifest" in dm.obsolete_reason("matbench_discovery_mp_2022_0.01_1_63")
    with pytest.raises(dm.ObsoleteDatasetError):
        dm.refuse_if_obsolete_dataset("no_such_dataset")


@pytest.mark.parametrize("name", ["lemat_bulk_fmax1", "lemat_bulk_ehull", "alex_mp_20"])
def test_obsolete_datasets_are_refused_unless_allowed(name, caplog):
    with pytest.raises(dm.ObsoleteDatasetError, match="--allow-obsolete-dataset"):
        dm.refuse_if_obsolete_dataset(name, "training")
    with caplog.at_level(logging.WARNING):
        dm.refuse_if_obsolete_dataset(name, "training", allow=True)
    assert "used anyway" in caplog.text


def test_a_source_is_not_a_training_dataset():
    with pytest.raises(dm.ObsoleteDatasetError, match="source dataset"):
        dm.refuse_if_obsolete_dataset("lemat-bulk")


@pytest.mark.parametrize("name", sorted(CURRENT))
def test_current_datasets_pass(name):
    dm.refuse_if_obsolete_dataset(name)


def test_obsolescence_warns_once_per_dataset(caplog, monkeypatch):
    monkeypatch.setattr(dm, "_warned", set())
    with caplog.at_level(logging.WARNING):
        dm.warn_if_obsolete_dataset("perov_5", "eval")
        dm.warn_if_obsolete_dataset("/some/cache/perov_5/", "eval")
    assert caplog.text.count("is obsolete") == 1


def test_cache_records_are_checked_against_the_manifest():
    record = dm.load_manifest("mp_20").fields_record()
    dm.check_cache_matches("mp_20", record)
    dm.check_cache_matches("mp_20", None)
    changed = dict(record)
    changed["energy_above_hull"] = dm.load_manifest(
        "lemat_bulk_fmax1_stress").fields_record()["energy_above_hull"]
    with pytest.raises(dm.CacheManifestMismatch, match="energy_above_hull"):
        dm.check_cache_matches("mp_20", changed)
