"""A trained model's field provenance: resolved at training, recorded beside the
weights, inferred for older runs, and checked when the model meets other data."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf

from wyckoff_transformer import dataset_manifest as dm
from wyckoff_transformer import field_provenance as fp
from wyckoff_transformer.energy_fields import IncompatibleEnergyFieldError


def config(dataset: str, condition=None, target: str | None = None) -> dict:
    args = {"condition_feature": condition}
    if target is not None:
        args.update(target="Scalar", target_name=target)
    else:
        args["target"] = "NextToken"
    return {"dataset": dataset, "model": {"WyckoffTrainer_args": args}}


def test_roles_come_from_the_condition_and_the_scalar_target():
    roles = fp.requested_fields(config("mp_20", ["energy_above_hull", "band_gap"],
                                       "formation_energy_per_atom"))
    assert roles == {"condition:energy_above_hull": "energy_above_hull",
                     "condition:band_gap": "band_gap", "target": "formation_energy_per_atom"}
    assert fp.requested_fields(config("mp_20")) == {}


def test_an_old_name_resolves_to_the_canonical_field_and_its_meaning():
    provenance = fp.build_field_provenance(config("mp_20", "e_above_hull"))
    entry = provenance["fields"]["condition:e_above_hull"]
    assert entry["field"] == "energy_above_hull"
    assert entry["energy"]["source"]["correction"] == "MaterialsProject2020Compatibility"
    assert entry["energy"]["reference"] == "mp_cdvae_2021"


def test_a_dataset_without_labels_leaves_the_provenance_unknown():
    provenance = fp.build_field_provenance(config("alex_mp_20", "energy_above_hull"))
    assert provenance["fields"]["condition:energy_above_hull"]["energy"] is None


def test_training_refuses_a_field_the_manifest_does_not_declare():
    with pytest.raises(dm.ManifestError, match="declares no field"):
        fp.require_resolved(config("mp_20", "delta_e_polymorph"))
    fp.require_resolved(config("mp_20", "e_above_hull"))


def test_recorded_provenance_wins_over_inference(tmp_path, caplog):
    recorded = fp.build_field_provenance(config("mp_20", "energy_above_hull"))
    # From the W&B config...
    assert fp.load_field_provenance({**config("lemat_bulk_fmax1_stress", "energy_above_hull"),
                                     fp.CONFIG_KEY: recorded}) == recorded
    # ...or from the run directory.
    fp.write_field_provenance(recorded, tmp_path)
    loaded = fp.load_field_provenance(config("lemat_bulk_fmax1_stress", "energy_above_hull"),
                                      tmp_path)
    assert loaded == json.loads(json.dumps(recorded))
    with caplog.at_level(logging.WARNING):
        inferred = fp.load_field_provenance(config("mp_20", "energy_above_hull"))
    assert inferred["recorded"] is False and "inferred" in caplog.text


def test_a_model_trained_on_the_superseded_lemat_variant_fits_the_current_one():
    regressor = fp.build_field_provenance(
        config("lemat_bulk_fmax1", "max_force", "gene_min_formation_energy_per_atom"))
    assert fp.check_against_dataset(regressor, "lemat_bulk_fmax1_stress", "test") == []


def test_an_mp_20_model_does_not_fit_lemat():
    model = fp.build_field_provenance(config("mp_20", "energy_above_hull"))
    with pytest.raises(IncompatibleEnergyFieldError) as error:
        fp.check_against_dataset(model, "lemat_bulk_fmax1_stress", "test")
    assert "source.correction" in str(error.value) and "reference" in str(error.value)
    lines = fp.check_against_dataset(model, "lemat_bulk_fmax1_stress", "test", allow=True)
    assert len(lines) == 2


def test_an_unlabelled_field_fits_nothing():
    model = fp.build_field_provenance(config("alex_mp_20", "energy_above_hull"))
    with pytest.raises(IncompatibleEnergyFieldError, match="unknown"):
        fp.check_against_dataset(model, "alex_mp_20", "test")


def test_tensors_stored_under_an_old_name_are_found_under_the_new_one():
    train = {"e_above_hull": torch.zeros(3)}
    val = {"energy_above_hull": torch.ones(2)}
    fp.alias_stored_columns("mp_20", ["energy_above_hull", "e_above_hull"], [train, val])
    assert train["energy_above_hull"] is train["e_above_hull"]
    assert val["e_above_hull"] is val["energy_above_hull"]


def test_resuming_ignores_the_provenance_key(tmp_path):
    from wyckoff_transformer.trainer import check_resume_config

    base = {"dataset": "mp_20", "model": {"WyckoffTrainer_args": {"target": "NextToken"}}}
    OmegaConf.save(base, tmp_path / "config.yaml")
    check_resume_config({**base, fp.CONFIG_KEY: {"fields": {}}}, tmp_path / "config.yaml")


def test_a_legacy_formula_ensemble_is_inferred_from_its_energy_scale(tmp_path):
    from wyckoff_transformer.formula_energy import train as T

    path = tmp_path / "ensemble.pt"
    torch.save({"config": {"energy_scale": T.LEMAT_BULK_PBE_ENERGY_SCALE},
                "provenance_features": [], "state_dicts": []}, path)
    formula_table = (dm.load_manifest("formula_energy"), "formula_table")
    with patch.object(dm, "table_for_file", lambda table: formula_table):
        provenance = T.load_ensemble_field_provenance(path)
    assert provenance["recorded"] is False
    assert provenance["fields"]["target"]["reference"] == "lemat_bulk_pbe"
    torch.save({"config": {"energy_scale": None}, "provenance_features": [],
                "state_dicts": []}, path)
    assert T.load_ensemble_field_provenance(path) is None
