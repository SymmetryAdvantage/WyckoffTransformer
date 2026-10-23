"""Template novelty must ignore chemistry without losing orbit multiplicities."""
import base64
import hashlib
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from wyckoff_transformer.cli.template_novelty import (
    analyse_artifact,
    fetch_entry,
    source_destination,
)
from wyckoff_transformer.evaluation.novelty import (
    record_to_anonymous_fingerprint,
    record_to_augmented_fingerprint,
)
from wyckoff_transformer.evaluation.template_novelty import (
    anonymize_fingerprint,
    attach_labels,
    label_genes,
    label_structures,
    label_templates,
    load_template_reference,
    template_id,
    template_metrics,
    template_summary,
)


def test_anonymization_preserves_counts_and_equivalent_enumerations():
    record = {"spacegroup_number": 1, "elements": ["Na", "Cl", "Na"],
              "site_symmetries_augmented": [["1", "1", "1"], ["1", "1", "1"]],
              "sites_enumeration_augmented": [[0, 0, 1], [1, 1, 0]]}
    expected = record_to_anonymous_fingerprint(record)
    actual = anonymize_fingerprint(record_to_augmented_fingerprint(record))
    assert actual == expected
    assert template_id(actual) == template_id(expected)
    record["elements"] = ["K", "K", "K"]
    assert anonymize_fingerprint(record_to_augmented_fingerprint(record)) == expected
    record["elements"] = ["K", "K"]
    record["site_symmetries_augmented"] = [["1", "1"]]
    record["sites_enumeration_augmented"] = [[0, 1]]
    assert template_id(record_to_anonymous_fingerprint(record)) != template_id(expected)


def test_representatives_are_valid_deterministic_and_energy_independent():
    structures = pd.DataFrame({
        "has_structure": [True] * 8,
        "valid_structure": [True, True, False, True, True, True, True, True],
        "e_above_hull": [0.2, -0.1, -0.2, 0.1, None, 0.0, 0.0, float("-inf")],
    }, index=[1, 3, 0, 2, 4, 5, 6, 7])
    ids = {0: "a", 1: "a", 2: "b", 3: "a", 4: "c", 5: "known", 7: "d"}
    labels = label_structures(structures, ids, {"known"}, {6: "symmetry failed"})
    assert set(labels.index[labels.unique_template]) == {1, 2, 4, 5, 7}
    assert labels.loc[0, "template_representative_index"] == 1
    assert pd.isna(labels.loc[6, "template_novel"])
    assert labels.loc[6, "template_error"] == "symmetry failed"
    metrics = template_metrics(labels, structures)
    assert metrics["novel_unique"] == 4
    assert metrics["metastable_novel_unique"] == 1  # threshold is inclusive
    assert metrics["stable_novel_unique"] == 0  # the duplicate stable a is not selected
    assert metrics["metastable_fraction_novel_unique"] == 0.25
    assert metrics["no_hull_energy_novel_unique"] == 2  # missing and nonfinite
    assert metrics["unresolved"] == 1
    assert template_metrics(label_structures(structures.iloc[::-1], ids, {"known"}), structures) == metrics


def test_no_novel_templates_has_undefined_fractions():
    structures = pd.DataFrame({"has_structure": [True], "valid_structure": [True],
                               "e_above_hull": [0.0]})
    labels = label_structures(structures, {0: "known"}, {"known"})
    metrics = template_metrics(labels, structures)
    assert metrics["novel_unique"] == metrics["stable_novel_unique"] == 0
    assert metrics["stable_fraction_novel_unique"] is None
    assert metrics["metastable_fraction_novel_unique"] is None
    summary = template_summary({"free": {"template_novelty": metrics}}, "protocol_arm/")
    assert summary["protocol_arm/free/template_novelty/stable_fraction_novel_unique"] is None
    empty = label_templates([], {}, set())
    assert template_metrics(empty)["unique"] == 0


def test_all_sampled_genes_and_relaxed_novelty_are_independent():
    class Fingerprinter:
        def fingerprint(self, gene):
            if gene == "invalid":
                raise ValueError("bad gene")
            return 1, frozenset([frozenset([((gene, "1", 0), 1)])])
    genes = label_genes(["Na", "Cl", "invalid"], set(), Fingerprinter())
    assert len(genes) == 3
    assert genes.unique_template.tolist() == [True, False, False]
    structures = pd.DataFrame({"has_structure": [True], "valid_structure": [True],
                               "e_above_hull": [0.0]})
    labels = label_structures(structures, {0: "known-relaxed"}, {"known-relaxed"})
    attach_labels(structures, labels, genes)
    assert structures.loc[0, "gene_template_novel"]
    assert not structures.loc[0, "template_novel"]


def test_reference_uses_requested_columns_and_reuses_provenance(tmp_path, monkeypatch):
    import wyckoff_transformer.evaluation.template_novelty as module
    record = {"spacegroup_number": 1, "site_symmetries_augmented": [["1"]],
              "sites_enumeration_augmented": [[0]]}
    monkeypatch.setattr(module, "resolve_cache", lambda _: tmp_path)
    monkeypatch.setattr(module, "cache_root", lambda: tmp_path)
    build = {"tool": "test", "options": {"revision": 1}}
    monkeypatch.setattr(module, "build_info", lambda *args: build.copy())
    calls = []
    def splits(cache, names, columns):
        calls.append((names, columns))
        yield "test", pd.DataFrame([record])[list(columns)]
    monkeypatch.setattr(module, "iter_splits", splits)
    first, provenance = load_template_reference(tmp_path, ["test"])
    assert first == {template_id(record_to_anonymous_fingerprint(record))}
    assert "elements" not in calls[0][1]
    assert load_template_reference(tmp_path, ["test"]) == (first, provenance)
    assert len(calls) == 1
    build["options"] = {"revision": 2}
    load_template_reference(tmp_path, ["test"])
    assert len(calls) == 2


@pytest.mark.parametrize("collection,expected", [
    ("protocol_run", ("run", "protocol/")),
    ("protocol_run.arm", ("run", "protocol_arm/")),
    ("protocol_run_ehull005", ("run", "protocol_ehull005/")),
    ("protocol_unknown", (None, None)),
])
def test_source_summary_destination(collection, expected):
    assert source_destination(collection, {"run"}) == expected


def test_backfill_reads_bound_artifact_entries_and_preserves_missing_data(tmp_path, monkeypatch):
    """Exercise saved-artifact inputs through labelled output and provenance."""
    import wyckoff_transformer.cli.template_novelty as cli
    genes = [{"group": 1, "species": ["H"], "numIons": [1], "sites": [["1a"]]}] * 2
    manifest = {"reference_cache": "cache/reference", "reference_splits": "test",
                "mlip": "orb_conserv_inf", "hull": {"revision": "test"}}
    payloads = {
        "manifest.json": json.dumps(manifest).encode(),
        "screen.json": json.dumps({"n_sampled": 2}).encode(),
        "structures.csv": b"index,has_structure,valid_structure,e_above_hull\n0,True,True,0.0\n1,True,True,\n",
        "cifs/0.cif": b"a CIF placeholder read by the mocked worker",
    }
    import gzip
    payloads["wyckoff_genes.json.gz"] = gzip.compress(json.dumps(genes).encode())
    class Entry:
        def __init__(self, name):
            self.extra = {}
            self.path = name
            self.digest = base64.b64encode(hashlib.md5(payloads[name]).digest()).decode()
        def download(self, root):
            path = tmp_path / root / self.path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payloads[self.path])
    class Artifact:
        name = "protocol_run:latest"
        version = "v2"
        entity = "entity"
        project = "project"
        id = "artifact-id"
        digest = "artifact-digest"
        # Raw manifest entries deliberately cannot be downloaded.
        manifest = SimpleNamespace(entries={name: None for name in payloads})
        def get_entry(self, name):
            return Entry(name)
    class Pool:
        def map(self, function, tasks, **kwargs):
            return [{"template_id": "novel", "error": None} for _ in tasks]
    class Downloader:
        def map(self, function, tasks):
            return map(function, tasks)
    class Fingerprinter:
        def fingerprint(self, gene):
            return 1, frozenset([frozenset([(("H", "1", 0), 1)])])
    monkeypatch.setattr(cli, "load_template_reference", lambda *args: (set(), {"test": "reference"}))
    root = tmp_path / "analysis"
    out, report, record = analyse_artifact(Artifact(), root, Pool(), Downloader(), {}, Fingerprinter())
    assert report["gene"]["template_novelty"]["rows"] == 2
    assert report["free"]["template_novelty"]["stable_novel_unique"] == 1
    assert report["free"]["template_novelty"]["unresolved_valid_structure"] == 1
    assert record["source_artifact"] == "entity/project/protocol_run:v2"
    assert record["missing_data"] == ["cifs/1.cif", "structures_fixed_symmetry.csv"]
    assert len(pd.read_csv(out / "gene_templates.csv")) == 2
    result = pd.read_csv(out / "structures.csv")
    assert result.template_resolved.tolist() == [True, False]
    assert pd.isna(result.loc[1, "template_novel"])
    assert "missing" in result.loc[1, "template_error"]
    assert json.loads((out / "template_novelty.json").read_text()) == report
    # An existing local file is only reused when it has the source digest.
    entry = Entry("screen.json")
    (root / "screen.json").write_text("corrupt")
    fetch_entry(entry, root)
    assert (root / "screen.json").read_bytes() == payloads["screen.json"]


def test_score_stage_persists_template_metrics_for_both_tracks(tmp_path, monkeypatch):
    """Run the score stage on a real CIF with an in-memory hull and reference."""
    import gzip

    from pymatgen.core import Lattice, Structure

    from wyckoff_transformer.cli import protocol as cli
    from wyckoff_transformer.evaluation import (
        hull_energy,
        structure_novelty,
        structure_validity,
    )
    from wyckoff_transformer.evaluation import template_novelty as templates
    from wyckoff_transformer.evaluation.protocol import GeneScreen, write_screen

    genes = [{"group": 225, "species": ["Na", "Cl"], "numIons": [4, 4],
              "sites": [["4a"], ["4b"]]}]
    gene_path = tmp_path / "wyckoff_genes.json.gz"
    with gzip.open(gene_path, "wt") as handle:
        json.dump(genes, handle)
    write_screen(GeneScreen(n_sampled=1, valid=[0], counts={0: 1}, novel=[0]), tmp_path / "screen.json")
    identity = {"cache": "reference", "splits": ["train", "val", "test"]}
    manifest = {"lineage": {"screen.json": {"reference": identity}}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    structure = Structure.from_spacegroup(225, Lattice.cubic(5.64), ["Na", "Cl"],
                                         [[0, 0, 0], [0.5, 0.5, 0.5]])
    for directory, filename in (("cifs", "structures.csv"), ("cifs_fixed_symmetry", "structures_fixed_symmetry.csv")):
        (tmp_path / directory).mkdir()
        structure.to(filename=str(tmp_path / directory / "0.cif"))
        pd.DataFrame({"index": [0], "has_structure": [True], "energy": [-10.0]}).to_csv(tmp_path / filename, index=False)
    class Hull:
        def __init__(self, *args):
            self.provenance = {"source": "test"}
        def energy_above_hull(self, *args):
            return 0.1
    monkeypatch.setattr(hull_energy, "HullEnergyCalculator", Hull)
    monkeypatch.setattr(templates, "load_template_reference", lambda *args: (set(), {"source": identity}))
    monkeypatch.setattr(structure_novelty, "build_novelty_reference", lambda *args, **kw: pd.DataFrame(columns=["fingerprint", "structure"]))
    monkeypatch.setattr(structure_validity, "is_valid", lambda structure: True)
    cli.stage_score(SimpleNamespace(input=gene_path, output_dir=tmp_path,
                                   mlip="orb_conserv_inf", reference_cache=tmp_path / "reference",
                                   reference_splits="train,val,test", lemat_cif_csv=tmp_path / "reference.csv.gz"))
    report = json.loads((tmp_path / "funnel.json").read_text())
    assert report["gene"]["template_novelty"]["novel_unique"] == 1
    for section, filename in (("free", "structures.csv"), ("fixed_symmetry", "structures_fixed_symmetry.csv")):
        assert report[section]["template_novelty"]["metastable_fraction_novel_unique"] == 1.0
        assert report[section]["template_novelty"]["stable_fraction_novel_unique"] == 0.0
        labels = pd.read_csv(tmp_path / filename)
        assert labels.loc[0, "template_id"] == labels.loc[0, "gene_template_id"]
        assert labels.loc[0, "novel_unique_template"]
    assert "template_novelty" in json.loads((tmp_path / "manifest.json").read_text())
    assert (tmp_path / "gene_templates.csv").is_file()


def test_publish_only_updates_template_keys_and_retains_undefined_rates(tmp_path, monkeypatch):
    import wandb

    from wyckoff_transformer.cli.template_novelty import publish_result

    class Summary:
        """Like the W&B public Summary, supports keys() but not iteration."""
        def __init__(self):
            self.data = {"training/loss": 1.0, "protocol/free/metasun_per_sampled_gene": 0.2,
                         "protocol/free/template_novelty/stale": 42}
        def keys(self):
            return self.data.keys()
        def __delitem__(self, key):
            del self.data[key]
        def update(self, data):
            self.data.update(data)
    summary = Summary()
    class Artifact:
        version = "v3"
        def __init__(self, **kwargs):
            self.kwargs = kwargs
        def add_dir(self, directory):
            assert directory == str(tmp_path)
        def wait(self):
            return self
    monkeypatch.setattr(wandb, "Artifact", Artifact)
    source = SimpleNamespace(summary=summary)
    api = SimpleNamespace(run=lambda path: source)
    analysis = SimpleNamespace(use_artifact=lambda path: None, log_artifact=lambda artifact: artifact)
    report = {"free": {"template_novelty": {"novel_unique": 0,
                                           "stable_fraction_novel_unique": None}}}
    run_id, result = publish_result(analysis, api, SimpleNamespace(name="protocol_run:v0"),
                                    tmp_path, report, {"source_artifact": "e/p/protocol_run:v0"}, {"run"}, "e/p")
    assert run_id == "run"
    assert result == "e/p/template_novelty_protocol_run:v3"
    assert summary.data["training/loss"] == 1.0
    assert summary.data["protocol/free/metasun_per_sampled_gene"] == 0.2
    assert "protocol/free/template_novelty/stale" not in summary.data
    assert summary.data["protocol/free/template_novelty/novel_unique"] == 0
    assert summary.data["protocol/free/template_novelty/stable_fraction_novel_unique"] is None
