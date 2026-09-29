"""An ordered alloy series should recover its shared substitutional parent."""

import importlib.util
from pathlib import Path

from pymatgen.core import Lattice, Structure

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts/infer_cu_ge_te_disorder_parents.py"
_SPEC = importlib.util.spec_from_file_location("infer_cu_ge_te_disorder_parents", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_cluster_signature = _MODULE._cluster_signature
_read_child = _MODULE._read_child
_write_parent_cif = _MODULE._write_parent_cif


def test_cu_ge_orderings_share_fcc_parent_across_compositions(tmp_path):
    fcc = Structure.from_spacegroup(
        "Fm-3m", Lattice.cubic(3.62), ["Cu"], [[0, 0, 0]],
    )
    first = fcc.copy()
    first.make_supercell([2, 2, 1])
    first.replace(0, "Ge")
    second = first.copy()
    second.replace(1, "Ge")

    candidates = []
    for number, structure in enumerate((first, second)):
        path = tmp_path / f"alloy_{number}.cif"
        structure.to(filename=str(path), fmt="cif")
        result = _read_child((f"ordered_{number}", str(path),
                              structure.composition.reduced_formula, 0.02, 0.15))
        assert result[-1] is None
        assert len(result[4]) == 1
        candidates.append((result[0], result[1], result[2], *result[4][0][1:]))

    signature = result[4][0][0]
    assert signature[0] == "Cu-Ge"
    assert signature[1] == 225
    assert signature[2] == 1
    _, families, memberships = _cluster_signature((signature, candidates, 1.15))
    assert len(families) == 1
    assert len(families[0]["formulas"]) == 2
    assert {row["local_id"] for row in memberships} == {0}
    families[0]["id"] = "test_parent"
    parent_cif = tmp_path / "parent.cif"
    _write_parent_cif(families[0], parent_cif)
    assert "_symmetry_Int_Tables_number   225" in parent_cif.read_text()


def test_binary_and_ternary_masks_are_kept_separate(tmp_path):
    structure = Structure(
        Lattice.cubic(5.7), ["Cu", "Ge", "Te"],
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]],
    )
    path = tmp_path / "ternary.cif"
    structure.to(filename=str(path), fmt="cif")
    result = _read_child(("ternary", str(path), "CuGeTe", 0.03, 0.15))
    assert {candidate[0][0] for candidate in result[4]} == {
        "Cu-Ge", "Cu-Te", "Ge-Te",
    }
