"""Gene-level motif flags on genes whose answer is known."""
import pytest

from wyckoff_transformer.evaluation.gene_motifs import (
    gene_flags,
    prototype_key,
    site_point_group,
)
from wyckoff_transformer.evaluation.protocol import GeneFingerprinter


@pytest.fixture(scope="module")
def fp():
    return GeneFingerprinter()


def test_site_point_groups():
    assert site_point_group(221, "c") == "4/mmm"
    assert site_point_group(194, "a") == "-3m"
    assert site_point_group(194, "d") == "-6m2"
    assert site_point_group(167, "c") == "3"
    assert site_point_group(225, "c") == "-43m"


def test_2h_d0_chain(fp):
    f = gene_flags(fp.record({"group": 194, "sites": [["2c"], ["2a"], ["6h"]],
                              "species": ["Ba", "Ti", "O"], "numIons": [2, 2, 6]}))
    assert f["reference_prototype"] == "2H_BaNiO3" and f["fs_d0_chain_2h"]


def test_2h_not_d0(fp):
    f = gene_flags(fp.record({"group": 194, "sites": [["2d"], ["2a"], ["6h"]],
                              "species": ["Ba", "Mn", "O"], "numIons": [2, 2, 6]}))
    assert f["reference_prototype"] == "2H_BaNiO3" and not f["fs_d0_chain_2h"]


def test_cubic_perovskite_tolerance(fp):
    f = gene_flags(fp.record({"group": 221, "sites": [["1b"], ["1a"], ["3d"]],
                              "species": ["Sr", "Ti", "O"], "numIons": [1, 1, 3]}))
    assert f["reference_prototype"] == "cubic_perovskite"
    assert f["tolerance_t"] == pytest.approx(1.0, abs=0.02) and not f["perovskite_t_out"]
    assert f["sojt_pinned_d0"]          # Ti4+ on m-3m cannot off-centre


def test_prototype_key_ignores_elements_and_setting(fp):
    a = fp.record({"group": 221, "sites": [["1a"], ["1b"], ["3c"]],
                   "species": ["Sr", "Ti", "O"], "numIons": [1, 1, 3]})
    b = fp.record({"group": 221, "sites": [["1b"], ["1a"], ["3d"]],
                   "species": ["Ba", "Zr", "F"], "numIons": [1, 1, 3]})
    assert prototype_key(a) == prototype_key(b)


def test_intermetallic_has_no_flags(fp):
    f = gene_flags(fp.record({"group": 221, "sites": [["1a"], ["1b"]],
                              "species": ["Cu", "Zn"], "numIons": [1, 1]}))
    assert not f["oxidation_ok"]
