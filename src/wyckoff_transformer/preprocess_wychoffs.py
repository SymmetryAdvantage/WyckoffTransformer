from collections import defaultdict, Counter
import json
import string
from pathlib import Path
import numpy as np
from pyxtal import Group

from wyckoff_transformer.wyckoff_processor import (
    ENGINEERS_DIR, FeatureEngineer, WyckoffProcessor, save_frozen_table)


N_3D_SPACEGROUPS = 230
WYCKOFF_MAPPINGS_FILENAME = "wyckoffs_enumerated_by_ss.json"
_PACKAGE_MAPPINGS_PATH = Path(__file__).parent / WYCKOFF_MAPPINGS_FILENAME
#: Harmonic signatures are rounded to this many decimals. Unrounded, they carry ~1e-16 of
#: floating-point noise that moves with the numpy/scipy/libm build, and that noise is
#: what used to decide which of two degenerate Wyckoff positions got which cluster.
#: Distances to cluster centres closer than this are a tie. KMeans' multithreaded
#: reductions move the centres by ~1e-15 from one run to the next, so exact comparison
#: of distances is not reproducible even within one environment.


def generate_wyckoff_mappings(
    output_file: Path = _PACKAGE_MAPPINGS_PATH) -> None:
    """Generate Wyckoff position mappings and save as JSON.

    Produces three mappings for all 230 3-D space groups:
      - enum_from_ss_letter[sg][letter]       -> enumeration index
      - letter_from_ss_enum[sg][site_symm][i] -> Wyckoff letter
      - ss_from_letter[sg][letter]            -> site symmetry string

    No dependency on FeatureEngineer, safe to call during package build.
    """
    enum_from_ss_letter = defaultdict(dict)
    ss_from_letter = defaultdict(dict)
    letter_from_ss_enum = defaultdict(lambda: defaultdict(dict))

    for spacegroup_number in range(1, N_3D_SPACEGROUPS + 1):
        group = Group(spacegroup_number)
        ss_counts = Counter()
        for wp in group.Wyckoff_positions[::-1]:
            wp.get_site_symmetry()
            site_symm = wp.site_symm
            ss_from_letter[spacegroup_number][wp.letter] = site_symm
            enum_from_ss_letter[spacegroup_number][wp.letter] = ss_counts[site_symm]
            letter_from_ss_enum[spacegroup_number][site_symm][ss_counts[site_symm]] = wp.letter
            ss_counts[site_symm] += 1

    output_file = Path(output_file)
    output_file.parent.mkdir(exist_ok=True, parents=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "enum_from_ss_letter": {
                str(sg): v for sg, v in enum_from_ss_letter.items()},
            "letter_from_ss_enum": {
                str(sg): {ss: {str(e): letter for e, letter in ed.items()}
                          for ss, ed in sd.items()}
                for sg, sd in letter_from_ss_enum.items()},
            "ss_from_letter": {
                str(sg): v for sg, v in ss_from_letter.items()},
        }, f)


def enumerate_wychoffs_by_ss(
    output_file: Path = _PACKAGE_MAPPINGS_PATH,
    engineers_dir: Path = ENGINEERS_DIR):
    """
    Enumerates all Wyckoff positions by site symmetry.

    Args:
        output_file (Path, optional): The output file for Wyckoff mappings JSON.
        engineers_dir (Path, optional): Directory to write FeatureEngineer JSON files.
    """
    engineers_dir = Path(engineers_dir)
    engineers_dir.mkdir(exist_ok=True, parents=True)

    enum_from_ss_letter = defaultdict(dict)
    ss_from_letter = defaultdict(dict)
    letter_from_ss_enum = defaultdict(lambda: defaultdict(dict))
    multiplicity_from_ss_enum = dict()
    max_multiplicity = 0
    for spacegroup_number in range(1, N_3D_SPACEGROUPS + 1):
        group = Group(spacegroup_number)
        ss_counts = Counter()
        # [::-1] doesn't matter in principle,
        # but serves a cosmetic purpose, so that
        # a comes before b, etc.
        for wp in group.Wyckoff_positions[::-1]:
            wp.get_site_symmetry()
            site_symm = wp.site_symm
            ss_from_letter[spacegroup_number][wp.letter] = site_symm
            enum_from_ss_letter[spacegroup_number][wp.letter] = ss_counts[site_symm]
            letter_from_ss_enum[spacegroup_number][site_symm][ss_counts[site_symm]] = wp.letter
            multiplicity_from_ss_enum[(spacegroup_number, site_symm, ss_counts[site_symm])] = wp.multiplicity
            max_multiplicity = max(max_multiplicity, wp.multiplicity)
            ss_counts[site_symm] += 1
    generate_wyckoff_mappings(output_file)

    def _save_engineer(engineer: FeatureEngineer, name: str) -> None:
        serialised = WyckoffProcessor._serialise_feature_engineer(engineer)
        (engineers_dir / f"{name}.json").write_text(
            serialised.model_dump_json(indent=2), encoding="utf-8")

    multiplicity_engineer = FeatureEngineer(
        multiplicity_from_ss_enum, ("spacegroup_number", "site_symmetries", "sites_enumeration"),
        name="multiplicity",
        stop_token=max_multiplicity + 1, mask_token=max_multiplicity + 2, pad_token=0, default_value=0)
    _save_engineer(multiplicity_engineer, "multiplicity")


def site_symmetry_ops_vectors() -> tuple[dict, int]:
    """The site symmetry operations encoding, shared by both engineers built from it.

    For every (sg, site_symm) pair across the 230 3-D space groups, looks up the
    canonical (first-encountered) Wyckoff position and stores
    ``wp.get_site_symmetry_object().to_matrix_representation().ravel()`` with constant
    columns dropped across the full enumeration. Mirrors the constant-removal logic in
    ``SpaceGroupEncoder.from_sg_set``. Per-SG uniqueness is asserted because the model
    receives the SG via the start token, and that is the relevant disambiguation scope.

    Returns:
        (reduced, n_varying): the (sg, site_symm) -> float32 vector mapping, and the
        length of those vectors.
    """
    raw = {}
    for spacegroup_number in range(1, N_3D_SPACEGROUPS + 1):
        group = Group(spacegroup_number)
        seen_ss = set()
        for wp in group.Wyckoff_positions:
            wp.get_site_symmetry()
            site_symm = wp.site_symm
            if site_symm in seen_ss:
                continue
            seen_ss.add(site_symm)
            sso = wp.get_site_symmetry_object()
            raw[(spacegroup_number, site_symm)] = sso.to_matrix_representation().ravel().astype(np.float32)

    matrix = np.stack(list(raw.values()))
    col_sum = matrix.sum(axis=0)
    varying = (col_sum > 0) & (col_sum < matrix.shape[0])
    n_varying = int(varying.sum())

    reduced = {key: vec[varying] for key, vec in raw.items()}

    # Sanity: within each SG the resulting vectors must be pairwise distinct
    by_sg = defaultdict(dict)
    for (sg, ss), vec in reduced.items():
        key = vec.tobytes()
        if key in by_sg[sg] and by_sg[sg][key] != ss:
            raise ValueError(
                f"Site symmetry encoding is not unique within SG {sg}: "
                f"{by_sg[sg][key]!r} and {ss!r} share a vector")
        by_sg[sg][key] = ss

    return reduced, n_varying


def flagged_site_symmetry_ops_vectors() -> tuple[dict, int]:
    """``site_symmetry_ops_vectors`` plus the column that separates it from a service token.

    Site symmetry "1" -- the general position, present in every one of the 230 groups and
    the commonest site symmetry there is -- has no operations beyond the identity, so its
    reduced vector is all zeros. The sibling engineers give STOP and PAD all-zero vectors
    too, so without the extra column 230 real (sg, site_symm) pairs encode identically to
    the end of a sequence. That matters most where this field is the *only* representation
    of the site symmetry (embedding_size.site_symmetries: 0), since then nothing else in the
    input distinguishes them.

    One extra column carries the distinction: 1 for a real (sg, site_symm) pair, 0 for the
    service tokens. Per-SG uniqueness is unaffected -- a constant column cannot merge rows.

    The collision is a defect by inspection; its cost was not established by measurement. On
    a 4k-structure LeMat-Bulk subset over 1200 steps, with and without the flag came out at
    95 and 109 valid structures out of 256, inside the run-to-run spread of that setup.
    """
    reduced, n_varying = site_symmetry_ops_vectors()
    width = n_varying + 1
    flagged = {
        key: np.concatenate([vec, np.ones(1, dtype=np.float32)]).astype(np.float32)
        for key, vec in reduced.items()}
    return flagged, width


def service_ops_vectors(width: int) -> dict:
    """MASK, STOP and PAD vectors for a site-symmetry-operations field of this width.

    MASK is all ones over the operation columns, following the sibling engineers, and clears
    the real-value flag, which keeps it distinct from every real vector (no site symmetry
    fills every column). STOP and PAD stay all zeros, including the flag: they are
    interchangeable to this field, and the categorical cascade fields carry their own STOP
    and PAD tokens to tell them apart.
    """
    mask = np.ones(width, dtype=np.float32)
    mask[-1] = 0.
    return {
        "mask": mask,
        "stop": np.zeros(width, dtype=np.float32),
        "pad": np.zeros(width, dtype=np.float32)}


def build_site_symmetry_ops_engineer(
    engineers_dir: Path = ENGINEERS_DIR) -> "FeatureEngineer":
    """Build and save the ``site_symmetry_ops`` engineered field: the dense form.

    Each token carries the full operations vector, so the field is stored per
    (structure, position) and consumed by the model as a ``pass_through_vector``. That
    costs n_structures * max_len * n_varying values of resident memory, which is fine
    for mp_20 and prohibitive for LeMat-Bulk -- see
    ``build_site_symmetry_ops_id_engineer`` for the form that scales.
    """
    engineers_dir = Path(engineers_dir)
    engineers_dir.mkdir(exist_ok=True, parents=True)

    values, width = flagged_site_symmetry_ops_vectors()
    service = service_ops_vectors(width)

    engineer = FeatureEngineer(
        values,
        ("spacegroup_number", "site_symmetries"),
        name="site_symmetry_ops",
        pad_token=service["pad"],
        stop_token=service["stop"],
        mask_token=service["mask"],
        # An unknown (sg, ss) has to resolve to something; PAD is the zero vector.
        default_value=service["pad"])

    serialised = WyckoffProcessor._serialise_feature_engineer(engineer)
    (engineers_dir / "site_symmetry_ops.json").write_text(
        serialised.model_dump_json(indent=2), encoding="utf-8")
    return engineer


def build_site_symmetry_ops_id_engineer(
    engineers_dir: Path = ENGINEERS_DIR) -> "FeatureEngineer":
    """Build and save ``site_symmetry_ops_id``: the same encoding, stored once.

    Identical information to ``site_symmetry_ops``, factored into a per-token integer
    id and a lookup table written next to the engineer as
    ``site_symmetry_ops_id_table.json``. The model expands the id through the table
    (``embedding_size: {frozen_table: site_symmetry_ops_id}``), so the vectors are held
    once as [n_ids, n_varying] instead of once per (structure, position): on LeMat-Bulk
    that is 85 GiB of resident int64 against 2 GB, for the same input to the encoder.

    Ids run 0..n-1 over the (sg, site_symm) pairs in the engineer's own (lexsorted)
    order, followed by MASK, STOP and PAD at n, n+1 and n+2 -- the convention
    ``WyckoffProcessor.tokenise_dataset`` assumes when it sizes a
    ``PassThroughTokeniser`` as ``db.max() + 1 + 3``. Their table rows reproduce the
    dense engineer's service vectors: ones for MASK, zeros for STOP and PAD.
    """
    engineers_dir = Path(engineers_dir)
    engineers_dir.mkdir(exist_ok=True, parents=True)

    values, width = flagged_site_symmetry_ops_vectors()
    service = service_ops_vectors(width)

    # FeatureEngineer lexsorts its db, so build the ids from the same sorted order the
    # table rows are written in; anything else would silently permute the lookup.
    keys = sorted(values.keys())
    ids = {key: index for index, key in enumerate(keys)}
    n_real = len(keys)
    mask_token, stop_token, pad_token = n_real, n_real + 1, n_real + 2

    table = np.zeros((n_real + 3, width), dtype=np.float32)
    for key, index in ids.items():
        table[index] = values[key]
    table[mask_token] = service["mask"]
    table[stop_token] = service["stop"]
    table[pad_token] = service["pad"]

    engineer = FeatureEngineer(
        ids,
        ("spacegroup_number", "site_symmetries"),
        name="site_symmetry_ops_id",
        mask_token=mask_token,
        stop_token=stop_token,
        # An unknown (sg, ss) has to resolve to *some* vector, and the dense engineer
        # answers zeros; PAD is the id whose row is zeros.
        pad_token=pad_token,
        default_value=pad_token)

    serialised = WyckoffProcessor._serialise_feature_engineer(engineer)
    (engineers_dir / "site_symmetry_ops_id.json").write_text(
        serialised.model_dump_json(indent=2), encoding="utf-8")
    save_frozen_table("site_symmetry_ops_id", table, engineers_dir=engineers_dir)
    return engineer


def get_augmentation_dict():
    ascii_range = tuple(string.ascii_letters)
    alternatives_by_sg = {}
    for spacegroup_number in range(1, N_3D_SPACEGROUPS + 1):
        alternatives_letters = tuple(tuple(x.split()) for x in Group(spacegroup_number).get_alternatives()['Transformed WP'])
        reference_order = alternatives_letters[0]
        assert reference_order == ascii_range[:len(reference_order)]
        # There are transformations that don't rearrange Wychoff letters, e. g.
        # Group(1) has
        # {'No.': ['1', '2'],
        # 'Coset Representative': ['x,y,z', '-x,-y,-z'],
        # 'Geometrical Interpretation': ['1', '-1 0,0,0'],
        # 'Transformed WP': ['a', 'a']}
        alternatives_letters_set = frozenset(alternatives_letters)
        alternatives_this_sg = []
        for this_alternative in alternatives_letters_set:
            this_augmentator = {}
            for new_letter, old_letter in zip(this_alternative, reference_order):
                this_augmentator[old_letter] = new_letter
            alternatives_this_sg.append(this_augmentator)
        alternatives_by_sg[spacegroup_number] = alternatives_this_sg
    return alternatives_by_sg


def main():
    generate_wyckoff_mappings()
    print("Done generating Wyckoff mappings JSON.")
    enumerate_wychoffs_by_ss()
    print("Done enumerating Wyckoff positions inside site symmetry.")
    build_site_symmetry_ops_engineer()
    print("Done building site_symmetry_ops engineer.")
    build_site_symmetry_ops_id_engineer()
    print("Done building site_symmetry_ops_id engineer and its lookup table.")
    get_augmentation_dict()
    print("Done test-running Wyckoff positions augmentation.")


if __name__ == "__main__":
    main()
