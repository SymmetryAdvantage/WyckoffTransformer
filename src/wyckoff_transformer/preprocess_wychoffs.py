from collections import defaultdict, Counter
import json
import string
from pathlib import Path
import numpy as np
import pandas as pd
from pyxtal import Group
from sklearn.cluster import KMeans
from scipy.special import sph_harm_y

from wyckoff_transformer.wyckoff_processor import (
    ENGINEERS_DIR, FeatureEngineer, WyckoffProcessor, save_frozen_table)


N_3D_SPACEGROUPS = 230
WYCKOFF_MAPPINGS_FILENAME = "wyckoffs_enumerated_by_ss.json"
_PACKAGE_MAPPINGS_PATH = Path(__file__).parent / WYCKOFF_MAPPINGS_FILENAME


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


def convolve_vectors_with_spherical_harmonics(vectors_batch, degree):
    """
    Convolves a batch of 3D vectors with spherical harmonics without explicit loops.

    Parameters:
    vectors_batch : ndarray
        A 3D array of shape (num_batches, num_objects, 3) representing the vectors
    degree : int
        The degree of the spherical harmonics

    Returns:
    ndarray
        A 1D array of shape (num_batches,) with the convolved values for each batch.
    """
    # Normalize the vectors
    norms = np.linalg.norm(vectors_batch, axis=-1, keepdims=True)
    x, y, z = vectors_batch[..., 0], vectors_batch[..., 1], vectors_batch[..., 2]
    theta = np.arctan2(np.hypot(x, y), z)
    phi = np.mod(np.arctan2(y, x), 2 * np.pi)

    # Compute spherical harmonics for all vectors
    res = np.array([sph_harm_y(degree, order, theta, phi) for order in range(degree+1)])
    res *= np.expand_dims(norms.squeeze(-1), 0)
    return res.mean(axis=-1)

def enumerate_wychoffs_by_ss(
    output_file: Path = _PACKAGE_MAPPINGS_PATH,
    engineers_dir: Path = ENGINEERS_DIR,
    spherical_harmonics_degree: int = 2):
    """
    Enumerates all Wyckoff positions by site symmetry.

    Args:
        output_file (Path, optional): The output file for Wyckoff mappings JSON.
        engineers_dir (Path, optional): Directory to write FeatureEngineer JSON files.
        spherical_harmonics_degree (int, optional): The degree of the spherical harmonics
            used to disabiguate the Wyckoff positions with the same site symmetry.
    """
    engineers_dir = Path(engineers_dir)
    engineers_dir.mkdir(exist_ok=True, parents=True)

    enum_from_ss_letter = defaultdict(dict)
    ss_from_letter = defaultdict(dict)
    letter_from_ss_enum = defaultdict(lambda: defaultdict(dict))
    multiplicity_from_ss_enum = dict()
    max_multiplicity = 0
    reference_vectors = (
        np.array([0, 0, 0]),
        np.array([1, 1, 1]),
    )
    signature_by_sg_ss_enum = {}
    for spacegroup_number in range(1, N_3D_SPACEGROUPS + 1):
        group = Group(spacegroup_number)
        ss_counts = Counter()
        opres_by_ss_enum = defaultdict(dict)
        # [::-1] doesn't matter in principle,
        # but serves a cosmetic purpose, so that
        # a comes before b, etc.
        for wp in group.Wyckoff_positions[::-1]:
            wp.get_site_symmetry()
            site_symm = wp.site_symm
            ss_from_letter[spacegroup_number][wp.letter] = site_symm
            enum_from_ss_letter[spacegroup_number][wp.letter] = ss_counts[site_symm]
            opres_by_ss_enum[site_symm][ss_counts[site_symm]] = \
                [[op.operate(v) for op in wp] for v in reference_vectors]
            letter_from_ss_enum[spacegroup_number][site_symm][ss_counts[site_symm]] = wp.letter
            multiplicity_from_ss_enum[(spacegroup_number, site_symm, ss_counts[site_symm])] = wp.multiplicity
            max_multiplicity = max(max_multiplicity, wp.multiplicity)
            ss_counts[site_symm] += 1
        for ss, opres_by_enum in opres_by_ss_enum.items():
            print(f"Spacegroup {spacegroup_number}, wp {ss} {letter_from_ss_enum[spacegroup_number][ss]}")
            # Step 1: find the position closest to the origin
            all_ops = np.concatenate([np.expand_dims(a, 0) for a in opres_by_enum.values()], axis=0)
            # [enum][ref_vector][op][xyz]
            print("Ops [enum][ref_vector][op][xyz]:")
            print(all_ops.shape)
            signatures = convolve_vectors_with_spherical_harmonics(all_ops, spherical_harmonics_degree)
            print("Signatures [degree][enum][ref_vector]")
            print(signatures.shape)
            signatures = signatures.reshape(
                spherical_harmonics_degree + 1, len(opres_by_enum), len(reference_vectors))
            assert np.unique(signatures, axis=1).shape == signatures.shape
            for enum in opres_by_enum.keys():
                signature_by_sg_ss_enum[(spacegroup_number, ss, enum)] = \
                    np.concatenate([signatures[:, enum, :].real.ravel(), signatures[:, enum, :].imag.ravel()])

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

    harmonic_size = 2 * (spherical_harmonics_degree + 1) * len(reference_vectors)
    harmonic_engineer = FeatureEngineer(
        signature_by_sg_ss_enum, ("spacegroup_number", "site_symmetries", "sites_enumeration"),
        name="harmonic_site_symmetries",
        # This requires some thouhgt. PAD = 0, OK
        pad_token=np.zeros(harmonic_size),
        # STOP does not necessarily need to be different from PAD, so OK
        stop_token=np.zeros(harmonic_size),
        # Usually, the models are not supposed to see MASK, STOP, and PAD togeher
        mask_token=np.ones(harmonic_size),
        # In case of making an invalid request, we need to have a default value
        # CONSIDER using nan
        default_value=np.zeros(harmonic_size))
    _save_engineer(harmonic_engineer, "harmonic_site_symmetries")

    enum_to_cluster, cluster_to_enum = clasterize_harmonics(harmonic_engineer)
    # Here we actually know the tokens - as this is their birthplace
    max_cluster_id = enum_to_cluster.max()
    enum_to_cluster_engineer = FeatureEngineer(
        enum_to_cluster,
        mask_token=max_cluster_id + 1,
        stop_token=max_cluster_id + 2,
        pad_token=max_cluster_id + 3)
    _save_engineer(enum_to_cluster_engineer, "harmonic_cluster")

    # We don't know yet the tokenization of enums, so we'll need to fill in the tokens later
    cluster_to_enum_engineer = FeatureEngineer(
        cluster_to_enum, mask_token=None, stop_token=None, pad_token=None,
        default_value=np.zeros_like(cluster_to_enum.iloc[0]))
    _save_engineer(cluster_to_enum_engineer, "sites_enumeration")


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


def assign_to_clusters(
    distances: pd.DataFrame):

    remaining_distances = distances.copy().droplevel((0, 1), axis=0)
    assert (remaining_distances.index == np.arange(remaining_distances.shape[0])).all()
    assert (remaining_distances.columns == np.arange(remaining_distances.shape[1])).all()
    mapping = np.empty(distances.shape[0], dtype=int)

    while not remaining_distances.empty:
        row, col = np.unravel_index(np.argmin(remaining_distances.values), remaining_distances.shape)
        row_label = remaining_distances.index[row]
        col_label = remaining_distances.columns[col]
        # enum -> cluster
        mapping[row_label] = col_label
        remaining_distances = remaining_distances.drop(row_label, axis=0).drop(col_label, axis=1)
    # inverse = pd.Series(distances.index.get_level_values(2), index=mapping)
    return pd.Series(mapping, index=distances.index.get_level_values(2))


def inverse_series(input_series: pd.Series) -> pd.Series:
    """
    Transforms a Series with a MultiIndex of 3 levels into a new Series with the last level
    of the index as the values and the first two levels as the new index.
    """
    if input_series.index.nlevels != 3:
        raise ValueError("Input series must have 3 levels in the index.")
    inverse_index = pd.MultiIndex.from_arrays(
        [input_series.index.get_level_values(0), input_series.index.get_level_values(1), input_series.values],
        names=[input_series.index.names[0], input_series.index.names[1], input_series.name])
    return pd.Series(input_series.index.get_level_values(2), index=inverse_index, name=input_series.index.names[2])


def clasterize_harmonics(
    harmonic_engineer,  # FeatureEngineer
    random_state: int = 42):
    """
    Harmonic fetures are nice and float, but when we predict the next token, we need
    to predict a set of distinct values. Morever, we need to predict the probability
    as enumeration can genuinly take several values, especially in the beginning.
    """
    n_enums = len(harmonic_engineer.db.index.get_level_values("sites_enumeration").unique())
    clusters = KMeans(n_clusters=n_enums, random_state=random_state).fit(
        harmonic_engineer.db.to_list())
    cluster_distances = clusters.transform(np.array(harmonic_engineer.db.to_list()))
    cluster_db = pd.DataFrame(cluster_distances, index=harmonic_engineer.db.index)
    # Clusters are global, but enumeration is local per spacegroup and site symmetry
    enum_to_cluster = cluster_db.groupby(
        level=["spacegroup_number", "site_symmetries"]).apply(assign_to_clusters).sort_index()
    enum_to_cluster.name = "harmonic_cluster"
    cluster_to_enum = inverse_series(enum_to_cluster)
    return enum_to_cluster, cluster_to_enum


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
