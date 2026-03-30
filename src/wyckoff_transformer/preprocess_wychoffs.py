from collections import defaultdict, Counter
import json
import string
from pathlib import Path
import numpy as np
import pandas as pd
from pyxtal import Group
from sklearn.cluster import KMeans
from scipy.special import sph_harm_y

from wyckoff_transformer.wyckoff_processor import FeatureEngineer, WyckoffProcessor


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
                str(sg): {ss: {str(e): l for e, l in ed.items()}
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
    engineers_dir: Path = Path(__file__).parent / "engineers",
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
    get_augmentation_dict()
    print("Done test-running Wyckoff positions augmentation.")


if __name__ == "__main__":
    main()
