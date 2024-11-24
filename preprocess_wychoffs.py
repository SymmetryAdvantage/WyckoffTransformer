from collections import defaultdict, Counter
import string
from pathlib import Path
import pickle
import gzip
import numpy as np
import pandas as pd
from pyxtal import Group
from sklearn.cluster import KMeans
from scipy.special import sph_harm

from wyckoff_transformer.tokenization import FeatureEngineer
from wyckoff_transformer.pyxtal_fix import SS_CORRECTIONS

N_3D_SPACEGROUPS = 230


def convolve_vectors_with_spherical_harmonics(vectors_batch, degree):
    """
    Convolves a batch of 3D vectors with spherical harmonics without explicit loops.
    
    Parameters:
    vectors_batch : ndarray
        A 3D array of shape (num_batches, num_objects, 3) representing the vectors.
    degree : int
        The degree of the spherical harmonics.
    order : int
        The order of the spherical harmonics.
    
    Returns:
    ndarray
        A 1D array of shape (num_batches,) with the convolved values for each batch.
    """
    # Normalize the vectors
    norms = np.linalg.norm(vectors_batch, axis=-1, keepdims=True)
    x, y = vectors_batch[..., 0], vectors_batch[..., 1]
    phi = np.arctan2(y, x)
    # Avoid division by zero
    nonzero_norms = norms.copy()
    nonzero_norms[nonzero_norms == 0] = 1
    z = vectors_batch[..., 2] / nonzero_norms.squeeze(-1)
    theta = np.arccos(z)

    # Compute spherical harmonics for all vectors
    res = np.array([sph_harm(order, degree, phi, theta) for order in range(degree+1)])
    res *= np.expand_dims(norms.squeeze(-1), 0)
    return res.mean(axis=-1)

def enumerate_wychoffs_by_ss(
    output_file: Path = Path("cache", "wychoffs_enumerated_by_ss.pkl.gz"),
    spherical_harmonics_degree: int = 2):
    """
    Enumerates all Wyckoff positions by site symmetry.

    Args:
        output_file (Path, optional): The output file.
        spherical_harmonics_degree (int, optional): The degree of the spherical harmonics
            used to disabiguate the Wyckoff positions with the same site symmetry.
    """
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
            # https://github.com/MaterSim/PyXtal/issues/295
            try:
                site_symm = SS_CORRECTIONS[spacegroup_number][wp.letter]
            except KeyError:
                site_symm = wp.site_symm
            ss_from_letter[spacegroup_number][wp.letter] = site_symm
            enum_from_ss_letter[spacegroup_number][wp.letter] = ss_counts[site_symm]
            opres_by_ss_enum[site_symm][ss_counts[site_symm]] = \
                [[op.operate(v) for op in wp] for v in reference_vectors]
            letter_from_ss_enum[spacegroup_number][site_symm][ss_counts[site_symm]] = wp.letter
            multiplicity_from_ss_enum[(spacegroup_number, site_symm, ss_counts[site_symm])] = wp.multiplicity
            max_multiplicity = max(max_multiplicity, wp.multiplicity)
            ss_counts[site_symm] += 1
        #signature_by_sg_ss_enum[spacegroup_number] = defaultdict(dict)
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
                
    if output_file.suffix == ".gz":
        opener = gzip.open
    else:
        opener = open
    with opener(output_file, "wb") as f:
        pickle.dump((dict(enum_from_ss_letter), dict(letter_from_ss_enum),
            dict(ss_from_letter)), f)
    engineered_path = Path("cache", "engineers")
    engineered_path.mkdir(exist_ok=True)
    multiplicity_engineer = FeatureEngineer(
        multiplicity_from_ss_enum, ("spacegroup_number", "site_symmetries", "sites_enumeration"),
        name="multiplicity",
        stop_token=max_multiplicity + 1, mask_token=max_multiplicity + 2, pad_token=0)
    with gzip.open(engineered_path / "multiplicity.pkl.gz", "wb") as f:
        pickle.dump(multiplicity_engineer, f)
    harmonic_size = 2 * (spherical_harmonics_degree + 1) * len(reference_vectors)
    harmonic_engineer = FeatureEngineer(
        signature_by_sg_ss_enum, ("spacegroup_number", "site_symmetries", "sites_enumeration"),
        name="harmonic_site_symmetries",
        # This requires some thouhgt. PAD = 0, OK
        pad_token=np.zeros(harmonic_size),
        # STOP does not necessarily need to be different from PAD, so OK
        stop_token=np.zeros(harmonic_size),
        # Usually, the models are not supposed to see MASK, STOP, and PAD togeher
        mask_token=np.ones(harmonic_size))
    with gzip.open(engineered_path / "harmonic_site_symmetries.pkl.gz", "wb") as f:
        pickle.dump(harmonic_engineer, f)
    enum_to_cluster, cluster_to_enum = clasterize_harmonics(harmonic_engineer)
    # Here we actually know the tokens - as this is their birthplace
    max_cluster_id = enum_to_cluster.max()
    enum_to_cluster_engineer = FeatureEngineer(
        enum_to_cluster,
        mask_token=max_cluster_id + 1,
        stop_token=max_cluster_id + 2,
        pad_token=max_cluster_id + 3)
    with gzip.open(engineered_path / "harmonic_cluster.pkl.gz", "wb") as f:
        pickle.dump(enum_to_cluster_engineer, f)
    # We don't know yet the tokenization of enums, so we'll need to fill in the tokens later
    cluster_to_enum_engineer = FeatureEngineer(
        cluster_to_enum, mask_token=None, stop_token=None, pad_token=None)
    with gzip.open(engineered_path / "sites_enumeration.pkl.gz", "wb") as f:
        pickle.dump(cluster_to_enum_engineer, f)


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
    inverse = pd.Series(distances.index.get_level_values(2), index=mapping)
    return pd.Series(mapping, index=distances.index.get_level_values(2))


def clasterize_harmonics(
    harmonic_engineer: FeatureEngineer,
    random_state: int = 42):
    """
    Harmoic fetures are nice and float, but when we predict the next token, we need
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
    inverse_index = pd.MultiIndex.from_arrays(
        [enum_to_cluster.index.get_level_values(0), enum_to_cluster.index.get_level_values(1), enum_to_cluster.values],
        names=[enum_to_cluster.index.names[0], enum_to_cluster.index.names[1], enum_to_cluster.name])
    cluster_to_enum = pd.Series(enum_to_cluster.index.get_level_values(2), index=inverse_index, name="sites_enumeration")
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


if __name__ == "__main__":
    enumerate_wychoffs_by_ss()
    print("Done enumerating Wyckoff positions inside site symmetry.")
    get_augmentation_dict()
    print("Done test-running Wyckoff positions augmentation.")
