if __name__ == "__main__":
    # We want to avoid messing with the environment variables in case we are used as a module.
    # The code is parallelised by structure
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OMP_THREAD_LIMIT"] = "1"
from typing import Dict, List
from collections import defaultdict, Counter
import argparse
from pathlib import Path
import torch
import json
import gzip
import pickle
import logging
from multiprocessing import Pool
from pymatgen.core.structure import Structure, Lattice
from data import structure_to_sites

logger = logging.getLogger(__name__)

# https://github.com/jiaor17/DiffCSP/blob/main/scripts/compute_metrics.py#L29
def get_crystals_list(
        frac_coords, atom_types, lengths, angles, num_atoms):
    """
    args:
        frac_coords: (num_atoms, 3)
        atom_types: (num_atoms)
        lengths: (num_crystals)
        angles: (num_crystals)
        num_atoms: (num_crystals)
    """
    assert frac_coords.size(0) == atom_types.size(0) == num_atoms.sum()
    assert lengths.size(0) == angles.size(0) == num_atoms.size(0)

    start_idx = 0
    crystal_array_list = []
    for batch_idx, num_atom in enumerate(num_atoms.tolist()):
        cur_frac_coords = frac_coords.narrow(0, start_idx, num_atom)
        cur_atom_types = atom_types.narrow(0, start_idx, num_atom)
        cur_lengths = lengths[batch_idx]
        cur_angles = angles[batch_idx]

        crystal_array_list.append({
            'frac_coords': cur_frac_coords.detach().cpu().numpy(),
            'atom_types': cur_atom_types.detach().cpu().numpy(),
            'lengths': cur_lengths.detach().cpu().numpy(),
            'angles': cur_angles.detach().cpu().numpy(),
        })
        start_idx = start_idx + num_atom
    return crystal_array_list


class StructureToSites():
    def __init__(self, tol=0.1):
        with open(Path(__file__).parent.joinpath("cache", "wychoffs_enumerated_by_ss.pkl.gz"), "rb") as f:
            self.wychoffs_enumerated_by_ss = pickle.load(f)[0]
        self.tol = tol
    
    def structure_to_sites(self, structure):
        try:
            return record_to_pyxtal(structure_to_sites(
                structure, wychoffs_enumerated_by_ss=self.wychoffs_enumerated_by_ss, tol=self.tol))
        except Exception as e:
            logger.error("Error processing %s", e)
            logger.debug(structure)


def get_structure(record: Dict) -> Structure:
    """
    Converts DiffCSP custom format to pymatgen Structure
    """
    return Structure(
                lattice=Lattice.from_parameters(
                    *(record['lengths'].tolist() + record['angles'].tolist())),
                species=record['atom_types'], coords=record['frac_coords'], coords_are_cartesian=False)


def record_to_pyxtal(record: Dict) -> Dict:
    """
    From a record in the sites dataset, returns a dictionary with the format expected by pyxtal.from_random.
    """
    sites = defaultdict(list)
    numIons = Counter()
    for element, letter, multiplicity in zip(record["elements"], record["wyckoff_letters"], record["multiplicity"]):
        sites[str(element)].append(f"{multiplicity}{letter}")
        numIons[str(element)] += multiplicity
    return {
        "group": record["spacegroup_number"],
        "sites": list(sites.values()),
        "species": list(sites.keys()),
        "numIons": list(numIons.values()),
    }


def load_diffcsp_dataset(dataset: Path) -> List[Structure]:
    data = torch.load(dataset, map_location='cpu')
    crystals_list = get_crystals_list(
        data['frac_coords'], data['atom_types'], data['lengths'], data['angles'], data['num_atoms'])
    print(crystals_list[0])
    with Pool() as pool:
        structures = pool.map(get_structure, crystals_list)
    return structures


def main():
    parser = argparse.ArgumentParser("Converts DiffCSP custom format to CIF and sites")
    parser.add_argument("dataset", type=Path, help="The dataset to process")
    parser.add_argument("--log-level", type=str, default=logging.WARNING)
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level)
    structures = load_diffcsp_dataset(args.dataset)
    print(structures[0])
    return
    converter = StructureToSites()
    with Pool() as pool:
        sites = pool.map(converter.structure_to_sites, structures)
    sites = filter(lambda x: x is not None, sites)
    with gzip.open(args.dataset.with_suffix(".sites.json.gz"), "wt") as f:
        json.dump(list(sites), f)

if __name__ == "__main__":
    main()