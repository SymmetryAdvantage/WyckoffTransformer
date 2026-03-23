#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Processes a batch of LeMaterial Parquet files to generate and append CIF data.

This script reads each Parquet file from a specified input directory,
generates a Crystallographic Information File (CIF) string for each row,
and stores it in a new 'cif' column.

If CIF generation fails for a row, the error message is stored instead,
ensuring the process does not halt. The updated DataFrame is then saved
to a specified output directory with the same filename.

Updated to use pandarallel for parallel processing.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from pandarallel import pandarallel

# Prerequisites: Ensure you have these libraries installed
# pip install pandas pyarrow numpy pymatgen tqdm pandarallel
try:
    from pymatgen.core.structure import Structure
    from pymatgen.io.cif import CifWriter
except ImportError:
    print("Pymatgen not found. Please install it using: pip install pymatgen")
    exit()


# --- Configuration ---
# PLEASE UPDATE THESE PATHS ACCORDING TO YOUR SYSTEM
INPUT_DIR = Path("/mnt/data/shares/database/LeMat/LeMat-Bulk-v2-Test/compatible_pbe/")
OUTPUT_DIR = Path("/home/users/shuya001/WyckoffTransformer/data/lemat_unique_v2test_cpbe/")

# List of files to be processed
# FILENAMES = [
#     'lemat_v2test.parquet'
# ]

FILENAMES = [
 'train-00000-of-00018.parquet',
 'train-00001-of-00018.parquet',
 'train-00002-of-00018.parquet',
 'train-00003-of-00018.parquet',
 'train-00004-of-00018.parquet',
 'train-00005-of-00018.parquet',
 'train-00006-of-00018.parquet',
 'train-00007-of-00018.parquet',
 'train-00008-of-00018.parquet',
 'train-00009-of-00018.parquet',
 'train-00010-of-00018.parquet',
 'train-00011-of-00018.parquet',
 'train-00012-of-00018.parquet',
 'train-00013-of-00018.parquet',
 'train-00014-of-00018.parquet',
 'train-00015-of-00018.parquet',
 'train-00016-of-00018.parquet',
 'train-00017-of-00018.parquet'
]
# --------------------


def parse_species(species_at_sites):
    """
    Parses the species_at_sites data from LeMaterial into a format
    compatible with pymatgen, handling multiple data formats.
    """
    if not isinstance(species_at_sites, (list, np.ndarray)) or len(species_at_sites) == 0:
        return []

    if all(isinstance(site, str) for site in species_at_sites):
        return list(species_at_sites)

    species_list = []
    for site in species_at_sites:
        if not isinstance(site, (list, np.ndarray)):
            continue
            
        if len(site) == 2 and isinstance(site[1], (int, float)):
            species_list.append(site[0])
        else:
            try:
                species_dict = {site[i]: site[i+1] for i in range(0, len(site), 2)}
                species_list.append(species_dict)
            except (IndexError, TypeError):
                continue
                
    return species_list


def create_cif_entry(row):
    """
    Takes a DataFrame row and returns a CIF string.
    If an error occurs, it returns a formatted error string.
    """
    try:
        lattice_vectors = np.array(row['lattice_vectors']).squeeze().tolist()
        cartesian_site_positions = np.array(row['cartesian_site_positions']).squeeze().tolist()

        if np.array(cartesian_site_positions).ndim == 1:
             cartesian_site_positions = [cartesian_site_positions]

        species = parse_species(row['species_at_sites'])
        
        if len(species) != len(cartesian_site_positions):
            raise ValueError(
                f"Length mismatch: len(species)={len(species)} vs len(coords)={len(cartesian_site_positions)}"
            )

        structure = Structure(
            lattice=lattice_vectors,
            species=species,
            coords=cartesian_site_positions,
            coords_are_cartesian=True
        )

        cif_writer = CifWriter(structure, symprec=None)  # None for Raw, 0.1 for Symmetry
        return str(cif_writer)

    except Exception as e:
        # If any error occurs during the process, record the error message.
        return f"ERROR: {e}"


def main():
    """
    Main function to run the batch processing.
    """
    # Initialize pandarallel
    pandarallel.initialize(progress_bar=True, verbose=1)
    
    # Create the output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output will be saved to: {OUTPUT_DIR.resolve()}")

    for filename in FILENAMES:
        input_path = INPUT_DIR / filename
        output_path = OUTPUT_DIR / filename

        if not input_path.exists():
            print(f"Warning: Input file not found, skipping: {input_path}")
            continue

        print(f"\nProcessing file: {filename}...")

        # Load the dataframe
        df = pd.read_parquet(input_path)

        # Apply the function to each row to create the 'cif' column using parallel processing
        # parallel_apply automatically distributes the work across available CPU cores
        df['cif'] = df.parallel_apply(create_cif_entry, axis=1)

        # Save the processed dataframe to the output directory
        df.to_parquet(output_path)
        print(f"Successfully processed and saved to: {output_path}")

    print("\nBatch processing complete.")


if __name__ == "__main__":
    main()