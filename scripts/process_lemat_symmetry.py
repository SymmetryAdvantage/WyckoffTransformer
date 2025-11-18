#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Processes a Parquet file of crystallographic structures to extract and append
symmetry information for each entry.

This script reads a dataset of materials from a predefined list of Parquet files.
For each entry, it attempts to parse the CIF (Crystallographic Information File)
string into a pymatgen Structure object. It then applies a function to convert
the structure into a dictionary of symmetry-related properties using pymatgen
and pyxtal.

The script is designed for parallel execution using pandarallel to accelerate
the computationally intensive steps. If an error occurs during CIF parsing or
symmetry analysis for a specific entry, it logs the 'immutable_id' and the error,
stores None for that entry, and continues processing the rest of the data.
Finally, it reports the total count of errors for each stage.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional, Dict

import pandas as pd
from pandarallel import pandarallel
from pymatgen.core import Structure, Element
from pyxtal import pyxtal

# Assume these functions are in a local 'data.py' file as per the original script.
# If they are located elsewhere, please adjust the import path.
try:
    # These functions are assumed to be defined externally and handle
    # the core logic of reading CIFs and running pyxtal.
    from data import read_cif, kick_pyxtal_until_it_works
except ImportError:
    sys.exit(
        "Error: Could not import 'read_cif' and 'kick_pyxtal_until_it_works'. "
        "Please ensure 'data.py' is in the correct path."
    )

# --- Configuration ---
# Configure logging to provide informative output.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    stream=sys.stdout,
)

# PLEASE UPDATE THESE PATHS ACCORDING TO YOUR SYSTEM
INPUT_DIR = Path("/home/users/shuya001/WyckoffTransformer/data/lemat_unique_v2test_cpbe/")
OUTPUT_DIR = Path("/home/users/shuya001/WyckoffTransformer/data/lemat_unique_v2test_cpbe/")

# List of files to be processed
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

# --- Core Functions (Do Not Modify) ---
def structure_to_sites(
    structure: Structure,
    tol: float = 0.1,
    a_tol: float = 5.0,
    max_wp: Optional[int] = None,
) -> Optional[Dict]:
    """
    Converts a pymatgen structure to a dictionary of symmetry sites.
    (Docstring shortened here for brevity.)
    """
    # This function's internal logic is preserved exactly as requested.
    try:
        pyxtal_structure = kick_pyxtal_until_it_works(structure, tol=tol, a_tol=a_tol)
        if len(pyxtal_structure.atom_sites) == 0:
            raise ValueError("pyxtal failed to convert the structure to symmetry sites.")

        if max_wp is None:
            atom_sites = pyxtal_structure.atom_sites
        else:
            atom_sites = sorted(
                pyxtal_structure.atom_sites, key=lambda x: x.wp.letter
            )[:max_wp]

        elements = []
        wyckoffs = []
        site_symmetries = []
        for site in atom_sites:
            site.wp.get_site_symmetry()
            wyckoffs.append(site.wp)
            elements.append(Element(site.specie))
            site_symmetries.append(site.wp.site_symm)

        multiplicity = [wp.multiplicity for wp in wyckoffs]
        dof = [wp.get_dof() for wp in wyckoffs]

        sites_dict = {
            "site_symmetries": site_symmetries,
            "elements": elements,
            "multiplicity": multiplicity,
            "wyckoff_letters": [wp.letter for wp in wyckoffs],
            "dof": dof,
            "spacegroup_number": pyxtal_structure.group.number,
        }

        return sites_dict

    except Exception as e:
        # NOTE: The original function already returns None on error, which is the
        # desired behavior. We add logging here to capture the specific error message.
        logging.error(f"Error in structure_to_sites: {e}", exc_info=False)
        return None

# --- Helper Functions for Safe Parallel Application ---
def safe_read_cif_from_row(row: pd.Series) -> Optional[Structure]:
    """
    Safely applies the read_cif function to a DataFrame row.

    Args:
        row (pd.Series): A row of the DataFrame, must contain 'cif' and 'immutable_id'.

    Returns:
        Optional[Structure]: A pymatgen Structure object, or None if parsing fails.
    """
    try:
        # The external read_cif function is called here.
        return read_cif(row["cif"])
    except Exception as e:
        logging.error(f"CIF parsing failed for immutable_id: {row['immutable_id']}. Error: {e}")
        return None

def safe_structure_to_sites_from_row(row: pd.Series) -> Optional[Dict]:
    """
    Safely applies the structure_to_sites function to a DataFrame row.

    Args:
        row (pd.Series): A row of the DataFrame, must contain 'structure' and 'immutable_id'.

    Returns:
        Optional[Dict]: A dictionary of symmetry info, or None if analysis fails.
    """
    structure = row["structure"]
    # If the structure is already None from the previous step, just pass it through.
    if not isinstance(structure, Structure):
        return None
    try:
        # The core symmetry analysis function is called here.
        return structure_to_sites(structure)
    except Exception as e:
        logging.error(f"Symmetry analysis failed for immutable_id: {row['immutable_id']}. Error: {e}")
        return None

def main():
    """
    Main function to run the data processing pipeline.
    """
    # --- Initialize Pandarallel ---
    # Determine the number of workers from environment variables or CPU count.
    nb_workers = int(os.environ.get("NP", os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))
    
    logging.info(f"Initializing pandarallel with {nb_workers} workers.")
    pandarallel.initialize(nb_workers=nb_workers, progress_bar=True)

    # Ensure the output directory exists.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    total_cif_errors = 0
    total_symmetry_errors = 0

    # --- Main Processing Loop ---
    for filename in FILENAMES:
        input_path = INPUT_DIR / filename
        # output_path = OUTPUT_DIR / filename.replace('.parquet', '_symmetry.parquet')
        output_path = OUTPUT_DIR / filename.replace('.parquet', '_symmetry_raw.pkl.gz') # _symmetry for symprec=0.1, _symmetry_raw for symprec=None

        if not input_path.exists():
            logging.warning(f"Input file not found, skipping: {input_path}")
            continue

        logging.info(f"--- Processing file: {filename} ---")
        
        # --- Data Loading ---
        logging.info(f"Loading data from {input_path}")
        df = pd.read_parquet(input_path)
        logging.info(f"Initial DataFrame shape: {df.shape}")

        # --- Structure Processing ---
        logging.info("Reading CIF strings into pymatgen Structure objects...")
        df["structure"] = df.parallel_apply(safe_read_cif_from_row, axis=1)
        
        # Count and report CIF parsing errors for this file
        cif_failures = df['structure'].isnull().sum()
        total_cif_errors += cif_failures
        logging.info(f"Shape after CIF parsing: {df.shape}")
        if cif_failures > 0:
            logging.warning(f"Found {cif_failures} CIF parsing errors in this file.")

        # --- Symmetry Analysis ---
        logging.info("Applying symmetry analysis to structures...")
        df["symmetry_dict"] = df.parallel_apply(safe_structure_to_sites_from_row, axis=1)

        # Count and report symmetry analysis errors for this file
        # We count where 'structure' was valid but 'symmetry_dict' is now null.
        symmetry_failures = df[df['structure'].notnull() & df['symmetry_dict'].isnull()].shape[0]
        total_symmetry_errors += symmetry_failures
        if symmetry_failures > 0:
            logging.warning(f"Found {symmetry_failures} symmetry analysis errors in this file.")
            
        # --- Save Results ---
        logging.info(f"Saving final processed data to {output_path}")
        # Drop the intermediate 'structure' column as it's a complex object not needed in the final file.
        # df_processed = df.drop(columns=["structure"])
        df.to_pickle(output_path, compression="gzip")
        logging.info(f"--- Finished processing {filename} ---")
        # df_processed.to_parquet(output_path, index=False)
        # logging.info(f"--- Finished processing {filename} ---")

    # --- Final Summary ---
    logging.info("=" * 50)
    logging.info("          Processing Summary")
    logging.info("=" * 50)
    logging.info(f"All files have been processed.")
    logging.info(f"Total CIF parsing errors across all files: {total_cif_errors}")
    logging.info(f"Total symmetry analysis errors across all files: {total_symmetry_errors}")
    logging.info("Processing complete.")


if __name__ == "__main__":
    main()