#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Calculates formation energy and energy above the convex hull (E-hull) for a list
of materials using pymatgen.

This script reads material data from a CSV file, constructs a phase diagram for
each chemical system, and computes the formation energy and E-hull for each entry.
It is optimized for performance on large datasets using pandarallel for parallel
processing.

The script is designed to be flexible, allowing users to specify file paths,
column names, and the number of processing workers via command-line arguments.
"""

import argparse
import pandas as pd
import sys
from typing import Set, Tuple, Optional

try:
    from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
    from pandarallel import pandarallel
except ImportError as e:
    print(f"Error: A required library is not installed. {e}")
    print("Please install the necessary libraries, e.g., pip install pandas pymatgen pandarallel")
    sys.exit(1)


def get_ef_ehull(
    id_in: str,
    formula_in: str,
    energy_in: float,
    chemsys_in_set: Set[str],
    data_refs: pd.DataFrame,
    col_name_formula: str,
    col_name_energy: str,
    col_name_id: str,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculates the formation energy and energy above the hull for a single material entry.

    Args:
        id_in: The unique identifier for the input material.
        formula_in: The chemical formula of the input material.
        energy_in: The total energy of the input material.
        chemsys_in_set: A set of elements in the chemical system.
        data_refs: A DataFrame containing the reference data for building the phase diagram.
        col_name_formula: The column name for the formula in the reference DataFrame.
        col_name_energy: The column name for the total energy in the reference DataFrame.
        col_name_id: The column name for the ID in the reference DataFrame.

    Returns:
        A tuple containing the formation energy per atom and the energy above hull.
        Returns (None, None) if the calculation cannot be completed.
    """
    # Define elements that are radioactive or not well-supported in common databases.
    NA_ELEMENTS = {
        "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu",
        "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db",
        "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv",
        "Ts", "Og"
    }

    # Skip calculations for systems containing certain elements like Yb or rare elements.
    if "Yb" in chemsys_in_set:
        return None, None
    
    if_na_elements = NA_ELEMENTS.intersection(chemsys_in_set)
    if if_na_elements:
        print(f"Warning: Skipping {id_in} ({formula_in}) due to unsupported element(s): {if_na_elements}", file=sys.stderr)
        return None, None

    # Qhull, used by pymatgen, can have issues with high-dimensional systems.
    if len(chemsys_in_set) >= 10:
        print(f"Warning: Skipping {id_in} ({formula_in}) due to high number of elements ({len(chemsys_in_set)}).", file=sys.stderr)
        return None, None

    # Filter the reference data to include only entries that are subsets of the current chemical system.
    # This is the most computationally intensive part of the setup.
    mask = data_refs["chemsys_set"].apply(lambda x: x.issubset(chemsys_in_set))
    data_select = data_refs[mask]

    if data_select.shape[0] == 0:
        print(f"Warning: No reference entries found for {id_in} ({formula_in}).", file=sys.stderr)
        return None, None

    # Create a list of PDEntry objects for the phase diagram.
    entries = [
        PDEntry(composition, energy, material_id)
        for composition, energy, material_id in zip(
            data_select[col_name_formula],
            data_select[col_name_energy],
            data_select[col_name_id]
        )
    ]

    if not entries:
        return None, None

    try:
        # Construct the phase diagram
        phase_diagram = PhaseDiagram(entries=entries)
        entry = PDEntry(formula_in, energy_in, id_in)

        # Calculate formation energy and energy above hull
        e_form = phase_diagram.get_form_energy_per_atom(entry)
        e_hull = phase_diagram.get_e_above_hull(entry)

    except Exception as e:
        # Catch potential errors during phase diagram construction or calculation.
        print(f"Error processing {id_in} ({formula_in}): {e}", file=sys.stderr)
        return None, None
    
    return e_form, e_hull


def main():
    """
    Main function to parse arguments and run the E-hull calculation workflow.
    """
    parser = argparse.ArgumentParser(
        description="Calculate formation energy and energy above hull for materials in a CSV file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input-file",
        required=True,
        help="Path to the input CSV file containing target materials."
    )
    parser.add_argument(
        "--ref-file",
        help="Path to the reference data CSV file. If not provided, the input file will be used as the reference."
    )
    parser.add_argument(
        "--output-file",
        required=True,
        help="Path to save the output CSV file with the new e_form and e_hull columns."
    )
    parser.add_argument("--id-col", default="immutable_id", help="Column name for the material identifier.")
    parser.add_argument("--formula-col", default="full_formula", help="Column name for the chemical formula.")
    parser.add_argument("--chemsys-col", default="chemsys", help="Column name for the chemical system (e.g., 'Fe-O').")
    parser.add_argument("--energy-col", default="energy_corrected", help="Column name for the total energy.")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers for pandarallel.")

    args = parser.parse_args()

    # --- 1. Load Data ---
    print("Loading data...")
    try:
        data_target = pd.read_csv(args.input_file, compression='gzip' if args.input_file.endswith('.gz') else None)
        
        if args.ref_file:
            data_refs = pd.read_csv(args.ref_file, compression='gzip' if args.ref_file.endswith('.gz') else None)
        else:
            print("No reference file provided. Using the input file as reference data.")
            data_refs = data_target.copy()
    except FileNotFoundError as e:
        print(f"Error: File not found. {e}", file=sys.stderr)
        sys.exit(1)

    # --- 2. Pre-process Data for Efficiency ---
    print("Preprocessing reference data for faster lookups...")
    # Create a set of elements for each entry in the reference data for faster subset checking.
    data_refs['chemsys_set'] = data_refs[args.chemsys_col].apply(lambda x: set(x.split('-')))

    # --- 3. Initialize Parallel Processing ---
    print(f"Initializing pandarallel with {args.workers} workers...")
    pandarallel.initialize(nb_workers=args.workers, progress_bar=True)

    # --- 4. Apply Calculation ---
    print("Calculating formation energy and E-hull for target materials...")
    
    # Create a temporary column with element sets for the target data
    data_target['chemsys_set'] = data_target[args.chemsys_col].apply(lambda x: set(x.split('-')))
    
    results = data_target.parallel_apply(
        lambda row: get_ef_ehull(
            id_in=row[args.id_col],
            formula_in=row[args.formula_col],
            energy_in=row[args.energy_col],
            chemsys_in_set=row['chemsys_set'],
            data_refs=data_refs,
            col_name_formula=args.formula_col,
            col_name_energy=args.energy_col,
            col_name_id=args.id_col
        ),
        axis=1,
        result_type='expand'
    )
    
    # Drop the temporary column
    data_target = data_target.drop(columns=['chemsys_set'])

    # Assign the new calculated columns
    data_target[['e_form', 'e_hull']] = results

    # --- 5. Save Results ---
    print(f"Saving results to {args.output_file}...")
    try:
        data_target.to_csv(args.output_file, compression='gzip' if args.output_file.endswith('.gz') else None, index=False)
        print("Calculation complete. Output saved successfully.")
    except Exception as e:
        print(f"Error saving file: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
