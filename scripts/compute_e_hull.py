#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Calculates formation energy and energy above the convex hull (E-hull) for a list
of materials using pymatgen.

Optimized with a global dictionary and PhaseDiagram caching for O(1) lookups.
"""

import argparse
import pandas as pd
import sys
from typing import Set, Tuple, Optional
from collections import defaultdict
from itertools import chain, combinations

try:
    from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
    from pandarallel import pandarallel
except ImportError as e:
    print(f"Error: A required library is not installed. {e}")
    print("Please install the necessary libraries, e.g., pip install pandas pymatgen pandarallel")
    sys.exit(1)

GLOBAL_ENTRIES_DICT = defaultdict(list)
CACHE = {}

def get_ef_ehull(
    id_in: str,
    formula_in: str,
    energy_in: float,
    chemsys_in_set: Set[str]
) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculates the formation energy and energy above the hull for a single material entry.
    """
    if not chemsys_in_set:
        return None, None

    NA_ELEMENTS = {
        "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu",
        "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db",
        "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv",
        "Ts", "Og"
    }

    if "Yb" in chemsys_in_set:
        return None, None
    
    if_na_elements = NA_ELEMENTS.intersection(chemsys_in_set)
    if if_na_elements:
        return None, None

    if len(chemsys_in_set) >= 10:
        return None, None

    fs_chemsys = frozenset(chemsys_in_set)
    
    if fs_chemsys in CACHE:
        phase_diagram = CACHE[fs_chemsys]
    else:
        subsets = chain(*map(lambda x: combinations(chemsys_in_set, x), range(1, len(chemsys_in_set)+1)))
        entries = []
        for subset in subsets:
            entries.extend(GLOBAL_ENTRIES_DICT.get(frozenset(subset), []))

        if not entries:
            CACHE[fs_chemsys] = None
            return None, None
            
        try:
            phase_diagram = PhaseDiagram(entries=entries)
            CACHE[fs_chemsys] = phase_diagram
        except Exception:
            CACHE[fs_chemsys] = None
            return None, None

    if phase_diagram is None:
        return None, None

    try:
        entry = PDEntry(formula_in, energy_in, id_in)
        e_form = phase_diagram.get_form_energy_per_atom(entry)
        e_hull = phase_diagram.get_e_above_hull(entry)
    except Exception:
        return None, None
    
    return e_form, e_hull


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--ref-file", help="Path to reference data CSV. If None, input-file is used.")
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--id-col", default="immutable_id")
    parser.add_argument("--formula-col", default="full_formula")
    parser.add_argument("--chemsys-col", default="chemsys")
    parser.add_argument("--energy-col", default="energy_corrected")
    parser.add_argument("--workers", type=int, default=4)

    args = parser.parse_args()

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

    print("Preprocessing reference data into global dictionary...")
    data_refs = data_refs.dropna(subset=[args.chemsys_col, args.energy_col, args.formula_col])
    
    for _, row in data_refs.iterrows():
        comp = row[args.formula_col]
        energy = row[args.energy_col]
        mat_id = row[args.id_col]
        chemsys = frozenset(str(row[args.chemsys_col]).split('-'))
        try:
            GLOBAL_ENTRIES_DICT[chemsys].append(PDEntry(comp, energy, mat_id))
        except Exception:
            pass

    print(f"Initializing pandarallel with {args.workers} workers...")
    pandarallel.initialize(nb_workers=args.workers, progress_bar=False, verbose=0)

    print("Calculating formation energy and E-hull for target materials...")
    data_target['chemsys_set'] = data_target[args.chemsys_col].apply(lambda x: set(str(x).split('-')) if pd.notna(x) else set())
    
    results = data_target.parallel_apply(
        lambda row: get_ef_ehull(
            id_in=row[args.id_col],
            formula_in=row[args.formula_col],
            energy_in=row[args.energy_col],
            chemsys_in_set=row['chemsys_set']
        ),
        axis=1,
        result_type='expand'
    )
    
    data_target = data_target.drop(columns=['chemsys_set'])
    data_target[['e_form', 'e_hull']] = results

    print(f"Saving results to {args.output_file}...")
    try:
        data_target.to_csv(args.output_file, compression='gzip' if args.output_file.endswith('.gz') else None, index=False)
        print("Calculation complete. Output saved successfully.")
    except Exception as e:
        print(f"Error saving file: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()