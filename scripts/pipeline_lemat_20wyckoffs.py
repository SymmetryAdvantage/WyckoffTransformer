#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path
from datasets import load_dataset
import pandas as pd
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import numpy as np
from pandarallel import pandarallel

def run_command(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def count_wyckoffs(cif_str):
    try:
        struct = Structure.from_str(cif_str, fmt='cif')
        sga = SpacegroupAnalyzer(struct)
        sym_data = sga.get_symmetry_dataset()
        if sym_data is None:
            return 9999
        return len(set(sym_data['equivalent_atoms']))
    except Exception:
        return 9999

def main():
    base_dir = Path("data/lemat-bulk")
    raw_dir = base_dir / "raw"
    cif_dir = base_dir / "cif_prepared"
    output_dir = base_dir / "20_wyckoffs"
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    cif_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step A: Download
    print("Downloading dataset (skipped)...")
    # ds = load_dataset('LeMaterial/LeMat-Bulk', 'compatible_pbe', split='train')
    
    # parquet_path = raw_dir / 'data.parquet'
    # if not parquet_path.exists():
    #     ds.to_parquet(str(parquet_path))
    # else:
    #     print(f"{parquet_path} already exists, skipping download.")
    
    # Step B: Prepare CIFs
    print("Preparing CIFs (skipped)...")
    # run_command(f".venv/bin/python scripts/prepare_cif.py --input-dir {raw_dir} --output-dir {cif_dir}")
    
    # Step C: Process
    print("Processing LeMat (skipped)...")
    processed_csv = base_dir / "lemat_pbe.csv.gz"
    # run_command(f".venv/bin/python scripts/process_lemat.py --input-dir {cif_dir} --output-file {processed_csv}")
    
    # Step D: Compute e_hull
    print("Computing E-hull (skipped)...")
    ehull_csv = base_dir / "lemat_pbe_ehull.csv.gz"
    # Need to specify correct columns as default ones might be different
    # run_command(f".venv/bin/python scripts/compute_e_hull.py --workers 40 --input-file {processed_csv} --output-file {ehull_csv} --id-col immutable_id --formula-col full_formula --chemsys-col chemsys --energy-col energy_corrected")
    
    # Step E & F: Filters
    print("Loading data for filtering...")
    df_ehull = pd.read_csv(ehull_csv, usecols=['immutable_id', 'e_hull', 'e_form'])
    df_base = pd.read_csv(processed_csv)
    
    print("Merging dataframes...")
    df = pd.merge(df_base, df_ehull, on='immutable_id', how='inner')
    
    # Filter max_force
    print(f"Total rows before force filter: {len(df)}")
    df['max_force'] = pd.to_numeric(df['max_force'], errors='coerce')
    df = df[df['max_force'] <= 0.02]
    print(f"Rows after force filter: {len(df)}")
    
    # Wyckoff filtering is handled by cache_a_dataset.py
    
    final_csv = output_dir / "lemat_pbe_20wyckoffs.csv.gz"
    df.to_csv(final_csv, index=False, compression='gzip')
    print(f"Final dataset saved to {final_csv}")

if __name__ == '__main__':
    main()