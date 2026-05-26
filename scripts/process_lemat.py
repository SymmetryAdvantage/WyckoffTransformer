#!/usr/bin/env python3
""" 
LeMat Dataset Processor 
Processes parquet files to extract structure information and creates a consolidated CSV. 
Compatible with PBS job scheduler environment. 
"""

import argparse
import pandas as pd
from pathlib import Path
from pymatgen.core import Structure
from pandarallel import pandarallel
import numpy as np
import logging
from tqdm import tqdm
import gc
import warnings
import os
import sys

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Functions ---

def parse_cif_safe(row):
    """
    Safely parse CIF string from a DataFrame row and extract composition information.
    Returns tuple: (full_formula, chemsys)
    """
    cif_string = row['cif']
    immutable_id = row['immutable_id']
    try:
        if pd.isna(cif_string) or cif_string == "":
            return np.nan, np.nan
        
        structure = Structure.from_str(cif_string, fmt='cif')
        full_formula = structure.composition.formula
        chemical_system = structure.composition.chemical_system
        
        return full_formula, chemical_system
    
    except Exception as e:
        logger.warning(f"Failed to parse CIF for immutable_id {immutable_id}: {str(e)[:100]}...")
        return np.nan, np.nan

def get_max_force(forces_array):
    try:
        if forces_array is None or len(forces_array) == 0:
            return np.nan
        f = np.stack(forces_array)
        if f.size == 0:
            return np.nan
        return float(np.max(np.abs(f)))
    except Exception:
        return np.nan

def process_cif_batch(df_batch, use_parallel=True):
    """
    Process a batch of CIF strings in parallel or sequentially.
    """
    logger.info(f"Processing batch of {len(df_batch)} CIF strings...")
    
    try:
        if use_parallel:
            # Apply the parsing function in parallel
            results = df_batch.parallel_apply(parse_cif_safe, axis=1)
        else:
            # Fallback to sequential processing with progress bar
            tqdm.pandas(desc="Processing CIF strings")
            results = df_batch.progress_apply(parse_cif_safe, axis=1)
    except Exception as e:
        logger.warning(f"Parallel processing failed: {e}. Falling back to sequential processing.")
        tqdm.pandas(desc="Processing CIF strings")
        results = df_batch.progress_apply(parse_cif_safe, axis=1)
    
    # Split the results into separate columns
    full_formulas = [r[0] for r in results]
    chemical_systems = [r[1] for r in results]
    
    return pd.DataFrame({
        'full_formula': full_formulas,
        'chemsys': chemical_systems
    })

def process_single_file(file_path, use_parallel=True):
    """
    Process a single parquet file and return extracted data.
    """
    logger.info(f"Processing {file_path.name}...")
    
    try:
        # Read parquet file
        df = pd.read_parquet(file_path)
        logger.info(f"Loaded {len(df)} rows from {file_path.name}")
        
        # Check required columns
        required_cols = ['immutable_id', 'cif', 'energy']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            # Maybe it is energy_corrected, lemat original just has energy
            if 'energy_corrected' in df.columns and 'energy' in missing_cols:
                 required_cols[required_cols.index('energy')] = 'energy_corrected'
            else:
                 logger.error(f"Missing columns in {file_path.name}: {missing_cols}")
                 return None
        
        energy_col = 'energy_corrected' if 'energy_corrected' in df.columns else 'energy'
        extract_cols = ['immutable_id', 'cif', energy_col]
        if 'forces' in df.columns:
            extract_cols.append('forces')

        # Extract only required columns first to reduce memory usage
        df_subset = df[extract_cols].copy()
        del df  # Free memory
        gc.collect()
        
        # Calculate max_force if forces available
        if 'forces' in df_subset.columns:
            if use_parallel:
                max_forces = df_subset['forces'].parallel_apply(get_max_force)
            else:
                max_forces = df_subset['forces'].apply(get_max_force)
        else:
            max_forces = np.full(len(df_subset), np.nan)
        
        # Process CIF strings to get formulas and chemical systems
        cif_results = process_cif_batch(df_subset[['immutable_id', 'cif']], use_parallel=use_parallel)
        
        # Combine results
        result_df = pd.DataFrame({
            'immutable_id': df_subset['immutable_id'],
            'full_formula': cif_results['full_formula'],
            'chemsys': cif_results['chemsys'],
            'energy_corrected': df_subset[energy_col],
            'max_force': max_forces,
            'cif': df_subset['cif']
        })
        
        # Remove rows where parsing failed
        initial_count = len(result_df)
        final_count = len(result_df)
        
        if initial_count != final_count:
            logger.warning(f"Dropped {initial_count - final_count} rows due to CIF parsing failures in {file_path.name}")
        
        logger.info(f"Successfully processed {file_path.name}: {final_count} valid rows")
        return result_df
        
    except Exception as e:
        logger.error(f"Error processing {file_path.name}: {str(e)}")
        return None

def main():
    """
    Main processing function.
    """
    parser = argparse.ArgumentParser(description="Process LeMat dataset CIF strings to extract features.")
    parser.add_argument("--input-dir", type=str, required=True, help="Input directory containing Parquet files.")
    parser.add_argument("--output-file", type=str, required=True, help="Output compressed CSV file path (.csv.gz).")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)

    logger.info("Starting LeMat dataset processing...")
    
    # Try to initialize pandarallel
    use_parallel = True
    try:
        import multiprocessing
        
        # Check for PBS environment variables first
        pbs_ncpus = os.environ.get('PBS_NCPUS')
        np_cores = os.environ.get('NP')
        
        if pbs_ncpus:
            n_cores = int(pbs_ncpus)
            logger.info(f"Using PBS_NCPUS: {n_cores} cores")
        elif np_cores:
            n_cores = int(np_cores)
            logger.info(f"Using NP environment variable: {n_cores} cores")
        else:
            n_cores = multiprocessing.cpu_count()
            logger.info(f"Detected {n_cores} CPU cores from system")
        
        # Use most cores but leave 1-2 free for system processes
        workers = max(1, min(n_cores - 1, 20))  # Cap at 20 to avoid overhead
        logger.info(f"Using {workers} workers for parallel processing")
        
        # Initialize pandarallel with explicit core count
        pandarallel.initialize(
            progress_bar=False, 
            nb_workers=workers,
            verbose=0
        )
        logger.info("Pandarallel initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize pandarallel: {e}")
        logger.info("Will use sequential processing instead")
        use_parallel = False
    
    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Process all files
    all_dataframes = []
    failed_files = []
    
    filenames = list(input_dir.glob("*.parquet"))
    if not filenames:
        logger.error(f"No parquet files found in {input_dir}")
        return

    for file_path in tqdm(filenames, desc="Processing files"):
        result_df = process_single_file(file_path, use_parallel=use_parallel)
        
        if result_df is not None:
            all_dataframes.append(result_df)
            logger.info(f"Added {len(result_df)} rows from {file_path.name}")
        else:
            failed_files.append(file_path.name)
        
        # Force garbage collection after each file
        gc.collect()
    
    if not all_dataframes:
        logger.error("No files were successfully processed!")
        return
    
    # Concatenate all dataframes
    logger.info("Concatenating all processed data...")
    final_df = pd.concat(all_dataframes, ignore_index=True)
    
    # Clean up memory
    del all_dataframes
    gc.collect()
    
    logger.info(f"Total processed rows: {len(final_df)}")
    logger.info(f"Columns: {list(final_df.columns)}")
    
    # Remove any remaining duplicates based on immutable_id
    initial_count = len(final_df)
    # final_df = final_df.drop_duplicates(subset=['immutable_id'], keep='first')
    final_count = len(final_df)
    
    if initial_count != final_count:
        logger.info(f"Removed {initial_count - final_count} duplicate entries")
    
    # Save to compressed CSV
    logger.info(f"Saving results to {output_file}...")
    
    final_df.to_csv(output_file, index=False, compression='gzip')
    
    logger.info("Processing complete!")
    logger.info(f"Final dataset: {len(final_df)} rows saved to {output_file}")
    logger.info(f"File size: {output_file.stat().st_size / (1024*1024):.2f} MB")
    
    if failed_files:
        logger.warning(f"Failed to process {len(failed_files)} files: {failed_files}")

if __name__ == "__main__":
    main()