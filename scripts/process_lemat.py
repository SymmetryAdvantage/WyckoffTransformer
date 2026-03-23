#!/usr/bin/env python3
""" 
LeMat Dataset Processor 
Processes parquet files to extract structure information and creates a consolidated CSV. 
Compatible with PBS job scheduler environment. 
"""

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

# Log environment information
logger.info(f"Python version: {sys.version}")
logger.info(f"Working directory: {os.getcwd()}")
logger.info(f"PBS_NCPUS: {os.environ.get('PBS_NCPUS', 'Not set')}")
logger.info(f"NP: {os.environ.get('NP', 'Not set')}")
logger.info(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'Not set')}")

# --- Configuration ---
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

# INPUT_DIR = Path("/home/users/shuya001/WyckoffTransformer/data/lemat_unique_pbe/")
# OUTPUT_DIR = Path("/home/users/shuya001/WyckoffTransformer/data/lemat_unique_pbe/")

# # List of files to be processed
# FILENAMES = [
#     'train-00000-of-00016.parquet', 'train-00001-of-00016.parquet',
#     'train-00002-of-00016.parquet', 'train-00003-of-00016.parquet',
#     'train-00004-of-00016.parquet', 'train-00005-of-00016.parquet',
#     'train-00006-of-00016.parquet', 'train-00007-of-00016.parquet',
#     'train-00008-of-00016.parquet', 'train-00009-of-00016.parquet',
#     'train-00010-of-00016.parquet', 'train-00011-of-00016.parquet',
#     'train-00012-of-00016.parquet', 'train-00013-of-00016.parquet',
#     'train-00014-of-00016.parquet', 'train-00015-of-00016.parquet'
# ]

OUTPUT_FILENAME = "lemat_v2test_compatible_pbe_id_formula_chemsys_energy_corrected.csv.gz"

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

def process_single_file(filename, use_parallel=True):
    """
    Process a single parquet file and return extracted data.
    """
    file_path = INPUT_DIR / filename
    
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return None
    
    logger.info(f"Processing {filename}...")
    
    try:
        # Read parquet file
        df = pd.read_parquet(file_path)
        logger.info(f"Loaded {len(df)} rows from {filename}")
        
        # Check required columns
        required_cols = ['immutable_id', 'cif', 'energy_corrected']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing columns in {filename}: {missing_cols}")
            return None
        
        # Extract only required columns first to reduce memory usage
        df_subset = df[required_cols].copy()
        del df  # Free memory
        gc.collect()
        
        # Process CIF strings to get formulas and chemical systems
        cif_results = process_cif_batch(df_subset[['immutable_id', 'cif']], use_parallel=use_parallel)
        
        # Combine results
        result_df = pd.DataFrame({
            'immutable_id': df_subset['immutable_id'],
            'full_formula': cif_results['full_formula'],
            'chemsys': cif_results['chemsys'],
            'energy_corrected': df_subset['energy_corrected']
        })
        
        # Remove rows where parsing failed
        initial_count = len(result_df)
        # result_df = result_df.dropna(subset=['full_formula', 'chemsys'])
        final_count = len(result_df)
        
        if initial_count != final_count:
            logger.warning(f"Dropped {initial_count - final_count} rows due to CIF parsing failures in {filename}")
        
        logger.info(f"Successfully processed {filename}: {final_count} valid rows")
        return result_df
        
    except Exception as e:
        logger.error(f"Error processing {filename}: {str(e)}")
        return None

def main():
    """
    Main processing function.
    """
    logger.info("Starting LeMat dataset processing...")
    
    # Try to initialize pandarallel
    use_parallel = True
    try:
        import multiprocessing
        import os
        
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
            progress_bar=True, 
            nb_workers=workers,
            verbose=1
        )
        logger.info("Pandarallel initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize pandarallel: {e}")
        logger.info("Will use sequential processing instead")
        use_parallel = False
    
    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process all files
    all_dataframes = []
    failed_files = []
    
    for filename in tqdm(FILENAMES, desc="Processing files"):
        result_df = process_single_file(filename, use_parallel=use_parallel)
        
        if result_df is not None:
            all_dataframes.append(result_df)
            logger.info(f"Added {len(result_df)} rows from {filename}")
        else:
            failed_files.append(filename)
        
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
    output_path = OUTPUT_DIR / OUTPUT_FILENAME
    logger.info(f"Saving results to {output_path}...")
    
    final_df.to_csv(output_path, index=False, compression='gzip')
    
    logger.info(f"Processing complete!")
    logger.info(f"Final dataset: {len(final_df)} rows saved to {output_path}")
    logger.info(f"File size: {output_path.stat().st_size / (1024*1024):.2f} MB")
    
    if failed_files:
        logger.warning(f"Failed to process {len(failed_files)} files: {failed_files}")
    
    # Display sample of results
    logger.info("Sample of processed data:")
    logger.info(f"\n{final_df.head()}")
    
    # Display statistics
    logger.info("\nDataset statistics:")
    logger.info(f"Unique chemical systems: {final_df['chemsys'].nunique()}")
    logger.info(f"Energy range: {final_df['energy_corrected'].min():.4f} to {final_df['energy_corrected'].max():.4f}")
    logger.info(f"Missing values:\n{final_df.isnull().sum()}")

if __name__ == "__main__":
    main()