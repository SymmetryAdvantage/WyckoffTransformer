#!/usr/bin/env python3
# cryspr_orb_correction.py

# ASE
import ase
from packaging import version

if version.parse(ase.__version__) > version.parse("3.22.1"):
    from ase.constraints import FixSymmetry, FixAtoms
    from ase.filters import ExpCellFilter as CellFilter
else:
    from ase.spacegroup.symmetrize import FixSymmetry
    from ase.constraints import ExpCellFilter as CellFilter
    from ase.constraints import FixAtoms

    print("Warning: No FrechetCellFilter in ase with version ",
          f"{ase.__version__}, the ExpCellFilter will be used instead.")
from ase.optimize import FIRE
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.optimize.optimize import Optimizer
from ase.optimize import FIRE
from ase.io import write
from ase.spacegroup import get_spacegroup

# MLFF ASE caculator
# Orb-v3
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator

from typing import Optional

# pymatgen IO
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
# --- Added for MP2020 Correction ---
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.io.vasp.inputs import Incar, Poscar
from pymatgen.io.vasp.sets import MPRelaxSet
import tempfile
# --- End Correction Imports ---


# Time
from datetime import datetime


def now():
    return datetime.now().strftime("%Y-%b-%d %H:%M:%S")


def get_mp2020_corrected_cse(
    structure: Structure, total_energy: float
) -> ComputedStructureEntry:
    """
    Generate MP2020-compatible CSE with proper DFT/DFT+U parameters
    and apply the default MP2020 correction scheme.
    
    Adapted from compare_UMA_formation_energy_with_MP_GGA.py
    """
    # Write VASP inputs files as if we were going to do a standard MP run
    # this is mainly necessary to get the right U values / etc
    b = MPRelaxSet(structure)
    with tempfile.TemporaryDirectory() as tmpdirname:
        b.write_input(f"{tmpdirname}/", potcar_spec=True)
        poscar = Poscar.from_file(f"{tmpdirname}/POSCAR")
        incar = Incar.from_file(f"{tmpdirname}/INCAR")
        clean_structure = Structure.from_file(f"{tmpdirname}/POSCAR")

    # Get the U values and figure out if we should have run a GGA+U calc
    param = {"hubbards": {}}
    if "LDAUU" in incar:
        param["hubbards"] = dict(zip(poscar.site_symbols, incar["LDAUU"], strict=False))
    param["is_hubbard"] = (
        incar.get("LDAU", True) and sum(param["hubbards"].values()) > 0
    )
    if param["is_hubbard"]:
        param["run_type"] = "GGA+U"
    else:
        param["run_type"] = "GGA"

    # Make a ComputedStructureEntry without the correction
    oxidation_states = structure.composition.oxi_state_guesses()
    if len(oxidation_states) == 0:
        oxidation_states = {}
    else:
        oxidation_states = oxidation_states[0]
    cse_d = {
        "structure": clean_structure,
        "energy": total_energy, # This is the uncorrected energy
        "correction": 0.0,
        "parameters": param,
        "data": {"oxidation_states": oxidation_states},
    }

    # Apply the default MP 2020 correction scheme
    cse = ComputedStructureEntry.from_dict(cse_d)
    MaterialsProject2020Compatibility(
        check_potcar=False,
    ).process_entries(
        cse, clean=True, inplace=True, on_error="raise"
    )  # noqa: PD002, RUF100

    # cse.energy is now the *corrected* energy
    return cse


FIX_SYMMETRY = True  # Warning: apply the symmetry constraint
# Main codes
# --------------------------------------------------------------------------- #
import os


def run_ase_relaxer(
        atoms_in: Atoms,
        calculator: Calculator,
        optimizer: Optimizer = FIRE,
        cell_filter=None,
        fix_symmetry: bool = True,
        fix_fractional: bool = False,
        hydrostatic_strain: bool = False,
        symprec: float = 1e-3,
        fmax: float = 0.02,
        steps_limit: int = 500,
        logfile: str = "-",
        wdir: str = "./",
) -> Atoms:
    atoms = atoms_in.copy()
    full_formula = atoms.get_chemical_formula(mode="metal")
    reduced_formula = atoms.get_chemical_formula(mode="metal", empirical=True)
    atoms.calc = calculator
    if fix_fractional:
        atoms.set_constraint([FixAtoms(indices=[atom.index for atom in atoms])])
    spg0 = get_spacegroup(atoms, symprec=symprec)
    if fix_symmetry:
        atoms.set_constraint([FixSymmetry(atoms, symprec=symprec)])
    if cell_filter is not None:
        target = cell_filter(atoms, hydrostatic_strain=hydrostatic_strain)
    else:
        target = atoms

    E0 = atoms.get_potential_energy()
    logcontent1 = "\n".join([
        f"[{now()}] CrySPR Info: Start structure relaxation.",
        f"[{now()}] CrySPR Info: Total energy for initial input = {E0:12.5f} eV",
        f"[{now()}] CrySPR Info: Initial symmetry {spg0.symbol} ({spg0.no})",
        f"[{now()}] CrySPR Info: Symmetry tolerance {symprec})",
        f"[{now()}] CrySPR Info: Symmetry constraint? {'Yes' if fix_symmetry else 'No'}",
        f"[{now()}] CrySPR Info: Relax cell? {'Yes' if cell_filter is not None else 'No'}",
        f"[{now()}] CrySPR Info: Relax atomic postions? {'Yes' if not fix_fractional else 'No'}",
        f"#{'-' * 60}#",
        f"\n",
    ])
    if logfile == "-":
        print(logcontent1)
    else:
        with open(f"{wdir}/{logfile}", mode='at') as f:
            f.write(logcontent1)
    opt = optimizer(atoms=target,
                    #                    trajectory=f"{wdir}/{reduced_formula}_{full_formula}_opt.traj",
                    logfile=f"{wdir}/{logfile}",
                    )
    opt.run(fmax=fmax, steps=steps_limit)
    if cell_filter is None:
        write(filename=f'{wdir}/{reduced_formula}_{full_formula}_fix-cell.cif',
              images=atoms,
              format="cif",
              )
    else:
        write(filename=f'{wdir}/{reduced_formula}_{full_formula}_cell+pos.cif',
              images=atoms,
              format="cif",
              )
    cell_diff = (atoms.cell.cellpar() / atoms_in.cell.cellpar() - 1.0) * 100
    E1 = atoms.get_potential_energy()
    spg1 = get_spacegroup(atoms, symprec=symprec)

    logcontent2 = "\n".join([
        f"#{'-' * 60}#",
        f"[{now()}] CrySPR Info: End structure relaxation.",
        f"[{now()}] CrySPR Info: Total energy for final structure = {E1:12.5f} eV",
        f"[{now()}] CrySPR Info: Final symmetry {spg1.symbol} ({spg1.no})",
        f"[{now()}] CrySPR Info: Symmetry tolerance {symprec}",
        f"[{now()}] CrySPR Info: The max absolute force {abs(atoms.get_forces()).max()}",
        f"Optimized Cell: {atoms.cell.cellpar()}",
        f"Cell diff (%): {cell_diff}",
        # f"Scaled positions:\n{atoms.get_scaled_positions()}", # comment out to minimize the file size
        f"\n",
    ]
    )
    if logfile == "-":
        print(logcontent2)
    else:
        with open(f"{wdir}/{logfile}", mode='at') as f:
            f.write(logcontent2)
    return atoms


def stepwise_relax(
        atoms_in: Atoms,
        calculator: Calculator,
        optimizer: Optimizer = FIRE,
        hydrostatic_strain: bool = False,
        symprec: float = 1e-3,
        fmax: float = 0.02,
        steps_limit: int = 500,
        logfile_prefix: str = "",
        logfile_postfix: str = "",
        wdir: str = "./",
) -> Atoms:
    """
    Do fix-cell relaxation first then cell + atomic postions.
    :param atoms_in: an input ase.Atoms object
    :param calculator: an ase calculator to be used
    :param optimizer: a local optimization algorithm, default FIRE
    :param hydrostatic_strain: if do isometrically cell-scaled relaxation, default True
    :param fmax: the max force per atom (unit as defined by the calculator), default 0.02
    :param steps_limit: the max steps to break the relaxation loop, default 500
    :param logfile_prefix: a prefix of the log file, default ""
    :param logfile_postfix: a postfix of the log file, default ""
    :param wdir: string of working directory, default "./" (current)
    :return: the last ase.Atoms trajectory
    """

    if not os.path.exists(wdir):
        os.makedirs(wdir)
    atoms = atoms_in.copy()
    full_formula = atoms.get_chemical_formula(mode="metal")
    reduced_formula = atoms.get_chemical_formula(mode="metal", empirical=True)
    structure0 = AseAtomsAdaptor.get_structure(atoms)
    structure0.to(filename=f'{wdir}/{reduced_formula}_{full_formula}_0_initial_symmetrized.cif', symprec=symprec)

    # fix cell relaxation
    logfile1 = "_".join([logfile_prefix, "fix-cell", logfile_postfix, ]).strip("_") + ".log"
    atoms1 = run_ase_relaxer(
        atoms_in=atoms,
        calculator=calculator,
        optimizer=optimizer,
        fix_symmetry=FIX_SYMMETRY,
        cell_filter=None,
        fix_fractional=False,
        hydrostatic_strain=hydrostatic_strain,
        fmax=fmax,
        steps_limit=steps_limit,
        logfile=logfile1,
        wdir=wdir,
    )

    atoms = atoms1.copy()
    structure1 = AseAtomsAdaptor.get_structure(atoms)
    _ = structure1.to(filename=f'{wdir}/{reduced_formula}_{full_formula}_1_fix-cell_symmetrized.cif', symprec=symprec)

    # relax both cell and atomic positions
    logfile2 = "_".join([logfile_prefix, "cell+positions", logfile_postfix, ]).strip("_") + ".log"
    atoms2 = run_ase_relaxer(
        atoms_in=atoms,
        calculator=calculator,
        optimizer=optimizer,
        fix_symmetry=FIX_SYMMETRY,
        cell_filter=CellFilter,
        fix_fractional=False,
        hydrostatic_strain=hydrostatic_strain,
        fmax=fmax,
        steps_limit=steps_limit,
        logfile=logfile2,
        wdir=wdir,
    )
    structure2 = AseAtomsAdaptor.get_structure(atoms2)
    _ = structure2.to(filename=f'{wdir}/{reduced_formula}_{full_formula}_2_cell+pos_symmetrized.cif', symprec=1e-3)

    return atoms2


def single_run(
        atoms_in: Atoms,
        relax_calculator: Optional[Calculator] = None,
        optimizer: Optimizer = FIRE,
        fmax: float = 0.02,
        verbose: bool = False,
        wdir: str = "./",
        logfile: str = "-",
        relax_logfile_prefix: str = "",
        relax_logfile_postfix: str = "",
        write_cif: bool = True,
        cif_prefix: str = "",
        cif_posfix: str = "",
):
    if relax_calculator is None:
        relax_calculator = build_orb_calculator()

    content = "\n".join(
        [
            f"[{now()}] CrySPR Info: Use ML-IAP = {relax_calculator.__class__.__name__}",
            f"[{now()}] CrySPR Info: Use local optimization algorithm = {optimizer.__name__}",
            f"[{now()}] CrySPR Info: Use fmax = {fmax}",
            f"\n",
        ]
    )
    if verbose:
        print(content)
    if logfile != "-":
        with open(logfile, mode='at') as f:
            f.write(content)
    elif not verbose:
        print(content)

    atoms_relaxed: Atoms = stepwise_relax(
        atoms_in=atoms_in,
        calculator=relax_calculator,
        optimizer=optimizer,
        fmax=fmax,
        wdir=wdir,
        logfile_prefix=relax_logfile_prefix,
        logfile_postfix=relax_logfile_postfix,
    )

    # log
    content = "\n".join(
        [
            f"[{now()}] CrySPR Info: Done structure relaxation.",
            f"#{'-' * 60}#",
            f"\n",
        ]
    )
    if verbose:
        print(content)
    if logfile != "-":
        with open(logfile, mode='at') as f:
            f.write(content)
    elif not verbose:
        print(content)

    return atoms_relaxed


def get_spg_num_pymatgen(structure: Structure, symprec: float = 0.1, comment="id"):
    if structure is None:
        return None
    # Analyze the symmetry
    try:
        spga: SpacegroupAnalyzer = SpacegroupAnalyzer(structure, symprec=symprec)
        spg_num = spga.get_space_group_number()
    except:
        print(f"Warning: symprec = {symprec} not work, roll back to the default (0.01)")
        try:
            spga: SpacegroupAnalyzer = SpacegroupAnalyzer(structure)
            spg_num = spga.get_space_group_number()
        except Exception as e:
            print(f"SpacegroupAnalyzer error for {comment}! Set space group number as 0! Also deprecated.\n")
            print(e)
            spg_num = 0
    return spg_num


def write_cif(structure: Structure, idx: int):
    _ = structure.to(f"./WTFL-DCPP_generated-{idx}.cif", fmt="cif")
    return 0


# -------------------------------- Preparations ---------------------------------#
# for consistency, ExpCellFilter is used
# from ase.filters import ExpCellFilter as CellFilter

import sys
import json
import pandas as pd
import os
import warnings

rootdir = os.getcwd()

DEFAULT_ORB_DEVICE = os.environ.get("ORB_DEVICE", "cpu")
DEFAULT_ORB_PRECISION = os.environ.get("ORB_PRECISION", "float32-high")


def build_orb_calculator(
        device: Optional[str] = None,
        precision: Optional[str] = None,
) -> ORBCalculator:
    """Return an ORB ASE calculator with the desired device/precision."""
    selected_device = device or DEFAULT_ORB_DEVICE
    selected_precision = precision or DEFAULT_ORB_PRECISION
    orbff = pretrained.orb_v3_conservative_inf_mpa(
        device=selected_device,
        precision=selected_precision,
    )
    return ORBCalculator(orbff, device=selected_device)

def func_run(
        structure: Structure,
        space_group_number: int,
        idx: int,
        model="wtf",
        cryspr_log_prefix="WyckoffTransformer",
        tmp_csv_file=f"{rootdir}/cache_id_formula_energy_energy_per_atom.csv",
        calculator: Optional[Calculator] = None,
        calculator_device: Optional[str] = None,
        calculator_precision: Optional[str] = None,
):
    if calculator is None:
        calculator = build_orb_calculator(
            device=calculator_device,
            precision=calculator_precision,
        )

    if space_group_number == 0:
        with open(f"{rootdir}/error_list.out", mode='a+') as f:
            f.write(
                f"[{now()}] CrySPR Error: {idx} symmetry undetermined error.\n"
            )
        return None, None, None, None, None, None # Added None for corrected vals

    work_dir = f"{idx}"
    if not os.path.exists(f"{rootdir}/{work_dir}"):
        os.makedirs(f"{rootdir}/{work_dir}")
    os.chdir(f"{rootdir}/{work_dir}")

    # generate structure
    write_cif(structure, idx)
    atoms_in: Atoms = AseAtomsAdaptor.get_atoms(structure)
    if atoms_in is None:
        os.chdir(f"{rootdir}/{work_dir}")
        return None, None, None, None, None, None # Added None for corrected vals

    formula = atoms_in.get_chemical_formula(mode="metal")
    # Initialize all return values
    atoms, energy, energy_per_atom = None, None, None
    energy_corrected, energy_per_atom_corrected = None, None
    
    try:
        with open(f"{rootdir}/cryspr.log", mode='a+') as f:
            f.write(
                f"[{now()}] CrySPR Info: Starting {model}-{idx}\n"
            )
        atoms_relaxed = single_run(
            atoms_in=atoms_in,
            relax_calculator=calculator,
            fmax=0.05,  # for consistency with previous
            logfile=f"{cryspr_log_prefix}_{model}-{idx}.log",
            relax_logfile_prefix=f"{formula}",
            relax_logfile_postfix="relax",
        )
        atoms = atoms_relaxed.copy()
        energy = atoms_relaxed.get_potential_energy()
        energy_per_atom = energy / len(atoms)

        # --- NEW: Calculate MP2020 Corrected Energy ---
        try:
            # Convert relaxed ASE Atoms back to Pymatgen Structure
            pmg_structure_relaxed = AseAtomsAdaptor.get_structure(atoms)
            
            # Get the MP2020 corrected ComputedStructureEntry
            # We pass the uncorrected energy (energy)
            corrected_cse = get_mp2020_corrected_cse(pmg_structure_relaxed, energy) 
            
            # The .energy attribute is now the corrected energy
            energy_corrected = corrected_cse.energy
            energy_per_atom_corrected = corrected_cse.energy_per_atom
        
        except Exception as e:
            # Handle potential failures in the correction (e.g., VASP input gen fails)
            with open(f"{rootdir}/cryspr.log", mode='a+') as f:
                f.write(
                    f"[{now()}] CrySPR Error: Failed MP2020 correction for {model}-{idx}: {e}\n"
                )
            # energy_corrected and energy_per_atom_corrected remain None
        # --- END NEW ---

        with open(f"{rootdir}/cryspr.log", mode='a+') as f:
            f.write(
                f"[{now()}] CrySPR Info: Done {model}-{idx}\n"
            )
    except Exception as e_relax:
        with open(f"{rootdir}/cryspr.log", mode='a+') as f:
            f.write(
                f"[{now()}] CrySPR Error: Failed for {model}-{idx} with error: {e_relax}"
            )
        # All values remain None as initialized

    os.chdir(f"{rootdir}")

    with open(f"{rootdir}/finished_list.out", mode='a+') as f:
        f.write(
            f"[{now()}] CrySPR Info: Finished {model}-{idx}\n"
        )

    with open(tmp_csv_file, 'a+') as f:
        f.write(
            ",".join([
                f"{model}", f"{idx}", f"{formula}", 
                f"{energy}", f"{energy_per_atom}",
                f"{energy_corrected}", f"{energy_per_atom_corrected}"
            ]) + "\n"
        )

    return atoms, formula, energy, energy_per_atom, energy_corrected, energy_per_atom_corrected


def main():
    # impose single thread
    if os.environ.get("OMP_NUM_THREADS", None) != "1":
        warnings.warn(
            message="Serious warning: Please set environment var \n\
                `OMP_NUM_THREADS` to 1, otherwise you will get very slow run ... :("
        )
    try:
        nb_workers = int(os.environ["NP"])
    except:
        print("Warning: NP variable unspecified, set as all CPU cores available.")
        nb_workers = os.cpu_count()

    # use pandarallel for auto multi-processing
    from pandarallel import pandarallel
    pandarallel.initialize(progress_bar=False, nb_workers=nb_workers)
    calculator_device = os.environ.get("ORB_DEVICE", DEFAULT_ORB_DEVICE)
    calculator_precision = os.environ.get("ORB_PRECISION", DEFAULT_ORB_PRECISION)

    # arguments from command line
    if len(sys.argv) >= 4:
        index_start, index_end = int(sys.argv[1]), int(sys.argv[2])
        filepath = sys.argv[3]
        if len(sys.argv) == 5:
            model = sys.argv[4]
        else:
            model = "model_name"
    else:
        index_start, index_end = 0, -1
        filepath = "./sg_letters-uy6zg9y9.diffcsp-pp.json"
        warnings.warn("Start and end indices (Python style) and input file should be specified as cli arguments:\n" +
                      "./this_script.py index_start index_end json_file_name [model_name] \n" +
                      "./this_script.py 0 -1 mp_20/WyckoffTransformer-letters/DiffCSP++/sg_letters-uy6zg9y9.diffcsp-pp.json wtfl-dcpp \n")
        warnings.warn(
            "Setting start and end index back to 0 to -1 (all), and default filename (sg_letters-uy6zg9y9.diffcsp-pp.json)")

    # read json file
    with open(filepath) as f:
        data = json.load(f)
    if index_end == -1:
        index_end = len(data)

    # define something
    indices = list(range(len(data)))
    df_data = pd.DataFrame(
        {
            "model": [f"{model}"] * len(data),
            "id": indices,
            "strc_dict": data,
        }
    )

    df_data["structure"] = df_data["strc_dict"].parallel_apply(Structure.from_dict)
    df_data["space_group_number"] = df_data.parallel_apply(
        lambda df: get_spg_num_pymatgen(structure=df["structure"], comment=df["id"]),
        axis=1,
    )

    df_data_in = df_data.iloc[index_start:index_end]
    # Multi-processing
    series = df_data_in.parallel_apply(
        lambda df: func_run(
            structure=df["structure"],
            space_group_number=df["space_group_number"],
            idx=df["id"],
            model=df["model"],
            # cryspr_log_prefix="test-code",
            calculator_device=calculator_device,
            calculator_precision=calculator_precision,
        ),
        axis=1,
    )

    # Save output
    output_csv_file = f"{rootdir}/{model}_id_formula_energy_corrected_{index_start}-{index_end}.csv"
    df_data_in["formula"] = series.apply(lambda x: x[1])
    df_data_in["energy"] = series.apply(lambda x: x[2])
    df_data_in["energy_per_atom"] = series.apply(lambda x: x[3])
    df_data_in["energy_corrected"] = series.apply(lambda x: x[4])
    df_data_in["energy_per_atom_corrected"] = series.apply(lambda x: x[5])

    output_columns = [
        "model", "id", "formula", 
        "energy", "energy_per_atom", 
        "energy_corrected", "energy_per_atom_corrected"
    ]
    df_data_in[output_columns].to_csv(output_csv_file, mode="w", index=False)

    return 0


# -------------------------------- Calculations ---------------------------------#
if __name__ == "__main__":
    main()