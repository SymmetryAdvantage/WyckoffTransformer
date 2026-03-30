#!/usr/bin/env python3

# PyXtal
from pyxtal import pyxtal
from pyxtal.tolerance import Tol_matrix

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
from ase.io import write
from ase.spacegroup import get_spacegroup

# MLFF ASE caculator
# CHGNet
from chgnet.model import CHGNetCalculator

# pymatgen IO
from pymatgen.io.ase import AseAtomsAdaptor

# Time
from datetime import datetime


def now():
    return datetime.now().strftime("%Y-%b-%d %H:%M:%S")


FIX_SYMMETRY = True  # Warning: apply the symmetry constraint
# Main codes
# --------------------------------------------------------------------------- #
import os  # noqa: E402


def single_pyxtal(
        wyckoffgene: dict,
        iadm: Tol_matrix = Tol_matrix(prototype="atomic", factor=1.3),
        nlimit: int = 20,
) -> Atoms:
    dim = 3
    spg: int = wyckoffgene["group"]
    species_in: list[str] = wyckoffgene["species"]
    numIons_in: list[int] = wyckoffgene["numIons"]
    Wyckoff_in: list[list] = wyckoffgene["sites"]
    try:
        candidate = pyxtal()
        candidate.from_random(
            dim=dim,
            group=spg,
            species=species_in,
            numIons=numIons_in,
            sites=Wyckoff_in,
            tm=iadm,
            max_count=nlimit,
        )
        atoms: Atoms = candidate.to_ase()
        formula = atoms.get_chemical_formula(mode="metal")
        candidate.to_file("pyxtal_generated_" + formula + ".cif")
        return atoms
    except Exception as exc:
        print(f"Raised an exception: {exc} in single PyXtal generation.")
        return None


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
        "\n",
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
        "\n",
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
        relax_calculator: Calculator = CHGNetCalculator(use_device="cpu"),
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
    content = "\n".join(
        [
            f"[{now()}] CrySPR Info: Use ML-IAP = {relax_calculator.__class__.__name__}",
            f"[{now()}] CrySPR Info: Use local optimization algorithm = {optimizer.__name__}",
            f"[{now()}] CrySPR Info: Use fmax = {fmax}",
            "\n",
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
            "\n",
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


def find_key_lowest_e(e_dict: dict):
    values: list = list(e_dict.values())
    array: np.array = np.array(values)
    e_min: float = array.min()
    for key, value in e_dict.items():
        if value <= e_min:
            return key


# -------------------------------- Preparations ---------------------------------#
# for consistency, ExpCellFilter is used
# from ase.filters import ExpCellFilter as CellFilter

import sys  # noqa: E402
import json  # noqa: E402
import pandas as pd  # noqa: E402
import numpy as np  # noqa: E402
import warnings  # noqa: E402
import glob  # noqa: E402

rootdir = os.getcwd()

def func_run(
        id_gene,  # Wyckoff gene id: str or int
        wyckoffgene,  # Wyckoff representation of structure
        model="wtf",
        n_trial_each_wyckoff_gene=6,
        cryspr_log_prefix="WyckoffTransformer",
        tmp_csv_file=f"{rootdir}/cache_id_formula_energy_energy_per_atom.csv",
        calculator: Calculator = CHGNetCalculator(use_device="cpu"),
):
    # layer for each Wyckoff gene
    wdir1 = f"{id_gene}"
    if not os.path.exists(f"{rootdir}/{wdir1}"):
        os.makedirs(f"{rootdir}/{wdir1}")
    os.chdir(f"{rootdir}/{wdir1}")

    # loop layer for some trials for each Wyckoff gene
    atoms_relaxed_dict = {}
    energy_dict = {}
    for i_trial in range(n_trial_each_wyckoff_gene):
        wdir2 = f"trial-{i_trial}"
        if not os.path.exists(f"{rootdir}/{wdir1}/{wdir2}"):
            os.makedirs(f"{rootdir}/{wdir1}/{wdir2}")
        os.chdir(f"{rootdir}/{wdir1}/{wdir2}")

        # generate structure
        atoms_in = single_pyxtal(
            wyckoffgene=wyckoffgene,
            nlimit=30,
        )
        if atoms_in is None:
            os.chdir(f"{rootdir}/{wdir1}")
            continue
        formula = atoms_in.get_chemical_formula(mode="metal")
        try:
            with open(f"{rootdir}/cryspr.log", mode='a+') as f:
                f.write(
                    f"[{now()}] CrySPR Info: Starting {model}-{id_gene}, trial-{i_trial}\n"
                )
            atoms_relaxed = single_run(
                atoms_in=atoms_in,
                relax_calculator=calculator,
                fmax=0.05,  # for consistency with previous
                logfile=f"{cryspr_log_prefix}_{model}-{id_gene}_trial-{i_trial}.log",
                relax_logfile_prefix=f"{formula}",
                relax_logfile_postfix="relax",
            )
            atoms_relaxed_dict[f"trial-{i_trial}"] = atoms_relaxed
            energy_dict[f"trial-{i_trial}"] = atoms_relaxed.get_potential_energy()
            with open(f"{rootdir}/cryspr.log", mode='a+') as f:
                f.write(
                    f"[{now()}] CrySPR Info: Done {model}-{id_gene}, trial-{i_trial}\n"
                )
        except:  # noqa: E722
            with open(f"{rootdir}/cryspr.log", mode='a+') as f:
                f.write(
                    f"[{now()}] CrySPR Error: Failed for {model}-{id_gene}, trial-{i_trial}\n"
                )

        os.chdir(f"{rootdir}/{wdir1}")

    if len(atoms_relaxed_dict) == 0:
        os.chdir(rootdir)
        atoms, energy, energy_per_atom = None, None, None
    else:
        with open(f"{rootdir}/finished_list.out", mode='a+') as f:
            f.write(
                f"[{now()}] CrySPR Info: Finished {model}-{id_gene}\n"
            )
        lowest_trial = find_key_lowest_e(energy_dict)
        os.symlink(lowest_trial, "./trial-lowest", target_is_directory=True)
        cif_lowest = glob.glob("./trial-lowest/*_cell+pos.cif")[0]
        os.symlink(cif_lowest, "./min_e_strc.cif")

        atoms = atoms_relaxed_dict[lowest_trial]
        energy = energy_dict[lowest_trial]
        energy_per_atom = energy_dict[lowest_trial] / len(atoms_relaxed_dict[lowest_trial])

    with open(tmp_csv_file, 'a+') as f:
        f.write(
            ",".join([f"{model}", f"{id_gene}", f"{formula}", f"{energy}", f"{energy_per_atom}"]) + "\n"
        )

    return atoms, formula, energy, energy_per_atom


def main():
    # impose single thread
    if os.environ.get("OMP_NUM_THREADS", None) != "1":
        warnings.warn(
            message="Serious warning: Please set environment var \n\
                `OMP_NUM_THREADS` to 1, otherwise you will get very slow run ... :("
        )
    try:
        nb_workers = int(os.environ["NP"])
    except:  # noqa: E722
        print("Warning: NP variable unspecified, set as all CPU cores available.")
        nb_workers = os.cpu_count()

    # use pandarallel for auto multi-processing
    from pandarallel import pandarallel

    pandarallel.initialize(progress_bar=False, nb_workers=nb_workers)

    # initialize calculator
    calculator = CHGNetCalculator(use_device="cpu")

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
        filepath = "./WyckoffTransformer_mp_20.json"
        warnings.warn("Start and end indices (Python style) and input file should be specified as cli arguments:\n" +
                      "./this_script.py index_start index_end json_file_name [model_name] \n" +
                      "./this_script.py 0 -1 mp_20/WyckoffTransformer/WyckoffTransformer_mp_20.json wtf \n")
        warnings.warn(
            "Setting start and end index back to 0 to -1 (all), and default filename (WyckoffTransformer_mp_20.json)")

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
            "wyckoffgene": data,
        }
    )
    df_data_in = df_data.iloc[index_start:index_end]
    # Multi-processing
    series = df_data_in.parallel_apply(
        lambda df: func_run(
            id_gene=df["id"],
            wyckoffgene=df["wyckoffgene"],
            n_trial_each_wyckoff_gene=6,
            # cryspr_log_prefix="test_code",
            calculator=calculator,
        ),
        axis=1,
    )

    # Save output
    output_csv_file = f"{rootdir}/{model}_id_formula_energy_{index_start}-{index_end}.csv"
    df_data_in["formula"] = series.apply(lambda x: x[1])
    df_data_in["energy"] = series.apply(lambda x: x[2])
    # df_data_in["energy_per_atom"] = series.apply(lambda x: x[3])
    df_data_in[["model", "id", "formula", "energy"]].to_csv(output_csv_file, mode="w", index=False)

    return 0


# -------------------------------- Calculations ---------------------------------#
if __name__ == "__main__":
    main()