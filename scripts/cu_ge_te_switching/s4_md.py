"""Stage 4: ORB molecular dynamics -- the amorphous state and heating runs.

``meltquench``: melt the crystalline ground state, equilibrate the liquid, quench
it to 300 K, anneal and relax.  The relaxed glass gives the amorphous-crystal
energy and density differences, the two numbers a phase-change memory lives on.

``anneal``: hold a relaxed glass from a ``meltquench`` run at a fixed
temperature and log its potential energy; a drop towards the crystal's energy is
crystallisation, the other half of a phase-change memory cycle.

``heat``: ramp the crystal at zero pressure with a fully flexible cell and log
the cell shape, so that structural transitions (GeTe's rhombohedral-to-cubic
one at ~700 K in experiment, any order-disorder transition of the ternaries)
show up as kinks.  This is the check of whether ORB gets finite-temperature
behaviour right at all.

Both modes use Langevin BAOAB at zero external stress.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from ase import units
from ase.build import find_optimal_cell_shape, make_supercell
from ase.io import read, write
from ase.md.langevinbaoab import LangevinBAOAB
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from common import OUT_DIR, REPO_ROOT, orb_calculator, relax, setup_logging

log = setup_logging()
TIMESTEP_FS = 2.0


def ground_state(formula: str):
    """The lowest-energy entry of *formula* on the corrected hull."""
    entries = pd.read_csv(OUT_DIR / "hull_entries_corrected.csv")
    rows = entries[entries["reduced_formula"] == formula].sort_values("e_above_hull")
    if rows.empty:
        raise KeyError(formula)
    row = rows.iloc[0]
    return read(REPO_ROOT / row["cif"]), row


def build_supercell(atoms, target_atoms: int, shape: str = "sc"):
    n = max(1, round(target_atoms / len(atoms)))
    matrix = find_optimal_cell_shape(atoms.cell, n, shape, verbose=False)
    return make_supercell(atoms, matrix)


class Recorder:
    def __init__(self, atoms, dyn, path, every):
        self.atoms, self.dyn, self.every = atoms, dyn, every
        self.rows, self.path = [], path
        self.t0 = time.time()

    def __call__(self, stage, target_t):
        a = self.atoms
        cellpar = a.cell.cellpar()
        self.rows.append({
            "stage": stage,
            "time_ps": self.dyn.get_time() / (1000 * units.fs),
            "target_T": target_t,
            "T": a.get_temperature(),
            "epot_per_atom": a.get_potential_energy() / len(a),
            "volume_per_atom": a.get_volume() / len(a),
            "a": cellpar[0], "b": cellpar[1], "c": cellpar[2],
            "alpha": cellpar[3], "beta": cellpar[4], "gamma": cellpar[5],
            "wall_s": time.time() - self.t0,
        })


def run_segment(atoms, dyn, recorder, stage, t_start, t_end, ps, traj=None,
                frames_per_ps=1):
    steps = round(ps * 1000 / TIMESTEP_FS)
    record_every = max(1, round(1000 / TIMESTEP_FS / frames_per_ps))
    for i in range(steps):
        t = t_start + (t_end - t_start) * i / max(1, steps - 1)
        if i % 25 == 0:
            dyn.set_temperature(temperature_K=t)
        dyn.run(1)
        if (i + 1) % record_every == 0:
            recorder(stage, t)
            if traj is not None:
                write(traj, atoms, append=True)
    log.info("%s done: T=%.0f K, V=%.2f A3/atom, Epot=%.4f eV/atom", stage,
             atoms.get_temperature(), atoms.get_volume() / len(atoms),
             atoms.get_potential_energy() / len(atoms))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formula", required=True)
    parser.add_argument("--mode", choices=["meltquench", "heat", "anneal"], required=True)
    parser.add_argument("--atoms", type=int, default=216, help="Target supercell size")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--t-melt", type=float, default=1500.0)
    parser.add_argument("--t-liquid", type=float, default=1100.0)
    parser.add_argument("--quench-rate", type=float, default=10.0, help="K/ps")
    parser.add_argument("--t-max", type=float, default=1000.0, help="heat mode: final T")
    parser.add_argument("--heat-rate", type=float, default=5.0, help="K/ps")
    parser.add_argument("--conventional", action="store_true",
                        help="Build the supercell from the conventional standard cell")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tag", default="")
    parser.add_argument("--d3", action="store_true", help="ORB + D3(BJ) dispersion")
    parser.add_argument("--from-run", help="anneal mode: meltquench run directory name")
    parser.add_argument("--t-anneal", type=float, default=600.0)
    parser.add_argument("--anneal-ps", type=float, default=200.0)
    args = parser.parse_args()

    crystal, row = ground_state(args.formula)
    if args.conventional:
        from pymatgen.io.ase import AseAtomsAdaptor
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        conv = SpacegroupAnalyzer(AseAtomsAdaptor.get_structure(crystal), symprec=0.1)
        crystal = AseAtomsAdaptor.get_atoms(conv.get_conventional_standard_structure())
    calculator = orb_calculator(args.device, d3=args.d3)
    # the crystal reference is relaxed with the very calculator the MD uses
    crystal, _ = relax(crystal, calculator, fmax=0.002)
    crystal_energy = crystal.get_potential_energy() / len(crystal)
    if args.mode == "anneal":
        atoms = read(OUT_DIR / "md" / args.from_run / "amorphous_relaxed.extxyz")
    else:
        atoms = build_supercell(crystal, args.atoms)
    tag = ("_d3" if args.d3 else "") + args.tag
    if args.mode == "anneal":
        tag = f"_{int(args.t_anneal)}K" + tag
    name = f"{args.mode}_{args.formula}_{len(atoms)}_s{args.seed}{tag}"
    out = OUT_DIR / "md" / name
    out.mkdir(parents=True, exist_ok=True)
    traj = out / "trajectory.extxyz"
    traj.unlink(missing_ok=True)
    write(out / "start.extxyz", atoms)
    log.info("%s: %d atoms from %s (%s)", name, len(atoms), row["name"], row["cif"])

    atoms.calc = calculator
    rng = np.random.default_rng(args.seed)
    t0 = {"heat": 300.0, "meltquench": args.t_melt, "anneal": args.t_anneal}[args.mode]
    MaxwellBoltzmannDistribution(atoms, temperature_K=t0, rng=rng)
    Stationary(atoms)
    dyn = LangevinBAOAB(atoms, TIMESTEP_FS * units.fs, temperature_K=t0, externalstress=0.0,
                        hydrostatic=args.mode != "heat", T_tau=100 * units.fs,
                        P_tau=2000 * units.fs, rng=rng)
    rec = Recorder(atoms, dyn, out, 1)
    summary = {"name": name, "formula": args.formula, "n_atoms": len(atoms),
               "crystal_entry": row["name"], "crystal_e_above_hull": float(row["e_above_hull"]),
               "d3": args.d3, "crystal_energy_per_atom": float(crystal_energy),
               "crystal_volume_per_atom": crystal.get_volume() / len(crystal)}

    if args.mode == "meltquench":
        run_segment(atoms, dyn, rec, "melt", args.t_melt, args.t_melt, 10, traj, frames_per_ps=0.5)
        run_segment(atoms, dyn, rec, "liquid", args.t_liquid, args.t_liquid, 15, traj, frames_per_ps=0.5)
        quench_ps = (args.t_liquid - 300.0) / args.quench_rate
        run_segment(atoms, dyn, rec, "quench", args.t_liquid, 300.0, quench_ps, traj, frames_per_ps=0.2)
        run_segment(atoms, dyn, rec, "anneal300", 300.0, 300.0, 10, traj, frames_per_ps=0.5)
        glass, converged = relax(atoms, calculator, fmax=0.01, steps=3000)
        write(out / "amorphous_relaxed.extxyz", glass)
        e_a = glass.get_potential_energy() / len(glass)
        summary.update({
            "amorphous_energy_per_atom": e_a,
            "amorphous_minus_crystal_meV": 1000 * (e_a - summary["crystal_energy_per_atom"]),
            "amorphous_volume_per_atom": glass.get_volume() / len(glass),
            "relax_converged": converged,
        })
        summary["amorphous_volume_change"] = (summary["amorphous_volume_per_atom"]
                                              / summary["crystal_volume_per_atom"] - 1.0)
        liquid = [r for r in rec.rows if r["stage"] == "liquid"]
        summary["liquid_volume_per_atom"] = float(np.mean([r["volume_per_atom"] for r in liquid[len(liquid) // 2:]]))
    elif args.mode == "anneal":
        run_segment(atoms, dyn, rec, "anneal", args.t_anneal, args.t_anneal, args.anneal_ps, traj,
                    frames_per_ps=0.5)
        final, converged = relax(atoms, calculator, fmax=0.01, steps=3000)
        write(out / "annealed_relaxed.extxyz", final)
        e_f = final.get_potential_energy() / len(final)
        summary.update({"t_anneal": args.t_anneal, "anneal_ps": args.anneal_ps,
                        "annealed_minus_crystal_meV": 1000 * (e_f - summary["crystal_energy_per_atom"]),
                        "annealed_volume_per_atom": final.get_volume() / len(final),
                        "relax_converged": converged})
    else:
        run_segment(atoms, dyn, rec, "equilibrate", 300.0, 300.0, 10, traj, frames_per_ps=0.5)
        ramp_ps = (args.t_max - 300.0) / args.heat_rate
        run_segment(atoms, dyn, rec, "heat", 300.0, args.t_max, ramp_ps, traj, frames_per_ps=0.5)
        write(out / "final.extxyz", atoms)

    pd.DataFrame(rec.rows).to_csv(out / "log.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("summary: %s", summary)


if __name__ == "__main__":
    main()
