"""Stage 5: PBE (Quantum ESPRESSO) checks on ORB geometries.

ORB carries a 10-30 meV/atom error, the same size as the polymorph gaps the
switching candidates rest on, so the energy order of each candidate pair is
re-checked with PBE single points at the ORB geometries, using the same
pslibrary 1.0.0 PAW potentials for every structure.  The same calculation gives
the Kohn-Sham band edges, from which a PBE gap is read (zero when a band crosses
the Fermi level).  PBE without spin-orbit underestimates gaps in tellurides;
the comparison between states is the point, not the absolute value.

Input: a list of CIFs (``--cifs``) or state ids from ``states.csv``.
Output: ``dft/<label>/`` with the pw.x run and a line in ``dft/dft.jsonl``.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess

import numpy as np
import pandas as pd
from ase.io import read, write
from common import OUT_DIR, REPO_ROOT, setup_logging

log = setup_logging()
PSEUDO_DIR = OUT_DIR / "pseudo"
PSEUDOS = {"Cu": "Cu.pbe-dn-kjpaw_psl.1.0.0.UPF", "Ge": "Ge.pbe-dn-kjpaw_psl.1.0.0.UPF",
           "Te": "Te.pbe-n-kjpaw_psl.1.0.0.UPF"}
VALENCE = {"Cu": 11, "Ge": 14, "Te": 6}
RY = 13.605693122994
QE_BIN = "/home/kna/mamba/envs/qe/bin"  # QE 7.5; the system pw.x 6.7 aborts on input


def kpoint_grid(atoms, spacing: float) -> list[int]:
    recip = 2 * np.pi * np.linalg.norm(atoms.cell.reciprocal(), axis=1)
    return [max(1, int(np.ceil(r / spacing))) for r in recip]


def xml_bands(path):
    """Fermi level and band energies (eV, k x band) from QE's data-file-schema.xml."""
    import xml.etree.ElementTree as ET

    hartree = 27.211386245988
    root = ET.parse(path).getroot()
    bs = root.find("output/band_structure")
    fermi = float(bs.find("fermi_energy").text) * hartree
    eigs = [np.array(ks.find("eigenvalues").text.split(), dtype=float) * hartree
            for ks in bs.findall("ks_energies")]
    return fermi, np.array(eigs)


def run_pw(atoms, label: str, nproc: int, spacing: float, ecut: float, d3: bool = False,
           gamma: bool = False) -> dict:
    work = OUT_DIR / "dft" / label
    work.mkdir(parents=True, exist_ok=True)
    nelec = sum(VALENCE[s] for s in atoms.get_chemical_symbols())
    kpts = None if gamma else kpoint_grid(atoms, spacing)  # None -> K_POINTS gamma
    input_data = {
        "control": {"calculation": "scf", "prefix": "pw", "outdir": "./tmp",
                    "pseudo_dir": str(PSEUDO_DIR), "tprnfor": True, "tstress": True},
        "system": {"ecutwfc": ecut, "ecutrho": 8 * ecut, "occupations": "smearing",
                   "smearing": "mv", "degauss": 0.005, "nbnd": int(nelec / 2 * 1.2) + 8,
                   **({"vdw_corr": "dft-d3", "dftd3_version": 4} if d3 else {})},
        "electrons": {"conv_thr": 1e-8 * len(atoms), "mixing_beta": 0.3},
    }
    write(work / "pw.in", atoms, format="espresso-in", input_data=input_data,
          pseudopotentials=PSEUDOS, kpts=kpts)
    out = work / "pw.out"
    if not (out.exists() and "JOB DONE" in out.read_text()):
        npool = 1 if kpts is None else min(nproc, int(np.prod(kpts)), 4)
        with (work / "pw.out").open("w") as fh:
            subprocess.run([f"{QE_BIN}/mpirun", "--bind-to", "none", "-np", str(nproc), f"{QE_BIN}/pw.x", "-nk", str(npool),
                            "-in", "pw.in"], cwd=work, stdout=fh, stderr=subprocess.STDOUT, check=True,
                           env={**os.environ, "OMP_NUM_THREADS": "1", "PATH": f"{QE_BIN}:/usr/bin:/bin"})
    result = read(out, format="espresso-out")
    energy = result.get_potential_energy()
    fermi, eigs = xml_bands(work / "tmp" / "pw.save" / "data-file-schema.xml")
    below, above = eigs[eigs <= fermi], eigs[eigs > fermi]
    vbm, cbm = below.max(), above.min()
    # a band that has states on both sides of E_F makes the system metallic
    crossing = any((band.min() < fermi < band.max()) for band in eigs.T)
    gap = 0.0 if crossing else float(cbm - vbm)
    return {"energy_per_atom": energy / len(atoms), "fermi": fermi, "vbm": float(vbm), "cbm": float(cbm),
            "pbe_gap": gap, "metallic": bool(crossing), "kpts": kpts, "n_atoms": len(atoms)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", nargs="*", default=[], help="state ids from states.csv")
    parser.add_argument("--cifs", nargs="*", default=[], help="label=path pairs")
    parser.add_argument("--nproc", type=int, default=16)
    parser.add_argument("--kspacing", type=float, default=0.25, help="1/A, 2pi included")
    parser.add_argument("--ecut", type=float, default=50.0, help="Ry")
    parser.add_argument("--d3", action="store_true", help="PBE-D3(BJ)")
    parser.add_argument("--gamma", action="store_true", help="Gamma-point only (large cells)")
    args = parser.parse_args()

    jobs = []
    if args.states:
        states = pd.read_csv(OUT_DIR / "states.csv").set_index("state_id")
        for sid in args.states:
            jobs.append((sid.replace("#", "_"), REPO_ROOT / states.loc[sid, "cif"], sid))
    for item in args.cifs:
        label, path = item.split("=", 1)
        jobs.append((label, path, label))
    for label, path, sid in jobs:
        atoms = read(path)
        log.info("%s: %d atoms", label, len(atoms))
        try:
            result = run_pw(atoms, label, args.nproc, args.kspacing, args.ecut, d3=args.d3,
                            gamma=args.gamma)
        except Exception as exc:  # noqa: BLE001
            log.warning("%s failed: %s", label, exc)
            continue
        record = {"label": label, "state_id": sid, "cif": str(path), "d3": args.d3, **result}
        with (OUT_DIR / "dft" / "dft.jsonl").open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        log.info("%s: E=%.4f eV/atom, gap=%.3f eV, metallic=%s", label, result["energy_per_atom"],
                 result["pbe_gap"], result["metallic"])


if __name__ == "__main__":
    main()
