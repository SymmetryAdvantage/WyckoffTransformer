"""Rows removed for a calculation-level defect, each with its evidence.

Every entry was first flagged by being > 0.5 eV/atom below all five MLIPs of
LeMat-Bulk-MLIP-Hull-All, but none is removed for that reason: the cause listed is a
property of the DFT record itself. Flagged rows without such a cause are kept
(see UNEXPLAINED).
"""

_CR_GHOST = ("spurious Cr-centred electron population: Alexandria's Bader charges put {pop} e on Cr "
             "against 10.3-11.7 e in 24 normal Alexandria Cr fluorides/oxides, i.e. {exc} excess "
             "electrons; the energy deficit ({deficit} eV/cell) is {epe} eV per excess electron, the "
             "same 4.0-4.8 eV quantum as in the other affected Cr rows, which a variational PBE+U "
             "calculation with a physical PAW setup cannot produce (consistent with a PAW ghost state)")

CAUSES = {
    "mp-1518724": (
        "collapsed relaxation: MP's only task for this material went 209 -> 150 -> 40 -> 142 A^3 in "
        "4 of 99 ionic steps (step 2 at -20,101 kbar, 1.25 A contacts); the final energy (-106.96 eV) "
        "is 34.5 eV below step 1 at a volume within 5%, the final SCF state has -0.4 muB on every O, "
        "and MP's parser flags 'Volume change > 20%'; Alexandria's identical structure "
        "(agm005277521) is 3.40 eV/atom higher"),
    "agm004911156": _CR_GHOST.format(pop="16.52", exc="6.1", deficit="25.9", epe="4.28"),
    "agm005102383": _CR_GHOST.format(pop="16.40", exc="11.8", deficit="47.6", epe="4.02")
        + "; present from the first step of its relaxation",
    "agm005277902": _CR_GHOST.format(pop="16.58", exc="12.2", deficit="53.5", epe="4.38")
        + "; present from the first step of its relaxation",
    "agm005289887": _CR_GHOST.format(pop="22.94", exc="11.7", deficit="49.6", epe="4.25")
        + "; Alexandria's own first relaxation run ends at -6.190 eV/atom, the restart at the same "
          "lattice drops to -6.86; MP's identical structure (mp-770668) is 0.67 eV/atom higher",
    "agm004676761": _CR_GHOST.format(pop="22.65", exc="12.2", deficit="54.3", epe="4.46")
        + "; present throughout its recorded relaxation (steps 15-33)",
    "agm004890438": _CR_GHOST.format(pop="16.35", exc="5.9", deficit="27.6", epe="4.69"),
    "agm004968779": _CR_GHOST.format(pop="16.92", exc="11.3", deficit="52.6", epe="4.65")
        + "; within one relaxation the energy jumps from -12.65 to -10.28 eV/atom (~11 electrons' "
          "worth), i.e. the run hops between discrete spurious states",
    "agm004969345": _CR_GHOST.format(pop="16.75", exc="5.5", deficit="26.3", epe="4.78")
        + "; within one relaxation the energy jumps from -12.00 to -9.68 eV/atom",
    "agm002116455": _CR_GHOST.format(pop="13.78", exc="10.1 (4 Cr)", deficit="52.8", epe="5.24"),
    "agm003235847": _CR_GHOST.format(pop="12.58", exc="4.0 (3 Cr)", deficit="25.4", epe="6.43")
        + "; MP's identical structure (mp-772506) is 0.99 eV/atom higher",
    "agm001439530": ("overlapping atoms: Bi-Cu at 1.086 A (0.37 x the sum of atomic radii); PAW "
                     "augmentation spheres overlap, so the energy is not a valid PBE energy (Alexandria's "
                     "Bader count of 23.4 e on Cu reflects the same overlap)"),
    "oqmd-7966826": ("overlapping atoms: Fe-Mg at 0.897 A (0.31 x the sum of atomic radii); PAW spheres "
                     "overlap; the MLIPs return +41 to +261 eV for this cell; duplicate of oqmd-5708135"),
    "oqmd-5708135": ("overlapping atoms: Fe-Mg at 0.897 A (0.31 x the sum of atomic radii); PAW spheres "
                     "overlap; duplicate of oqmd-7966826"),
    "oqmd-1922538": ("energy belongs to another structure: total energy -24.0129 eV is identical to "
                     "oqmd-2011591 (same 5-atom composition) whose cell is 43% smaller "
                     "(10.79 vs 18.81 A^3/atom); the MLIPs agree with oqmd-2011591's structure"),
    "oqmd-2002394": ("energy belongs to another structure: total energy -30.794 eV is identical to "
                     "oqmd-2826002, a different CoC3 structure (C-C 1.43 vs 1.77 A), with which the "
                     "MLIPs agree (-0.04 eV/atom) while they put this one 1.15 eV/atom higher"),
    "oqmd-3005551": ("not reproducible: the identical crystal (volume within 0.3%) is computed by MP "
                     "(mp-865098), Alexandria (agm002711640) and three more OQMD entries, all 2.13 eV/atom "
                     "higher; this record also carries 1.38 eV/A residual forces and 177 kbar"),
    "oqmd-3245677": ("not reproducible: three other OQMD calculations of the identical structure "
                     "(oqmd-5104427, oqmd-3119544, oqmd-7093706) are 0.73 eV/atom higher"),
}

#: Flagged by the MLIP gap, but no calculation-level defect was found: kept.
UNEXPLAINED = {
    "oqmd-3911617": "OQMD's own entry 1109963 (P2_1) carries the same value; residual forces 2.7 eV/A",
    "oqmd-5269036": "OQMD's own entry 1345850 (Fddd) carries the same value; residual forces 11.3 eV/A, 655 kbar",
    "oqmd-2982764": "no identical structure among OQMD's current AlCoIr entries",
    "agm001685190": "no trajectory, sibling or charge anomaly found",
}

#: Non-standard POTCARs on MP-deprecated materials with no valid MP replacement, and their
#: measured offset against the standard POTCAR on same-structure pairs (eV per atom of that element).
POTCAR_OFFSETS = {
    "PAW_PBE Mg_pv 13Apr2007": (0.097, 116),
    "PAW_PBE Ge_d 03Jul2007": (0.107, 22),
}
