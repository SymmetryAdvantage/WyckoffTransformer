"""
Publication-quality plotting for the Cu-Ge-Te convex hull and pseudobinary cut.
Overwrites artifacts/cu_ge_te_hull/cu_ge_te_hull.png with a clean, two-panel figure.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.patches import Polygon
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Composition

# Set style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.titlesize": 16,
    "figure.dpi": 300,
})

def to_ternary(x_cu, x_ge, x_te):
    """Transform ternary fractions (sum=1) to 2D Cartesian coords in equilateral triangle."""
    # Cu at (0, 0), Ge at (1, 0), Te at (0.5, sqrt(3)/2)
    x = x_ge + 0.5 * x_te
    y = (np.sqrt(3) / 2.0) * x_te
    return x, y

def comp_to_xy(comp):
    c = Composition(comp)
    total = c.get("Cu", 0) + c.get("Ge", 0) + c.get("Te", 0)
    x_cu = c.get("Cu", 0) / total
    x_ge = c.get("Ge", 0) / total
    x_te = c.get("Te", 0) / total
    return to_ternary(x_cu, x_ge, x_te)

def main():
    csv_path = "artifacts/cu_ge_te_hull/cu_ge_te_hull_entries.csv"
    df = pd.read_csv(csv_path)

    # Reconstruct PhaseDiagram to get facets
    entries = []
    for _, r in df.iterrows():
        c = Composition(r["reduced_formula"])
        entries.append(PDEntry(c, r["energy_per_atom"] * c.num_atoms, name=r["name"]))
    pd_hull = PhaseDiagram(entries)

    # Create figure
    fig = plt.figure(figsize=(18, 8.5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 1.0], wspace=0.25)

    # =========================================================================
    # Panel A: Ternary Phase Diagram
    # =========================================================================
    ax1 = fig.add_subplot(gs[0])
    ax1.set_aspect("equal")

    h = np.sqrt(3) / 2.0
    triangle = np.array([[0, 0], [1, 0], [0.5, h]])
    poly = Polygon(triangle, closed=True, facecolor="#fbfbfd", edgecolor="#1b2a4a", lw=2.5, zorder=1)
    ax1.add_patch(poly)

    # Gridlines
    for f in [0.2, 0.4, 0.6, 0.8]:
        # Const Te lines (horizontal)
        x_left, y_te = to_ternary(1.0 - f, 0, f)
        x_right, _ = to_ternary(0, 1.0 - f, f)
        ax1.plot([x_left, x_right], [y_te, y_te], color="#d0d5dd", ls="--", lw=0.8, zorder=2)

        # Const Ge lines
        p1 = to_ternary(1.0 - f, f, 0)
        p2 = to_ternary(0, f, 1.0 - f)
        ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#d0d5dd", ls="--", lw=0.8, zorder=2)

        # Const Cu lines
        p1 = to_ternary(f, 1.0 - f, 0)
        p2 = to_ternary(f, 0, 1.0 - f)
        ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#d0d5dd", ls="--", lw=0.8, zorder=2)

    # Plot Tie Lines (Facets)
    for facet in pd_hull.facets:
        pts = []
        for idx in facet:
            e = pd_hull.qhull_entries[idx]
            x, y = comp_to_xy(e.composition)
            pts.append([x, y])
        pts.append(pts[0])
        pts = np.array(pts)
        ax1.plot(pts[:, 0], pts[:, 1], color="#1b2a4a", lw=1.8, ls="-", zorder=4)

    # Plot Metastable entries (e_above_hull <= 100 meV)
    meta_df = df[(df["e_above_hull"] <= 0.100) & (df["e_above_hull"] > 1e-4)].copy()
    meta_x, meta_y, meta_eh = [], [], []
    for _, r in meta_df.iterrows():
        x, y = comp_to_xy(r["reduced_formula"])
        meta_x.append(x)
        meta_y.append(y)
        meta_eh.append(r["e_above_hull"] * 1000.0) # meV

    sc = ax1.scatter(
        meta_x, meta_y, c=meta_eh, cmap="viridis_r",
        s=16, alpha=0.55, edgecolors="none", zorder=3, vmin=0, vmax=100
    )
    cbar = plt.colorbar(sc, ax=ax1, fraction=0.035, pad=0.04)
    cbar.set_label(r"$E_{\mathrm{above\ hull}}$ (meV/atom)", fontsize=12)

    # Stable Vertices
    stable_positions = {
        "Cu": (0.0, 0.0),
        "Ge": (1.0, 0.0),
        "Te": (0.5, h),
    }
    for e in pd_hull.stable_entries:
        f = e.composition.reduced_formula
        stable_positions[f] = comp_to_xy(e.composition)

    # Plot Stable Points
    for f, (x, y) in stable_positions.items():
        if f in ["Cu", "Ge", "Te"]:
            ax1.scatter(x, y, s=160, marker="o", facecolor="#ffffff", edgecolor="#1b2a4a", lw=2.5, zorder=6)
        else:
            ax1.scatter(x, y, s=200, marker="*", facecolor="#ff3366", edgecolor="#1b2a4a", lw=1.5, zorder=6)

    # Annotate Stable Vertices with clean offsets
    label_offsets = {
        "Cu": (-0.06, -0.04),
        "Ge": (0.02, -0.04),
        "Te": (0.0, 0.03),
        "Cu5Ge": (-0.02, -0.045),
        "Cu3Te2": (-0.11, 0.01),
        "CuTe": (-0.09, 0.03),
        "GeTe": (0.03, 0.02),
        "Cu3GeTe4": (0.03, -0.03),
    }
    for f, (x, y) in stable_positions.items():
        dx, dy = label_offsets.get(f, (0.02, 0.02))
        ha = "center" if "Te" == f or "Cu5Ge" == f else ("right" if dx < 0 else "left")
        ax1.text(
            x + dx,
            y + dy,
            rf"$\mathbf{{{f}}}$",
            fontsize=11,
            fontweight="bold",
            color="#0f172a",
            ha=ha,
            va="center",
            zorder=7,
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "#ffffff",
                "alpha": 0.85,
                "edgecolor": "none",
            },
        )

    # Highlight Cu5Ge2Te7 (the Chem. Mater. 2026 material!)
    x_paper, y_paper = comp_to_xy("Cu5Ge2Te7")
    ax1.scatter(
        x_paper,
        y_paper,
        s=260,
        marker="D",
        facecolor="#00e5ff",
        edgecolor="#0f172a",
        lw=2.0,
        zorder=8,
        label=r"$\mathrm{Cu_5Ge_2Te_7}$ (Dutta et al. 2026)",
    )
    ax1.annotate(
        r"$\mathbf{Cu_5Ge_2Te_7}$"
        + "\n"
        + r"(Dutta et al. 2026)"
        + "\n"
        + r"$e_{\mathrm{hull}} = 89.7\ \mathrm{meV/at}$",
        xy=(x_paper, y_paper),
        xytext=(x_paper - 0.22, y_paper + 0.12),
        arrowprops={
            "facecolor": "#0f172a",
            "edgecolor": "#0f172a",
            "arrowstyle": "->",
            "lw": 1.8,
        },
        fontsize=10,
        fontweight="semibold",
        color="#0f172a",
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "#e0f7fa",
            "edgecolor": "#00bcd4",
            "lw": 1.5,
        },
        zorder=9,
    )

    # Highlight Cu2GeTe3 (near hull)
    x_213, y_213 = comp_to_xy("Cu2GeTe3")
    ax1.scatter(
        x_213,
        y_213,
        s=180,
        marker="^",
        facecolor="#ffea00",
        edgecolor="#0f172a",
        lw=1.8,
        zorder=8,
    )
    ax1.annotate(
        r"$\mathbf{Cu_2GeTe_3}$"
        + "\n"
        + r"$e_{\mathrm{hull}} = 15.1\ \mathrm{meV/at}$",
        xy=(x_213, y_213),
        xytext=(x_213 + 0.08, y_213 + 0.09),
        arrowprops={
            "facecolor": "#0f172a",
            "edgecolor": "#0f172a",
            "arrowstyle": "->",
            "lw": 1.5,
        },
        fontsize=9.5,
        fontweight="semibold",
        color="#0f172a",
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "#fffde7",
            "edgecolor": "#fbc02d",
            "lw": 1.2,
        },
        zorder=9,
    )

    ax1.set_xlim(-0.15, 1.15)
    ax1.set_ylim(-0.08, h + 0.10)
    ax1.axis("off")
    ax1.set_title(r"$\mathbf{(a)}$ Cu–Ge–Te 0 K Convex Hull & Metastable Landscape", pad=15, loc="left", fontweight="bold")

    # Legend for Panel A
    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#ffffff", markeredgecolor="#1b2a4a", markersize=10, lw=2, label="Elemental Reference"),
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="#ff3366", markeredgecolor="#1b2a4a", markersize=14, label="Stable Phase (SUN, Hull Vertex)"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="#00e5ff", markeredgecolor="#0f172a", markersize=11, label=r"$\mathrm{Cu_5Ge_2Te_7}$ (Chem. Mater. 2026)"),
        plt.Line2D([0], [0], color="#1b2a4a", lw=2, label="Convex Hull Tie Line"),
    ]
    ax1.legend(handles=legend_elements, loc="upper right", frameon=True, framealpha=0.9, fontsize=9.5)

    # =========================================================================
    # Panel B: Pseudo-binary Join (CuTe)_x (GeTe)_(1-x) at 50% Te
    # =========================================================================
    ax2 = fig.add_subplot(gs[1])

    # Along the CuTe - GeTe join:
    # x = 0 is GeTe, x = 1 is CuTe
    # Cu3GeTe4 is x = 0.75
    # Cu5Ge2Te7 is x = 5/7 = 0.7143
    # Cu2GeTe3 is x = 2/3 = 0.6667
    # CuGeTe2 is x = 0.50

    join_phases = [
        {"formula": "GeTe", "x": 0.0, "ef": -0.1260, "eh": 0.0, "stable": True, "label": r"$\mathrm{GeTe}$"},
        {"formula": "CuGeTe2", "x": 0.50, "ef": -0.0794, "eh": 0.0473, "stable": False, "label": r"$\mathrm{CuGeTe_2}$"},
        {"formula": "Cu2GeTe3", "x": 2.0/3.0, "ef": -0.1119, "eh": 0.0151, "stable": False, "label": r"$\mathrm{Cu_2GeTe_3}$"},
        {"formula": "Cu5Ge2Te7", "x": 5.0/7.0, "ef": -0.0373, "eh": 0.0897, "stable": False, "label": r"$\mathbf{Cu_5Ge_2Te_7}$" + "\n(Dutta 2026)"},
        {"formula": "Cu3GeTe4", "x": 0.75, "ef": -0.1271, "eh": 0.0, "stable": True, "label": r"$\mathrm{Cu_3GeTe_4}$"},
        {"formula": "CuTe", "x": 1.0, "ef": -0.0792, "eh": 0.0, "stable": True, "label": r"$\mathrm{CuTe}$"},
    ]

    # Convex hull line along join
    hull_x = [0.0, 0.75, 1.0]
    hull_ef = [-0.1260, -0.1271, -0.0792]
    ax2.plot(hull_x, hull_ef, color="#1b2a4a", lw=2.5, ls="-", zorder=3, label="0 K Ground-State Hull")

    # Plot all phases along the join
    for p in join_phases:
        x, ef, _ = p["x"], p["ef"], p["eh"]
        if p["stable"]:
            ax2.scatter(x, ef, s=200, marker="*", facecolor="#ff3366", edgecolor="#1b2a4a", lw=1.5, zorder=5)
        elif p["formula"] == "Cu5Ge2Te7":
            ax2.scatter(x, ef, s=240, marker="D", facecolor="#00e5ff", edgecolor="#0f172a", lw=2.0, zorder=6)
            # Draw vertical distance line
            hull_at_x = -0.1260 + (p["x"] / 0.75) * (-0.1271 - (-0.1260))
            ax2.plot([x, x], [hull_at_x, ef], color="#00bcd4", lw=2.0, ls="--", zorder=4)
            ax2.text(x + 0.02, (hull_at_x + ef) / 2.0, r"$\Delta E_{\mathrm{hull}} = 89.7\ \mathrm{meV/at}$",
                     fontsize=9.5, fontweight="bold", color="#00838f", ha="left", va="center")
        else:
            ax2.scatter(x, ef, s=120, marker="o", facecolor="#fbc02d", edgecolor="#1b2a4a", lw=1.5, zorder=5)
            hull_at_x = -0.1260 + (p["x"] / 0.75) * (-0.1271 - (-0.1260))
            ax2.plot([x, x], [hull_at_x, ef], color="#fbc02d", lw=1.2, ls=":", zorder=4)

        # Label position
        y_offset = 0.008 if ef > -0.10 else -0.012
        if p["formula"] == "Cu5Ge2Te7":
            y_offset = 0.012
        ax2.text(x, ef + y_offset, p["label"], fontsize=10, ha="center", va="bottom" if y_offset > 0 else "top",
                 fontweight="bold" if p["formula"] == "Cu5Ge2Te7" else "normal")

    ax2.set_xlabel(r"Composition along Join: $x$ in $(\mathrm{CuTe})_x(\mathrm{GeTe})_{1-x}$", fontsize=12)
    ax2.set_ylabel(r"Formation Energy $E_f$ (eV/atom)", fontsize=12)
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-0.160, 0.005)
    ax2.grid(True, ls=":", alpha=0.6, color="#94a3b8")
    ax2.set_title(r"$\mathbf{(b)}$ Pseudo-Binary Cut $(\mathrm{CuTe})_x(\mathrm{GeTe})_{1-x}$ (50% Te Join)", pad=15, loc="left", fontweight="bold")

    # Note annotation
    note_txt = (
        r"$\mathbf{Thermodynamic\ Context:}$" + "\n"
        r"$\bullet\ \mathrm{Cu_3GeTe_4}$ and $\mathrm{GeTe}$ define the ground-state envelope." + "\n"
        r"$\bullet\ \mathrm{Cu_5Ge_2Te_7}$ lies $89.7\ \mathrm{meV/atom}$ above the hull," + "\n"
        r"  explaining why non-equilibrium Direct Joule Synthesis" + "\n"
        r"  (DJS) + rapid quenching was required (Dutta et al. 2026)."
    )
    ax2.text(
        0.03,
        0.96,
        note_txt,
        transform=ax2.transAxes,
        fontsize=9.5,
        verticalalignment="top",
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "#f8fafc",
            "edgecolor": "#cbd5e1",
            "lw": 1.2,
        },
    )

    out_png = "artifacts/cu_ge_te_hull/cu_ge_te_hull.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print("Successfully generated publication-quality figure at", out_png)

if __name__ == "__main__":
    main()
