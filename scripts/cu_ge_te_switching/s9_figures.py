"""Stage 9: figures for the report (static PNGs in ``figures/``).

pressure.png  -- enthalpy above the lowest state vs pressure, GeTe and Cu2GeTe3
heating.png   -- GeTe rhombohedral strain and Peierls distortion vs T (ORB and
                 ORB+D3), and the crystalline Te fraction of every heating run
anneal.png    -- crystalline Te fraction of the glasses held at 600 K
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from common import OUT_DIR

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "legend.frameon": False,
})


def label_end(ax, x, y, text, color):
    ax.annotate(text, (x, y), xytext=(4, 0), textcoords="offset points", va="center",
                fontsize=8, color=INK)


def pressure_figure(out):
    table = pd.read_csv(OUT_DIR / "pressure.csv")
    panels = {
        "GeTe": {"GeTe#0": "R3m (ground state)", "GeTe#rocksalt": "Fm-3m rocksalt",
                 "GeTe#4": "P2$_1$ (domain variant)", "GeTe#9": "P6$_3$mc"},
        "Cu2GeTe3": {"Cu2GeTe3#0": "Imm2, tetrahedral", "Cu2GeTe3#1": "Cmc2$_1$, tetrahedral",
                     "Cu2GeTe3#2": "Imm2, dense", "Cu2GeTe3#3": "Cmcm, dense (CN 8-9)"},
    }
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
    for ax, (formula, states) in zip(axes, panels.items()):
        for k, (sid, name) in enumerate(states.items()):
            rows = table[table["state_id"] == sid].sort_values("P_GPa")
            if rows.empty:
                continue
            ax.plot(rows["P_GPa"], rows["dH_meV"], color=SERIES[k], marker="o", markersize=4, label=name)
        ax.set_title(formula.replace("2", "$_2$").replace("3", "$_3$"), loc="left", color=INK)
        ax.set_xlabel("pressure (GPa)")
        ax.set_ylabel("H - H$_{min}$ (meV/atom, ORB)")
        ax.legend(fontsize=8)
    fig.savefig(out / "pressure.png", dpi=200)


def heating_figure(out):
    heat = pd.read_csv(OUT_DIR / "md_heat.csv")
    heat["T_bin"] = (heat["target_T"] // 25) * 25
    binned = heat.groupby(["run", "T_bin"]).mean(numeric_only=True).reset_index()
    names = {"heat_GeTe_216_s0": "GeTe, ORB", "heat_GeTe_216_s0_d3": "GeTe, ORB+D3",
             "heat_Cu2GeTe3_216_s0_d3": "Cu$_2$GeTe$_3$, ORB+D3"}
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), constrained_layout=True)
    for k, (run, name) in enumerate(names.items()):
        rows = binned[binned["run"] == run]
        if rows.empty:
            continue
        if run.startswith("heat_GeTe"):
            axes[0].plot(rows["T_bin"], rows["angle_dev_deg"], color=SERIES[k], label=name)
            axes[1].plot(rows["T_bin"], rows["peierls_A"], color=SERIES[k], label=name)
        axes[2].plot(rows["T_bin"], rows["te_crystalline"], color=SERIES[k], label=name)
    axes[0].set_ylabel("|cell angle - 90°| (deg)")
    axes[1].set_ylabel("Ge-Te long - short bond (Å)")
    axes[2].set_ylabel("crystalline Te fraction")
    for ax in axes:
        ax.set_xlabel("temperature (K)")
        ax.legend(fontsize=8)
    axes[0].axvline(700, color=MUTED, linewidth=1, linestyle="--")
    axes[0].annotate("experiment ~700 K", (700, axes[0].get_ylim()[1] * 0.92), xytext=(4, 0),
                     textcoords="offset points", fontsize=8, color=MUTED)
    fig.savefig(out / "heating.png", dpi=200)


def anneal_figure(out):
    path = OUT_DIR / "md_anneal.csv"
    if not path.exists():
        return
    anneal = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(5.5, 3.3), constrained_layout=True)
    for k, (run, rows) in enumerate(anneal.groupby("run")):
        name = run.split("_")[1].replace("2", "$_2$").replace("3", "$_3$").replace("4", "$_4$")
        ax.plot(rows["time_ps"], rows["te_crystalline"], color=SERIES[k], label=f"{name}, 600 K")
    ax.set_xlabel("time (ps)")
    ax.set_ylabel("crystalline Te fraction")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8)
    fig.savefig(out / "anneal.png", dpi=200)


def main() -> None:
    out = OUT_DIR / "figures"
    out.mkdir(exist_ok=True)
    pressure_figure(out)
    if (OUT_DIR / "md_heat.csv").exists():
        heating_figure(out)
    anneal_figure(out)


if __name__ == "__main__":
    main()
