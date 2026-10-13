"""Figures : DP exacte vs DADP et SDDP, avec les deux optimiseurs du dual (plans coupants / Uzawa)
Ce fichier est fait pour être appelé dans experiments.py, il aurait pu être carrément intégré à Experiments.py mais on a préféré le laisser indépendant
NB (Convention graphique) : la couleur designe la methode (bleu = DP exacte,
orange = plans coupants, vert = Uzawa), le STYLE designe la quantité
(trait plein = politique admissible, tirets = borne inferieure)
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Couleurs
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d2"

METHODS = {   # clé de solve_dadp -> (libelle, couleur, marqueur)
    "bundle": ("Cutting planes", ORANGE, "s"),
    "uzawa": ("Uzawa (decreasing step)", AQUA, "^"),
}


# Utilitaires
def read_csv(name):
    with open(name, newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        d = {}
        for k, x in r.items():
            if x in ("True", "False"):
                d[k] = (x == "True")
            else:
                try:
                    d[k] = float(x)
                except ValueError:
                    d[k] = x
        out.append(d)
    return out


def _style(ax, xlabel, ylabel, title):
    ax.set_title(title, loc="left", fontsize=10, color=INK, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9, color=INK2)
    ax.set_ylabel(ylabel, fontsize=9, color=INK2)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8.5, length=3)


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path + ".png", dpi=170)
    fig.savefig(path + ".pdf")
    plt.close(fig)
    print(f"Figure written: {path}.png / {path}.pdf")


def _merge(rowsA, rowsB):
    """Une ligne par N : B (grands N) prioritaire, A complete les N absents de B"""
    allN = {int(r["N"]): r for r in rowsB}
    for r in rowsA:
        allN.setdefault(int(r["N"]), r)
    return allN


# ------------------------------------------------------------------ figures de synthese (6 figures separees)
def summary_figure(rowsA, rowsB, path="fig_summary"):
    """Ecrit six figures separées """
    allN = _merge(rowsA, rowsB)
    Ns = sorted(allN)
    NA = [int(r["N"]) for r in rowsA]
    pct = lambda x, ref: 100 * x / abs(ref)

    def new():
        return plt.subplots(figsize=(6.4, 4.6))

    # 1. cout minimal
    fig, ax = new()
    ax.plot(NA, [r["DP_value"] for r in rowsA], "-o", color=BLUE, lw=2, ms=6,
            label="Exact DP (optimum)")
    for m, (lab, col, mk) in METHODS.items():
        ax.plot(Ns, [allN[n][f"LB_{m}"] for n in Ns], "--" + mk, color=col, lw=1.6, ms=5,
                mfc="white", label=f"{lab}: lower bound")
        ax.errorbar(Ns, [allN[n][f"policy_{m}"] for n in Ns],
                    yerr=[allN[n][f"ci_{m}"] for n in Ns], fmt="-" + mk, color=col, lw=2,
                    ms=6, capsize=2, elinewidth=1, label=f"{lab}: policy")
    _style(ax, "number of dams N", "expected cost (negative = profit)", "Minimal cost")
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    ax.text(0.98, 0.97, "curves almost coincide:\nsee the relative-gap figures",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK2)
    ax.set_xticks(Ns)
    _save(fig, f"{path}_1_cost")

    # 2. ecart a la DP exacte (petits N)
    fig, ax = new()
    for m, (lab, col, mk) in METHODS.items():
        ax.plot(NA, [pct(r[f"policy_{m}"] - r["DP_value"], r["DP_value"]) for r in rowsA],
                "-" + mk, color=col, lw=2, ms=6, label=f"{lab}: policy − DP")
        ax.plot(NA, [pct(r["DP_value"] - r[f"LB_{m}"], r["DP_value"]) for r in rowsA],
                "--" + mk, color=col, lw=1.6, ms=5, mfc="white",
                label=f"{lab}: DP − lower bound")
    ax.axhline(0, color=INK2, lw=0.8)
    _style(ax, "number of dams N", "relative gap to exact DP (%)",
           "Distance to the optimum (N where DP is tractable)")
    ax.legend(fontsize=7.5, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.08))
    ax.set_xticks(NA)
    _save(fig, f"{path}_2_gap_to_DP")

    # 3. certificat : politique vs sa propre borne inferieure, tout N
    fig, ax = new()
    for m, (lab, col, mk) in METHODS.items():
        g = [pct(allN[n][f"policy_{m}"] - allN[n][f"LB_{m}"], allN[n][f"LB_{m}"]) for n in Ns]
        ax.plot(Ns, g, "-" + mk, color=col, lw=2, ms=6, label=lab)
    _style(ax, "number of dams N", "(policy − lower bound) / |lower bound|  (%)",
           "Optimality certificate, available for every N")
    ax.legend(fontsize=8, frameon=False, loc="best")
    ax.set_xticks(Ns)
    _save(fig, f"{path}_3_certificate")

    # 4. finesse des bornes : Uzawa vs plans coupants
    fig, ax = new()
    diff = [pct(allN[n]["LB_uzawa"] - allN[n]["LB_bundle"], allN[n]["LB_bundle"]) for n in Ns]
    ax.bar(range(len(Ns)), diff, color=AQUA, width=0.55)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(range(len(Ns)))
    ax.set_xticklabels(Ns)
    _style(ax, "number of dams N", "gap between lower bounds (%)",
           "Uzawa bound − cutting-plane bound")
    ax.text(0.02, 0.03, "> 0: Uzawa's bound is tighter\n< 0: the cutting-plane bound is tighter",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=8, color=INK2)
    _save(fig, f"{path}_4_bounds")

    # 5. temps de calcul
    fig, ax = new()
    ax.semilogy(NA, [max(r["DP_time_s"], 1e-3) for r in rowsA], "-o", color=BLUE, lw=2,
                ms=6, label="Exact DP")
    for m, (lab, col, mk) in METHODS.items():
        ax.semilogy(Ns, [allN[n][f"time_{m}"] for n in Ns], "-" + mk, color=col, lw=2,
                    ms=6, label=lab)
    _style(ax, "number of dams N", "computation time (s, log scale)",
           "Optimization time")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.grid(axis="y", which="minor", color=GRID, lw=0.3)
    ax.set_xticks(Ns)
    _save(fig, f"{path}_5_time")

    # 6.  Iterations avant convergence
    fig, ax = new()
    for m, (lab, col, mk) in METHODS.items():
        ev = [allN[n][f"evals_{m}"] for n in Ns]
        ax.semilogy(Ns, ev, "-", color=col, lw=2)
        for n, v_ in zip(Ns, ev):
            ok = allN[n].get(f"conv_{m}", True)
            ax.plot(n, v_, mk, ms=7, color=col if ok else "white", mec=col, mew=1.8)
        ax.plot([], [], "-" + mk, color=col, lw=2, ms=6, label=lab)
    ax.plot([], [], "s", color="white", mec=INK2, mew=1.5,
            label="hollow: iteration budget exhausted")
    _style(ax, "number of dams N", "dual oracle evaluations (log scale)",
           "Number of iterations")
    ax.legend(fontsize=8, frameon=False, loc="best")
    ax.set_xticks(Ns)
    _save(fig, f"{path}_6_iterations")


# Convergence + multiplicateurs
def convergence_figure(v, runs, path="fig_convergence_multipliers"):
    """runs : dict methode -> resultat de solve_dadp (cles 'bundle', 'uzawa', 'lbfgs')."""
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axs[0]
    styles = dict(METHODS)
    styles["lbfgs"] = ("L-BFGS (paper)", YELLOW, "o")
    for m, (lab, col, mk) in styles.items():
        if m not in runs:
            continue
        h = np.maximum.accumulate(runs[m]["hist"])
        ax.semilogx(np.arange(1, len(h) + 1), h, "-", color=col, lw=2, label=lab)
    _style(ax, "dual oracle evaluation (log scale)",
           "best lower bound θ(λ)", f"(a) Dual convergence, N = {v.N}")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ymax = max(np.max(runs[m]["hist"]) for m in runs)
    ax.set_ylim(ymax - 0.12 * abs(ymax), ymax + 0.01 * abs(ymax))
    ax.text(0.02, 0.04, "L-BFGS stalls: piecewise-linear\ndual function", transform=ax.transAxes,
            fontsize=8, color=INK2)

    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
    lo = min(runs[m]["lam"].min() for m in ("bundle", "uzawa"))
    hi = max(runs[m]["lam"].max() for m in ("bundle", "uzawa"))
    pad = 0.05 * (hi - lo + 1e-9)
    for ax, m in zip(axs[1:], ("bundle", "uzawa")):
        for j in range(v.N - 1):
            ax.plot(range(v.T), runs[m]["lam"][j], color=cols[j % len(cols)], lw=1.8,
                    label=f"link {j + 1}→{j + 2}")
        _style(ax, "month t", "optimal λₜ",
               f"({'b' if m == 'bundle' else 'c'}) Multipliers: {METHODS[m][0]}")
        ax.set_ylim(lo - pad, hi + 0.30 * (hi - lo) + pad)
        ax.legend(fontsize=7.5, frameon=False, ncol=3, loc="upper center")
    _save(fig, path)


# Distributions de gains
def payoffs_figure(costs, path="fig_payoffs_N3"):
    """costs : dict etiquette -> tableau de couts simules (memes scenarios)."""
    fig, axs = plt.subplots(1, len(costs), figsize=(4.4 * len(costs), 3.6), sharey=True,
                            sharex=True)
    cols = [BLUE, ORANGE, AQUA]
    lo = min(c.min() for c in costs.values())
    hi = max(c.max() for c in costs.values())
    for ax, (lab, c), col in zip(np.atleast_1d(axs), costs.items(), cols):
        ax.hist(c, bins=np.linspace(lo, hi, 60), density=True, color=col, alpha=0.9)
        ax.axvline(c.mean(), color=INK, ls="--", lw=1.2)
        _style(ax, "cost (= −profit)", "density", f"{lab}\nmean {c.mean():.2f}")
    _save(fig, path)


if __name__ == "__main__":
    summary_figure(read_csv("results_A_dp_vs_dadp.csv"), read_csv("results_B_scaling.csv"))