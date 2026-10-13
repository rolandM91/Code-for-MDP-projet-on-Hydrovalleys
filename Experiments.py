"""
Compare la DP exacte avec deux optimiseurs du dual : plans coupants et Uzawa
--> tableaux (affiches + results_*.csv) et figures (fig_*.png / .pdf) avec toutes les infos dont on a besoin
"""
import sys
import csv
import numpy as np

from dadp_hydro import (make_valley, solve_dadp, solve_dp, sample_noise,
                        evaluate_dadp_policy, evaluate_dp_policy, mean_ci)
from plots import summary_figure, convergence_figure, payoffs_figure, METHODS

QUICK = "--quick" in sys.argv
S_SMALL = 20_000 if QUICK else 100_000     # scenarios de simu
S_BIG = 5_000 if QUICK else 20_000
N_DP = (1, 2, 3) if QUICK else (1, 2, 3, 4)          # DP exacte (N=4 : ~30 s)
N_SCAL = (2, 4, 6) if QUICK else (2, 4, 6, 8, 10, 12)   # N=12 : ~2 min (cutting plane)

# parametres des deux optimiseurs du dual
SOLVER_KW = {"bundle": {}, "uzawa": dict(maxiter=5000, rho=0.05)}


def save_csv(name, rows):
    with open(name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def run_dadp(v, nidx):
    """On résout avec chaque optimiseur et on simule la politique admissible associée."""
    out = {}
    for m in METHODS:
        r = solve_dadp(v, method=m, **SOLVER_KW[m])
        c, spills = evaluate_dadp_policy(v, r["Vall"], nidx)
        mean, ci = mean_ci(c)
        out[m] = dict(res=r, costs=c, spills=spills, mean=mean, ci=ci)
    return out


def method_columns(out):
    row = {}
    for m, o in out.items():
        r = o["res"]
        row.update({f"LB_{m}": r["theta"], f"policy_{m}": o["mean"], f"ci_{m}": o["ci"],
                    f"time_{m}": r["time"], f"evals_{m}": len(r["hist"]),
                    f"conv_{m}": bool(r["converged"])})
    return row


def show(row):
    print({k: (round(float(x), 3) if isinstance(x, (float, np.floating)) else x)
           for k, x in row.items()})


# Echauffement 
for _N in (1, 2):
    _v = make_valley(_N)
    solve_dp(_v)
    for _m in METHODS:
        _r = solve_dadp(_v, method=_m, **SOLVER_KW[_m])
    evaluate_dadp_policy(_v, _r["Vall"], sample_noise(_v, 10, 0))

#  DP vs DADP
print("\n A. DP exacte vs DADP (plans coupants et Uzawa) ")
rowsA = []
for N in N_DP:
    v = make_valley(N)
    dp = solve_dp(v)
    nidx = sample_noise(v, S_SMALL, seed=1)
    out = run_dadp(v, nidx)
    row = dict(N=N, DP_value=dp["value"], DP_time_s=dp["time"])
    row.update(method_columns(out))
    for m, o in out.items():
        row[f"gap_policy_vs_DP_pct_{m}"] = 100 * (o["mean"] - dp["value"]) / abs(dp["value"])
        row[f"gap_LB_vs_DP_pct_{m}"] = 100 * (dp["value"] - o["res"]["theta"]) / abs(dp["value"])
    rowsA.append(row)
    show(row)
save_csv("results_A_dp_vs_dadp.csv", rowsA)

# Distributions de gains a N=3 (comme la Fig. 7 du papier en gros)
v = make_valley(3)
dp = solve_dp(v)
nidx = sample_noise(v, S_SMALL, seed=2)
out = run_dadp(v, nidx)
payoffs_figure({"Exact DP": evaluate_dp_policy(v, dp, nidx),
                "DADP – cutting planes": out["bundle"]["costs"],
                "DADP – Uzawa": out["uzawa"]["costs"]})

# Passage a l'echelle
print("\n B. Passage a l'echelle en N (DADP seul, deux optimiseurs) ")
rowsB = []
for N in N_SCAL:
    v = make_valley(N)
    nidx = sample_noise(v, S_BIG, seed=3)
    out = run_dadp(v, nidx)
    row = dict(N=N)
    row.update(method_columns(out))
    rowsB.append(row)
    show(row)
save_csv("results_B_scaling.csv", rowsB)

summary_figure(rowsA, rowsB)      # figure de synthese (voir plots.py)

#Convergence et multiplicateurs
print("\n C. Convergence duale et multiplicateurs (N=6) ")
v = make_valley(6)
runs = {}
for method, kw in (("bundle", {}), ("uzawa", SOLVER_KW["uzawa"]), ("lbfgs", {})):
    runs[method] = solve_dadp(v, method=method, **kw)
    r = runs[method]
    print(f"  {method:7s}: theta = {r['theta']:.4f} en {len(r['hist'])} evaluations, {r['time']:.2f} s")
convergence_figure(v, runs, "fig_convergence_multipliers_N6")

# Sensibilite (N=3, plans coupants)
print("\n D. Sensibilite aux parametres (N=3) ")
rowsD = []
sweeps = {"price_amp": [0.0, 0.2, 0.4, 0.8], "eps": [0.0, 0.05, 0.2],
          "alpha": [0.0, 0.5, 2.0], "inflow_scale": [1.0, 2.0]}
for name, vals in sweeps.items():
    for val in vals:
        v = make_valley(3, **{name: val})
        dp, da = solve_dp(v), solve_dadp(v)
        nidx = sample_noise(v, S_SMALL // 2, seed=4)
        c_da, _ = evaluate_dadp_policy(v, da["Vall"], nidx)
        m, e = mean_ci(c_da)
        row = dict(param=name, value=val, DP=dp["value"], DADP_LB=da["theta"],
                   DADP_policy=m, ci95=e,
                   gap_policy_pct=100 * (m - dp["value"]) / abs(dp["value"]),
                   gap_LB_pct=100 * (dp["value"] - da["theta"]) / abs(dp["value"]))
        rowsD.append(row)
        print(f"  {name}={val}: DP={dp['value']:.2f}  LB={da['theta']:.2f}  "
              f"politique={m:.2f}  (ecarts : {row['gap_policy_pct']:.2f} % / "
              f"{row['gap_LB_pct']:.2f} %)")
save_csv("results_D_sensitivity.csv", rowsD)

# Optimiseurs du dual (tableau)
print("\n E. Optimiseurs du dual : Uzawa vs plans coupants vs L-BFGS ")
rowsE = []
for N in (2, 3, 4, 6):
    v = make_valley(N)
    for method, kw in (("uzawa", SOLVER_KW["uzawa"]), ("bundle", {}), ("lbfgs", {})):
        r = solve_dadp(v, method=method, **kw)
        rowsE.append(dict(N=N, method=method, theta=r["theta"], n_evals=len(r["hist"]),
                          time_s=r["time"]))
        print(f"  N={N} {method:7s}: borne inf = {r['theta']:.3f}  "
              f"({len(r['hist'])} evaluations, {r['time']:.2f} s)")
save_csv("results_E_dual_optimizers.csv", rowsE)
print("\nTermine. Fichiers : results_*.csv, fig_*.png / fig_*.pdf")