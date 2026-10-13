"""Tests de coherence pour DADP (typiquement, on vérifie que la borne inf est bien plus petite que la politique optimale et que la valeur DP exacte)"""
import numpy as np
from dadp_hydro import (make_valley, dual_oracle, solve_dadp, solve_dp, sample_noise,
                        evaluate_dadp_policy, evaluate_dp_policy, mean_ci)


def test_zmax_bound():
    """g = u + s <= zmax du barrage suivant, pour tout (x, a, z, u) admissible."""
    v = make_valley(8)
    for i in range(v.N - 1):
        for x in range(v.X + 1):
            for k in range(v.na[i]):
                for z in range(v.zmax[i] + 1):
                    for u in range(min(v.umax[i], x + v.A[i, k] + z) + 1):
                        xs = x - u + v.A[i, k] + z
                        s = max(0, xs - v.X)
                        assert u + s <= v.zmax[i + 1]
    print("OK  borne zmax")


def test_gradient():
    """Le gradient analytique (signe compris) correspond aux differences finies."""
    v = make_valley(3)
    rng = np.random.default_rng(0)
    lam = rng.uniform(0.2, 1.5, size=(v.N - 1, v.T))
    th, gr, _ = dual_oracle(v, lam)
    h = 1e-6
    err = 0.0
    for j in range(v.N - 1):
        for t in range(v.T):
            l2 = lam.copy()
            l2[j, t] += h
            th2, _, _ = dual_oracle(v, l2)
            err = max(err, abs((th2 - th) / h - gr[j, t]))
    print(f"OK  gradient (ecart max aux differences finies : {err:.2e})")
    assert err < 1e-4


def test_chain(N, S=100_000):
    v = make_valley(N)
    dp = solve_dp(v)
    da = solve_dadp(v)
    nidx = sample_noise(v, S, seed=1)
    c_dadp, _ = evaluate_dadp_policy(v, da["Vall"], nidx)
    c_dp = evaluate_dp_policy(v, dp, nidx)
    m1, e1 = mean_ci(c_dadp)
    m2, e2 = mean_ci(c_dp)
    print(f"N={N}: borne inf DADP={da['theta']:.3f} | DP={dp['value']:.3f} "
          f"(MC politique DP={m2:.3f}+-{e2:.3f}) | politique DADP={m1:.3f}+-{e1:.3f}")
    assert da["theta"] <= dp["value"] + 1e-6, "borne inf > DP : bug"
    assert abs(m2 - dp["value"]) < 3 * e2 + 1e-9, "simulation incoherente avec la DP"
    assert m1 >= dp["value"] - 3 * e1, "politique admissible meilleure que l'optimum : bug"
    if N == 1:
        assert abs(da["theta"] - dp["value"]) < 1e-9, "N=1 : DADP doit egaler la DP"
    print(f"    ecart politique DADP / DP : {100 * (m1 - dp['value']) / abs(dp['value']):.2f} %")


def test_uzawa(N=2):
    """Uzawa (pas decroissant) : borne inf valide et proche de celle des plans coupants"""
    v = make_valley(N)
    dp = solve_dp(v)
    uz = solve_dadp(v, method="uzawa", maxiter=5000)
    bd = solve_dadp(v, method="bundle")
    print(f"N={N}: Uzawa={uz['theta']:.3f} | plans coupants={bd['theta']:.3f} | DP={dp['value']:.3f}")
    assert uz["theta"] <= dp["value"] + 1e-6, "borne inf Uzawa > DP : bug"
    assert uz["theta"] <= bd["theta"] + 1e-6 and bd["theta"] - uz["theta"] < 0.01 * abs(bd["theta"])


if __name__ == "__main__":
    test_zmax_bound()
    test_gradient()
    for N in (1, 2, 3):
        test_chain(N)
    test_uzawa(2)
    test_uzawa(3)
    print("Tous les tests passent")