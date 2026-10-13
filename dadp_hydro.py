"""
DADP sur une vallee hydraulique : 

Modèle utilisé : cf section 2.2 du rapport

Notez que dans un premier temps l'esperance du gradient est calculée exactement (propagation de la loi du stock),
sans Monte-Carlo parce que c'est plus direct et surtout nous n'avons pas suivi de cours de MC. 
La simulation Monte-Carlo est tout de même assez indispensable pour un tel problème donc nous l'avons utilisée en fin de code pour retrouver
#la politique optimale 
"""
import time
from dataclasses import dataclass

import numpy as np
from numba import njit, prange
from scipy.optimize import minimize



# Description de la vallee

@dataclass
class Valley:
    N: int
    T: int
    X: int
    x0: int
    target: int
    umax: np.ndarray   # (N,) turbinage max
    A: np.ndarray      # (N, K) valeurs d'apports (entiers)
    PA: np.ndarray     # (N, K) probabilites (0 pour le padding)
    na: np.ndarray     # (N,) nombre de valeurs d'apport
    p: np.ndarray      # (T,) prix
    eps: float
    alpha: float
    zmax: np.ndarray   # (N,) borne sur l'eau recue de l'amont


def make_valley(N, T=12, X=10, x0=5, target=5, price_amp=0.4, eps=0.05,
                alpha=0.5, umax_cap=5, inflow_scale=1.0):
    """Vallee academique : turbinage croissant vers l'aval, apports plus forts en amont."""
    umax = np.array([min(2 + i, umax_cap) for i in range(N)], dtype=np.int64)
    K = 4
    A = np.zeros((N, K), dtype=np.int64)
    PA = np.zeros((N, K))
    na = np.zeros(N, dtype=np.int64)
    for i in range(N):
        if i == 0:
            vals, pr = [0, 1, 2, 3], [0.2, 0.3, 0.3, 0.2]
        elif i == 1:
            vals, pr = [0, 1, 2], [0.3, 0.4, 0.3]
        else:
            vals, pr = [0, 1], [0.5, 0.5]
        vals = [int(round(v * inflow_scale)) for v in vals]
        na[i] = len(vals)
        A[i, :len(vals)] = vals
        PA[i, :len(vals)] = pr
    p = 1.0 + price_amp * np.cos(2 * np.pi * np.arange(T) / 12.0)
    # g = u + s <= max(umax, a + z) quand x <= X  ->  borne sur z du barrage suivant
    zmax = np.zeros(N, dtype=np.int64)
    for i in range(1, N):
        amax = A[i - 1, na[i - 1] - 1]
        zmax[i] = max(umax[i - 1], amax + zmax[i - 1])
    return Valley(N, T, X, x0, target, umax, A, PA, na, p, eps, alpha, zmax)



# DADP : sous-problemes locaux (un barrage, état = stock)

@njit(cache=True)
def local_solve(X, T, umax, zmax, A, PA, na, p, eps, alpha, target,
                lam_in, lam_out, x0):
    """Bellman d'un barrage avec prix lam_in (eau recue z) et lam_out (eau sortante g).
    Renvoie V, politiques (u,z) et E[z_t], E[g_t] sous la politique optimale (exact)."""
    V = np.zeros((T + 1, X + 1))
    for x in range(X + 1):
        d = target - x
        V[T, x] = alpha * d * d if d > 0 else 0.0
    pu = np.zeros((T, X + 1, na), dtype=np.int64)
    pz = np.zeros((T, X + 1, na), dtype=np.int64)
    for t in range(T - 1, -1, -1):
        for x in range(X + 1):
            acc = 0.0
            for k in range(na):
                a = A[k]
                best = 1e300
                bu = 0
                bz = 0
                for z in range(zmax + 1):
                    cap = min(umax, x + a + z)
                    for u in range(cap + 1):
                        xs = x - u + a + z
                        s = xs - X if xs > X else 0
                        xn = xs - s
                        g = u + s
                        c = (-p[t] * u + eps * u * u + lam_in[t] * z
                             - lam_out[t] * g + V[t + 1, xn])
                        if c < best:
                            best = c
                            bu = u
                            bz = z
                pu[t, x, k] = bu
                pz[t, x, k] = bz
                acc += PA[k] * best
            V[t, x] = acc
    # propagation exacte de la loi du stock sous la politique optimale
    P = np.zeros(X + 1)
    P[x0] = 1.0
    Ez = np.zeros(T)
    Eg = np.zeros(T)
    for t in range(T):
        Pn = np.zeros(X + 1)
        for x in range(X + 1):
            if P[x] == 0.0:
                continue
            for k in range(na):
                pr = P[x] * PA[k]
                u = pu[t, x, k]
                z = pz[t, x, k]
                xs = x - u + A[k] + z
                s = xs - X if xs > X else 0
                xn = xs - s
                Ez[t] += pr * z
                Eg[t] += pr * (u + s)
                Pn[xn] += pr
        P = Pn
    return V, pu, pz, Ez, Eg


def dual_oracle(v, lam):
   
    N, T, X = v.N, v.T, v.X
    zeros = np.zeros(T)
    theta = 0.0
    Ez = np.zeros((N, T))
    Eg = np.zeros((N, T))
    Vall = np.zeros((N, T + 1, X + 1))
    for i in range(N):
        lam_in = lam[i - 1] if i > 0 else zeros
        lam_out = lam[i] if i < N - 1 else zeros
        V, _, _, ez, eg = local_solve(X, T, int(v.umax[i]), int(v.zmax[i]), v.A[i],
                                      v.PA[i], int(v.na[i]), v.p, v.eps, v.alpha,
                                      v.target, lam_in, lam_out, v.x0)
        theta += V[0, v.x0]
        Vall[i] = V
        Ez[i], Eg[i] = ez, eg
    grad = Ez[1:] - Eg[:-1]
    return theta, grad, Vall


def solve_dadp(v, method="bundle", maxiter=1500, tol=1e-6, rho=0.05, verbose=False):
    """Commentaire sur la théorie derrière : on max theta(lam) (concave, lineaire par morceaux car les controles sont discrets)

    method="bundle" : plans coupants stabilisés (boite de confiance + LP HiGHS)
    method="lbfgs"  : L-BFGS-B comme dans le papier (peut stagner ici : fonction non lisse)
    method="uzawa"  : algorithme d'Uzawa de l'article (eq. 5 / 16 / 11 avec Y = 0) :
                      lam <- lam + rho_k * (E[Z^{i+1}] - E[g^i]),  rho_k = rho / sqrt(k+1)
                      On renvoie le meilleur iterate (donc theta(lam) est une borne inf. pour tout lam)
    """
    from scipy.optimize import linprog
    t0 = time.time()
    N, T = v.N, v.T
    hist = []
    lam0 = np.zeros((N - 1, T))
    if N == 1:
        th, _, Vall = dual_oracle(v, lam0)
        return dict(lam=lam0, theta=th, Vall=Vall, hist=[th], time=time.time() - t0,
                    converged=True)
    d = (N - 1) * T

    if method == "lbfgs":
        def f(l):
            th, gr, _ = dual_oracle(v, l.reshape(N - 1, T))
            hist.append(th)
            return -th, -gr.ravel()
        res = minimize(f, lam0.ravel(), jac=True, method="L-BFGS-B",
                       options=dict(maxiter=maxiter, ftol=1e-13, gtol=1e-9))
        best_lam = res.x.reshape(N - 1, T)
        converged = bool(res.success)
    elif method == "uzawa":
        lam = lam0.copy()
        best_th, best_lam = -np.inf, lam.copy()
        converged = False
        for k in range(maxiter):
            th, g, _ = dual_oracle(v, lam)
            hist.append(th)
            if th > best_th:
                best_th, best_lam = th, lam.copy()
            if np.abs(g).max() < tol:           # contrainte de couplage satisfaite en moyenne
                converged = True
                break
            lam = lam + rho / np.sqrt(k + 1.0) * g     # pas de montee duale
    else:
        # lam est borne : la valeur marginale de l'eau ne depasse pas N * prix max
        Lam = float(N * v.p.max() + 1.0)
        cen = lam0.ravel().copy()
        th_c, g_c, _ = dual_oracle(v, cen.reshape(N - 1, T))
        hist.append(th_c)
        G = [g_c.ravel()]                      # sous-gradients (coupes)
        B = [th_c - g_c.ravel() @ cen]         # coupe : theta(l) <= B_k + G_k . l
        delta = 0.2                            # rayon de la boite de confiance
        converged = False
        c = np.zeros(d + 1)
        c[0] = -1.0
        for it in range(maxiter):
            # LP maitre : max tau  s.t. tau - G_k.l <= B_k,  |l - cen|_inf <= delta
            A_ub = np.hstack([np.ones((len(G), 1)), -np.array(G)])
            lo = np.maximum(cen - delta, -Lam)
            hi = np.minimum(cen + delta, Lam)
            bounds = [(None, None)] + list(zip(lo, hi))
            r = linprog(c, A_ub=A_ub, b_ub=np.array(B), bounds=bounds, method="highs")
            if r.status != 0:
                break
            cand = r.x[1:]
            pred = r.x[0] - th_c               # gain predit par le modele
            if pred < tol * max(1.0, abs(th_c)):
                converged = True
                break                           # le modele ne voit plus d'amelioration
            th, g, _ = dual_oracle(v, cand.reshape(N - 1, T))
            hist.append(th)
            G.append(g.ravel())
            B.append(th - g.ravel() @ cand)
            ratio = (th - th_c) / pred
            if ratio >= 0.1:                    # pas serieux : on deplace le centre
                cen, th_c = cand.copy(), th
                if ratio >= 0.5:
                    delta = min(2 * delta, 2 * Lam)
            elif ratio < 0:                     # la coupe ajoutée corrigera le modele
                delta = max(0.7 * delta, 1e-4)
            if len(G) > 800:                    # on garde la mémoire bornée
                G, B = G[-600:], B[-600:]
        best_lam = cen.reshape(N - 1, T)
    th, _, Vall = dual_oracle(v, best_lam)
    if verbose:
        print(f"  DADP N={N} ({method}): theta={th:.4f}, {len(hist)} evals")
    return dict(lam=best_lam, theta=th, Vall=Vall, hist=hist, time=time.time() - t0,
                converged=converged)



# Politique admissible (eq. 17) et simulation Monte-Carlo 

def sample_noise(v, S, seed):
    """Indices d'apports (S, T, N) : memes scenarios pour tous les algorithmes."""
    rng = np.random.default_rng(seed)
    idx = np.zeros((S, v.T, v.N), dtype=np.int64)
    for i in range(v.N):
        idx[:, :, i] = rng.choice(int(v.na[i]), size=(S, v.T), p=v.PA[i, :v.na[i]])
    return idx


@njit(parallel=True, cache=True)
def simulate_dadp(Vall, nidx, A, X, T, N, umax, zmax, p, eps, alpha, target, x0,
                  Xw, Fw, BUw):
    """Politique : V_inf = somme des V^i ; a chaque pas, probleme a un pas avec la
    contrainte exacte z^{i+1} = u^i + s^i, resolu par DP le long de la chaine
    (exact, lineaire en N, pas d'enumeration des controles conjoints)."""
    S = nidx.shape[0]
    ZM = 0
    for i in range(N):
        ZM = max(ZM, zmax[i])
    costs = np.zeros(S)
    spills = np.zeros(S)
    for sc in prange(S):
        x = Xw[sc]          # vues sur des tableaux de travail propres a ce scenario
        F = Fw[sc]
        BU = BUw[sc]
        F[:, :] = 0.0
        for i in range(N):
            x[i] = x0
        tot = 0.0
        sp = 0.0
        for t in range(T):
            for i in range(N - 1, -1, -1):
                a = A[i, nidx[sc, t, i]]
                for z in range(zmax[i] + 1):
                    best = 1e300
                    bu = 0
                    cap = min(umax[i], x[i] + a + z)
                    for u in range(cap + 1):
                        xs = x[i] - u + a + z
                        s = xs - X if xs > X else 0
                        xn = xs - s
                        c = (-p[t] * u + eps * u * u + Vall[i, t + 1, xn]
                             + F[i + 1, u + s])
                        if c < best:
                            best = c
                            bu = u
                    F[i, z] = best
                    BU[i, z] = bu
            z = 0
            for i in range(N):
                a = A[i, nidx[sc, t, i]]
                u = BU[i, z]
                xs = x[i] - u + a + z
                s = xs - X if xs > X else 0
                tot += -p[t] * u + eps * u * u
                if s > 0:
                    sp += 1.0
                x[i] = xs - s
                z = u + s
        for i in range(N):
            d = target - x[i]
            if d > 0:
                tot += alpha * d * d
        costs[sc] = tot
        spills[sc] = sp
    return costs, spills


def evaluate_dadp_policy(v, Vall, nidx, batch=20000):
    ZM = int(v.zmax.max())
    costs, spills = [], []
    for b in range(0, nidx.shape[0], batch):
        nb = nidx[b:b + batch]
        B = nb.shape[0]
        c, sp = simulate_dadp(Vall, nb, v.A, v.X, v.T, v.N, v.umax, v.zmax, v.p,
                              v.eps, v.alpha, v.target, v.x0,
                              np.zeros((B, v.N), dtype=np.int64),
                              np.zeros((B, v.N + 1, ZM + 2)),
                              np.zeros((B, v.N, ZM + 2), dtype=np.int64))
        costs.append(c)
        spills.append(sp)
    return np.concatenate(costs), np.concatenate(spills)



# DP exacte sur l'état conjoint (reference, exponentielle en N)

def joint_noise_table(v):
    """Table de bruit conjoint : k = sum_i idx_i * prod_{j<i} na_j."""
    nn = int(np.prod(v.na))
    NA = np.zeros((nn, v.N), dtype=np.int64)
    PN = np.ones(nn)
    for k in range(nn):
        kk = k
        for i in range(v.N):
            j = kk % int(v.na[i])
            kk //= int(v.na[i])
            NA[k, i] = v.A[i, j]
            PN[k] *= v.PA[i, j]
    return NA, PN


@njit(parallel=True, cache=True)
def dp_joint(N, T, X, umax, p, eps, alpha, target, NA, PN, Xw):
    nS = (X + 1) ** N
    nn = NA.shape[0]
    ncomb = 1
    for i in range(N):
        ncomb *= umax[i] + 1
    V = np.zeros((T + 1, nS))
    pol = np.zeros((T, nS, nn), dtype=np.int32)
    for s in prange(nS):
        c = 0.0
        pw = 1
        for i in range(N):
            d = target - (s // pw) % (X + 1)
            pw *= (X + 1)
            if d > 0:
                c += alpha * d * d
        V[T, s] = c
    for t in range(T - 1, -1, -1):
        for s in prange(nS):
            x = Xw[s]
            pw = 1
            for i in range(N):
                x[i] = (s // pw) % (X + 1)
                pw *= (X + 1)
            acc = 0.0
            for k in range(nn):
                best = 1e300
                bc = 0
                for c in range(ncomb):
                    cc = c
                    z = 0
                    cost = 0.0
                    idx = 0
                    mult = 1
                    ok = True
                    for i in range(N):
                        u = cc % (umax[i] + 1)
                        cc //= (umax[i] + 1)
                        a = NA[k, i]
                        if u > x[i] + a + z:
                            ok = False
                            break
                        xs = x[i] - u + a + z
                        sp = xs - X if xs > X else 0
                        cost += -p[t] * u + eps * u * u
                        idx += (xs - sp) * mult
                        mult *= (X + 1)
                        z = u + sp
                    if ok:
                        cost += V[t + 1, idx]
                        if cost < best:
                            best = cost
                            bc = c
                acc += PN[k] * best
                pol[t, s, k] = bc
            V[t, s] = acc
    return V, pol


def solve_dp(v):
    t0 = time.time()
    NA, PN = joint_noise_table(v)
    Xw = np.zeros(((v.X + 1) ** v.N, v.N), dtype=np.int64)
    V, pol = dp_joint(v.N, v.T, v.X, v.umax, v.p, v.eps, v.alpha, v.target, NA, PN, Xw)
    s0 = sum(v.x0 * (v.X + 1) ** i for i in range(v.N))
    return dict(V=V, pol=pol, value=V[0, s0], time=time.time() - t0)


@njit(parallel=True, cache=True)
def simulate_dp(pol, nidx, A, na, X, T, N, umax, p, eps, alpha, target, x0, Xw):
    S = nidx.shape[0]
    costs = np.zeros(S)
    for sc in prange(S):
        x = Xw[sc]
        for i in range(N):
            x[i] = x0
        tot = 0.0
        for t in range(T):
            sidx = 0
            mult = 1
            k = 0
            kmult = 1
            for i in range(N):
                sidx += x[i] * mult
                mult *= (X + 1)
                k += nidx[sc, t, i] * kmult
                kmult *= na[i]
            cc = pol[t, sidx, k]
            z = 0
            for i in range(N):
                u = cc % (umax[i] + 1)
                cc //= (umax[i] + 1)
                xs = x[i] - u + A[i, nidx[sc, t, i]] + z
                s = xs - X if xs > X else 0
                tot += -p[t] * u + eps * u * u
                x[i] = xs - s
                z = u + s
        for i in range(N):
            d = target - x[i]
            if d > 0:
                tot += alpha * d * d
        costs[sc] = tot
    return costs


def evaluate_dp_policy(v, dp, nidx):
    Xw = np.zeros((nidx.shape[0], v.N), dtype=np.int64)
    return simulate_dp(dp["pol"], nidx, v.A, v.na, v.X, v.T, v.N, v.umax, v.p,
                       v.eps, v.alpha, v.target, v.x0, Xw)


def mean_ci(c):
    m = c.mean()
    return m, 1.96 * c.std(ddof=1) / np.sqrt(len(c))