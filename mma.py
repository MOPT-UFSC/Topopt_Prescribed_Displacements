# SPDX-License-Identifier: GPL-3.0-or-later
"""
GCMMA-MMA-Python (consolidated / corrected / reviewed)

This module is a hardened drop-in replacement of the MMA/GCMMA Python translation
you sent (originally based on Krister Svanberg's MATLAB code, via Arjen Deetman).

What was improved (while preserving MATLAB logic):
- Strict column-shape normalization (prevents silent NumPy broadcasting bugs)
- Robust block-KKT assembly using np.block (fixes fragile m<n branch assembly)
- Denominator clamping to avoid NaN/Inf in edge cases (does not change math in healthy runs)
- Consistent float dtype + safe scalar casts
- Preserved function signatures/arguments so you can just replace your mma.py

Functions:
- mmasub
- gcmmasub (kept from your version; lightly hardened)
- subsolv (heavily hardened, core Newton solver)
- kktcheck
- raaupdate
- concheck
- asymp
"""

from __future__ import division

from typing import Tuple
import numpy as np
from scipy.sparse import diags
from scipy.linalg import solve


# =============================================================================
# Helpers (internal)
# =============================================================================
def _col(v, dtype=float) -> np.ndarray:
    """Return v as (k,1) column array (MATLAB-like)."""
    a = np.asarray(v, dtype=dtype)
    if a.ndim == 0:
        return a.reshape(1, 1)
    if a.ndim == 1:
        return a.reshape(-1, 1)
    if a.ndim == 2 and a.shape[1] == 1:
        return a
    if a.ndim == 2 and a.shape[0] == 1:
        return a.T
    return a.reshape(-1, 1)


def _as_float_scalar(x) -> float:
    """Extract python float from numpy scalar/1x1 arrays safely."""
    a = np.asarray(x)
    if a.size == 1:
        return float(a.reshape(-1)[0])
    return float(x)


def _assert_shape(name: str, arr: np.ndarray, shape: Tuple[int, ...]) -> None:
    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")


# =============================================================================
# MMA
# =============================================================================
def mmasub(m: int, n: int, iter: int, xval: np.ndarray, xmin: np.ndarray, xmax: np.ndarray,
           xold1: np.ndarray, xold2: np.ndarray, f0val: float,  df0dx: np.ndarray, fval: np.ndarray,
           dfdx: np.ndarray, low: np.ndarray, upp: np.ndarray, a0: float, a: np.ndarray, c: np.ndarray,
           d: np.ndarray, move: float = 0.5, asyinit: float = 0.01, asydecr: float = 0.65,
           asyincr: float = 1.06, asymin: float = 0.01, asymax: float = 10, raa0: float = 0.00001,
           albefa: float = 0.1, **kwargs) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray,
                                                   np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Solve the MMA subproblem for one outer iteration.
    Signature/returns preserved from your version.
    """

    # ---- Shape/dtype hardening (critical in NumPy) ----
    xval = _col(xval); xmin = _col(xmin); xmax = _col(xmax)
    xold1 = _col(xold1); xold2 = _col(xold2)
    df0dx = _col(df0dx)
    fval = _col(fval)
    low = _col(low); upp = _col(upp)

    a = _col(a); c = _col(c); d = _col(d)
    dfdx = np.asarray(dfdx, dtype=float)
    _assert_shape("dfdx", dfdx, (m, n))

    f0val = _as_float_scalar(f0val)
    a0 = _as_float_scalar(a0)
    raa0 = _as_float_scalar(raa0)

    epsimin = 1e-7
    eeen = np.ones((n, 1), dtype=float)
    eeem = np.ones((m, 1), dtype=float)
    zeron = np.zeros((n, 1), dtype=float)

    # ---- Asymptotes low/upp (same as MATLAB) ----
    if iter <= 2:
        low = xval - asyinit * (xmax - xmin)
        upp = xval + asyinit * (xmax - xmin)
    else:
        zzz = (xval - xold1) * (xold1 - xold2)
        factor = eeen.copy()
        factor[zzz > 0] = asyincr
        factor[zzz < 0] = asydecr
        low = xval - factor * (xold1 - low)
        upp = xval + factor * (upp - xold1)

        lowmin = xval - asymax * (xmax - xmin)
        lowmax = xval - asymin * (xmax - xmin)
        uppmin = xval + asymin * (xmax - xmin)
        uppmax = xval + asymax * (xmax - xmin)

        low = np.maximum(low, lowmin)
        low = np.minimum(low, lowmax)
        upp = np.minimum(upp, uppmax)
        upp = np.maximum(upp, uppmin)

    # ---- alfa/beta bounds (same as MATLAB) ----
    zzz1 = low + albefa * (xval - low)
    zzz2 = xval - move * (xmax - xmin)
    alfa = np.maximum(np.maximum(zzz1, zzz2), xmin)

    zzz1 = upp - albefa * (upp - xval)
    zzz2 = xval + move * (xmax - xmin)
    beta = np.minimum(np.minimum(zzz1, zzz2), xmax)

    # ---- p0, q0, P, Q, b (same as MATLAB; safer numerics) ----
    xmami = xmax - xmin
    xmami = np.maximum(xmami, 1e-5 * eeen)
    xmami_inv = eeen / xmami

    tiny = 1e-14
    ux1 = np.maximum(upp - xval, tiny)
    xl1 = np.maximum(xval - low, tiny)

    ux2 = ux1 * ux1
    xl2 = xl1 * xl1
    ux_inv = eeen / ux1
    xl_inv = eeen / xl1

    p0 = np.maximum(df0dx, 0.0)
    q0 = np.maximum(-df0dx, 0.0)
    pq0 = 0.001 * (p0 + q0) + raa0 * xmami_inv
    p0 = (p0 + pq0) * ux2
    q0 = (q0 + pq0) * xl2

    P = np.maximum(dfdx, 0.0)
    Q = np.maximum(-dfdx, 0.0)
    PQ = 0.001 * (P + Q) + raa0 * (eeem @ xmami_inv.T)
    P = P + PQ
    Q = Q + PQ

    P = (diags(ux2.flatten(), 0).dot(P.T)).T
    Q = (diags(xl2.flatten(), 0).dot(Q.T)).T

    # IMPORTANT: MATLAB convention f <= 0.  Violations -> positive fval.
    b = (P @ ux_inv) + (Q @ xl_inv) - fval

    # ---- Solve subproblem (Newton primal-dual) ----
    xmma, ymma, zmma, lam, xsi, eta, mu, zet, s = subsolv(
        m, n, epsimin, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d
    )

    return xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp


# =============================================================================
# GCMMA (kept; lightly hardened) – you said to disregard, but kept for compatibility
# =============================================================================
def gcmmasub(m: int, n: int, iter: int, epsimin: float, xval: np.ndarray, xmin: np.ndarray,
             xmax: np.ndarray, low: np.ndarray, upp: np.ndarray, raa0: float, raa: np.ndarray,
             f0val: np.ndarray, df0dx: np.ndarray, fval: np.ndarray, dfdx: np.ndarray, a0: float,
             a: np.ndarray, c: np.ndarray, d: np.ndarray, albefa: float = 0.1, **kwargs) -> Tuple[np.ndarray,
            np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:

    # Shape hardening
    xval = _col(xval); xmin = _col(xmin); xmax = _col(xmax)
    low = _col(low); upp = _col(upp)
    df0dx = _col(df0dx); fval = _col(fval)
    a = _col(a); c = _col(c); d = _col(d)
    raa = _col(raa)
    dfdx = np.asarray(dfdx, dtype=float)
    _assert_shape("dfdx", dfdx, (m, n))

    a0 = _as_float_scalar(a0)
    raa0 = _as_float_scalar(raa0)
    epsimin = float(epsimin)

    eeen = np.ones((n, 1))
    eeem = np.ones((m, 1))
    zeron = np.zeros((n, 1))

    # alfa/beta
    alfa = np.maximum(low + albefa * (xval - low), xmin)
    beta = np.minimum(upp - albefa * (upp - xval), xmax)

    # p0,q0,r0,P,Q,r,b
    xmami = xmax - xmin
    xmami = np.maximum(xmami, 1e-5 * eeen)
    xmami_inv = eeen / xmami

    tiny = 1e-14
    ux1 = np.maximum(upp - xval, tiny)
    xl1 = np.maximum(xval - low, tiny)
    ux2 = ux1 * ux1
    xl2 = xl1 * xl1
    ux_inv = eeen / ux1
    xl_inv = eeen / xl1

    p0 = np.maximum(df0dx, 0.0)
    q0 = np.maximum(-df0dx, 0.0)
    pq0 = p0 + q0
    p0 = (p0 + 0.001 * pq0 + raa0 * xmami_inv) * ux2
    q0 = (q0 + 0.001 * pq0 + raa0 * xmami_inv) * xl2

    r0 = f0val - (p0.T @ ux_inv) - (q0.T @ xl_inv)

    P = np.maximum(dfdx, 0.0)
    Q = np.maximum(-dfdx, 0.0)
    PQ = P + Q
    P = P + 0.001 * PQ + (raa @ xmami_inv.T)
    Q = Q + 0.001 * PQ + (raa @ xmami_inv.T)

    P = (diags(ux2.flatten(), 0).dot(P.T)).T
    Q = (diags(xl2.flatten(), 0).dot(Q.T)).T

    r = fval - (P @ ux_inv) - (Q @ xl_inv)
    b = -r

    xmma, ymma, zmma, lam, xsi, eta, mu, zet, s = subsolv(
        m, n, epsimin, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d
    )

    # f0app, fapp
    ux1 = np.maximum(upp - xmma, tiny)
    xl1 = np.maximum(xmma - low, tiny)
    ux_inv = eeen / ux1
    xl_inv = eeen / xl1
    f0app = float(r0 + (p0.T @ ux_inv) + (q0.T @ xl_inv))
    fapp = r + (P @ ux_inv) + (Q @ xl_inv)

    return xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, f0app, fapp


# =============================================================================
# Core Newton primal-dual solver (MMA/GCMMA subproblem)
# =============================================================================
def subsolv(m: int, n: int, epsimin: float, low: np.ndarray, upp: np.ndarray, alfa: np.ndarray,
            beta: np.ndarray, p0: np.ndarray, q0: np.ndarray, P: np.ndarray, Q: np.ndarray,
            a0: float, a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, **kwargs) -> Tuple[np.ndarray,
            np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Primal-dual Newton solver for the MMA/GCMMA convex subproblem.
    Signature preserved.
    """

    # Shape/dtype hardening
    low = _col(low); upp = _col(upp)
    alfa = _col(alfa); beta = _col(beta)
    p0 = _col(p0); q0 = _col(q0)
    a = _col(a); b = _col(b)
    c = _col(c); d = _col(d)

    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)

    a0 = _as_float_scalar(a0)
    epsimin = float(epsimin)

    een = np.ones((n, 1), dtype=float)
    eem = np.ones((m, 1), dtype=float)

    tiny = 1e-14

    # Initialization (as in typical Svanberg code)
    epsi = 1.0
    x = 0.5 * (alfa + beta)
    y = eem.copy()
    z = np.array([[1.0]], dtype=float)
    lam = eem.copy()

    xsi = een / np.maximum(x - alfa, tiny)
    xsi = np.maximum(xsi, een)
    eta = een / np.maximum(beta - x, tiny)
    eta = np.maximum(eta, een)

    mu = np.maximum(eem, 0.5 * c)
    zet = np.array([[1.0]], dtype=float)
    s = eem.copy()

    while epsi > epsimin:
        epsvecn = epsi * een
        epsvecm = epsi * eem

        # Residuals
        ux1 = np.maximum(upp - x, tiny)
        xl1 = np.maximum(x - low, tiny)
        ux2 = ux1 * ux1
        xl2 = xl1 * xl1
        uxinv1 = een / ux1
        xlinv1 = een / xl1

        plam = p0 + (P.T @ lam)
        qlam = q0 + (Q.T @ lam)
        gvec = (P @ uxinv1) + (Q @ xlinv1)
        dpsidx = plam / ux2 - qlam / xl2

        rex = dpsidx - xsi + eta
        rey = c + d * y - mu - lam
        rez = a0 - zet - (a.T @ lam)
        relam = gvec - a * z - y + s - b

        rexsi = xsi * (x - alfa) - epsvecn
        reeta = eta * (beta - x) - epsvecn
        remu = mu * y - epsvecm
        rezet = zet * z - epsi
        res = lam * s - epsvecm

        residu = np.vstack([rex, rey, rez, relam, rexsi, reeta, remu, rezet, res])
        residunorm = float(np.sqrt((residu.T @ residu).item()))
        residumax = float(np.max(np.abs(residu)))

        ittt = 0
        while (residumax > 0.9 * epsi) and (ittt < 200):
            ittt += 1

            ux1 = np.maximum(upp - x, tiny)
            xl1 = np.maximum(x - low, tiny)
            ux2 = ux1 * ux1
            xl2 = xl1 * xl1
            ux3 = ux1 * ux2
            xl3 = xl1 * xl2

            uxinv1 = een / ux1
            xlinv1 = een / xl1
            uxinv2 = een / ux2
            xlinv2 = een / xl2

            plam = p0 + (P.T @ lam)
            qlam = q0 + (Q.T @ lam)

            gvec = (P @ uxinv1) + (Q @ xlinv1)

            # GG(i,j) = P(i,j)/(upp_j-x_j)^2 - Q(i,j)/(x_j-low_j)^2
            GG = (diags(uxinv2.flatten(), 0).dot(P.T)).T - (diags(xlinv2.flatten(), 0).dot(Q.T)).T

            dpsidx = plam / ux2 - qlam / xl2

            delx = dpsidx - epsvecn / np.maximum(x - alfa, tiny) + epsvecn / np.maximum(beta - x, tiny)
            dely = c + d * y - lam - epsvecm / np.maximum(y, tiny)
            delz = a0 - (a.T @ lam) - epsi / np.maximum(z, tiny)
            dellam = gvec - a * z - y - b + epsvecm / np.maximum(lam, tiny)

            diagx = plam / ux3 + qlam / xl3
            diagx = 2.0 * diagx + xsi / np.maximum(x - alfa, tiny) + eta / np.maximum(beta - x, tiny)
            diagx = np.maximum(diagx, tiny)
            diagxinv = een / diagx

            diagy = d + mu / np.maximum(y, tiny)
            diagy = np.maximum(diagy, tiny)
            diagyinv = eem / diagy

            diaglam = s / np.maximum(lam, tiny)
            diaglamyi = diaglam + diagyinv
            diaglamyi = np.maximum(diaglamyi, tiny)

            if m < n:
                blam = dellam + dely / diagy - (GG @ (delx / diagx))
                bb = np.vstack([blam, delz])

                # Alam = diag(diaglamyi) + GG*diag(diagxinv)*GG^T
                Alam = diags(diaglamyi.flatten(), 0).toarray() + (GG * diagxinv.T) @ GG.T

                # Robust block system (MATLAB-equivalent):
                AA = np.block([
                    [Alam,               a],
                    [a.T,   -zet / np.maximum(z, tiny)]
                ])

                solut = solve(AA, bb, assume_a='sym')
                dlam = solut[:m]
                dz = solut[m:m + 1]
                dx = -delx / diagx - (GG.T @ dlam) / diagx
            else:
                diaglamyiinv = eem / diaglamyi
                dellamyi = dellam + dely / diagy

                # Axx = diag(diagx) + GG^T*diag(diaglamyiinv)*GG
                Axx = diags(diagx.flatten(), 0).toarray() + (GG.T * diaglamyiinv.T) @ GG

                azz = zet / np.maximum(z, tiny) + (a.T @ (a * diaglamyiinv))
                axz = -(GG.T @ (a * diaglamyiinv))

                bx = delx + GG.T @ (dellamyi * diaglamyiinv)
                bz = delz - a.T @ (dellamyi * diaglamyiinv)

                AA = np.block([
                    [Axx, axz],
                    [axz.T, azz]
                ])
                bb = np.vstack([-bx, -bz])

                solut = solve(AA, bb, assume_a='sym')
                dx = solut[:n]
                dz = solut[n:n + 1]

                dlam = (GG @ dx) * diaglamyiinv - dz * (a * diaglamyiinv) + dellamyi * diaglamyiinv

            dy = -dely / diagy + dlam / diagy

            dxsi = -xsi + epsvecn / np.maximum(x - alfa, tiny) - (xsi * dx) / np.maximum(x - alfa, tiny)
            deta = -eta + epsvecn / np.maximum(beta - x, tiny) + (eta * dx) / np.maximum(beta - x, tiny)
            dmu = -mu + epsvecm / np.maximum(y, tiny) - (mu * dy) / np.maximum(y, tiny)
            dzet = -zet + epsi / np.maximum(z, tiny) - zet * dz / np.maximum(z, tiny)
            ds = -s + epsvecm / np.maximum(lam, tiny) - (s * dlam) / np.maximum(lam, tiny)

            # Step length
            xx = np.vstack([y, z, lam, xsi, eta, mu, zet, s])
            dxx = np.vstack([dy, dz, dlam, dxsi, deta, dmu, dzet, ds])

            stepxx = -1.01 * dxx / np.maximum(xx, tiny)
            stmxx = float(np.max(stepxx))

            stepalfa = -1.01 * dx / np.maximum(x - alfa, tiny)
            stmalfa = float(np.max(stepalfa))

            stepbeta = 1.01 * dx / np.maximum(beta - x, tiny)
            stmbeta = float(np.max(stepbeta))

            stmalbe = max(stmalfa, stmbeta)
            stmalbexx = max(stmalbe, stmxx)
            stminv = max(stmalbexx, 1.0)
            steg = 1.0 / stminv

            # Backtracking on residual norm
            xold = x.copy(); yold = y.copy(); zold = z.copy()
            lamold = lam.copy(); xsiold = xsi.copy(); etaold = eta.copy()
            muold = mu.copy(); zetold = zet.copy(); sold = s.copy()

            itto = 0
            resinew = 2.0 * residunorm

            while (resinew > residunorm) and (itto < 50):
                itto += 1

                x = xold + steg * dx
                y = yold + steg * dy
                z = zold + steg * dz
                lam = lamold + steg * dlam
                xsi = xsiold + steg * dxsi
                eta = etaold + steg * deta
                mu = muold + steg * dmu
                zet = zetold + steg * dzet
                s = sold + steg * ds

                ux1 = np.maximum(upp - x, tiny)
                xl1 = np.maximum(x - low, tiny)
                ux2 = ux1 * ux1
                xl2 = xl1 * xl1
                uxinv1 = een / ux1
                xlinv1 = een / xl1

                plam = p0 + (P.T @ lam)
                qlam = q0 + (Q.T @ lam)
                gvec = (P @ uxinv1) + (Q @ xlinv1)
                dpsidx = plam / ux2 - qlam / xl2

                rex = dpsidx - xsi + eta
                rey = c + d * y - mu - lam
                rez = a0 - zet - (a.T @ lam)
                relam = gvec - a * z - y + s - b
                rexsi = xsi * (x - alfa) - epsvecn
                reeta = eta * (beta - x) - epsvecn
                remu = mu * y - epsvecm
                rezet = zet * z - epsi
                res = lam * s - epsvecm

                residu = np.vstack([rex, rey, rez, relam, rexsi, reeta, remu, rezet, res])
                resinew = float(np.sqrt((residu.T @ residu).item()))

                steg *= 0.5

            residunorm = resinew
            residumax = float(np.max(np.abs(residu)))
            steg *= 2.0

        epsi *= 0.1

    return x.copy(), y.copy(), z.copy(), lam.copy(), xsi.copy(), eta.copy(), mu.copy(), zet.copy(), s.copy()


# =============================================================================
# KKT check (kept; minor hardening)
# =============================================================================
def kktcheck(m: int, n: int, x: np.ndarray, y: np.ndarray, z: float, lam: np.ndarray, xsi: np.ndarray,
            eta: np.ndarray, mu: np.ndarray, zet: float, s: np.ndarray, xmin: np.ndarray, xmax: np.ndarray,
            df0dx: np.ndarray, fval: np.ndarray, dfdx: np.ndarray, a0: float, a: np.ndarray, c: np.ndarray,
            d: np.ndarray, **kwargs) -> Tuple[np.ndarray, float, float]:

    x = _col(x); y = _col(y); lam = _col(lam); xsi = _col(xsi); eta = _col(eta)
    mu = _col(mu); s = _col(s)
    xmin = _col(xmin); xmax = _col(xmax)
    df0dx = _col(df0dx); fval = _col(fval)
    a = _col(a); c = _col(c); d = _col(d)

    dfdx = np.asarray(dfdx, dtype=float)
    _assert_shape("dfdx", dfdx, (m, n))

    z = _col(z)
    zet = _col(zet)
    a0 = _as_float_scalar(a0)

    rex = df0dx + dfdx.T @ lam - xsi + eta
    rey = c + d * y - mu - lam
    rez = a0 - zet - a.T @ lam
    relam = fval - a * z - y + s
    rexsi = xsi * (x - xmin)
    reeta = eta * (xmax - x)
    remu = mu * y
    rezet = zet * z
    res = lam * s

    residu = np.vstack([rex, rey, rez, relam, rexsi, reeta, remu, rezet, res])
    residunorm = float(np.sqrt((residu.T @ residu).item()))
    residumax = float(np.max(np.abs(residu)))

    return residu, residunorm, residumax


# =============================================================================
# RAA update / conservativeness check / asymptotes (kept; minor hardening)
# =============================================================================
def raaupdate(xmma: np.ndarray, xval: np.ndarray, xmin: np.ndarray, xmax: np.ndarray, low: np.ndarray, upp: np.ndarray,
              f0valnew: np.ndarray, fvalnew: np.ndarray, f0app: np.ndarray, fapp: np.ndarray, raa0: np.ndarray,
              raa: np.ndarray, raa0eps: np.ndarray, raaeps: np.ndarray,  epsimin: float, **kwargs) -> Tuple[np.ndarray, np.ndarray]:

    raacofmin = 1e-12

    xmma = _col(xmma); xval = _col(xval)
    xmin = _col(xmin); xmax = _col(xmax)
    low = _col(low); upp = _col(upp)

    raa0 = _col(raa0)
    raa = _col(raa)
    raa0eps = _col(raa0eps)
    raaeps = _col(raaeps)

    f0valnew = _col(f0valnew)
    fvalnew = _col(fvalnew)
    f0app = _col(f0app)
    fapp = _col(fapp)

    epsimin = float(epsimin)

    eeem = np.ones((raa.size, 1))
    eeen = np.ones((xmma.size, 1))

    xmami = xmax - xmin
    xmami = np.maximum(xmami, 1e-5 * eeen)

    xxux = (xmma - xval) / np.maximum((upp - xmma), 1e-14)
    xxxl = (xmma - xval) / np.maximum((xmma - low), 1e-14)
    xxul = xxux * xxxl
    ulxx = (upp - low) / xmami

    raacof = float((xxul.T @ ulxx).item())
    raacof = max(raacof, raacofmin)

    f0appe = f0app + 0.5 * epsimin

    if np.all(f0valnew > f0appe):
        deltaraa0 = (1.0 / raacof) * (f0valnew - f0app)
        zz0 = 1.1 * (raa0 + deltaraa0)
        zz0 = np.minimum(zz0, 10 * raa0)
        raa0 = zz0

    fappe = fapp + 0.5 * epsimin * eeem
    fdelta = fvalnew - fappe
    deltaraa = (1.0 / raacof) * (fvalnew - fapp)
    zzz = 1.1 * (raa + deltaraa)
    zzz = np.minimum(zzz, 10 * raa)

    idx = np.where(fdelta > 0)[0]
    if idx.size > 0:
        raa[idx, :] = zzz[idx, :]

    return raa0, raa


def concheck(m: int, epsimin: np.ndarray, f0app: np.ndarray, f0valnew: np.ndarray,
             fapp: np.ndarray, fvalnew: np.ndarray, **kwargs) -> int:

    eeem = np.ones((m, 1))
    f0app = _col(f0app)
    f0valnew = _col(f0valnew)
    fapp = _col(fapp)
    fvalnew = _col(fvalnew)

    epsimin = float(np.asarray(epsimin).reshape(-1)[0])

    f0appe = f0app + epsimin
    fappe = fapp + epsimin * eeem

    arr1 = np.concatenate((f0appe.flatten(), fappe.flatten()))
    arr2 = np.concatenate((f0valnew.flatten(), fvalnew.flatten()))

    conserv = 1 if np.all(arr1 >= arr2) else 0
    return conserv


def asymp(outeriter: int, n: int, xval: np.ndarray, xold1: np.ndarray, xold2: np.ndarray, xmin: np.ndarray,
          xmax: np.ndarray, low: np.ndarray, upp: np.ndarray, raa0: float, raa: np.ndarray, raa0eps: float,
          raaeps: float, df0dx: np.ndarray, dfdx: np.ndarray, asyinit: float = 0.5, asydecr: float = 0.7,
          asyincr: float = 1.2, asymin: float = 0.01, asymax: float = 10, **kwargs) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:

    xval = _col(xval); xold1 = _col(xold1); xold2 = _col(xold2)
    xmin = _col(xmin); xmax = _col(xmax)
    low = _col(low); upp = _col(upp)
    df0dx = _col(df0dx)
    dfdx = np.asarray(dfdx, dtype=float)

    eeen = np.ones((n, 1))
    xmami = xmax - xmin
    xmami = np.maximum(xmami, 1e-5 * eeen)

    raa0 = float((np.abs(df0dx).T @ xmami).item())
    raa0 = max(raa0eps, (0.1 / n) * raa0)

    raa = np.abs(dfdx) @ xmami
    raa = np.maximum(raaeps, (0.1 / n) * raa)

    if outeriter <= 2:
        low = xval - asyinit * xmami
        upp = xval + asyinit * xmami
    else:
        xxx = (xval - xold1) * (xold1 - xold2)
        factor = eeen.copy()
        factor[xxx > 0] = asyincr
        factor[xxx < 0] = asydecr
        low = xval - factor * (xold1 - low)
        upp = xval + factor * (upp - xold1)

        lowmin = xval - asymax * xmami
        lowmax = xval - asymin * xmami
        uppmin = xval + asymin * xmami
        uppmax = xval + asymax * xmami

        low = np.maximum(low, lowmin)
        low = np.minimum(low, lowmax)
        upp = np.minimum(upp, uppmax)
        upp = np.maximum(upp, uppmin)

    return low, upp, raa0, raa