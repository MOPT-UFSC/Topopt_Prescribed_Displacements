"""
FD_TEST.py — Validação das sensibilidades adjuntas (Comment 2.1, SAMO-D-26-00705)

Adjunto vs. diferenças finitas centrais, na variável de projeto NÃO filtrada,
atravessando a cadeia completa: filtro de densidades -> projeção Heaviside
(robusta, física no campo erodido) -> RAMP/SIMP -> análise harmônica com
deslocamentos prescritos -> potência de entrada ativa W.

Malha reduzida do Exemplo 1 (LC1(a)); módulos do artigo importados sem
modificação (PLANE4Bolha_PT, FILTER, SOLVE, OBJETIVO). Apenas a fatoração
linear usa um shim scipy (mesma álgebra, ver pypardiso/pardiso_wrapper.py).
"""
import sys, os, json, time
# the solver modules live in the repository root (one level up)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

import numpy as np
import PLANE4Bolha_PT as femp4
import FILTER as filt
import SOLVE as solv
import OBJETIVO as obj

t0 = time.time()
np.set_printoptions(precision=6)

# ------------------------------------------------------------------
# 1. Parâmetros (Exemplo 1 do artigo; malha reduzida p/ FD)
# ------------------------------------------------------------------
E, Emin = 210e9, 210e9 * 1e-7
rho, rhomin = 7850.0, 7850.0 * 1e-8
v_poisson = 0.3
h = 1.0
alpha_ = 0.0
xsi = 0.05                      # amortecimento usado na OTIMIZAÇÃO do Ex.1
p_ramp, q_mass, ps_ramp = 4, 1, 4
THRSHLD = 0
eta_ = (0.5, 0.05)
beta_proj = 8.0                 # continuação intermediária (representativo)

LX, LY = 0.5, 0.7               # domínio do Exemplo 1 (a+b+a = d = 0.5; c = 0.7)
ELEM = 0.01                     # malha reduzida (artigo: 0.0025)
Y_BASE = 0.07                   # faixa passiva inferior (viga horizontal, e = 0.07)
X_COL0, X_COL1 = 0.225, 0.275   # coluna passiva central (viga vertical, b = 0.05)
R_MIN = 10.0 * np.sqrt(2.0) * 0.0025 + 1e-9   # raio FÍSICO do artigo (~0.0354 m)

FREQS = [280.0, 380.0, 1050.0, 1180.0, 1440.0]
UX_PRESC = 1e-6                 # LC1(a): ux = 1e-6, uy = 0 na aresta inferior

# ------------------------------------------------------------------
# 2. Malha estruturada Q4 (mesmo formato do MESH.py: coord (nn,2), connect 1-based ccw)
# ------------------------------------------------------------------
nx, ny = round(LX / ELEM), round(LY / ELEM)
xs = np.linspace(0.0, LX, nx + 1)
ys = np.linspace(0.0, LY, ny + 1)
XX, YY = np.meshgrid(xs, ys, indexing="xy")
coord = np.column_stack([XX.ravel(), YY.ravel()])          # nó k = j*(nx+1)+i
nnode = coord.shape[0]

connect = np.zeros(((nx * ny), 4), dtype=int)
el = 0
for j in range(ny):
    for i in range(nx):
        n1 = j * (nx + 1) + i
        connect[el] = [n1 + 1, n1 + 2, n1 + nx + 2 + 1, n1 + nx + 1 + 1]  # ccw, 1-based
        el += 1
nel = connect.shape[0]
print(f"[malha] {nx}x{ny} = {nel} elementos, {nnode} nos, {2*nnode} gdl", flush=True)

cent = np.zeros((nel, 2))
idx0 = connect - 1
cent[:, 0] = coord[idx0, 0].mean(axis=1)
cent[:, 1] = coord[idx0, 1].mean(axis=1)

# regiões
passive = np.where((cent[:, 1] < Y_BASE) |
                   ((cent[:, 0] > X_COL0) & (cent[:, 0] < X_COL1) & (cent[:, 1] >= Y_BASE)))[0]
active = np.setdiff1d(np.arange(nel), passive)
print(f"[regioes] passivos={len(passive)}  ativos={len(active)}", flush=True)

# ------------------------------------------------------------------
# 3. Condições de contorno LC1(a) + carga estática (ponta, unitária vertical)
# ------------------------------------------------------------------
nodes_base = np.where(np.abs(coord[:, 1]) < 1e-12)[0]
fixed_x = 2 * nodes_base
fixed_y = 2 * nodes_base + 1
fixed_dofs = np.unique(np.concatenate([fixed_x, fixed_y]))
free_dofs = np.setdiff1d(np.arange(2 * nnode), fixed_dofs)

U0 = np.zeros((2 * nnode, 1), dtype=complex)
U0[fixed_x, :] = UX_PRESC
U0[fixed_y, :] = 0.0
Up = U0[fixed_dofs, :].copy()

Lp, Lf = femp4.build_Lp_Lf(2 * nnode, fixed_dofs)
F = np.zeros((2 * nnode, 1), dtype=complex)     # sem forças externas dinâmicas

tip_node = int(np.argmin((coord[:, 0] - 0.25) ** 2 + (coord[:, 1] - LY) ** 2))
F_est = np.zeros((2 * nnode, 1), dtype=complex)
F_est[2 * tip_node + 1, 0] = -1.0               # carga unitária vertical na ponta

# ------------------------------------------------------------------
# 4. Filtro (raio físico do artigo) + campo de projeto heterogêneo determinístico
# ------------------------------------------------------------------
print("[filtro] montando vizinhanças...", flush=True)
neighbors, H, _ = filt.get_neighbors_radius(nel, coord, connect, R_MIN)

mu = 0.3 + 0.25 * np.sin(2 * np.pi * cent[:, 0] / LX) * np.sin(np.pi * cent[:, 1] / LY)
mu = np.clip(mu, 0.05, 0.95).reshape(-1, 1)
mu[passive] = 1.0

# ------------------------------------------------------------------
# 5. Matrizes-base elementares (independentes do projeto) — uma vez
# ------------------------------------------------------------------
print("[assembly] matrizes-base...", flush=True)
ind_rows, ind_cols = femp4.generate_ind_rows_cols(connect)
_ , _ , _ , data_k_, data_m_ = femp4.stif_mass_matrices_0(
    coord, connect, nnode, nel, ind_rows, ind_cols,
    E, Emin, v_poisson, rho, rhomin, h, p_ramp, q_mass, ps_ramp,
    np.ones((nel, 1)), THRSHLD)

def project(mu_unf):
    """cadeia filtro+projecao robusta com sobrescrita dos passivos (como no MAIN)."""
    xE, dE, xI, dI, xD, dD, _ = filt.calc_xnew_robust(nel, H, neighbors, mu_unf, beta_proj, eta_)
    xE[passive] = 1.0
    dE[passive] = 0.0
    return xE, dE

def assemble(xE):
    return femp4.stif_mass_matrices_OPT(
        data_k_, data_m_, nnode, nel, ind_rows, ind_cols,
        p_ramp, q_mass, ps_ramp, xE, E, Emin, rho, rhomin, THRSHLD)

def W_of_U(Ufull, K, M, f):
    """potência de entrada ativa — fórmula idêntica à do OBJETIVO.objetivo."""
    omega = 2 * np.pi * f
    beta_d = xsi * (2.0 / omega)
    C = alpha_ * M + beta_d * K
    Kd = K - (omega ** 2) * M + 1j * omega * C
    r = Lp @ Kd @ Ufull
    F_ = F + r
    return float((0.5 * omega * np.real(1j * Ufull.T @ np.conj(F_)))[0, 0])

def evaluate_W(mu_unf, freqs):
    """pipeline completo: projeto nao filtrado -> W em cada frequencia."""
    xE, _ = project(mu_unf)
    K, M, Ks = assemble(xE)
    Ufr = solv.harmonic_analysys(freqs, K, M, alpha_, xsi, F, Up, free_dofs, fixed_dofs)
    Ws, Ufull_list = [], []
    for i, f in enumerate(freqs):
        Ufull = U0.copy()
        Ufull[free_dofs, 0] = Ufr[:, i]
        Ws.append(W_of_U(Ufull, K, M, f))
        Ufull_list.append(Ufull)
    return np.array(Ws), Ufull_list, (K, M, Ks)

def evaluate_C(mu_unf):
    """compliance estática F^T u (rigidez com RAMP ps)."""
    xE, _ = project(mu_unf)
    _, _, Ks = assemble(xE)
    Ue = np.zeros((2 * nnode, 1), dtype=complex)
    Ue[free_dofs, :] = solv.static_analysys(Ks[free_dofs, :][:, free_dofs], F_est[free_dofs, :])
    return float(np.real(F_est.T @ Ue)[0, 0])

# ------------------------------------------------------------------
# 6. Baseline: estado + adjunto + gradientes adjuntos com cadeia
# ------------------------------------------------------------------
print("[baseline] estado, adjunto e gradientes...", flush=True)
xE_b, dE_b = project(mu)
K_b, M_b, Ks_b = assemble(xE_b)
W_base, Ufull_base, _ = evaluate_W(mu, FREQS)
C_base = evaluate_C(mu)

U_est = np.zeros((2 * nnode, 1), dtype=complex)
U_est[free_dofs, :] = solv.static_analysys(Ks_b[free_dofs, :][:, free_dofs], F_est[free_dofs, :])

grad_mu = {}        # freq -> (nel,1) dPsi/dmu (adjunto, pos-cadeia)
identity = {}       # freq -> (W, xi*omega*Re(U^H K U), rel.diff)
for f in FREQS:
    Ufr = solv.harmonic_analysys([f], K_b, M_b, alpha_, xsi, F, Up, free_dofs, fixed_dofs)
    Uadj_fr = solv.harmonic_adjoint_analysys([f], K_b, M_b, alpha_, xsi, F, Up, free_dofs, fixed_dofs)
    Ufull = U0.copy();  Ufull[free_dofs, 0] = Ufr[:, 0]
    Uadj = np.zeros((2 * nnode, 1), dtype=complex);  Uadj[free_dofs, 0] = Uadj_fr[:, 0]

    W_f, C_f, dW_phys, dC_phys = obj.objetivo(
        [f], alpha_, xsi, data_k_, data_m_, K_b, M_b, Lp, Lf, F, F_est,
        Ufull, Uadj, U_est, connect, nel, p_ramp, q_mass, ps_ramp,
        xE_b, E, Emin, rho, rhomin, THRSHLD)

    g = filt.density_filter_with_dHdx(nel, H, neighbors,
                                      np.asarray(dW_phys).reshape(1, -1), dE_b).reshape(-1, 1)
    grad_mu[f] = g

    omega = 2 * np.pi * f
    Wc = xsi * omega * float(np.real(np.conj(Ufull).T @ (K_b @ Ufull))[0, 0])
    identity[f] = (W_f, Wc, abs(W_f - Wc) / abs(Wc))
    print(f"  f={f:7.1f} Hz  W={W_f:.8e}  identidade rel.diff={identity[f][2]:.2e}", flush=True)

gC_mu = filt.density_filter_with_dHdx(nel, H, neighbors,
                                      np.asarray(dC_phys).reshape(1, -1), dE_b).reshape(-1, 1)

# ------------------------------------------------------------------
# 7. Seleção dos 5 elementos (ao longo da peça, todos no domínio de projeto)
# ------------------------------------------------------------------
targets = [(0.11, 0.10), (0.39, 0.24), (0.11, 0.39), (0.285, 0.55), (0.39, 0.66)]
sel = []
for (tx, ty) in targets:
    d2 = (cent[active, 0] - tx) ** 2 + (cent[active, 1] - ty) ** 2
    sel.append(int(active[np.argmin(d2)]))
print("[elementos] selecionados:")
for k, e_ in enumerate(sel):
    print(f"  E{k+1}: el {e_}  centro=({cent[e_,0]:.3f},{cent[e_,1]:.3f})  mu={mu[e_,0]:.4f}", flush=True)

# ------------------------------------------------------------------
# 8. Varredura do passo (elemento central, 1050 Hz)
# ------------------------------------------------------------------
e_sw, f_sw = sel[2], 1050.0
i_fsw = FREQS.index(f_sw)
g_ref = float(grad_mu[f_sw][e_sw, 0])
deltas = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
sweep_err = []
print(f"[sweep] elemento {e_sw} @ {f_sw:.0f} Hz  (adjunto = {g_ref:.8e})", flush=True)
for dlt in deltas:
    mup = mu.copy(); mup[e_sw, 0] += dlt
    mum = mu.copy(); mum[e_sw, 0] -= dlt
    Wp, _, _ = evaluate_W(mup, [f_sw])
    Wm, _, _ = evaluate_W(mum, [f_sw])
    g_fd = (Wp[0] - Wm[0]) / (2 * dlt)
    err = abs(g_fd - g_ref) / abs(g_ref)
    sweep_err.append(err)
    print(f"  d={dlt:.0e}  FD={g_fd:.8e}  rel.err={err:.3e}", flush=True)

DELTA = deltas[int(np.argmin(sweep_err))]
print(f"[sweep] passo escolhido: {DELTA:.0e}", flush=True)

# ------------------------------------------------------------------
# 9. Tabela 5 elementos x 5 frequências (+ compliance estática)
# ------------------------------------------------------------------
tab_adj = np.zeros((5, len(FREQS)))
tab_fd = np.zeros((5, len(FREQS)))
tab_err = np.zeros((5, len(FREQS)))
stat_adj = np.zeros(5); stat_fd = np.zeros(5); stat_err = np.zeros(5)

for k, e_ in enumerate(sel):
    mup = mu.copy(); mup[e_, 0] += DELTA
    mum = mu.copy(); mum[e_, 0] -= DELTA
    Wp, _, _ = evaluate_W(mup, FREQS)
    Wm, _, _ = evaluate_W(mum, FREQS)
    for j, f in enumerate(FREQS):
        tab_adj[k, j] = grad_mu[f][e_, 0]
        tab_fd[k, j] = (Wp[j] - Wm[j]) / (2 * DELTA)
        tab_err[k, j] = abs(tab_fd[k, j] - tab_adj[k, j]) / abs(tab_fd[k, j])
    Cp = evaluate_C(mup); Cm = evaluate_C(mum)
    stat_adj[k] = gC_mu[e_, 0]
    stat_fd[k] = (Cp - Cm) / (2 * DELTA)
    stat_err[k] = abs(stat_fd[k] - stat_adj[k]) / abs(stat_fd[k])
    print(f"[tabela] E{k+1} concluido  (max rel.err din = {tab_err[k].max():.2e} | est = {stat_err[k]:.2e})", flush=True)

# ------------------------------------------------------------------
# 10. Persistência dos resultados
# ------------------------------------------------------------------
out = {
    "config": {"mesh": [nx, ny], "elem_size": ELEM, "r_min": R_MIN, "beta": beta_proj,
               "eta": list(eta_), "xsi": xsi, "p": p_ramp, "q": q_mass, "ps": ps_ramp,
               "delta_fd": DELTA, "freqs": FREQS, "ux_presc": UX_PRESC,
               "design_field": "mu = clip(0.3 + 0.25 sin(2 pi x/LX) sin(pi y/LY), 0.05, 0.95)"},
    "elements": [{"id": int(e_), "x": float(cent[e_, 0]), "y": float(cent[e_, 1]),
                  "mu": float(mu[e_, 0])} for e_ in sel],
    "W_base": {str(f): W_base[i] for i, f in enumerate(FREQS)},
    "identity_W_vs_quadC": {str(f): {"W": identity[f][0], "xi_omega_UKU": identity[f][1],
                                     "rel_diff": identity[f][2]} for f in FREQS},
    "sweep": {"element": int(e_sw), "freq": f_sw, "deltas": deltas, "rel_err": sweep_err},
    "table": {"adjoint": tab_adj.tolist(), "fd": tab_fd.tolist(), "rel_err": tab_err.tolist()},
    "static": {"C_base": C_base, "adjoint": stat_adj.tolist(), "fd": stat_fd.tolist(),
               "rel_err": stat_err.tolist()},
    "runtime_s": time.time() - t0,
}
with open("FD_validation_results.json", "w") as fjs:
    json.dump(out, fjs, indent=1)

print(f"\n[resumo] max rel.err dinamico (5x5): {tab_err.max():.3e}")
print(f"[resumo] max rel.err estatico:        {stat_err.max():.3e}")
print(f"[resumo] identidade W=(w^2/2)u^H C u: pior rel.diff {max(v[2] for v in identity.values()):.3e}")
print(f"[tempo] {time.time()-t0:.1f} s", flush=True)
