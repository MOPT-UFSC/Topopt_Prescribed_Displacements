# import numpy as np
# from scipy.sparse import csr_matrix, diags
# from scipy.spatial import cKDTree

# # ============================================================
# # Cache interno (não mexe no MAIN)
# # ============================================================
# # Chave: (nel, id(neighbors), id(weights), need_norm)
# # Valor: (W_csr, Hi_vector, W_norm_csr)
# _FILTER_CACHE = {}

# # ============================================================
# # Heaviside + Volume-preserving eta (Xu, Cai, Cheng)
# # ============================================================

# # Liga/desliga volume-preserving sem mudar a MAIN:
# # True  -> eta é resolvido por bisseção para preservar volume médio
# # False -> usa eta passado pela MAIN (comportamento clássico)
# VOLUME_PRESERVING = True


# def _to_scalar(x, name="param"):
#     if x is None:
#         return None
#     if np.isscalar(x):
#         return float(x)
#     arr = np.asarray(x).ravel()
#     if arr.size != 1:
#         raise TypeError(f"{name} deve ser escalar ou ter 1 elemento; veio com size={arr.size}.")
#     return float(arr[0])


# def _heaviside(x_tilde, beta, eta):
#     """Heaviside suave (Sigmund/Guest): retorna x_phys"""
#     xt = np.asarray(x_tilde).reshape(-1)
#     den = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
#     if den == 0.0:
#         den = 1.0
#     return (np.tanh(beta * eta) + np.tanh(beta * (xt - eta))) / den


# def _heaviside_deriv(x_tilde, beta, eta):
#     """Derivada d(x_phys)/d(x_tilde)"""
#     xt = np.asarray(x_tilde).reshape(-1)
#     den = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
#     if den == 0.0:
#         den = 1.0

#     z = beta * (xt - eta)
#     # robustez numérica (não muda o resultado prático, só evita overflow)
#     z = np.clip(z, -50.0, 50.0)

#     sech2 = 1.0 / np.cosh(z)**2
#     return (beta * sech2) / den


# def _solve_eta_volume_preserving(x_tilde, beta, v_target, tol=1e-6, maxit=60):
#     """
#     Resolve eta em [0,1] tal que mean( H(x_tilde; beta, eta) ) = v_target.
#     v_target tipicamente = mean(x_tilde).
#     """
#     # garante faixa admissível
#     v_target = float(np.clip(v_target, 0.0, 1.0))

#     lo, hi = 0.0, 1.0

#     # função monotônica decrescente em eta
#     def F(eta):
#         return float(np.mean(_heaviside(x_tilde, beta, eta)) - v_target)

#     Flo = F(lo)
#     Fhi = F(hi)

#     # Se por alguma razão numérica não houver "bracketing",
#     # retorna o melhor extremo (mais próximo).
#     if Flo * Fhi > 0.0:
#         return lo if abs(Flo) < abs(Fhi) else hi

#     for _ in range(maxit):
#         mid = 0.5 * (lo + hi)
#         Fmid = F(mid)
#         if abs(Fmid) < tol:
#             return mid
#         if Flo * Fmid > 0.0:
#             lo = mid
#             Flo = Fmid
#         else:
#             hi = mid
#     return 0.5 * (lo + hi)

# def _build_weight_matrix_cached(nel, neighbors, weights, need_norm=False):
#     """
#     Retorna (W, Hi, W_norm ou None) usando cache para evitar reconstruções.
#     - W: csr_matrix com pesos brutos
#     - Hi: soma por linha de W
#     - W_norm: diags(1/Hi) @ W (se need_norm=True)
#     """
#     key = (nel, int(id(neighbors)), int(id(weights)), bool(need_norm))
#     cached = _FILTER_CACHE.get(key, None)
#     if cached is not None:
#         return cached

#     valid = (neighbors != -1)

#     rows = np.repeat(np.arange(nel, dtype=int), valid.sum(axis=1))
#     cols = neighbors[valid].ravel()
#     data = weights[valid].ravel()

#     W = csr_matrix((data, (rows, cols)), shape=(nel, nel))

#     Hi = np.asarray(W.sum(axis=1)).reshape(-1)
#     Hi[Hi == 0.0] = 1.0

#     if need_norm:
#         W_norm = diags(1.0 / Hi) @ W
#         out = (W, Hi, W_norm)
#     else:
#         out = (W, Hi, None)

#     _FILTER_CACHE[key] = out
#     return out


# # ============================================================
# # Heaviside projection (Guest/Sigmund) + derivada
# # ============================================================
# def _to_scalar(x, name="param"):
#     if x is None:
#         return None
#     if np.isscalar(x):
#         return float(x)
#     arr = np.asarray(x).ravel()
#     if arr.size != 1:
#         raise TypeError(f"{name} deve ser escalar ou ter 1 elemento; veio com size={arr.size}.")
#     return float(arr[0])


# def heaviside_projection(x_tilde, beta=1.0, eta=0.5):
#     xt = np.asarray(x_tilde).reshape(-1)

#     den = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
#     if den == 0.0:
#         den = 1.0

#     x_bar = (np.tanh(beta * eta) + np.tanh(beta * (xt - eta))) / den
#     sech2 = 1.0 / np.cosh(beta * (xt - eta))**2
#     dHdx = (beta * sech2) / den

#     return x_bar, dHdx


# # ============================================================
# # 1) get_neighbors_radius com KDTree (mesma assinatura)
# # ============================================================
# def get_neighbors_radius(nel, coord, connect, radius):
#     """
#     Mesma assinatura e saídas do original:
#       neighbors: (nel, kmax) com índices 0-based e padding -1
#       H:         (nel, kmax) com pesos w = radius - dist
#       centroids: (nel, 2)
#     """
#     idx = connect - 1
#     nno = connect.shape[1]
#     centroids = np.empty((nel, 2), dtype=float)
#     centroids[:, 0] = np.sum(coord[idx, 0], axis=1) / nno
#     centroids[:, 1] = np.sum(coord[idx, 1], axis=1) / nno

#     tree = cKDTree(centroids)
#     neigh_lists = tree.query_ball_point(centroids, r=radius)

#     row = []
#     col = []
#     data = []
#     kmax = 0

#     for el, nn in enumerate(neigh_lists):
#         nn = np.asarray(nn, dtype=int)

#         d = np.linalg.norm(centroids[nn] - centroids[el], axis=1)

#         # mantém d < radius (como você fazia)
#         mask = d < radius
#         nn = nn[mask]
#         d = d[mask]

#         ww = radius - d

#         # ordena por distância (determinístico)
#         if nn.size > 1:
#             order = np.argsort(d)
#             nn = nn[order]
#             ww = ww[order]

#         kmax = max(kmax, nn.size)
#         row.append(np.full(nn.size, el, dtype=int))
#         col.append(nn)
#         data.append(ww)

#     if kmax == 0:
#         neighbors = np.full((nel, 1), -1, dtype=int)
#         H = np.zeros((nel, 1), dtype=float)
#         return neighbors, H, centroids

#     row = np.concatenate(row)
#     col = np.concatenate(col)
#     data = np.concatenate(data)

#     Hsquare = csr_matrix((data, (row, col)), shape=(nel, nel))

#     neighbors = np.full((nel, kmax), -1, dtype=int)
#     H = np.zeros((nel, kmax), dtype=float)

#     indptr = Hsquare.indptr
#     indices = Hsquare.indices
#     values = Hsquare.data

#     for i in range(nel):
#         start, end = indptr[i], indptr[i + 1]
#         js = indices[start:end]
#         ws = values[start:end]
#         k = js.size
#         neighbors[i, :k] = js
#         H[i, :k] = ws

#     return neighbors, H, centroids


# # ============================================================
# # 2) sensitivity_filter (mesma assinatura + beta/eta opcionais)
# #
# # Se beta is None: comportamento antigo.
# # Se beta definido: aplica Heaviside via cadeia:
# #   df/dx = W_norm^T * ( df/dx_phys * dH/dx_tilde )
# #
# # Obs.: Como seu sensitivity_filter já tem xval, dá pra fazer certo aqui.
# # ============================================================
# def sensitivity_filter(nel, df0dx, df0dx_est, dfdx, neighbors, H, xval, beta=None, eta=0.5):
#     x = np.asarray(xval).reshape(-1)
#     f0 = np.asarray(df0dx).reshape(-1)
#     f0_est = np.asarray(df0dx_est).reshape(-1)
#     g = np.asarray(dfdx).reshape(-1)  # MAIN típico: 1 restrição

#     # pesos brutos (W) e soma por linha (Hj)
#     W_mat, Hj, _ = _build_weight_matrix_cached(nel, neighbors, H, need_norm=False)

#     # === Caso antigo (sem Heaviside): exatamente como você tinha ===
#     if beta is None:
#         denom = np.maximum(1e-3, x) * Hj
#         denom[denom == 0.0] = 1.0

#         new_f0 = (W_mat @ (x * f0)) / denom
#         new_f0_est = (W_mat @ (x * f0_est)) / denom
#         new_g = (W_mat @ (x * g)) / denom

#         return new_f0.reshape(-1, 1), new_f0_est.reshape(-1, 1), new_g.reshape(1, -1)

#     # === Com Heaviside (recomendado) ===
#     # 1) x_tilde = (W @ x)/Hj  (equivalente a W_norm @ x)
#     x_tilde = (W_mat @ x) / Hj

#     # 2) derivada da projeção
#     _, dHdx = heaviside_projection(x_tilde, beta=float(beta), eta=float(eta))

#     # 3) sensitividades em relação a x (variável do MMA)
#     #    Usa a mesma forma do seu filtro de sensitividade, mas agora:
#     #    df/dx = (W_mat @ (x * (df/dx_phys * dHdx))) / (x * Hj)
#     #    (mantém sua estrutura "x*df" / "x*Hj", só que com o fator dHdx)
#     f0h = f0 * dHdx
#     f0eh = f0_est * dHdx
#     gh = g * dHdx

#     denom = np.maximum(1e-3, x) * Hj
#     denom[denom == 0.0] = 1.0

#     new_f0 = (W_mat @ (x * f0h)) / denom
#     new_f0_est = (W_mat @ (x * f0eh)) / denom
#     new_g = (W_mat @ (x * gh)) / denom

#     return new_f0.reshape(-1, 1), new_f0_est.reshape(-1, 1), new_g.reshape(1, -1)


# # ============================================================
# # 3) calc_xnew (mesma assinatura + beta/eta opcionais)
# #
# # Se beta is None: retorna x_tilde = W_norm @ x (como antes).
# # Se beta definido: retorna x_phys = Heaviside( x_tilde ).
# # ============================================================
# def calc_xnew(nel, H, neighbors, xval, beta=None, eta=0.5):
#     x_unf = np.asarray(xval).reshape(-1)
#     _, _, W_norm = _build_weight_matrix_cached(nel, neighbors, H, need_norm=True)

#     x_tilde = W_norm @ x_unf

#     if beta is None:
#         return np.asarray(x_tilde).reshape(-1, 1)

#     b = _to_scalar(beta, "beta")
#     eta_in = _to_scalar(eta, "eta")

#     # Volume-preserving: resolve eta para preservar mean(x)
#     if VOLUME_PRESERVING:
#         v_target = float(np.mean(x_tilde))          # volumes iguais
#         eta_eff = _solve_eta_volume_preserving(x_tilde, b, v_target=v_target)
#     else:
#         eta_eff = eta_in
    
#     print('eta_eff = ',eta_eff)

#     x_phys = _heaviside(x_tilde, b, eta_eff)
#     return np.asarray(x_phys).reshape(-1, 1)


# # ============================================================
# # 4) density_filter (mesma assinatura + (xval,beta,eta) opcionais)
# #
# # Se beta is None: comportamento antigo.
# # Se beta definido: aplica cadeia:
# #   df/dx = W_norm^T @ ( df/dx_phys * dHdx )
# #
# # Para isso, precisamos de xval (para computar x_tilde e dHdx).
# # Se beta for dado e xval=None -> gera erro claro.
# # ============================================================
# def density_filter(nel, H1, neighbors, df0dx, df0dx_est, dfdx, xval=None, beta=None, eta=0.5):
#     _, _, W_norm = _build_weight_matrix_cached(nel, neighbors, H1, need_norm=True)

#     f0  = np.real(np.asarray(df0dx).reshape(-1))
#     f0e = np.real(np.asarray(df0dx_est).reshape(-1))

#     if f0.size != nel or f0e.size != nel:
#         raise ValueError(f"df0dx/df0dx_est devem ter tamanho nel={nel}. "
#                          f"Recebido f0={f0.size}, f0e={f0e.size}")

#     dfdx_arr = np.asarray(dfdx)
#     if dfdx_arr.ndim == 1:
#         dfdx_arr = dfdx_arr.reshape(1, -1)
#     elif dfdx_arr.ndim == 2:
#         if dfdx_arr.shape == (nel, 1):
#             dfdx_arr = dfdx_arr.reshape(1, -1)
#     else:
#         raise ValueError(f"dfdx com ndim inesperado: {dfdx_arr.ndim}")

#     if dfdx_arr.shape[1] != nel:
#         raise ValueError(f"dfdx deve ter nel colunas. Recebido shape={dfdx_arr.shape}, nel={nel}")

#     dfdx_arr = np.real(dfdx_arr)
#     m = dfdx_arr.shape[0]

#     # ---------- sem Heaviside ----------
#     if beta is None:
#         new_f0  = W_norm.T @ f0
#         new_f0e = W_norm.T @ f0e

#         new_g = np.zeros((m, nel), dtype=float)
#         for i in range(m):
#             new_g[i, :] = (W_norm.T @ dfdx_arr[i, :]).reshape(-1)

#         return (np.asarray(new_f0).reshape(-1, 1),
#                 np.asarray(new_f0e).reshape(-1, 1),
#                 np.asarray(new_g).reshape(m, -1))

#     # ---------- com Heaviside ----------
#     if xval is None:
#         raise ValueError("density_filter com Heaviside requer xval = x_unf (variável do MMA).")

#     b = _to_scalar(beta, "beta")
#     eta_in = _to_scalar(eta, "eta")

#     x_unf = np.asarray(xval).reshape(-1)
#     if x_unf.size != nel:
#         raise ValueError(f"xval deve ter nel entradas. Recebido {x_unf.size}, nel={nel}")

#     x_tilde = W_norm @ x_unf

#     # usa o MESMO eta_eff do forward (volume-preserving) para consistência
#     if VOLUME_PRESERVING:
#         v_target = float(np.mean(x_tilde))
#         eta_eff = _solve_eta_volume_preserving(x_tilde, b, v_target=v_target)
#     else:
#         eta_eff = eta_in

#     dHdx = _heaviside_deriv(x_tilde, b, eta_eff)  # (nel,)

#     new_f0  = W_norm.T @ (f0  * dHdx)
#     new_f0e = W_norm.T @ (f0e * dHdx)

#     new_g = np.zeros((m, nel), dtype=float)
#     for i in range(m):
#         new_g[i, :] = (W_norm.T @ (dfdx_arr[i, :] * dHdx)).reshape(-1)

#     return (np.asarray(new_f0).reshape(-1, 1),
#             np.asarray(new_f0e).reshape(-1, 1),
#             np.asarray(new_g).reshape(m, -1))

###############################################################################################################

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.spatial import cKDTree

# ============================================================
# Cache interno
# ============================================================
# Chave: (nel, id(neighbors), id(weights), need_norm)
# Valor: (W_csr, Hi_vector, W_norm_csr)
_FILTER_CACHE = {}

def _build_weight_matrix_cached(nel, neighbors, weights, need_norm=False):
    key = (nel, int(id(neighbors)), int(id(weights)), bool(need_norm))
    cached = _FILTER_CACHE.get(key, None)
    if cached is not None:
        return cached

    valid = (neighbors != -1)

    rows = np.repeat(np.arange(nel, dtype=int), valid.sum(axis=1))
    cols = neighbors[valid].ravel()
    data = weights[valid].ravel()

    W = csr_matrix((data, (rows, cols)), shape=(nel, nel))

    Hi = np.asarray(W.sum(axis=1)).reshape(-1)
    Hi[Hi == 0.0] = 1.0

    if need_norm:
        W_norm = diags(1.0 / Hi) @ W
        out = (W, Hi, W_norm)
    else:
        out = (W, Hi, None)

    _FILTER_CACHE[key] = out
    return out


# ============================================================
# Utilitários robustos
# ============================================================
def _to_scalar(x, name="param"):
    if x is None:
        return None
    if np.isscalar(x):
        return float(x)
    arr = np.asarray(x).ravel()
    if arr.size != 1:
        raise TypeError(f"{name} deve ser escalar ou ter 1 elemento; veio com size={arr.size}.")
    return float(arr[0])


def _parse_eta(eta):
    """
    Aceita:
      eta = 0.5                    -> (eta0=0.5, deta=None)   (Heaviside normal)
      eta = (0.5, 0.05) ou [..]    -> (eta0=0.5, deta=0.05)   (robust E/I/D disponível)
    """
    if eta is None:
        return 0.5, None

    if np.isscalar(eta):
        return float(eta), None

    arr = np.asarray(eta).ravel()
    if arr.size == 1:
        return float(arr[0]), None
    if arr.size == 2:
        return float(arr[0]), float(arr[1])

    raise TypeError("eta deve ser escalar ou ter 2 elementos (eta0, deta).")


def heaviside_projection(x_tilde, beta=0.1, eta=0.5, clip_z=100.0):
    """
    Projeção Heaviside suave + derivada.
    x_tilde: (nel,) ou (nel,1)
    Retorna:
      x_phys (nel,)
      dHdx   (nel,)
    """
    b = _to_scalar(beta, "beta")
    e = _to_scalar(eta, "eta")

    xt = np.asarray(x_tilde).reshape(-1)

    den = np.tanh(b * e) + np.tanh(b * (1.0 - e))
    if den == 0.0:
        den = 1.0

    z = b * (xt - e)
    # evita overflow de cosh em beta altos
    z = np.clip(z, -clip_z, clip_z)

    x_phys = (np.tanh(b * e) + np.tanh(z)) / den

    sech2 = 1.0 / np.cosh(z)**2
    dHdx = (b * sech2) / den

    return x_phys, dHdx


def robust_heaviside_triplet(x_tilde, beta, eta0=0.5, deta=0.05):
    """
    Retorna (xE, dE), (xI, dI), (xD, dD) com:
      etaE = eta0 + deta  (erode)
      etaI = eta0         (intermediate)
      etaD = eta0 - deta  (dilate)
    """
    b = _to_scalar(beta, "beta")
    eta0 = _to_scalar(eta0, "eta0")
    deta = _to_scalar(deta, "deta")

    xt = np.asarray(x_tilde).reshape(-1)

    xE, dE = heaviside_projection(xt, beta=b, eta=eta0 + deta)
    xI, dI = heaviside_projection(xt, beta=b, eta=eta0)
    xD, dD = heaviside_projection(xt, beta=b, eta=eta0 - deta)

    return (xE, dE), (xI, dI), (xD, dD)


# ============================================================
# 1) get_neighbors_radius (mesma assinatura)
# ============================================================
def get_neighbors_radius(nel, coord, connect, radius):
    idx = connect - 1
    nno = connect.shape[1]
    centroids = np.empty((nel, 2), dtype=float)
    centroids[:, 0] = np.sum(coord[idx, 0], axis=1) / nno
    centroids[:, 1] = np.sum(coord[idx, 1], axis=1) / nno

    tree = cKDTree(centroids)
    neigh_lists = tree.query_ball_point(centroids, r=radius)

    row = []
    col = []
    data = []
    kmax = 0

    for el, nn in enumerate(neigh_lists):
        nn = np.asarray(nn, dtype=int)
        d = np.linalg.norm(centroids[nn] - centroids[el], axis=1)

        # mantém d < radius (igual ao seu)
        mask = d < radius + 1e-8
        nn = nn[mask]
        d = d[mask]

        ww = radius - d

        # ordena por distância (determinístico)
        if nn.size > 1:
            dx = centroids[nn, 0] - centroids[el, 0]
            dy = centroids[nn, 1] - centroids[el, 1]
            order = np.lexsort((dy, dx, d))
            nn = nn[order]
            ww = ww[order]

        kmax = max(kmax, nn.size)
        row.append(np.full(nn.size, el, dtype=int))
        col.append(nn)
        data.append(ww)

    if kmax == 0:
        neighbors = np.full((nel, 1), -1, dtype=int)
        H = np.zeros((nel, 1), dtype=float)
        return neighbors, H, centroids

    row = np.concatenate(row)
    col = np.concatenate(col)
    data = np.concatenate(data)

    Hsquare = csr_matrix((data, (row, col)), shape=(nel, nel))

    neighbors = np.full((nel, kmax), -1, dtype=int)
    H = np.zeros((nel, kmax), dtype=float)

    indptr = Hsquare.indptr
    indices = Hsquare.indices
    values = Hsquare.data

    for i in range(nel):
        start, end = indptr[i], indptr[i + 1]
        js = indices[start:end]
        ws = values[start:end]
        k = js.size
        neighbors[i, :k] = js
        H[i, :k] = ws

    return neighbors, H, centroids


# ============================================================
# 2) sensitivity_filter (mesma assinatura original)
#    OBS: Eu deixo como você tinha (sem Heaviside aqui).
# ============================================================
def sensitivity_filter(nel, df0dx, df0dx_est, dfdx, neighbors, H, xval):
    x = np.asarray(xval).reshape(-1)
    f0 = np.asarray(df0dx).reshape(-1)
    f0_est = np.asarray(df0dx_est).reshape(-1)
    g = np.asarray(dfdx).reshape(-1)  # seu uso típico (1 restrição)

    W_mat, Hj, _ = _build_weight_matrix_cached(nel, neighbors, H, need_norm=False)

    denom = np.maximum(1e-3, x) * Hj
    denom[denom == 0.0] = 1.0

    new_f0 = (W_mat @ (x * f0)) / denom
    new_f0_est = (W_mat @ (x * f0_est)) / denom
    new_g = (W_mat @ (x * g)) / denom

    return new_f0.reshape(-1, 1), new_f0_est.reshape(-1, 1), new_g.reshape(1, -1)


# ============================================================
# 3) calc_xnew (mesma assinatura, com beta/eta opcionais)
#    - Se beta is None: retorna x_tilde = W_norm @ x_unf
#    - Se beta definido: retorna x_phys = Heaviside(x_tilde, beta, eta0)
#    - Se eta for (eta0, deta): por padrão usa eta0 (campo "intermediate")
#      (robust completo use calc_xnew_robust)
# ============================================================
def calc_xnew(nel, H, neighbors, xval, beta=None, eta=0.5):
    x_unf = np.asarray(xval).reshape(-1)

    _, _, W_norm = _build_weight_matrix_cached(nel, neighbors, H, need_norm=True)
    x_tilde = W_norm @ x_unf

    if beta is None:
        return np.asarray(x_tilde).reshape(-1, 1)

    eta0, deta = _parse_eta(eta)  # deta ignorado aqui
    x_phys, _ = heaviside_projection(x_tilde, beta=beta, eta=eta0)
    return np.asarray(x_phys).reshape(-1, 1)


# ============================================================
# Extras: calc_xnew_robust (E/I/D) - para Opção C
# ============================================================
def calc_xnew_robust(nel, H, neighbors, xval, beta, eta=0.5):
    """
    eta pode ser:
      eta=0.5           -> usa deta default 0.05
      eta=(eta0,deta)
    Retorna:
      xE,dE,xI,dI,xD,dD,x_tilde  (todos como (nel,1))
    """
    x_unf = np.asarray(xval).reshape(-1)
    _, _, W_norm = _build_weight_matrix_cached(nel, neighbors, H, need_norm=True)
    x_tilde = W_norm @ x_unf

    eta0, deta = _parse_eta(eta)
    if deta is None:
        deta = 0.05

    (xE, dE), (xI, dI), (xD, dD) = robust_heaviside_triplet(x_tilde, beta=beta, eta0=eta0, deta=deta)

    return (xE.reshape(-1,1), dE.reshape(-1,1),
            xI.reshape(-1,1), dI.reshape(-1,1),
            xD.reshape(-1,1), dD.reshape(-1,1),
            x_tilde.reshape(-1,1))


# ============================================================
# Extra: filtrar gradientes com um dHdx já escolhido (E/I/D)
# ============================================================
def density_filter_with_dHdx(nel, H1, neighbors, grad_phys, dHdx):
    """
    grad_phys:
      - (nel,) ou (nel,1)  -> retorna (nel,1)
      - (m,nel)            -> retorna (m,nel)
    dHdx: (nel,) ou (nel,1)
    """
    _, _, W_norm = _build_weight_matrix_cached(nel, neighbors, H1, need_norm=True)

    d = np.asarray(dHdx).reshape(-1)
    G = np.asarray(grad_phys)

    if G.ndim == 1:
        g = np.real(G.reshape(-1)) * d
        return (W_norm.T @ g).reshape(-1,1)

    if G.ndim == 2:
        if G.shape[1] != nel:
            raise ValueError(f"grad_phys deve ter nel colunas. Recebido {G.shape}, nel={nel}")
        m = G.shape[0]
        out = np.zeros((m, nel), dtype=float)
        for i in range(m):
            gi = np.real(G[i, :]) * d
            out[i, :] = (W_norm.T @ gi).reshape(-1)
        return out

    raise ValueError("grad_phys ndim inválido.")


# ============================================================
# 4) density_filter (compatível com sua MAIN) + Heaviside (normal)
#    - Sem Heaviside: W_norm.T @ grad
#    - Com Heaviside: W_norm.T @ (grad * dHdx) com dHdx de eta0
# ============================================================
def density_filter(nel, H1, neighbors, df0dx, df0dx_est, dfdx, xval=None, beta=None, eta=0.5):
    """
    Mantém sua assinatura de uso:

      df0dx_din, df0dx_est, dfdx = density_filter(nel, H, neighbors, df0dx_din, df0dx_est, dfdx, xval_unf, bebeta, eta_)

    Observações:
    - Se beta != None, xval DEVE ser x_unf (variável do MMA).
    - eta pode ser escalar; se for (eta0,deta), aqui usa eta0 (intermediate).
      Para robust completo (E/D), use calc_xnew_robust + density_filter_with_dHdx.
    """
    _, _, W_norm = _build_weight_matrix_cached(nel, neighbors, H1, need_norm=True)

    f0  = np.real(np.asarray(df0dx).reshape(-1))
    f0e = np.real(np.asarray(df0dx_est).reshape(-1))

    if f0.size != nel or f0e.size != nel:
        raise ValueError(f"df0dx/df0dx_est devem ter tamanho nel={nel}. Recebido f0={f0.size}, f0e={f0e.size}")

    dfdx_arr = np.asarray(dfdx)
    if dfdx_arr.ndim == 1:
        dfdx_arr = dfdx_arr.reshape(1, -1)
    elif dfdx_arr.ndim == 2:
        if dfdx_arr.shape == (nel, 1):
            dfdx_arr = dfdx_arr.reshape(1, -1)
    else:
        raise ValueError(f"dfdx com ndim inesperado: {dfdx_arr.ndim}")

    if dfdx_arr.shape[1] != nel:
        raise ValueError(f"dfdx deve ter nel colunas. Recebido shape={dfdx_arr.shape}, nel={nel}")

    m = dfdx_arr.shape[0]

    # -------------------------
    # sem Heaviside
    # -------------------------
    if beta is None:
        new_f0  = W_norm.T @ f0
        new_f0e = W_norm.T @ f0e

        new_g = np.zeros((m, nel), dtype=float)
        for i in range(m):
            new_g[i, :] = (W_norm.T @ np.real(dfdx_arr[i, :])).reshape(-1)

        return (np.asarray(new_f0).reshape(-1, 1),
                np.asarray(new_f0e).reshape(-1, 1),
                np.asarray(new_g).reshape(m, -1))

    # -------------------------
    # com Heaviside (normal)
    # -------------------------
    if xval is None:
        raise ValueError("density_filter com Heaviside requer xval = x_unf (variável do MMA).")

    x_unf = np.asarray(xval).reshape(-1)
    if x_unf.size != nel:
        raise ValueError(f"xval deve ter nel entradas. Recebido {x_unf.size}, nel={nel}")

    eta0, deta = _parse_eta(eta)  # deta ignorado aqui

    x_tilde = W_norm @ x_unf
    _, dHdx = heaviside_projection(x_tilde, beta=beta, eta=eta0)  # (nel,)

    new_f0  = W_norm.T @ (f0  * dHdx)
    new_f0e = W_norm.T @ (f0e * dHdx)

    new_g = np.zeros((m, nel), dtype=float)
    for i in range(m):
        gi = np.real(dfdx_arr[i, :]) * dHdx
        new_g[i, :] = (W_norm.T @ gi).reshape(-1)

    return (np.asarray(new_f0).reshape(-1, 1),
            np.asarray(new_f0e).reshape(-1, 1),
            np.asarray(new_g).reshape(m, -1))




