import numpy as np

def split_mesh_three_parts(node_coords, quad_elements, ly, fractions=(0.25, 0.5, 0.25)):
    """
    Divide a malha em 3 partes horizontais (de baixo pra cima),
    de acordo com frações normalizadas da altura total ly.

    Parâmetros
    ----------
    node_coords : (Nn, 3) array
        Coordenadas dos nós (como retornado pelo gmsh).
    quad_elements : (Ne, 4) array (int)
        Conectividade dos elementos quadrangulares (tags de nós, 1-based).
    ly : float
        Altura total do domínio em y.
    fractions : tupla/list com 3 floats
        Frações da altura: (f1, f2, f3). Se não somarem 1,
        serão normalizadas automaticamente.

    Retorna
    -------
    elems_part1, nodes_part1,
    elems_part2, nodes_part2,
    elems_part3, nodes_part3
    
    Onde:
    - elems_partX : índices (0-based) das linhas de quad_elements
    - nodes_partX : índices (0-based) das linhas de node_coords
    """

    fractions = np.array(fractions, dtype=float)
    if fractions.size != 3:
        raise ValueError("fractions deve ter exatamente 3 valores (f1, f2, f3).")

    # Normaliza se não somar 1
    s = fractions.sum()
    if not np.isclose(s, 1.0):
        fractions = fractions / s
        print(f"Aviso: frações normalizadas internamente para somar 1. "
              f"Novos valores: {fractions}")

    f1, f2, f3 = fractions

    # Limites em y (de baixo pra cima)
    y0 = 0.0
    y1 = f1 * ly
    y2 = (f1 + f2) * ly
    y3 = ly

    # Listas de índices de elementos para cada parte (0-based nas linhas de quad_elements)
    elems_part1 = []
    elems_part2 = []
    elems_part3 = []

    # Pequena tolerância numérica
    tol = 1e-6

    # Loop nos elementos
    for e_idx in range(quad_elements.shape[0]):
        # Tags dos nós do elemento (1-based)
        elem_nodes_tags = quad_elements[e_idx, :]
        # Converte para índices 0-based para acessar node_coords
        elem_nodes_idx = elem_nodes_tags - 1

        # Coordenadas y dos 4 nós
        ys = node_coords[elem_nodes_idx, 1]
        y_mean = ys.mean()

        # Classificação pela posição do centro (y_mean)
        if y_mean <= y1 + tol:
            elems_part1.append(e_idx)
        elif y_mean <= y2 + tol:
            elems_part2.append(e_idx)
        else:
            elems_part3.append(e_idx)

    elems_part1 = np.array(elems_part1, dtype=int)
    elems_part2 = np.array(elems_part2, dtype=int)
    elems_part3 = np.array(elems_part3, dtype=int)

    # Nós usados em cada parte (tags de nó -> índices 0-based)
    if elems_part1.size > 0:
        nodes_tags_part1 = np.unique(quad_elements[elems_part1, :])
        nodes_part1 = nodes_tags_part1 - 1
    else:
        nodes_part1 = np.array([], dtype=int)

    if elems_part2.size > 0:
        nodes_tags_part2 = np.unique(quad_elements[elems_part2, :])
        nodes_part2 = nodes_tags_part2 - 1
    else:
        nodes_part2 = np.array([], dtype=int)

    if elems_part3.size > 0:
        nodes_tags_part3 = np.unique(quad_elements[elems_part3, :])
        nodes_part3 = nodes_tags_part3 - 1
    else:
        nodes_part3 = np.array([], dtype=int)

    return elems_part1, nodes_part1, elems_part2, nodes_part2, elems_part3, nodes_part3

def split_mesh_three_parts_xy(node_coords, quad_elements, lx, ly,
                              fractions_y=(0.25, 0.5, 0.25),
                              fractions_x=(0.25, 0.5, 0.25)):
    """
    Divide a malha em 3 faixas horizontais (Y) e 3 faixas verticais (X),
    retornando elementos e nós de cada faixa.

    node_coords: (Nn,3) com coordenadas
    quad_elements: (Ne,4) com tags de nós (1-based)
    lx, ly: dimensões do retângulo
    fractions_y: 3 frações da altura (somam 1 ou serão normalizadas)
    fractions_x: 3 frações do comprimento (somam 1 ou serão normalizadas)

    Retorna um dicionário com:
      out["Y"][k]["elems"], out["Y"][k]["nodes"]  para k=0,1,2 (de baixo pra cima)
      out["X"][k]["elems"], out["X"][k]["nodes"]  para k=0,1,2 (da esquerda pra direita)

    Observação: nós de interfaces aparecem em duas faixas, naturalmente.
    """

    def _normalize(fracs, name):
        fracs = np.array(fracs, dtype=float)
        if fracs.size != 3:
            raise ValueError(f"{name} deve ter exatamente 3 valores.")
        s = fracs.sum()
        if not np.isclose(s, 1.0):
            fracs = fracs / s
            print(f"Aviso: {name} normalizadas internamente para somar 1. Novos valores: {fracs}")
        return fracs

    fy = _normalize(fractions_y, "fractions_y")
    fx = _normalize(fractions_x, "fractions_x")

    # Limites em Y: y0<y1<y2<y3
    y1 = fy[0] * ly
    y2 = (fy[0] + fy[1]) * ly

    # Limites em X: x0<x1<x2<x3
    x1 = fx[0] * lx
    x2 = (fx[0] + fx[1]) * lx

    tol = 1e-6

    # Listas de elementos (índices 0-based) por faixa
    elemsY = [[], [], []]  # 0: baixo, 1: meio, 2: topo
    elemsX = [[], [], []]  # 0: esquerda, 1: meio, 2: direita

    for e_idx in range(quad_elements.shape[0]):
        # nós do elemento (tags 1-based)
        tags = quad_elements[e_idx, :]
        idx0 = tags - 1  # para acessar node_coords

        xs = node_coords[idx0, 0]
        ys = node_coords[idx0, 1]
        x_mean = xs.mean()
        y_mean = ys.mean()

        # Classificação em Y
        if y_mean <= y1 + tol:
            elemsY[0].append(e_idx)
        elif y_mean <= y2 + tol:
            elemsY[1].append(e_idx)
        else:
            elemsY[2].append(e_idx)

        # Classificação em X
        if x_mean <= x1 + tol:
            elemsX[0].append(e_idx)
        elif x_mean <= x2 + tol:
            elemsX[1].append(e_idx)
        else:
            elemsX[2].append(e_idx)

    # Converter para arrays e montar nós de cada faixa
    out = {"Y": [], "X": []}

    for k in range(3):
        e = np.array(elemsY[k], dtype=int)
        if e.size > 0:
            n_tags = np.unique(quad_elements[e, :])
            n = n_tags - 1  # índices 0-based
        else:
            n = np.array([], dtype=int)
        out["Y"].append({"elems": e, "nodes": n})

    for k in range(3):
        e = np.array(elemsX[k], dtype=int)
        if e.size > 0:
            n_tags = np.unique(quad_elements[e, :])
            n = n_tags - 1
        else:
            n = np.array([], dtype=int)
        out["X"].append({"elems": e, "nodes": n})

    return out

def add_all_sides_to_parts(parts, node_coords, lx, ly,
                           fractions_y=(0.25, 0.5, 0.25),
                           fractions_x=(0.25, 0.5, 0.25),
                           tol=1e-8,
                           origin_from_coords=True):
    """
    Para cada faixa em X e em Y, adiciona:
      left_nodes, right_nodes, bottom_nodes, top_nodes
    ao dicionário parts retornado por split_mesh_three_parts_xy.

    parts["X"][k] e parts["Y"][k] passam a ter:
      - elems, nodes (já existiam)
      - left_nodes, right_nodes, bottom_nodes, top_nodes (novos)
    """

    def _normalize(fracs, name):
        fracs = np.array(fracs, dtype=float)
        if fracs.size != 3:
            raise ValueError(f"{name} deve ter exatamente 3 valores.")
        s = fracs.sum()
        if not np.isclose(s, 1.0):
            fracs = fracs / s
            print(f"Aviso: {name} normalizadas internamente para somar 1. Novos valores: {fracs}")
        return fracs

    fy = _normalize(fractions_y, "fractions_y")
    fx = _normalize(fractions_x, "fractions_x")

    # Origem: ou assume (0,0) ou infere de node_coords
    if origin_from_coords:
        x0 = float(np.min(node_coords[:, 0]))
        y0 = float(np.min(node_coords[:, 1]))
    else:
        x0, y0 = 0.0, 0.0

    x3 = x0 + lx
    y3 = y0 + ly

    # Limites internos
    x1 = x0 + fx[0] * lx
    x2 = x0 + (fx[0] + fx[1]) * lx

    y1 = y0 + fy[0] * ly
    y2 = y0 + (fy[0] + fy[1]) * ly

    # Para cada faixa, (esquerda,direita) e (baixo,cima)
    x_bounds = [(x0, x1), (x1, x2), (x2, x3)]
    y_bounds = [(y0, y1), (y1, y2), (y2, y3)]

    def _add_sides_for_group(group_dict, xb, yb):
        nodes = np.asarray(group_dict["nodes"], dtype=int)  # <-- força int aqui
        group_dict["nodes"] = nodes  # opcional: garante que fica int no dicionário

        if nodes.size == 0:
            group_dict["left_nodes"]   = np.array([], dtype=int)
            group_dict["right_nodes"]  = np.array([], dtype=int)
            group_dict["bottom_nodes"] = np.array([], dtype=int)
            group_dict["top_nodes"]    = np.array([], dtype=int)
            return

        xs = node_coords[nodes, 0]
        ys = node_coords[nodes, 1]
        xl, xr = xb
        yb0, yt = yb

        group_dict["left_nodes"]   = np.unique(nodes[np.abs(xs - xl)  <= tol]).astype(int)
        group_dict["right_nodes"]  = np.unique(nodes[np.abs(xs - xr)  <= tol]).astype(int)
        group_dict["bottom_nodes"] = np.unique(nodes[np.abs(ys - yb0) <= tol]).astype(int)
        group_dict["top_nodes"]    = np.unique(nodes[np.abs(ys - yt)  <= tol]).astype(int)

    # Faixas em X: x_bounds depende de k; y é sempre domínio inteiro (y0,y3)
    for k in range(3):
        _add_sides_for_group(parts["X"][k], x_bounds[k], (y0, y3))

    # Faixas em Y: y_bounds depende de k; x é sempre domínio inteiro (x0,x3)
    for k in range(3):
        _add_sides_for_group(parts["Y"][k], (x0, x3), y_bounds[k])

    return parts

def build_symmetry_map_y(coord, connect, ly, tol=1e-8):
    """
    Constrói mapeamento de simetria em Y para cada elemento.
    
    Para cada elemento e com centroide (cx, cy), encontra o elemento e'
    cujo centroide é (cx, ly - cy).
    
    Parâmetros
    ----------
    coord : (nnode, 3) array
        Coordenadas dos nós.
    connect : (nel, 4) array (1-based)
        Conectividade dos elementos.
    ly : float
        Altura do domínio.
    tol : float
        Tolerância para matching de centroides.
    
    Retorna
    -------
    sym_map : (nel,) array int
        sym_map[e] = índice do elemento simétrico de e em Y.
        Se não encontrar par, sym_map[e] = e (mapeia para si mesmo).
    """
    nel = connect.shape[0]
    idx = connect - 1  # 0-based
    
    # Centroides
    cx = np.mean(coord[idx, 0], axis=1)
    cy = np.mean(coord[idx, 1], axis=1)
    
    # Centroides espelhados
    cy_mirror = ly - cy
    
    sym_map = np.arange(nel, dtype=int)  # default: mapeia para si mesmo
    
    for e in range(nel):
        dx = np.abs(cx - cx[e])
        dy = np.abs(cy - cy_mirror[e])
        dist = np.sqrt(dx**2 + dy**2)
        j = np.argmin(dist)
        if dist[j] < tol:
            sym_map[e] = j
        else:
            print(f"Aviso: elemento {e} sem par simétrico (dist_min = {dist[j]:.2e})")
    
    return sym_map