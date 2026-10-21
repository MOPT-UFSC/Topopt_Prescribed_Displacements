"""
FD_LAYOUT.py — Figura do layout de verificação: campo de projeto heterogêneo,
regiões passivas, aresta com deslocamento prescrito e os 5 elementos amostrados.
Convenção das figuras de layout do artigo: preto = sólido, cinza claro = vazio.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = ['Times New Roman', 'Liberation Serif', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 12

R = json.load(open("FD_validation_results.json"))
LX, LY = 0.5, 0.7
ELEM = R["config"]["elem_size"]
nx, ny = R["config"]["mesh"]
Y_BASE, X0, X1 = 0.07, 0.225, 0.275

# reconstruir o campo de projeto (mesma formula deterministica do FD_TEST.py)
cx = (np.arange(nx) + 0.5) * ELEM
cy = (np.arange(ny) + 0.5) * ELEM
CX, CY = np.meshgrid(cx, cy, indexing="xy")
MU = np.clip(0.3 + 0.25 * np.sin(2 * np.pi * CX / LX) * np.sin(np.pi * CY / LY), 0.05, 0.95)
PASS = (CY < Y_BASE) | ((CX > X0) & (CX < X1) & (CY >= Y_BASE))
MU[PASS] = 1.0

fig, ax = plt.subplots(figsize=(4.4, 5.6))
im = ax.imshow(MU, origin="lower", extent=[0, LX, 0, LY], cmap="gray_r",
               vmin=0.0, vmax=1.0, interpolation="nearest", aspect="equal")

# contornos das regioes passivas
ax.plot([0, LX], [Y_BASE, Y_BASE], color="k", linewidth=0.8)
ax.plot([X0, X0], [Y_BASE, LY], color="k", linewidth=0.8)
ax.plot([X1, X1], [Y_BASE, LY], color="k", linewidth=0.8)

# aresta com deslocamento prescrito (vermelha, como na Fig. do dominio)
ax.plot([0, LX], [0.0035, 0.0035], color="r", linewidth=4.0, solid_capstyle="butt", zorder=5)

# elementos amostrados — mesmos marcadores/cores da figura de erros
cores = ['k', 'r', 'b', 'm', 'g']
marcas = ['o', 's', '^', 'D', 'v']
for k, e in enumerate(R["elements"]):
    ax.plot(e["x"], e["y"], marker=marcas[k], color=cores[k], markersize=10,
            markerfacecolor='none', markeredgewidth=2.0, linestyle='None')
    # rotulo a direita para elementos colados na coluna passiva (evita texto sobre preto)
    if e["x"] < 0.25 or (e["x"] - 0.275) < 0.06:
        dx, ha = 0.022, 'left'
    else:
        dx, ha = -0.022, 'right'
    ax.text(e["x"] + dx, e["y"], f"E{k+1}", fontsize=13, ha=ha, va='center')

cb = plt.colorbar(im, ax=ax, fraction=0.052, pad=0.04)
cb.set_label(r"$\mu$", fontsize=16)
cb.ax.tick_params(labelsize=13)

ax.set_xlabel(r"$x$ [m]", fontsize=16)
ax.set_ylabel(r"$y$ [m]", fontsize=16, labelpad=1)
ax.tick_params(axis='both', which='major', labelsize=13)
plt.tight_layout()
plt.savefig("FD_layout.pdf")
plt.savefig("FD_layout.png", dpi=160)
plt.close()
print("figura gerada: FD_layout.[pdf|png]")
