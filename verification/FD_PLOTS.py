"""
FD_PLOTS.py — Figuras da validação FD no estilo das figuras do artigo
(plots.py: Times/stix, labels 18, ticks 16, legenda 14, grid pontilhado alfa 0.5).

No ambiente do artigo use 'Times New Roman'; aqui o fallback métrico é a
Liberation Serif. A linha do rcParams fica identica à do plots.py.
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
freqs = R["config"]["freqs"]
els = R["elements"]
sweep = R["sweep"]
tab_err = np.array(R["table"]["rel_err"])
stat_err = np.array(R["static"]["rel_err"])

# ------------------------------------------------------------------
# Fig. 1 — estudo do passo (curva V)
# ------------------------------------------------------------------
plt.figure(figsize=(6.0, 4.4))
plt.loglog(sweep["deltas"], sweep["rel_err"], 'k-o', markersize=6, linewidth=1.5,
           label=f"Element at (0.105, 0.385) m, {sweep['freq']:.0f} Hz")
plt.xlabel(r"Perturbation step $\Delta\mu$", fontsize=18)
plt.ylabel("Relative error", fontsize=18, labelpad=1)
plt.title("Central finite-difference step-size study", fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.legend(loc='upper center', fontsize=14)
plt.grid(True, which='both', linestyle=':', alpha=0.5)
plt.gca().invert_xaxis()
plt.tight_layout()
plt.savefig("FD_step_study.pdf")
plt.savefig("FD_step_study.png", dpi=160)
plt.close()

# ------------------------------------------------------------------
# Fig. 2 — erros relativos: 5 elementos x (5 frequências + estática)
# ------------------------------------------------------------------
cores = ['k', 'r', 'b', 'm', 'g']
marcas = ['o', 's', '^', 'D', 'v']
xcat = np.arange(len(freqs) + 1)
labels_x = [f"{f:.0f}" for f in freqs] + ["Static"]

plt.figure(figsize=(6.6, 4.4))
for k, e in enumerate(els):
    yy = np.concatenate([tab_err[k, :], [stat_err[k]]])
    plt.semilogy(xcat, yy, linestyle='None', marker=marcas[k], color=cores[k],
                 markersize=8, markerfacecolor='none', markeredgewidth=1.6,
                 label=f"El. {k+1} ({e['x']:.3f}, {e['y']:.3f}) m")
plt.xticks(xcat, labels_x)
plt.xlabel("Frequency [Hz]", fontsize=18)
plt.ylabel(r"$|\,1 - (\partial W/\partial\mu)_\mathrm{FD}\,/\,(\partial W/\partial\mu)_\mathrm{adj}\,|$",
           fontsize=16, labelpad=1)
plt.title("Adjoint vs. central finite differences", fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.ylim(1e-10, 1e-3)
plt.legend(loc='upper right', fontsize=12, ncol=1)
plt.grid(True, which='both', linestyle=':', alpha=0.5)
plt.tight_layout()
plt.savefig("FD_errors.pdf")
plt.savefig("FD_errors.png", dpi=160)
plt.close()

print("figuras geradas: FD_step_study.[pdf|png], FD_errors.[pdf|png]")
