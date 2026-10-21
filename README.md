# Topology optimization under time-harmonic prescribed displacements

Python implementation accompanying the article

> O. M. Silva, C. Wagner, J. G. Vargas, A. Lenzi, E. L. Cardoso, M. M. Neves,
> **Adjoint sensitivity formulation for topology optimization under time-harmonic prescribed displacements**,
> *Structural and Multidisciplinary Optimization*, 2026 (accepted). DOI: to be added.

The code performs density-based topology optimization of two-dimensional plane-stress structures excited by **prescribed harmonic displacements** (base excitation), optionally combined with external harmonic forces. The objective function is the **active input power** delivered to the structure at a driving frequency, and its gradient is obtained with the **adjoint formulation derived in the paper**, which carries the design dependence of the effective load vector through the dynamic stiffness matrix. The optimization uses RAMP material interpolation, density filtering with robust Heaviside projection (eroded/intermediate/dilated fields), an adaptive continuation of the projection parameter, and the Method of Moving Asymptotes (MMA).

## Repository layout

| File | Content |
|---|---|
| `MAIN.py` | Driver script: problem definition (geometry, passive regions, boundary conditions, prescribed displacements, loads), optimization loop with MMA and projection continuation (Algorithm 1 of the paper), post-processing of the frequency response. |
| `PLANE4Bolha_PT.py` | Q4 plane-stress element with incompatible (bubble) modes; element stiffness and mass matrices, RAMP/SIMP interpolation, sparse assembly, selector matrices `Lp`/`Lf` for prescribed and free degrees of freedom. |
| `SOLVE.py` | Harmonic analysis with prescribed displacements, adjoint harmonic analysis (Eqs. 46 and 62 of the paper), static analysis, modal analysis. Uses a sparse direct solver (PARDISO). |
| `OBJETIVO.py` | Active input power and its element-level sensitivities (Eqs. 63–65); auxiliary static compliance; frequency sweep of the objective. |
| `RESTRICAO.py` | Volume constraint and its gradient. |
| `FILTER.py` | Density filter, Heaviside projection, robust eroded/intermediate/dilated triplet, and chain-rule back-propagation of sensitivities through filter and projection (Eqs. 74–75). |
| `MESH.py` | Structured Q4 mesh of a rectangular domain generated with Gmsh; identification of edge node sets. |
| `substruct.py` | Partition of the mesh into bands in `x` and `y` (used to define passive regions) and identification of their boundary nodes. |
| `VTK_FUNCS.py` | Export of the optimized density field and displacement field to `result.vtu` (ParaView). |
| `plots.py` | Convergence history and frequency response plots from the text files written by `MAIN.py`. |
| `mma.py`, `util.py` | Method of Moving Asymptotes (Svanberg) in the Python translation by Arjen Deetman, with shape/robustness hardening; logger. |
| `verification/` | Finite-difference verification of the adjoint sensitivities (Section 5, Table 1 and Fig. 3 of the paper). See below. |

## Requirements

Python ≥ 3.10 with

```
numpy  scipy  matplotlib  gmsh  vtk  pypardiso (MOPT-UFSC fork, see below)
```

```bash
pip install -r requirements.txt
```

### Linear solver

All state and adjoint systems are solved with Intel oneAPI MKL PARDISO through PyPardiso. The harmonic problems are **complex symmetric** (`PyPardisoSolver(mtype=6)` in `SOLVE.py`). The PyPardiso release on PyPI accepts only real `float64` matrices; this code requires the [MOPT-UFSC fork of PyPardiso](https://github.com/MOPT-UFSC/PyPardiso), which adds complex-matrix support:

```bash
pip install git+https://github.com/MOPT-UFSC/PyPardiso.git
```

`requirements.txt` already points to this fork. If PARDISO is not available on your platform, `SOLVE.py` can be adapted to `scipy.sparse.linalg.splu` (one factorization per frequency, reused for the state and adjoint solves). The algebra is unchanged; only the factorization backend differs.

## Running

```bash
python MAIN.py
```

`MAIN.py` is a self-contained script: every parameter of the problem is set at the top of the file. As distributed, it is configured for **Example 2, Load Case 2(b)** of the paper (rectangular support base, 1.5 m × 0.3 m, sinusoidal prescribed displacement along the bottom edge, target frequency 1815 Hz). The other examples of the paper are obtained by editing the variables below.

| Paper quantity | Variable(s) in `MAIN.py` | Example 1 | Example 2 |
|---|---|---|---|
| Design domain | `lx`, `ly` | 0.5, 0.7 | 1.5, 0.3 |
| Element size | `tamanho_elemento` | 0.0025 | 0.0025 |
| Passive regions | `fra_y`, `fra_x`, `passive` | bottom band `Y1` (0.07 m) and central column `X2` (0.05 m) | bottom and top bands `Y1`, `Y3` (0.03 m each) |
| Initial density of design elements | `xinit_value` | 0.20 | 0.25 |
| Volume bounds (fraction of design domain) | `volume_restante`, `minimo_volume` | 0.25, 0.05 | 0.25, 0.05 |
| Filter radius | `radius` | 10 × element diagonal | 15 × element diagonal |
| Driving frequency [Hz] | `freq` | 280, 380, 1050, 1180, 1440 | 1000, 1815 |
| Damping ratio during optimization | `xsi_s` | 0.05 | 0.05 |
| Prescribed displacements | `AAA`, `BBB`, `NNN`, assignments to `U[fixed_dofs_x]`, `U[fixed_dofs_y]` | Table 2 | Table 3 |
| External harmonic force | `F` | none | 1000 N at the centre of the top edge |
| Static weight φ and static load | `PARCELA_ESTATICO`, `F_est` | 0.1, unit load at the tip | 0.2, distributed unit load on the top edge |
| RAMP penalization (dynamic / static) | `p`, `ps`; mass exponent `q` | 4, 4; 1 | 4, 4; 1 |
| Heaviside projection | `bebeta` (initial β), `eta_` = (η, Δη) | 0.1, (0.5, 0.05) | 0.1, (0.5, 0.05) |
| Maximum outer iterations | `maxoutit` | 2000 | 2000 |

Material: steel (`E = 210e9`, `rho = 7850`, `v = 0.3`), with `Emin = 1e-7 E` and `rhomin = 1e-8 rho`.

### Outputs

Written to the working directory:

- `EPT_OPT.log` — iteration log (objective, volume, KKT residual, density change);
- `f0val_dyn.txt`, `f0val_est.txt`, `fval.txt`, `bebeta.txt`, `kktnorm.txt`, `change.txt` — convergence histories;
- `FRF_init.txt`, `FRF_opt.txt`, `Ux_*.txt`, `Uy_*.txt` — active input power and monitoring-point displacement versus frequency for the initial and optimized designs (evaluated with the damping ratio set in the post-processing block, ξ = 0.001 in the paper);
- `result.vtu` — optimized density field and displacement field, readable in ParaView.

## Sensitivity verification

`verification/FD_TEST.py` compares the adjoint sensitivities with central finite differences on a coarse mesh (element size 0.01 m) of the Example 1 geometry under Load Case 1(a). The perturbation is applied to the unfiltered design variable, so the test covers the full chain: density filter → robust Heaviside projection (physics on the eroded field) → RAMP → harmonic analysis with prescribed displacements → active input power. Five design elements are sampled at five frequencies (280, 380, 1050, 1180 and 1440 Hz), the static compliance is checked as well, a step-size study is performed, and the identity W = (ω²/2) uᴴ C u is verified.

```bash
cd verification
python FD_TEST.py     # writes FD_validation_results.json (a few minutes)
python FD_PLOTS.py    # FD_step_study.pdf, FD_errors.pdf
python FD_LAYOUT.py   # FD_layout.pdf
```

`FD_validation_results.json` and `FD_table.tex` contain the numbers reported in Table 1 of the paper. The scripts import the solver modules from the repository root without modification.

## License

This repository is distributed under the **GNU General Public License v3.0 or later** (see `LICENSE`).

The files `mma.py` and `util.py` are derived from [GCMMA-MMA-Python](https://github.com/arjendeetman/GCMMA-MMA-Python) by Arjen Deetman, a Python translation of Krister Svanberg's MATLAB implementation of the Method of Moving Asymptotes, which is itself released under GPL-3.0.

## Acknowledgments

The authors are grateful to Krister Svanberg (KTH) for providing the original FORTRAN codes of the MMA, and to Arjen Deetman (TU/e) for the Python implementation of the MMA.

## Citation

If you use this code, please cite the article above. A `CITATION.cff` file is provided.
