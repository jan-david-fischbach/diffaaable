---
jupytext:
  cell_metadata_filter: -all
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Pole-Expansion of a T-Matrix

This example reproduces Fig. 1 of [Pole-Expansion of the T-Matrix Based on a Matrix-Valued AAA-Algorithm](https://doi.org/10.1021/acsphotonics.6c00436).
A few samples of a frequency-dependent T-matrix are fed jointly to {func}`~diffaaable.tensor.tensor_aaa`.
The result is a pole-expansion with one set of poles shared by all matrix elements, which can be evaluated at arbitrary frequencies.

The scatterer is a sphere ($\varepsilon_\mathrm{r} = 12$, radius {math}`250\,\mathrm{nm}`) with a shell ($\varepsilon_\mathrm{r} = 12$, outer radius {math}`300\,\mathrm{nm}`) made of a fictitious chiral material with Pasteur parameter $\kappa = 2\times10^{-5}$.
The chirality couples the electric and magnetic dipole coefficients, so all resonances also show up in the (tiny) off-diagonal entries.
The T-matrices are computed with [treams](https://github.com/tfp-photonics/treams) (`pip install treams`).

## Sampling the T-matrix

We sample the dipolar T-matrix in the parity basis at only 13 wavenumbers $k_0$ and keep the $2\times2$ block coupling the $z$-oriented magnetic and electric dipoles.

```{code-cell}
:tags: [remove-stderr]
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, e, hbar
import treams
import diffaaable
from diffaaable.tensor import residues

radii = [0.25, 0.3]  # µm
materials = [12, treams.Material(12, 1, 2e-5), 1]  # core, chiral shell, embedding

def t_matrix(k0):
    """Dipolar T-matrix block of the chiral core-shell sphere at k0 (in 1/µm)."""
    T = treams.TMatrix.sphere(
        lmax=1, radii=radii, materials=materials, k0=k0, poltype="helicity"
    ).changepoltype()  # helicity -> parity basis
    return np.asarray(T)[:2, :2]

def k_to_ev(k0):
    """Wavenumber in 1/µm to photon energy in eV."""
    return hbar * c * k0 * 1e6 / e

k0s = np.linspace(0.8, 7.32, 13)
Ts = np.array([t_matrix(k0) for k0 in k0s])
Ts.shape
```

## Joint pole-expansion

{func}`~diffaaable.tensor.tensor_aaa` approximates all four entries at once.

```{code-cell}
z_j, f_j, w_j, z_n = diffaaable.tensor_aaa(k0s, Ts, tol_aaa=1e-5)

# keep the resonances: poles in the lower half-plane near the sampled interval
domain = (z_n.real > k0s.min()) & (z_n.real < k0s.max()) & (z_n.imag < 0) & (z_n.imag > -0.7)
poles = z_n[domain]
R_n = residues(z_j, f_j, w_j, poles)  # matrix-valued residues, shape (n, 2, 2)
k_to_ev(poles)
```

The three poles correspond to the three resonances (a, b, c) of the paper's figure.

## Evaluating the expansion

The barycentric representation can be evaluated at arbitrary frequencies practically for free with {func}`~diffaaable.tensor.tensor_baryrat`.
For comparison we also compute a finely resolved reference with treams.

```{code-cell}
:tags: [remove-stderr]
k0s_dense = np.linspace(k0s.min(), k0s.max(), 501)
T_aaa = diffaaable.tensor_baryrat(z_j, f_j, w_j)(k0s_dense)
T_ref = np.array([t_matrix(k0) for k0 in k0s_dense])

rel_err = np.max(np.abs(T_aaa - T_ref)) / np.max(np.abs(T_ref))
print(f"max. relative error: {rel_err:.1e}")
```

```{code-cell}
colors = [["C0", "C2"], ["C3", "C1"]]
scale = [[1, 1e10], [1e10, 1]]  # off-diagonal entries are tiny

fig = plt.figure(figsize=(9, 3.2), layout="constrained")
left, right = fig.subfigures(1, 2, width_ratios=[1.3, 1])
axs = left.subplots(2, 2, sharex=True, sharey=True)
for i in range(2):
    for j in range(2):
        ax, s = axs[i, j], scale[i][j]
        ax.plot(k_to_ev(k0s_dense), np.abs(T_aaa[:, i, j])**2 * s, color=colors[i][j])
        ax.plot(k_to_ev(k0s), np.abs(Ts[:, i, j])**2 * s, "x", color=colors[i][j])
        ax.plot(k_to_ev(k0s_dense), np.abs(T_ref[:, i, j])**2 * s, "k--", lw=1)
        label = f"$|T_{{{i}{j}}}|^2$" + (r"$\times10^{10}$" if s != 1 else "")
        ax.set_title(label, fontsize=9, color=colors[i][j])
left.supxlabel(r"$\hbar\omega$ [eV]")

ax = right.subplots()
weight = np.linalg.norm(R_n, axis=(1, 2)) / np.abs(poles.imag)
ax.scatter(k_to_ev(poles.real), k_to_ev(poles.imag), s=200 * weight / weight.max(), color="C4")
for label, p in zip("abc", np.sort_complex(poles)):
    ax.annotate(label, (k_to_ev(p.real), k_to_ev(p.imag)), xytext=(6, 6), textcoords="offset points")
ax.axhline(0, color="gray", lw=1)
ax.set_xlim(k_to_ev(k0s.min()), k_to_ev(k0s.max()))
ax.set_xlabel(r"$\Re\{\hbar\omega\}$ [eV]")
ax.set_ylabel(r"$\Im\{\hbar\omega\}$ [eV]")
ax.set_title("poles", fontsize=9);
```

Left: samples (crosses), pole-expansion (solid) and reference (dashed) for the four T-matrix entries.
Right: the shared poles in the complex frequency plane. The marker size is proportional to $\|\mathbf{R}_n\| / |\Im\{\omega_n\}|$, i.e. to the pole's influence on the real axis.
Note how peak c is reproduced although no sample lies close to its center.
