import jax;
jax.config.update('jax_platforms', 'cpu')
jax.config.update("jax_enable_x64", True)
from jax.tree_util import Partial
from diffaaable import selective
import jax.numpy as np

import matplotlib.pyplot as plt
import pytest

def f1(a, x):
    return np.tan(a*x)
f1 = Partial(f1, np.pi*(1-0.3j))

def f2(z, N=70):
    return 1/(1-z**N)

def poles_f2(N=70):
    return np.exp(2j*np.pi*np.linspace(0,1,N, endpoint=False))

def f3(z, N=70):
    return 1/(1-z**N) + 1j/(1-(z/1.5)**N)

def poles_f3(N=70):
    p1 = np.exp(2j*np.pi*np.linspace(0,1,N, endpoint=False))
    p2 = 1.5*np.exp(2j*np.pi*np.linspace(0,1,N, endpoint=False))
    return np.concat([p1, p2])

TREAMS_FOUND = False
try:
    import treams
    TREAMS_FOUND = True
except ModuleNotFoundError:
    print("treams not found, skipping test")

if TREAMS_FOUND:
    def f_mie(z):
        mie = []
        for x in z:
            mie.append(treams.coeffs.mie(1, [x], [4,1], [1,1], [0,0]))
        mie = np.array(mie)
        return mie[:,0,0]

    def test_mie():
        poles, residues, evals = selective.selective_subdivision_aaa(f_mie, (0.2-0.5j, 82+0.1j), debug_plot_domains=True, max_poles=60, tol_aaa=1e-10, N=200, evolutions_adaptive=10)
        plt.scatter(poles.real, poles.imag, marker=".")
        plt.savefig("debug_out/selective_mie.pdf")
        plt.close()


@pytest.mark.parametrize("f", [f2, f3])
@pytest.mark.parametrize("N", [5, 10, 30, 60])
def test_fwd(f, N):
    mi, ma = (-300, 400)
    mi, ma = (-6, 2)

    f = Partial(f, N=N)

    poles, residues, evals = selective.selective_subdivision_aaa(f, (mi-1.23+(mi-1.1)*1j, ma+1+(ma+2)*1j), debug_plot_domains=True, max_poles=30, cutoff=1e6, tol_aaa=1e-9, N=100, evolutions_adaptive=8)
    print(f"Poles: {poles}")
    print(f"Residues: {residues}")
    print(f"Evaluations: {evals}")
    print(f"Num Poles: {len(poles)}")

    plt.scatter(poles.real, poles.imag, marker=".")

    plt.savefig("debug_out/selective.pdf")
    plt.close()


if __name__ == "__main__":
    test_fwd()
