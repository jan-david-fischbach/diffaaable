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

def test_fwd():
    mi, ma = (-300, 400)
    mi, ma = (-6, 2)
    N = 300

    expected = poles_f3(N)
    f = Partial(f3, N=N)

    poles, residues, evals = selective.selective_refinement_aaa(f, (mi-1.23+(mi-1.1)*1j, ma+1+(ma+2)*1j))
    print(f"Poles: {poles}")
    print(f"Residues: {residues}")
    print(f"Evaluations: {evals}")
    print(f"Num Poles: {len(poles)}")

    print("Hall of shame:")
    for pole in expected:
        hit = np.any(np.abs(poles-pole)< 1e-5)
        if not hit:
            print(pole)
            plt.scatter([pole.real], [pole.imag], facecolor="none", edgecolor="k")

    plt.savefig("debug_out/selective.pdf")
    plt.close()


if __name__ == "__main__":
    test_fwd()
    #pytest.main()
