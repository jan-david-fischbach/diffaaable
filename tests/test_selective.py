import jax; jax.config.update('jax_platforms', 'cpu')
import jax.tree_util
from diffaaable import selective
import jax.numpy as np

import matplotlib.pyplot as plt
import pytest

def f(a, x):
    return np.tan(a*x)

def test_fwd():
    expected = np.arange(-40, 89)+0.5
    f_a = jax.tree_util.Partial(f, np.pi)
    poles, residues, evals = selective.selective_refinement_aaa(f_a, (-41.1-3j, 90+10j))
    print(f"Poles: {poles}")
    print(f"Residues: {residues}")
    print(f"Evaluations: {evals}")
    print(f"Num Poles: {len(expected)}")
    plt.savefig("debug_out/selective.pdf")
    plt.close()

    print("Hall of shame:")
    for pole in expected:
        hit = np.any(np.abs(poles-pole)< 1e-5)
        if not hit:
            print(pole)


if __name__ == "__main__":
    test_fwd()
    #pytest.main()
