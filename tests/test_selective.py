import jax; jax.config.update('jax_platforms', 'cpu')
import jax.tree_util
from diffaaable import selective
import jax.numpy as np

import matplotlib.pyplot as plt
import pytest

def f(a, x):
    return np.tan(a*x)

def test_fwd():
    f_a = jax.tree_util.Partial(f, np.pi)
    poles, residues = selective.selective_refinement_aaa(f_a, (-11-11j, 10+10j))
    print(f"Poles: {poles}")
    print(f"Residues: {residues}")
    plt.savefig("debug_out/selective.pdf")
    plt.close()

if __name__ == "__main__":
    test_fwd()
    #pytest.main()
