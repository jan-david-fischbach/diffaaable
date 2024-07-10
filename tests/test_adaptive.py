# Sample Test passing with nose and pytest
import jax.test_util
from diffaaable import adaptive
import jax.numpy as np
import jax
import jax; jax.config.update('jax_platforms', 'cpu')
import jax.tree_util
import pytest
from functools import partial
import pathlib

def f1(a, x):
    return 1/(x-a*2)

def f2(a, x):
    return np.tan(a*x)

def sort_poles(z_n):
    order = np.argsort(np.abs(z_n-0.5))
    return z_n[order]

@pytest.mark.parametrize("f", [f1, f2])
def test_fwd(f):
    f_a = jax.tree_util.Partial(f, np.pi)
    z_j, f_j, w_j, z_n = adaptive.adaptive_aaa(np.linspace(0,1,10, dtype=complex), f_a)
    z_n = sort_poles(z_n)
    print(f"Poles: {z_n}")

def test_fwd_heat():
    f_a = partial(f2, np.pi)
    known_poles = np.arange(-4.5, 5.5)

    folder = "debug_out"
    pathlib.Path(folder).mkdir(exist_ok=True)
    next_samples = partial(
        adaptive.next_samples_heat, batchsize=5,
        debug=folder, resolution=(201, 201), debug_known_poles=known_poles)
    z_j, f_j, w_j, z_n = adaptive.adaptive_aaa(
        np.linspace(-1,2,10, dtype=complex)+1j, f_a,
        sampling=next_samples, domain=[-5-1j, 7+1j],
        radius=0.1, evolutions=17)
    z_n = sort_poles(z_n)
    print(f"Poles: {z_n}")

    for pole in known_poles:
        assert np.min(np.abs(z_n-pole)) < 1e-8

def test_kwargs():
    f_a = jax.tree_util.Partial(f1, np.pi)
    z_j, f_j, w_j, z_n = adaptive.adaptive_aaa(
        np.linspace(0,1,10, dtype=complex),
        f_a,
        evolutions = 2,
        cutoff = 1e12,
        tol = 1e-9,
        radius = 1e-3)
    z_n = sort_poles(z_n)
    print(f"Poles: {z_n}")

@pytest.mark.parametrize("f", [f1, f2])
def test_grad(f):
    def p1_imag(a):
        f_a = jax.tree_util.Partial(f, a)
        z_k = np.linspace(0,0.5,10, dtype=complex)
        # Note: one of the samples lies directly on a pole.
        # Adaptive aaa should be robust against that due to masking.
        z_j, f_j, w_j, z_n = \
            adaptive.adaptive_aaa(z_k, f_a)

        z_n = sort_poles(z_n)
        #jax.debug.print("z_n: {}", z_n[0])

        return np.real(z_n[0])

    grad = jax.grad(p1_imag)

    a = np.array([np.pi], dtype=complex)
    # print("## Test Grad")
    print(f"grad: {grad(a)}")
    # print("## Check Grads")
    # jax.test_util.check_grads(p1_imag, (a,), 1)

if __name__ == "__main__":
    test_fwd_heat()
    #pytest.main()
