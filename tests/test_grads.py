# Sample Test passing with nose and pytest
from diffaaable import aaa
import jax.numpy as np
import jax
import numpy as onp

from jax import random
import pytest

n_sample = 200

def pole1(a):
    key = random.PRNGKey(0)
    key, subkey = random.split(key)
    z_k = random.uniform(key, (n_sample,))*2+0.5 + random.uniform(subkey, (n_sample,))*1j
    f_k = f(z_k, a)
    z_j, f_j, w_j, z_n = aaa(z_k, f_k)
    p_i = z_n.imag
    p_r = z_n.real
    selected_poles = z_n[p_r>0]
    selected_poles = np.sort(selected_poles)
    return np.real(selected_poles[0])

def f(x, a):
    return np.tan(a*x)

def test_tan():
    assert np.allclose(pole1(np.pi/2), 1)

def test_grad():
    g = jax.grad(pole1)
    g(np.pi/2)

@pytest.mark.xfail
def test_jacfwd():
    g = jax.jacfwd(pole1)
    assert np.allclose(g(np.pi/2), -0.63661977)

def test_jacrev():
    g = jax.jacrev(pole1)
    assert np.allclose(g(np.pi/2), -0.63661977)

def test_kwargs():
    z_k = np.linspace(0, 3, 20)
    f_k = f(z_k, np.pi)
    aaa(z_k, f_k, tol=1e-9, mmax=50)

@pytest.mark.xfail
def test_jvp():
    z_k = np.linspace(0, 3, 20)
    f_k = f(z_k, np.pi)
    f_dot = np.ones_like(z_k)
    jax.jvp(aaa, (z_k, f_k), (np.zeros_like(z_k), f_dot))

def test_jvp_complex_input():
    z_k = np.linspace(0, 3, 20, dtype=complex)
    f_k = f(z_k, np.pi)
    f_dot = np.ones_like(z_k)
    jax.jvp(aaa, (z_k, f_k), (np.zeros_like(z_k), f_dot))

def test_grad_simple():
    @jax.custom_jvp
    def foo(a):
        return onp.sum(a) * 2

    @foo.defjvp
    def foo_jvp(primals, tangents):
        primal_out = foo(*primals)

        a = primals[0]
        a_dot = tangents[0]

        with jax.disable_jit():
            a_dot_out, _, _, _ = np.linalg.lstsq(np.diag(a),a_dot)
        return primal_out, a_dot_out[0]*2

    def bar(a):
        return foo(a)

    g = jax.jacrev(bar)
    g(np.array([1.0, 1.0]))

def test_lstsq():
    def lstsq(A, b):
        res = np.linalg.lstsq(A,b)
        return np.sum(res[0])

    g = jax.jacrev(lstsq, argnums=1)
    g(np.eye(2), np.ones(2))


if __name__ == "__main__":
    #test_jvp_complex_input()
    test_grad()
