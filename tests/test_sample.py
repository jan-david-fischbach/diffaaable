# Sample Test passing with nose and pytest
from diffaaable import aaa
import jax.numpy as np
import jax
import numpy as onp

n_sample = 200
z_k = onp.random.rand(n_sample)*2+1 + onp.random.rand(n_sample)*1j-0.5

def pole1(a):
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

def test_jacfwd():
    g = jax.jacfwd(pole1)
    assert np.allclose(g(np.pi/2), -0.63661977)
