# Sample Test passing with nose and pytest
from diffaaable import adaptive
import jax.numpy as np
import jax

def f(x, a):
    return np.tan(a*x)

def test_tan():
    z_j, f_j, w_j, z_n = adaptive.adaptive_aaa(np.linspace(0,1,10), f, np.array([np.pi]))
    print(z_n)

def test_grad():
    def p1_imag(a):
        z_j, f_j, w_j, z_n = adaptive.adaptive_aaa(
            np.linspace(0,1,10), f, np.array([a]))
        return np.real(z_n[-1])
    grad = jax.grad(p1_imag)
    print(grad(np.pi))

if __name__ == "__main__":
    test_grad()
