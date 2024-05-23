# Sample Test passing with nose and pytest
from diffaaable import adaptive
import jax.numpy as np
import jax
import jax.tree_util

def f(x, a):
    return 1/(x-a) #np.tan(a*x)

def test_tan():
    f_a = jax.tree_util.Partial(f, a=np.pi)
    z_j, f_j, w_j, z_n = adaptive.adaptive_aaa(np.linspace(0,1,10), f_a)
    print(z_n)

def test_grad():
    def p1_imag(a):
        jax.debug.print("a {}", a)
        f_a = jax.tree_util.Partial(f, a=a)
        jax.debug.print("f_a {}", f_a)
        z_k = np.linspace(0,0.5,10)
        z_j, f_j, w_j, z_n = \
            adaptive.adaptive_aaa(z_k, f_a)
        return np.real(z_n[-1])
    
    fwd = jax.jacfwd(p1_imag)
    print(fwd(np.pi))

    grad = jax.grad(p1_imag)
    #print(grad(np.pi))

if __name__ == "__main__":
    #test_tan()
    test_grad()
