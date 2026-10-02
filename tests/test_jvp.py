import jax
import jax.numpy as jnp
import numpy as onp
import pytest
from diffaaable import aaa, vectorial_aaa, set_aaa, tensor_aaa

key1, key2 = jax.random.split(jax.random.PRNGKey(0))
z_k = (jax.random.uniform(key1, (120,))*3-1.5) + 1j*(jax.random.uniform(key2, (120,))*3-1.5)

def f_scalar(a, z):
  # poles of tan(a z) at z = pi/(2a) + n pi/a and a pole at z = a/2
  return jnp.tan(a*z) + 1/(z-a/2)

def f_vec(a, z):
  return jnp.stack([
    f_scalar(a, z),
    (2+1j)*jnp.tan(a*z) + 1/(z-3j),
    jnp.zeros_like(z), # numerical zero entry
  ], axis=-1)

def f_tensor(a, z):
  return f_vec(a, z).reshape(-1, 1, 3)

# fit(z_k, f_k) and the sampled function
variants = {
  "aaa": (lambda z, f: aaa(z, f, tol=1e-10), f_scalar),
  "vectorial_aaa": (lambda z, f: vectorial_aaa(z, f, tol=1e-10), f_vec),
  "set_aaa": (lambda z, f: set_aaa(z, f[:, :2], tol=1e-10), f_vec),
  "set_aaa_scalar": (lambda z, f: set_aaa(z, f[:, None], tol=1e-10), f_scalar),
  "tensor_aaa": (lambda z, f: tensor_aaa(z, f, tol_aaa=1e-10), f_tensor),
}

a0 = 1.1
target = jnp.pi/(2*a0)  # pole of tan(a z), d/da = -pole/a
dtarget = -target/a0

def pole(a, variant):
  fit, f = variants[variant]
  z_j, f_j, w_j, z_n = fit(z_k, f(a, z_k))
  return z_n[jnp.argmin(jnp.abs(z_n - target))]

@pytest.mark.parametrize("variant", variants)
def test_pole_jvp(variant):
  p, dp = jax.jvp(lambda a: pole(a, variant), (a0,), (1.0,))
  assert jnp.abs(p - target) < 1e-7
  assert jnp.abs(dp - dtarget) < 1e-5

@pytest.mark.parametrize("variant", variants)
def test_pole_grad(variant):
  g = jax.grad(lambda a: jnp.real(pole(a, variant)))(a0)
  assert jnp.abs(g - jnp.real(dtarget)) < 1e-5

@pytest.mark.parametrize("variant", variants)
def test_value_tangents_shape(variant):
  fit, f = variants[variant]
  primal, tangent = jax.jvp(lambda a: fit(z_k, f(a, z_k)), (a0,), (1.0,))
  for p, t in zip(primal, tangent):
    assert onp.shape(p) == onp.shape(t)
