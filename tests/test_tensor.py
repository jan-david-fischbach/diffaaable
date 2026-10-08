from baryrat import BarycentricRational
import jax.numpy as np
from diffaaable.tensor import tensor_aaa, tensor_baryrat


def f_test(z, r1, r2):

  return r1[None, ...]/(z[:, None, None]-(1+1j)) + r2[None, ...]/(z[:, None, None]-(1-1j))

z_k = np.linspace(-4, 4, 200) + 0.5j

def test_tensor_aaa():
  r1 = np.array([
    [1, 3j, 1+2j],
    [-4, 0, 1e-14]
  ])
  r2 = np.array([
    [3, 0, 2+1j],
    [-6, 2, 0]
  ])

  F_k = f_test(z_k, r1, r2)
  print(F_k.shape)

  z_j, f_j, w_j, poles = tensor_aaa(z_k, F_k)

  F = tensor_baryrat(z_j, f_j, w_j)

  print(f"{poles=}")
  print(F(0))

  print(F(np.linspace(1,2,2)))


def test_adaptive_tensor_aaa():
  from diffaaable.adaptive import adaptive_aaa
  r1 = np.array([
    [1, 3j, 1+2j],
    [-4, 0, 1e-14]
  ])
  r2 = np.array([
    [3, 0, 2+1j],
    [-6, 2, 0]
  ])

  def F(z):
    return f_test(z, r1, r2)

  z_k = np.linspace(-4, 4, 11) + 0.5j
  z_j, f_j, w_j, poles = adaptive_aaa(z_k, F, aaa=tensor_aaa, domain=[-4-2j, 4+2j], radius=1e-2)

  print(f"{poles=}")

if __name__ == "__main__":
  test_tensor_aaa()
  test_adaptive_tensor_aaa()


def test_tensor_baryrat():
  import numpy as onp
  import jax
  from diffaaable import tensor_baryrat as public_tensor_baryrat
  assert public_tensor_baryrat is tensor_baryrat

  r1 = np.array([[1, 3j, 1+2j], [-4, 0, 1e-14]])
  r2 = np.array([[3, 0, 2+1j], [-6, 2, 0]])
  z_j, f_j, w_j, _ = tensor_aaa(z_k, f_test(z_k, r1, r2))
  F = tensor_baryrat(z_j, f_j, w_j)

  # agrees with baryrat entry by entry
  z = np.linspace(-3, 3, 7) + 0.1j
  ref = onp.stack([
    BarycentricRational(onp.asarray(z_j), onp.asarray(f_j[:, i, j]), onp.asarray(w_j))(onp.asarray(z))
    for i in range(2) for j in range(3)
  ], axis=-1).reshape(len(z), 2, 3)
  assert F(z).shape == (7, 2, 3)
  assert np.allclose(F(z), ref)
  assert np.allclose(F(z), f_test(z, r1, r2))

  # scalar and 2D arguments, exact values at the nodes
  assert F(0.5j).shape == (2, 3)
  assert F(z.reshape(7, 1)).shape == (7, 1, 2, 3)
  assert np.allclose(F(z_j), f_j)

  # differentiable, also at a node
  d = jax.grad(lambda x: F(x)[0, 0].real)
  assert np.isfinite(d(0.3 + 0.5j))
  assert np.isfinite(d(z_j[0]))

def test_tensor_baryrat_scalar_values():
  z_j = np.array([0., 1., 2.])
  f_j = 1 + z_j**2
  w_j = np.array([1., -2., 1.])  # weights of polynomial interpolation
  F = tensor_baryrat(z_j, f_j, w_j)
  z = np.array([0.5, 1.5, -1.])
  assert np.allclose(F(z), 1 + z**2)
  assert np.allclose(F(np.array([0., 1., 2.])), f_j)
