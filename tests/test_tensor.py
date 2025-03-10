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
