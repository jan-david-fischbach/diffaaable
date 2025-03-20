from baryrat import BarycentricRational
import jax.numpy as np
from diffaaable import set_aaa
import logging
logging.getLogger("diffaaable.set_aaa.set_aaa").setLevel(logging.INFO)


def f_test(z, residues, n_poles):
  poles = (0.7+0.9j)**np.arange(n_poles)
  print(poles)
  return np.sum(residues.T[None, :, :]/(z[:, None, None]-poles[None, None, :]), axis=-1)

z_k = np.linspace(-4, 4, 1000) + 0.8j

def test_set_aaa():
  n_poles = 11
  residues = np.arange(n_poles*40_000).reshape(n_poles, -1)
  f_k = f_test(z_k, residues, n_poles)

  print(f_k.shape)
  z_j, f_j, w_j = set_aaa(z_k, f_k, tol=1e-10)

  for i, res in enumerate(residues[:1]):
    r = BarycentricRational(z_j, f_j[:,i], w_j)
    pol, res_found = r.polres()
    print(res_found)
    sorter = np.argsort(-np.abs(pol))
    sorted_poles = pol[sorter]
    print(f"{sorted_poles=}")
    #assert np.isclose(pol, 1+1j)
    #assert np.isclose(res, res_found)

if __name__ == "__main__":
  lmax = np.arange(1, 10)
  N = (2 * lmax * (lmax + 2))**2
  print(np.stack([lmax, N]))
  test_set_aaa()