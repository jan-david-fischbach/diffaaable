from baryrat import BarycentricRational
import jax.numpy as np
from diffaaable import vectorial_aaa

def test_vectorial_aaa():
  residues = np.array([1, 3j, 1+2j, -4, 6, 9, 1-3j])
  def f_test(z):
    return residues/(z[:, None]-(1+1j))

  z_k = np.linspace(-4, 4, 1000) + 0.5j

  z_j, f_j, w_j = vectorial_aaa(z_k, f_test(z_k))

  for i, res in enumerate(residues):
    r = BarycentricRational(z_j, f_j[:,i], w_j)
    pol, res_found = r.polres()
    assert np.isclose(pol, 1+1j)
    assert np.isclose(res, res_found)
