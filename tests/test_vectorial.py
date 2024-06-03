from baryrat import BarycentricRational
import jax.numpy as np
from diffaaable import vectorial_aaa, aaa


def f_test(z, residues):
  return residues/(z[:, None]-(1+1j))

z_k = np.linspace(-4, 4, 1000) + 0.5j

def test_vectorial_aaa():
  residues = np.array([1, 3j, 1+2j, -4, 6, 9, 1-3j])
  z_j, f_j, w_j = vectorial_aaa(z_k, f_test(z_k, residues))

  for i, res in enumerate(residues):
    r = BarycentricRational(z_j, f_j[:,i], w_j)
    pol, res_found = r.polres()
    assert np.isclose(pol, 1+1j)
    assert np.isclose(res, res_found)

def test_runtime(benchmark):
  residues = np.array([1])
  f_k_vec = f_test(z_k, residues)
  f_k = f_k_vec.squeeze()

  benchmark(lambda :aaa(z_k, f_k))

def test_runtime_vec(benchmark):
  residues = np.array([1, 1j, 2j])
  f_k_vec = f_test(z_k, residues)

  benchmark(lambda :vectorial_aaa(z_k, f_k_vec))
