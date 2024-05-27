from baryrat import BarycentricRational
import jax.numpy as np
from diffaaable.lorentz import lorentz_aaa


def f_test(z, res, pole):
  z = z[:, None]
  return 1j*res/(z-pole) + 1j*res.conj()/(z+np.conj(pole))

z_k = np.linspace(1, 5, 100) + 0.01j

def test_lorentz_aaa():
  pole = 2+0.1j
  residues = np.array([1j])
  z_j, f_j, w_j, errors = lorentz_aaa(z_k, f_test(z_k, residues, pole), mmax=10)

  print(f"{errors=}")

  z_j = np.concatenate([z_j, -np.conj(z_j)])
  f_j = f_j[:, 0]
  f_j = np.concatenate([f_j, np.conj(f_j)])
  w_j = np.concatenate([w_j, np.conj(w_j)])

  r = BarycentricRational(z_j, f_j, w_j)

  pol, res = r.polres()
  mask = np.abs(res) > 1e-13
  pol = pol[mask]
  res = res[mask]

  assert np.allclose(pol, np.array([-np.conj(pole), pole]))
  assert np.allclose(res, 1j*np.array([np.conj(residues[0]), residues[0]]))
  print(f"poles: {pol}, residues: {res}")

if __name__ == "__main__":
  test_lorentz_aaa()
