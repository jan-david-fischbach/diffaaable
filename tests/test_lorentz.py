from baryrat import BarycentricRational
import jax.numpy as np
from diffaaable.lorentz import lorentz_aaa


def f_test(z, res, poles):
  f = np.zeros_like(z)
  for r, p in zip(res, poles):
    f += 1j*r/(z-p) + 1j*np.conj(r)/(z+np.conj(p))
  return f

z_k = np.linspace(-3, 5, 100) + 0.05j

def test_lorentz_aaa():
  poles = np.array([2+0.1j, 2.7+0.3j])
  residues = np.array([1j, 2])
  z_j, f_j, w_j, z_n, errors = lorentz_aaa(z_k, f_test(z_k, residues, poles),
                                           mmax=10, return_errors=True)

  print(f"{errors=}")

  r = BarycentricRational(z_j, f_j, w_j)

  pol, res = r.polres()
  assert np.allclose(np.sort_complex(z_n), np.sort_complex(pol))

  for pole, residue in zip(poles, residues):
    assert np.any(np.isclose(pol, pole))
    assert np.any(np.isclose(pol, -np.conj(pole)))

    assert np.any(np.isclose(res, 1j*residue))
    assert np.any(np.isclose(res, 1j*np.conj(residue)))

  print(f"poles: {pol}, residues: {res}")

def test_lorentz_aaa_returns():
  f_k = f_test(z_k, np.array([1j]), np.array([2+0.1j]))
  z_j, f_j, w_j, z_n = lorentz_aaa(z_k, f_k, mmax=10)
  assert len(z_n) == len(z_j) - 1
  assert np.any(np.isclose(z_n, 2+0.1j))

if __name__ == "__main__":
  test_lorentz_aaa()
